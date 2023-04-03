/*
 * Copyright © 2015 Intel Corporation
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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include "vtn_private.h"
#include "nir/nir_vla.h"
#include "nir/nir_control_flow.h"
#include "nir/nir_constant_expressions.h"
#include "nir/nir_deref.h"
#include "spirv_info.h"

#include "util/format/u_format.h"
#include "util/u_math.h"
#include "util/u_string.h"
#include "util/u_debug.h"

#include <stdio.h>

#include "drivers/d3d12/d3d12_godot_nir_bridge.h"

#ifndef NDEBUG
uint32_t mesa_spirv_debug = 0;

static const struct debug_named_value mesa_spirv_debug_control[] = {
   DEBUG_NAMED_VALUE_END,
};

DEBUG_GET_ONCE_FLAGS_OPTION(mesa_spirv_debug, "MESA_SPIRV_DEBUG", mesa_spirv_debug_control, 0)

static enum nir_spirv_debug_level
vtn_default_log_level(void)
{
   enum nir_spirv_debug_level level = NIR_SPIRV_DEBUG_LEVEL_WARNING;
   const char *vtn_log_level_strings[] = {
      [NIR_SPIRV_DEBUG_LEVEL_WARNING] = "warning",
      [NIR_SPIRV_DEBUG_LEVEL_INFO]  = "info",
      [NIR_SPIRV_DEBUG_LEVEL_ERROR] = "error",
   };
   const char *str = getenv("MESA_SPIRV_LOG_LEVEL");

   if (str == NULL)
      return NIR_SPIRV_DEBUG_LEVEL_WARNING;

   for (int i = 0; i < ARRAY_SIZE(vtn_log_level_strings); i++) {
      if (strcasecmp(str, vtn_log_level_strings[i]) == 0) {
         level = i;
         break;
      }
   }

   return level;
}
#endif

void
vtn_log(struct vtn_builder *b, enum nir_spirv_debug_level level,
        size_t spirv_offset, const char *message)
{
   if (b->options->debug.func) {
      b->options->debug.func(b->options->debug.private_data,
                             level, spirv_offset, message);
   }

#ifndef NDEBUG
   static enum nir_spirv_debug_level default_level =
      NIR_SPIRV_DEBUG_LEVEL_INVALID;

   if (default_level == NIR_SPIRV_DEBUG_LEVEL_INVALID)
      default_level = vtn_default_log_level();

   if (level >= default_level)
      fprintf(stderr, "%s\n", message);
#endif
}

void
vtn_logf(struct vtn_builder *b, enum nir_spirv_debug_level level,
         size_t spirv_offset, const char *fmt, ...)
{
   va_list args;
   char *msg;

   va_start(args, fmt);
   msg = ralloc_vasprintf(NULL, fmt, args);
   va_end(args);

   vtn_log(b, level, spirv_offset, msg);

   ralloc_free(msg);
}

static void
vtn_log_err(struct vtn_builder *b,
            enum nir_spirv_debug_level level, const char *prefix,
            const char *file, unsigned line,
            const char *fmt, va_list args)
{
   char *msg;

   msg = ralloc_strdup(NULL, prefix);

#ifndef NDEBUG
   ralloc_asprintf_append(&msg, "    In file %s:%u\n", file, line);
#endif

   ralloc_asprintf_append(&msg, "    ");

   ralloc_vasprintf_append(&msg, fmt, args);

   ralloc_asprintf_append(&msg, "\n    %zu bytes into the SPIR-V binary",
                          b->spirv_offset);

   if (b->file) {
      ralloc_asprintf_append(&msg,
                             "\n    in SPIR-V source file %s, line %d, col %d",
                             b->file, b->line, b->col);
   }

   vtn_log(b, level, b->spirv_offset, msg);

   ralloc_free(msg);
}

static void
vtn_dump_shader(struct vtn_builder *b, const char *path, const char *prefix)
{
   static int idx = 0;

   char filename[1024];
   int len = snprintf(filename, sizeof(filename), "%s/%s-%d.spirv",
                      path, prefix, idx++);
   if (len < 0 || len >= sizeof(filename))
      return;

   FILE *f = fopen(filename, "w");
   if (f == NULL)
      return;

   fwrite(b->spirv, sizeof(*b->spirv), b->spirv_word_count, f);
   fclose(f);

   vtn_info("SPIR-V shader dumped to %s", filename);
}

void
_vtn_warn(struct vtn_builder *b, const char *file, unsigned line,
          const char *fmt, ...)
{
   va_list args;

   va_start(args, fmt);
   vtn_log_err(b, NIR_SPIRV_DEBUG_LEVEL_WARNING, "SPIR-V WARNING:\n",
               file, line, fmt, args);
   va_end(args);
}

void
_vtn_err(struct vtn_builder *b, const char *file, unsigned line,
          const char *fmt, ...)
{
   va_list args;

   va_start(args, fmt);
   vtn_log_err(b, NIR_SPIRV_DEBUG_LEVEL_ERROR, "SPIR-V ERROR:\n",
               file, line, fmt, args);
   va_end(args);
}

void
_vtn_fail(struct vtn_builder *b, const char *file, unsigned line,
          const char *fmt, ...)
{
   va_list args;

   va_start(args, fmt);
   vtn_log_err(b, NIR_SPIRV_DEBUG_LEVEL_ERROR, "SPIR-V parsing FAILED:\n",
               file, line, fmt, args);
   va_end(args);

   const char *dump_path = getenv("MESA_SPIRV_FAIL_DUMP_PATH");
   if (dump_path)
      vtn_dump_shader(b, dump_path, "fail");

#ifndef NDEBUG
   os_break();
#endif

   vtn_longjmp(b->fail_jump, 1);
}

static struct vtn_ssa_value *
vtn_undef_ssa_value(struct vtn_builder *b, const struct glsl_type *type)
{
   struct vtn_ssa_value *val = rzalloc(b, struct vtn_ssa_value);
   val->type = glsl_get_bare_type(type);

   if (glsl_type_is_vector_or_scalar(type)) {
      unsigned num_components = glsl_get_vector_elements(val->type);
      unsigned bit_size = glsl_get_bit_size(val->type);
      val->def = nir_ssa_undef(&b->nb, num_components, bit_size);
   } else {
      unsigned elems = glsl_get_length(val->type);
      val->elems = ralloc_array(b, struct vtn_ssa_value *, elems);
      if (glsl_type_is_array_or_matrix(type)) {
         const struct glsl_type *elem_type = glsl_get_array_element(type);
         for (unsigned i = 0; i < elems; i++)
            val->elems[i] = vtn_undef_ssa_value(b, elem_type);
      } else {
         vtn_assert(glsl_type_is_struct_or_ifc(type));
         for (unsigned i = 0; i < elems; i++) {
            const struct glsl_type *elem_type = glsl_get_struct_field(type, i);
            val->elems[i] = vtn_undef_ssa_value(b, elem_type);
         }
      }
   }

   return val;
}

struct vtn_ssa_value *
vtn_const_ssa_value(struct vtn_builder *b, nir_constant *constant,
                    const struct glsl_type *type)
{
   struct hash_entry *entry = _mesa_hash_table_search(b->const_table, constant);

   if (entry)
      return entry->data;

   struct vtn_ssa_value *val = rzalloc(b, struct vtn_ssa_value);
   val->type = glsl_get_bare_type(type);

   if (glsl_type_is_vector_or_scalar(type)) {
      unsigned num_components = glsl_get_vector_elements(val->type);
      unsigned bit_size = glsl_get_bit_size(type);
      nir_load_const_instr *load =
         nir_load_const_instr_create(b->shader, num_components, bit_size);

      memcpy(load->value, constant->values,
             sizeof(nir_const_value) * num_components);

      nir_instr_insert_before_cf_list(&b->nb.impl->body, &load->instr);
      val->def = &load->def;
   } else {
      unsigned elems = glsl_get_length(val->type);
      val->elems = ralloc_array(b, struct vtn_ssa_value *, elems);
      if (glsl_type_is_array_or_matrix(type)) {
         const struct glsl_type *elem_type = glsl_get_array_element(type);
         for (unsigned i = 0; i < elems; i++) {
            val->elems[i] = vtn_const_ssa_value(b, constant->elements[i],
                                                elem_type);
         }
      } else {
         vtn_assert(glsl_type_is_struct_or_ifc(type));
         for (unsigned i = 0; i < elems; i++) {
            const struct glsl_type *elem_type = glsl_get_struct_field(type, i);
            val->elems[i] = vtn_const_ssa_value(b, constant->elements[i],
                                                elem_type);
         }
      }
   }

   return val;
}

struct vtn_ssa_value *
vtn_ssa_value(struct vtn_builder *b, uint32_t value_id)
{
   struct vtn_value *val = vtn_untyped_value(b, value_id);
   switch (val->value_type) {
   case vtn_value_type_undef:
      return vtn_undef_ssa_value(b, val->type->type);

   case vtn_value_type_constant:
      return vtn_const_ssa_value(b, val->constant, val->type->type);

   case vtn_value_type_ssa:
      return val->ssa;

   case vtn_value_type_pointer:
      vtn_assert(val->pointer->ptr_type && val->pointer->ptr_type->type);
      struct vtn_ssa_value *ssa =
         vtn_create_ssa_value(b, val->pointer->ptr_type->type);
      ssa->def = vtn_pointer_to_ssa(b, val->pointer);
      return ssa;

   default:
      vtn_fail("Invalid type for an SSA value");
   }
}

struct vtn_value *
vtn_push_ssa_value(struct vtn_builder *b, uint32_t value_id,
                   struct vtn_ssa_value *ssa)
{
   struct vtn_type *type = vtn_get_value_type(b, value_id);

   /* See vtn_create_ssa_value */
   vtn_fail_if(ssa->type != glsl_get_bare_type(type->type),
               "Type mismatch for SPIR-V SSA value");

   struct vtn_value *val;
   if (type->base_type == vtn_base_type_pointer) {
      val = vtn_push_pointer(b, value_id, vtn_pointer_from_ssa(b, ssa->def, type));
   } else {
      /* Don't trip the value_type_ssa check in vtn_push_value */
      val = vtn_push_value(b, value_id, vtn_value_type_invalid);
      val->value_type = vtn_value_type_ssa;
      val->ssa = ssa;
   }

   return val;
}

nir_ssa_def *
vtn_get_nir_ssa(struct vtn_builder *b, uint32_t value_id)
{
   struct vtn_ssa_value *ssa = vtn_ssa_value(b, value_id);
   vtn_fail_if(!glsl_type_is_vector_or_scalar(ssa->type),
               "Expected a vector or scalar type");
   return ssa->def;
}

struct vtn_value *
vtn_push_nir_ssa(struct vtn_builder *b, uint32_t value_id, nir_ssa_def *def)
{
   /* Types for all SPIR-V SSA values are set as part of a pre-pass so the
    * type will be valid by the time we get here.
    */
   struct vtn_type *type = vtn_get_value_type(b, value_id);
   vtn_fail_if(def->num_components != glsl_get_vector_elements(type->type) ||
               def->bit_size != glsl_get_bit_size(type->type),
               "Mismatch between NIR and SPIR-V type.");
   struct vtn_ssa_value *ssa = vtn_create_ssa_value(b, type->type);
   ssa->def = def;
   return vtn_push_ssa_value(b, value_id, ssa);
}

static enum gl_access_qualifier
spirv_to_gl_access_qualifier(struct vtn_builder *b,
                             SpvAccessQualifier access_qualifier)
{
   switch (access_qualifier) {
   case SpvAccessQualifierReadOnly:
      return ACCESS_NON_WRITEABLE;
   case SpvAccessQualifierWriteOnly:
      return ACCESS_NON_READABLE;
   case SpvAccessQualifierReadWrite:
      return 0;
   default:
      vtn_fail("Invalid image access qualifier");
   }
}

static nir_deref_instr *
vtn_get_image(struct vtn_builder *b, uint32_t value_id,
              enum gl_access_qualifier *access)
{
   struct vtn_type *type = vtn_get_value_type(b, value_id);
   vtn_assert(type->base_type == vtn_base_type_image);
   if (access)
      *access |= spirv_to_gl_access_qualifier(b, type->access_qualifier);
   nir_variable_mode mode = glsl_type_is_image(type->glsl_image) ?
                            nir_var_image : nir_var_uniform;
   return nir_build_deref_cast(&b->nb, vtn_get_nir_ssa(b, value_id),
                               mode, type->glsl_image, 0);
}

static void
vtn_push_image(struct vtn_builder *b, uint32_t value_id,
               nir_deref_instr *deref, bool propagate_non_uniform)
{
   struct vtn_type *type = vtn_get_value_type(b, value_id);
   vtn_assert(type->base_type == vtn_base_type_image);
   struct vtn_value *value = vtn_push_nir_ssa(b, value_id, &deref->dest.ssa);
   value->propagated_non_uniform = propagate_non_uniform;
}

static nir_deref_instr *
vtn_get_sampler(struct vtn_builder *b, uint32_t value_id)
{
   struct vtn_type *type = vtn_get_value_type(b, value_id);
   vtn_assert(type->base_type == vtn_base_type_sampler);
   return nir_build_deref_cast(&b->nb, vtn_get_nir_ssa(b, value_id),
                               nir_var_uniform, glsl_bare_sampler_type(), 0);
}

nir_ssa_def *
vtn_sampled_image_to_nir_ssa(struct vtn_builder *b,
                             struct vtn_sampled_image si)
{
   return nir_vec2(&b->nb, &si.image->dest.ssa, &si.sampler->dest.ssa);
}

static void
vtn_push_sampled_image(struct vtn_builder *b, uint32_t value_id,
                       struct vtn_sampled_image si, bool propagate_non_uniform)
{
   struct vtn_type *type = vtn_get_value_type(b, value_id);
   vtn_assert(type->base_type == vtn_base_type_sampled_image);
   struct vtn_value *value = vtn_push_nir_ssa(b, value_id,
                                              vtn_sampled_image_to_nir_ssa(b, si));
   value->propagated_non_uniform = propagate_non_uniform;
}

static struct vtn_sampled_image
vtn_get_sampled_image(struct vtn_builder *b, uint32_t value_id)
{
   struct vtn_type *type = vtn_get_value_type(b, value_id);
   vtn_assert(type->base_type == vtn_base_type_sampled_image);
   nir_ssa_def *si_vec2 = vtn_get_nir_ssa(b, value_id);

   /* Even though this is a sampled image, we can end up here with a storage
    * image because OpenCL doesn't distinguish between the two.
    */
   const struct glsl_type *image_type = type->image->glsl_image;
   nir_variable_mode image_mode = glsl_type_is_image(image_type) ?
                                  nir_var_image : nir_var_uniform;

   struct vtn_sampled_image si = { NULL, };
   si.image = nir_build_deref_cast(&b->nb, nir_channel(&b->nb, si_vec2, 0),
                                   image_mode, image_type, 0);
   si.sampler = nir_build_deref_cast(&b->nb, nir_channel(&b->nb, si_vec2, 1),
                                     nir_var_uniform,
                                     glsl_bare_sampler_type(), 0);
   return si;
}

const char *
vtn_string_literal(struct vtn_builder *b, const uint32_t *words,
                   unsigned word_count, unsigned *words_used)
{
   /* From the SPIR-V spec:
    *
    *    "A string is interpreted as a nul-terminated stream of characters.
    *    The character set is Unicode in the UTF-8 encoding scheme. The UTF-8
    *    octets (8-bit bytes) are packed four per word, following the
    *    little-endian convention (i.e., the first octet is in the
    *    lowest-order 8 bits of the word). The final word contains the
    *    string’s nul-termination character (0), and all contents past the
    *    end of the string in the final word are padded with 0."
    *
    * On big-endian, we need to byte-swap.
    */
#if UTIL_ARCH_BIG_ENDIAN
   {
      uint32_t *copy = ralloc_array(b, uint32_t, word_count);
      for (unsigned i = 0; i < word_count; i++)
         copy[i] = util_bswap32(words[i]);
      words = copy;
   }
#endif

   const char *str = (char *)words;
   const char *end = memchr(str, 0, word_count * 4);
   vtn_fail_if(end == NULL, "String is not null-terminated");

   if (words_used)
      *words_used = DIV_ROUND_UP(end - str + 1, sizeof(*words));

   return str;
}

const uint32_t *
vtn_foreach_instruction(struct vtn_builder *b, const uint32_t *start,
                        const uint32_t *end, vtn_instruction_handler handler)
{
   b->file = NULL;
   b->line = -1;
   b->col = -1;

   const uint32_t *w = start;
   while (w < end) {
      SpvOp opcode = w[0] & SpvOpCodeMask;
      unsigned count = w[0] >> SpvWordCountShift;
      vtn_assert(count >= 1 && w + count <= end);

      b->spirv_offset = (uint8_t *)w - (uint8_t *)b->spirv;

      switch (opcode) {
      case SpvOpNop:
         break; /* Do nothing */

      case SpvOpLine:
         b->file = vtn_value(b, w[1], vtn_value_type_string)->str;
         b->line = w[2];
         b->col = w[3];
         break;

      case SpvOpNoLine:
         b->file = NULL;
         b->line = -1;
         b->col = -1;
         break;

      default:
         if (!handler(b, opcode, w, count))
            return w;
         break;
      }

      w += count;
   }

   b->spirv_offset = 0;
   b->file = NULL;
   b->line = -1;
   b->col = -1;

   assert(w == end);
   return w;
}

static bool
vtn_handle_non_semantic_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                                    const uint32_t *w, unsigned count)
{
   /* Do nothing. */
   return true;
}

static void
vtn_handle_extension(struct vtn_builder *b, SpvOp opcode,
                     const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpExtInstImport: {
      struct vtn_value *val = vtn_push_value(b, w[1], vtn_value_type_extension);
      const char *ext = vtn_string_literal(b, &w[2], count - 2, NULL);
      if (strcmp(ext, "GLSL.std.450") == 0) {
         val->ext_handler = vtn_handle_glsl450_instruction;
      } else if ((strcmp(ext, "SPV_AMD_gcn_shader") == 0)
                && (b->options && b->options->caps.amd_gcn_shader)) {
         val->ext_handler = vtn_handle_amd_gcn_shader_instruction;
      } else if ((strcmp(ext, "SPV_AMD_shader_ballot") == 0)
                && (b->options && b->options->caps.amd_shader_ballot)) {
         val->ext_handler = vtn_handle_amd_shader_ballot_instruction;
      } else if ((strcmp(ext, "SPV_AMD_shader_trinary_minmax") == 0)
                && (b->options && b->options->caps.amd_trinary_minmax)) {
         val->ext_handler = vtn_handle_amd_shader_trinary_minmax_instruction;
      } else if ((strcmp(ext, "SPV_AMD_shader_explicit_vertex_parameter") == 0)
                && (b->options && b->options->caps.amd_shader_explicit_vertex_parameter)) {
         val->ext_handler = vtn_handle_amd_shader_explicit_vertex_parameter_instruction;
      } else if (strcmp(ext, "OpenCL.std") == 0) {
         val->ext_handler = vtn_handle_opencl_instruction;
      } else if (strstr(ext, "NonSemantic.") == ext) {
         val->ext_handler = vtn_handle_non_semantic_instruction;
      } else {
         vtn_fail("Unsupported extension: %s", ext);
      }
      break;
   }

   case SpvOpExtInst: {
      struct vtn_value *val = vtn_value(b, w[3], vtn_value_type_extension);
      bool handled = val->ext_handler(b, w[4], w, count);
      vtn_assert(handled);
      break;
   }

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }
}

static void
_foreach_decoration_helper(struct vtn_builder *b,
                           struct vtn_value *base_value,
                           int parent_member,
                           struct vtn_value *value,
                           vtn_decoration_foreach_cb cb, void *data)
{
   for (struct vtn_decoration *dec = value->decoration; dec; dec = dec->next) {
      int member;
      if (dec->scope == VTN_DEC_DECORATION) {
         member = parent_member;
      } else if (dec->scope >= VTN_DEC_STRUCT_MEMBER0) {
         vtn_fail_if(value->value_type != vtn_value_type_type ||
                     value->type->base_type != vtn_base_type_struct,
                     "OpMemberDecorate and OpGroupMemberDecorate are only "
                     "allowed on OpTypeStruct");
         /* This means we haven't recursed yet */
         assert(value == base_value);

         member = dec->scope - VTN_DEC_STRUCT_MEMBER0;

         vtn_fail_if(member >= base_value->type->length,
                     "OpMemberDecorate specifies member %d but the "
                     "OpTypeStruct has only %u members",
                     member, base_value->type->length);
      } else {
         /* Not a decoration */
         assert(dec->scope == VTN_DEC_EXECUTION_MODE ||
                dec->scope <= VTN_DEC_STRUCT_MEMBER_NAME0);
         continue;
      }

      if (dec->group) {
         assert(dec->group->value_type == vtn_value_type_decoration_group);
         _foreach_decoration_helper(b, base_value, member, dec->group,
                                    cb, data);
      } else {
         cb(b, base_value, member, dec, data);
      }
   }
}

/** Iterates (recursively if needed) over all of the decorations on a value
 *
 * This function iterates over all of the decorations applied to a given
 * value.  If it encounters a decoration group, it recurses into the group
 * and iterates over all of those decorations as well.
 */
void
vtn_foreach_decoration(struct vtn_builder *b, struct vtn_value *value,
                       vtn_decoration_foreach_cb cb, void *data)
{
   _foreach_decoration_helper(b, value, -1, value, cb, data);
}

void
vtn_foreach_execution_mode(struct vtn_builder *b, struct vtn_value *value,
                           vtn_execution_mode_foreach_cb cb, void *data)
{
   for (struct vtn_decoration *dec = value->decoration; dec; dec = dec->next) {
      if (dec->scope != VTN_DEC_EXECUTION_MODE)
         continue;

      assert(dec->group == NULL);
      cb(b, value, dec, data);
   }
}

void
vtn_handle_decoration(struct vtn_builder *b, SpvOp opcode,
                      const uint32_t *w, unsigned count)
{
   const uint32_t *w_end = w + count;
   const uint32_t target = w[1];
   w += 2;

   switch (opcode) {
   case SpvOpDecorationGroup:
      vtn_push_value(b, target, vtn_value_type_decoration_group);
      break;

   case SpvOpDecorate:
   case SpvOpDecorateId:
   case SpvOpMemberDecorate:
   case SpvOpDecorateString:
   case SpvOpMemberDecorateString:
   case SpvOpExecutionMode:
   case SpvOpExecutionModeId: {
      struct vtn_value *val = vtn_untyped_value(b, target);

      struct vtn_decoration *dec = rzalloc(b, struct vtn_decoration);
      switch (opcode) {
      case SpvOpDecorate:
      case SpvOpDecorateId:
      case SpvOpDecorateString:
         dec->scope = VTN_DEC_DECORATION;
         break;
      case SpvOpMemberDecorate:
      case SpvOpMemberDecorateString:
         dec->scope = VTN_DEC_STRUCT_MEMBER0 + *(w++);
         vtn_fail_if(dec->scope < VTN_DEC_STRUCT_MEMBER0, /* overflow */
                     "Member argument of OpMemberDecorate too large");
         break;
      case SpvOpExecutionMode:
      case SpvOpExecutionModeId:
         dec->scope = VTN_DEC_EXECUTION_MODE;
         break;
      default:
         unreachable("Invalid decoration opcode");
      }
      dec->decoration = *(w++);
      dec->num_operands = w_end - w;
      dec->operands = w;

      /* Link into the list */
      dec->next = val->decoration;
      val->decoration = dec;
      break;
   }

   case SpvOpMemberName: {
      struct vtn_value *val = vtn_untyped_value(b, target);
      struct vtn_decoration *dec = rzalloc(b, struct vtn_decoration);

      dec->scope = VTN_DEC_STRUCT_MEMBER_NAME0 - *(w++);

      dec->member_name = vtn_string_literal(b, w, w_end - w, NULL);

      dec->next = val->decoration;
      val->decoration = dec;
      break;
   }

   case SpvOpGroupMemberDecorate:
   case SpvOpGroupDecorate: {
      struct vtn_value *group =
         vtn_value(b, target, vtn_value_type_decoration_group);

      for (; w < w_end; w++) {
         struct vtn_value *val = vtn_untyped_value(b, *w);
         struct vtn_decoration *dec = rzalloc(b, struct vtn_decoration);

         dec->group = group;
         if (opcode == SpvOpGroupDecorate) {
            dec->scope = VTN_DEC_DECORATION;
         } else {
            dec->scope = VTN_DEC_STRUCT_MEMBER0 + *(++w);
            vtn_fail_if(dec->scope < 0, /* Check for overflow */
                        "Member argument of OpGroupMemberDecorate too large");
         }

         /* Link into the list */
         dec->next = val->decoration;
         val->decoration = dec;
      }
      break;
   }

   default:
      unreachable("Unhandled opcode");
   }
}

struct member_decoration_ctx {
   unsigned num_fields;
   struct glsl_struct_field *fields;
   struct vtn_type *type;
};

/**
 * Returns true if the given type contains a struct decorated Block or
 * BufferBlock
 */
bool
vtn_type_contains_block(struct vtn_builder *b, struct vtn_type *type)
{
   switch (type->base_type) {
   case vtn_base_type_array:
      return vtn_type_contains_block(b, type->array_element);
   case vtn_base_type_struct:
      if (type->block || type->buffer_block)
         return true;
      for (unsigned i = 0; i < type->length; i++) {
         if (vtn_type_contains_block(b, type->members[i]))
            return true;
      }
      return false;
   default:
      return false;
   }
}

/** Returns true if two types are "compatible", i.e. you can do an OpLoad,
 * OpStore, or OpCopyMemory between them without breaking anything.
 * Technically, the SPIR-V rules require the exact same type ID but this lets
 * us internally be a bit looser.
 */
bool
vtn_types_compatible(struct vtn_builder *b,
                     struct vtn_type *t1, struct vtn_type *t2)
{
   if (t1->id == t2->id)
      return true;

   if (t1->base_type != t2->base_type)
      return false;

   switch (t1->base_type) {
   case vtn_base_type_void:
   case vtn_base_type_scalar:
   case vtn_base_type_vector:
   case vtn_base_type_matrix:
   case vtn_base_type_image:
   case vtn_base_type_sampler:
   case vtn_base_type_sampled_image:
   case vtn_base_type_event:
      return t1->type == t2->type;

   case vtn_base_type_array:
      return t1->length == t2->length &&
             vtn_types_compatible(b, t1->array_element, t2->array_element);

   case vtn_base_type_pointer:
      return vtn_types_compatible(b, t1->deref, t2->deref);

   case vtn_base_type_struct:
      if (t1->length != t2->length)
         return false;

      for (unsigned i = 0; i < t1->length; i++) {
         if (!vtn_types_compatible(b, t1->members[i], t2->members[i]))
            return false;
      }
      return true;

   case vtn_base_type_accel_struct:
   case vtn_base_type_ray_query:
      return true;

   case vtn_base_type_function:
      /* This case shouldn't get hit since you can't copy around function
       * types.  Just require them to be identical.
       */
      return false;
   }

   vtn_fail("Invalid base type");
}

struct vtn_type *
vtn_type_without_array(struct vtn_type *type)
{
   while (type->base_type == vtn_base_type_array)
      type = type->array_element;
   return type;
}

/* does a shallow copy of a vtn_type */

static struct vtn_type *
vtn_type_copy(struct vtn_builder *b, struct vtn_type *src)
{
   struct vtn_type *dest = ralloc(b, struct vtn_type);
   *dest = *src;

   switch (src->base_type) {
   case vtn_base_type_void:
   case vtn_base_type_scalar:
   case vtn_base_type_vector:
   case vtn_base_type_matrix:
   case vtn_base_type_array:
   case vtn_base_type_pointer:
   case vtn_base_type_image:
   case vtn_base_type_sampler:
   case vtn_base_type_sampled_image:
   case vtn_base_type_event:
   case vtn_base_type_accel_struct:
   case vtn_base_type_ray_query:
      /* Nothing more to do */
      break;

   case vtn_base_type_struct:
      dest->members = ralloc_array(b, struct vtn_type *, src->length);
      memcpy(dest->members, src->members,
             src->length * sizeof(src->members[0]));

      dest->offsets = ralloc_array(b, unsigned, src->length);
      memcpy(dest->offsets, src->offsets,
             src->length * sizeof(src->offsets[0]));
      break;

   case vtn_base_type_function:
      dest->params = ralloc_array(b, struct vtn_type *, src->length);
      memcpy(dest->params, src->params, src->length * sizeof(src->params[0]));
      break;
   }

   return dest;
}

static bool
vtn_type_needs_explicit_layout(struct vtn_builder *b, struct vtn_type *type,
                               enum vtn_variable_mode mode)
{
   /* For OpenCL we never want to strip the info from the types, and it makes
    * type comparisons easier in later stages.
    */
   if (b->options->environment == NIR_SPIRV_OPENCL)
      return true;

   switch (mode) {
   case vtn_variable_mode_input:
   case vtn_variable_mode_output:
      /* Layout decorations kept because we need offsets for XFB arrays of
       * blocks.
       */
      return b->shader->info.has_transform_feedback_varyings;

   case vtn_variable_mode_ssbo:
   case vtn_variable_mode_phys_ssbo:
   case vtn_variable_mode_ubo:
   case vtn_variable_mode_push_constant:
   case vtn_variable_mode_shader_record:
      return true;

   case vtn_variable_mode_workgroup:
      return b->options->caps.workgroup_memory_explicit_layout;

   default:
      return false;
   }
}

const struct glsl_type *
vtn_type_get_nir_type(struct vtn_builder *b, struct vtn_type *type,
                      enum vtn_variable_mode mode)
{
   if (mode == vtn_variable_mode_atomic_counter) {
      vtn_fail_if(glsl_without_array(type->type) != glsl_uint_type(),
                  "Variables in the AtomicCounter storage class should be "
                  "(possibly arrays of arrays of) uint.");
      return glsl_type_wrap_in_arrays(glsl_atomic_uint_type(), type->type);
   }

   if (mode == vtn_variable_mode_uniform) {
      switch (type->base_type) {
      case vtn_base_type_array: {
         const struct glsl_type *elem_type =
            vtn_type_get_nir_type(b, type->array_element, mode);

         return glsl_array_type(elem_type, type->length,
                                glsl_get_explicit_stride(type->type));
      }

      case vtn_base_type_struct: {
         bool need_new_struct = false;
         const uint32_t num_fields = type->length;
         NIR_VLA(struct glsl_struct_field, fields, num_fields);
         for (unsigned i = 0; i < num_fields; i++) {
            fields[i] = *glsl_get_struct_field_data(type->type, i);
            const struct glsl_type *field_nir_type =
               vtn_type_get_nir_type(b, type->members[i], mode);
            if (fields[i].type != field_nir_type) {
               fields[i].type = field_nir_type;
               need_new_struct = true;
            }
         }
         if (need_new_struct) {
            if (glsl_type_is_interface(type->type)) {
               return glsl_interface_type(fields, num_fields,
                                          /* packing */ 0, false,
                                          glsl_get_type_name(type->type));
            } else {
               return glsl_struct_type(fields, num_fields,
                                       glsl_get_type_name(type->type),
                                       glsl_struct_type_is_packed(type->type));
            }
         } else {
            /* No changes, just pass it on */
            return type->type;
         }
      }

      case vtn_base_type_image:
         vtn_assert(glsl_type_is_texture(type->glsl_image));
         return type->glsl_image;

      case vtn_base_type_sampler:
         return glsl_bare_sampler_type();

      case vtn_base_type_sampled_image:
         return glsl_texture_type_to_sampler(type->image->glsl_image,
                                             false /* is_shadow */);

      default:
         return type->type;
      }
   }

   if (mode == vtn_variable_mode_image) {
      struct vtn_type *image_type = vtn_type_without_array(type);
      vtn_assert(image_type->base_type == vtn_base_type_image);
      return glsl_type_wrap_in_arrays(image_type->glsl_image, type->type);
   }

   /* Layout decorations are allowed but ignored in certain conditions,
    * to allow SPIR-V generators perform type deduplication.  Discard
    * unnecessary ones when passing to NIR.
    */
   if (!vtn_type_needs_explicit_layout(b, type, mode))
      return glsl_get_bare_type(type->type);

   return type->type;
}

static struct vtn_type *
mutable_matrix_member(struct vtn_builder *b, struct vtn_type *type, int member)
{
   type->members[member] = vtn_type_copy(b, type->members[member]);
   type = type->members[member];

   /* We may have an array of matrices.... Oh, joy! */
   while (glsl_type_is_array(type->type)) {
      type->array_element = vtn_type_copy(b, type->array_element);
      type = type->array_element;
   }

   vtn_assert(glsl_type_is_matrix(type->type));

   return type;
}

static void
vtn_handle_access_qualifier(struct vtn_builder *b, struct vtn_type *type,
                            int member, enum gl_access_qualifier access)
{
   type->members[member] = vtn_type_copy(b, type->members[member]);
   type = type->members[member];

   type->access |= access;
}

static void
array_stride_decoration_cb(struct vtn_builder *b,
                           struct vtn_value *val, int member,
                           const struct vtn_decoration *dec, void *void_ctx)
{
   struct vtn_type *type = val->type;

   if (dec->decoration == SpvDecorationArrayStride) {
      if (vtn_type_contains_block(b, type)) {
         vtn_warn("The ArrayStride decoration cannot be applied to an array "
                  "type which contains a structure type decorated Block "
                  "or BufferBlock");
         /* Ignore the decoration */
      } else {
         vtn_fail_if(dec->operands[0] == 0, "ArrayStride must be non-zero");
         type->stride = dec->operands[0];
      }
   }
}

static void
struct_member_decoration_cb(struct vtn_builder *b,
                            UNUSED struct vtn_value *val, int member,
                            const struct vtn_decoration *dec, void *void_ctx)
{
   struct member_decoration_ctx *ctx = void_ctx;

   if (member < 0)
      return;

   assert(member < ctx->num_fields);

   switch (dec->decoration) {
   case SpvDecorationRelaxedPrecision:
   case SpvDecorationUniform:
   case SpvDecorationUniformId:
      break; /* FIXME: Do nothing with this for now. */
   case SpvDecorationNonWritable:
      vtn_handle_access_qualifier(b, ctx->type, member, ACCESS_NON_WRITEABLE);
      break;
   case SpvDecorationNonReadable:
      vtn_handle_access_qualifier(b, ctx->type, member, ACCESS_NON_READABLE);
      break;
   case SpvDecorationVolatile:
      vtn_handle_access_qualifier(b, ctx->type, member, ACCESS_VOLATILE);
      break;
   case SpvDecorationCoherent:
      vtn_handle_access_qualifier(b, ctx->type, member, ACCESS_COHERENT);
      break;
   case SpvDecorationNoPerspective:
      ctx->fields[member].interpolation = INTERP_MODE_NOPERSPECTIVE;
      break;
   case SpvDecorationFlat:
      ctx->fields[member].interpolation = INTERP_MODE_FLAT;
      break;
   case SpvDecorationExplicitInterpAMD:
      ctx->fields[member].interpolation = INTERP_MODE_EXPLICIT;
      break;
   case SpvDecorationCentroid:
      ctx->fields[member].centroid = true;
      break;
   case SpvDecorationSample:
      ctx->fields[member].sample = true;
      break;
   case SpvDecorationStream:
      /* This is handled later by var_decoration_cb in vtn_variables.c */
      break;
   case SpvDecorationLocation:
      ctx->fields[member].location = dec->operands[0];
      break;
   case SpvDecorationComponent:
      break; /* FIXME: What should we do with these? */
   case SpvDecorationBuiltIn:
      ctx->type->members[member] = vtn_type_copy(b, ctx->type->members[member]);
      ctx->type->members[member]->is_builtin = true;
      ctx->type->members[member]->builtin = dec->operands[0];
      ctx->type->builtin_block = true;
      break;
   case SpvDecorationOffset:
      ctx->type->offsets[member] = dec->operands[0];
      ctx->fields[member].offset = dec->operands[0];
      break;
   case SpvDecorationMatrixStride:
      /* Handled as a second pass */
      break;
   case SpvDecorationColMajor:
      break; /* Nothing to do here.  Column-major is the default. */
   case SpvDecorationRowMajor:
      mutable_matrix_member(b, ctx->type, member)->row_major = true;
      break;

   case SpvDecorationPatch:
   case SpvDecorationPerPrimitiveNV:
   case SpvDecorationPerTaskNV:
   case SpvDecorationPerViewNV:
   case SpvDecorationInvariant: /* Silence this one to avoid warning spam. */
      break;

   case SpvDecorationSpecId:
   case SpvDecorationBlock:
   case SpvDecorationBufferBlock:
   case SpvDecorationArrayStride:
   case SpvDecorationGLSLShared:
   case SpvDecorationGLSLPacked:
   case SpvDecorationAliased:
   case SpvDecorationConstant:
   case SpvDecorationIndex:
   case SpvDecorationBinding:
   case SpvDecorationDescriptorSet:
   case SpvDecorationLinkageAttributes:
   case SpvDecorationNoContraction:
   case SpvDecorationInputAttachmentIndex:
   case SpvDecorationCPacked:
      vtn_warn("Decoration not allowed on struct members: %s",
               spirv_decoration_to_string(dec->decoration));
      break;

   case SpvDecorationRestrict:
      /* While "Restrict" is invalid for struct members, glslang incorrectly
       * generates it and it ends up hiding actual driver issues in a wall of
       * spam from deqp-vk.  Return it to the above block once the issue is
       * resolved.  https://github.com/KhronosGroup/glslang/issues/703
       */
      break;

   case SpvDecorationXfbBuffer:
   case SpvDecorationXfbStride:
      /* This is handled later by var_decoration_cb in vtn_variables.c */
      break;

   case SpvDecorationSaturatedConversion:
   case SpvDecorationFuncParamAttr:
   case SpvDecorationFPRoundingMode:
   case SpvDecorationFPFastMathMode:
   case SpvDecorationAlignment:
      if (b->shader->info.stage != MESA_SHADER_KERNEL) {
         vtn_warn("Decoration only allowed for CL-style kernels: %s",
                  spirv_decoration_to_string(dec->decoration));
      }
      break;

   case SpvDecorationUserSemantic:
   case SpvDecorationUserTypeGOOGLE:
      /* User semantic decorations can safely be ignored by the driver. */
      break;

   default:
      vtn_fail_with_decoration("Unhandled decoration", dec->decoration);
   }
}

/** Chases the array type all the way down to the tail and rewrites the
 * glsl_types to be based off the tail's glsl_type.
 */
static void
vtn_array_type_rewrite_glsl_type(struct vtn_type *type)
{
   if (type->base_type != vtn_base_type_array)
      return;

   vtn_array_type_rewrite_glsl_type(type->array_element);

   type->type = glsl_array_type(type->array_element->type,
                                type->length, type->stride);
}

/* Matrix strides are handled as a separate pass because we need to know
 * whether the matrix is row-major or not first.
 */
static void
struct_member_matrix_stride_cb(struct vtn_builder *b,
                               UNUSED struct vtn_value *val, int member,
                               const struct vtn_decoration *dec,
                               void *void_ctx)
{
   if (dec->decoration != SpvDecorationMatrixStride)
      return;

   vtn_fail_if(member < 0,
               "The MatrixStride decoration is only allowed on members "
               "of OpTypeStruct");
   vtn_fail_if(dec->operands[0] == 0, "MatrixStride must be non-zero");

   struct member_decoration_ctx *ctx = void_ctx;

   struct vtn_type *mat_type = mutable_matrix_member(b, ctx->type, member);
   if (mat_type->row_major) {
      mat_type->array_element = vtn_type_copy(b, mat_type->array_element);
      mat_type->stride = mat_type->array_element->stride;
      mat_type->array_element->stride = dec->operands[0];

      mat_type->type = glsl_explicit_matrix_type(mat_type->type,
                                                 dec->operands[0], true);
      mat_type->array_element->type = glsl_get_column_type(mat_type->type);
   } else {
      vtn_assert(mat_type->array_element->stride > 0);
      mat_type->stride = dec->operands[0];

      mat_type->type = glsl_explicit_matrix_type(mat_type->type,
                                                 dec->operands[0], false);
   }

   /* Now that we've replaced the glsl_type with a properly strided matrix
    * type, rewrite the member type so that it's an array of the proper kind
    * of glsl_type.
    */
   vtn_array_type_rewrite_glsl_type(ctx->type->members[member]);
   ctx->fields[member].type = ctx->type->members[member]->type;
}

static void
struct_packed_decoration_cb(struct vtn_builder *b,
                            struct vtn_value *val, int member,
                            const struct vtn_decoration *dec, void *void_ctx)
{
   vtn_assert(val->type->base_type == vtn_base_type_struct);
   if (dec->decoration == SpvDecorationCPacked) {
      if (b->shader->info.stage != MESA_SHADER_KERNEL) {
         vtn_warn("Decoration only allowed for CL-style kernels: %s",
                  spirv_decoration_to_string(dec->decoration));
      }
      val->type->packed = true;
   }
}

static void
struct_block_decoration_cb(struct vtn_builder *b,
                           struct vtn_value *val, int member,
                           const struct vtn_decoration *dec, void *ctx)
{
   if (member != -1)
      return;

   struct vtn_type *type = val->type;
   if (dec->decoration == SpvDecorationBlock)
      type->block = true;
   else if (dec->decoration == SpvDecorationBufferBlock)
      type->buffer_block = true;
}

static void
type_decoration_cb(struct vtn_builder *b,
                   struct vtn_value *val, int member,
                   const struct vtn_decoration *dec, UNUSED void *ctx)
{
   struct vtn_type *type = val->type;

   if (member != -1) {
      /* This should have been handled by OpTypeStruct */
      assert(val->type->base_type == vtn_base_type_struct);
      assert(member >= 0 && member < val->type->length);
      return;
   }

   switch (dec->decoration) {
   case SpvDecorationArrayStride:
      vtn_assert(type->base_type == vtn_base_type_array ||
                 type->base_type == vtn_base_type_pointer);
      break;
   case SpvDecorationBlock:
      vtn_assert(type->base_type == vtn_base_type_struct);
      vtn_assert(type->block);
      break;
   case SpvDecorationBufferBlock:
      vtn_assert(type->base_type == vtn_base_type_struct);
      vtn_assert(type->buffer_block);
      break;
   case SpvDecorationGLSLShared:
   case SpvDecorationGLSLPacked:
      /* Ignore these, since we get explicit offsets anyways */
      break;

   case SpvDecorationRowMajor:
   case SpvDecorationColMajor:
   case SpvDecorationMatrixStride:
   case SpvDecorationBuiltIn:
   case SpvDecorationNoPerspective:
   case SpvDecorationFlat:
   case SpvDecorationPatch:
   case SpvDecorationCentroid:
   case SpvDecorationSample:
   case SpvDecorationExplicitInterpAMD:
   case SpvDecorationVolatile:
   case SpvDecorationCoherent:
   case SpvDecorationNonWritable:
   case SpvDecorationNonReadable:
   case SpvDecorationUniform:
   case SpvDecorationUniformId:
   case SpvDecorationLocation:
   case SpvDecorationComponent:
   case SpvDecorationOffset:
   case SpvDecorationXfbBuffer:
   case SpvDecorationXfbStride:
   case SpvDecorationUserSemantic:
      vtn_warn("Decoration only allowed for struct members: %s",
               spirv_decoration_to_string(dec->decoration));
      break;

   case SpvDecorationStream:
      /* We don't need to do anything here, as stream is filled up when
       * aplying the decoration to a variable, just check that if it is not a
       * struct member, it should be a struct.
       */
      vtn_assert(type->base_type == vtn_base_type_struct);
      break;

   case SpvDecorationRelaxedPrecision:
   case SpvDecorationSpecId:
   case SpvDecorationInvariant:
   case SpvDecorationRestrict:
   case SpvDecorationAliased:
   case SpvDecorationConstant:
   case SpvDecorationIndex:
   case SpvDecorationBinding:
   case SpvDecorationDescriptorSet:
   case SpvDecorationLinkageAttributes:
   case SpvDecorationNoContraction:
   case SpvDecorationInputAttachmentIndex:
      vtn_warn("Decoration not allowed on types: %s",
               spirv_decoration_to_string(dec->decoration));
      break;

   case SpvDecorationCPacked:
      /* Handled when parsing a struct type, nothing to do here. */
      break;

   case SpvDecorationSaturatedConversion:
   case SpvDecorationFuncParamAttr:
   case SpvDecorationFPRoundingMode:
   case SpvDecorationFPFastMathMode:
   case SpvDecorationAlignment:
      vtn_warn("Decoration only allowed for CL-style kernels: %s",
               spirv_decoration_to_string(dec->decoration));
      break;

   case SpvDecorationUserTypeGOOGLE:
      /* User semantic decorations can safely be ignored by the driver. */
      break;

   default:
      vtn_fail_with_decoration("Unhandled decoration", dec->decoration);
   }
}

static unsigned
translate_image_format(struct vtn_builder *b, SpvImageFormat format)
{
   switch (format) {
   case SpvImageFormatUnknown:      return PIPE_FORMAT_NONE;
   case SpvImageFormatRgba32f:      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   case SpvImageFormatRgba16f:      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case SpvImageFormatR32f:         return PIPE_FORMAT_R32_FLOAT;
   case SpvImageFormatRgba8:        return PIPE_FORMAT_R8G8B8A8_UNORM;
   case SpvImageFormatRgba8Snorm:   return PIPE_FORMAT_R8G8B8A8_SNORM;
   case SpvImageFormatRg32f:        return PIPE_FORMAT_R32G32_FLOAT;
   case SpvImageFormatRg16f:        return PIPE_FORMAT_R16G16_FLOAT;
   case SpvImageFormatR11fG11fB10f: return PIPE_FORMAT_R11G11B10_FLOAT;
   case SpvImageFormatR16f:         return PIPE_FORMAT_R16_FLOAT;
   case SpvImageFormatRgba16:       return PIPE_FORMAT_R16G16B16A16_UNORM;
   case SpvImageFormatRgb10A2:      return PIPE_FORMAT_R10G10B10A2_UNORM;
   case SpvImageFormatRg16:         return PIPE_FORMAT_R16G16_UNORM;
   case SpvImageFormatRg8:          return PIPE_FORMAT_R8G8_UNORM;
   case SpvImageFormatR16:          return PIPE_FORMAT_R16_UNORM;
   case SpvImageFormatR8:           return PIPE_FORMAT_R8_UNORM;
   case SpvImageFormatRgba16Snorm:  return PIPE_FORMAT_R16G16B16A16_SNORM;
   case SpvImageFormatRg16Snorm:    return PIPE_FORMAT_R16G16_SNORM;
   case SpvImageFormatRg8Snorm:     return PIPE_FORMAT_R8G8_SNORM;
   case SpvImageFormatR16Snorm:     return PIPE_FORMAT_R16_SNORM;
   case SpvImageFormatR8Snorm:      return PIPE_FORMAT_R8_SNORM;
   case SpvImageFormatRgba32i:      return PIPE_FORMAT_R32G32B32A32_SINT;
   case SpvImageFormatRgba16i:      return PIPE_FORMAT_R16G16B16A16_SINT;
   case SpvImageFormatRgba8i:       return PIPE_FORMAT_R8G8B8A8_SINT;
   case SpvImageFormatR32i:         return PIPE_FORMAT_R32_SINT;
   case SpvImageFormatRg32i:        return PIPE_FORMAT_R32G32_SINT;
   case SpvImageFormatRg16i:        return PIPE_FORMAT_R16G16_SINT;
   case SpvImageFormatRg8i:         return PIPE_FORMAT_R8G8_SINT;
   case SpvImageFormatR16i:         return PIPE_FORMAT_R16_SINT;
   case SpvImageFormatR8i:          return PIPE_FORMAT_R8_SINT;
   case SpvImageFormatRgba32ui:     return PIPE_FORMAT_R32G32B32A32_UINT;
   case SpvImageFormatRgba16ui:     return PIPE_FORMAT_R16G16B16A16_UINT;
   case SpvImageFormatRgba8ui:      return PIPE_FORMAT_R8G8B8A8_UINT;
   case SpvImageFormatR32ui:        return PIPE_FORMAT_R32_UINT;
   case SpvImageFormatRgb10a2ui:    return PIPE_FORMAT_R10G10B10A2_UINT;
   case SpvImageFormatRg32ui:       return PIPE_FORMAT_R32G32_UINT;
   case SpvImageFormatRg16ui:       return PIPE_FORMAT_R16G16_UINT;
   case SpvImageFormatRg8ui:        return PIPE_FORMAT_R8G8_UINT;
   case SpvImageFormatR16ui:        return PIPE_FORMAT_R16_UINT;
   case SpvImageFormatR8ui:         return PIPE_FORMAT_R8_UINT;
   case SpvImageFormatR64ui:        return PIPE_FORMAT_R64_UINT;
   case SpvImageFormatR64i:         return PIPE_FORMAT_R64_SINT;
   default:
      vtn_fail("Invalid image format: %s (%u)",
               spirv_imageformat_to_string(format), format);
   }
}

static void
vtn_handle_type(struct vtn_builder *b, SpvOp opcode,
                const uint32_t *w, unsigned count)
{
   struct vtn_value *val = NULL;

   /* In order to properly handle forward declarations, we have to defer
    * allocation for pointer types.
    */
   if (opcode != SpvOpTypePointer && opcode != SpvOpTypeForwardPointer) {
      val = vtn_push_value(b, w[1], vtn_value_type_type);
      vtn_fail_if(val->type != NULL,
                  "Only pointers can have forward declarations");
      val->type = rzalloc(b, struct vtn_type);
      val->type->id = w[1];
   }

   switch (opcode) {
   case SpvOpTypeVoid:
      val->type->base_type = vtn_base_type_void;
      val->type->type = glsl_void_type();
      break;
   case SpvOpTypeBool:
      val->type->base_type = vtn_base_type_scalar;
      val->type->type = glsl_bool_type();
      val->type->length = 1;
      break;
   case SpvOpTypeInt: {
      int bit_size = w[2];
      const bool signedness = w[3];
      vtn_fail_if(bit_size != 8 && bit_size != 16 &&
                  bit_size != 32 && bit_size != 64,
                  "Invalid int bit size: %u", bit_size);
      val->type->base_type = vtn_base_type_scalar;
      val->type->type = signedness ? glsl_intN_t_type(bit_size) :
                                     glsl_uintN_t_type(bit_size);
      val->type->length = 1;
      break;
   }

   case SpvOpTypeFloat: {
      int bit_size = w[2];
      val->type->base_type = vtn_base_type_scalar;
      vtn_fail_if(bit_size != 16 && bit_size != 32 && bit_size != 64,
                  "Invalid float bit size: %u", bit_size);
      val->type->type = glsl_floatN_t_type(bit_size);
      val->type->length = 1;
      break;
   }

   case SpvOpTypeVector: {
      struct vtn_type *base = vtn_get_type(b, w[2]);
      unsigned elems = w[3];

      vtn_fail_if(base->base_type != vtn_base_type_scalar,
                  "Base type for OpTypeVector must be a scalar");
      vtn_fail_if((elems < 2 || elems > 4) && (elems != 8) && (elems != 16),
                  "Invalid component count for OpTypeVector");

      val->type->base_type = vtn_base_type_vector;
      val->type->type = glsl_vector_type(glsl_get_base_type(base->type), elems);
      val->type->length = elems;
      val->type->stride = glsl_type_is_boolean(val->type->type)
         ? 4 : glsl_get_bit_size(base->type) / 8;
      val->type->array_element = base;
      break;
   }

   case SpvOpTypeMatrix: {
      struct vtn_type *base = vtn_get_type(b, w[2]);
      unsigned columns = w[3];

      vtn_fail_if(base->base_type != vtn_base_type_vector,
                  "Base type for OpTypeMatrix must be a vector");
      vtn_fail_if(columns < 2 || columns > 4,
                  "Invalid column count for OpTypeMatrix");

      val->type->base_type = vtn_base_type_matrix;
      val->type->type = glsl_matrix_type(glsl_get_base_type(base->type),
                                         glsl_get_vector_elements(base->type),
                                         columns);
      vtn_fail_if(glsl_type_is_error(val->type->type),
                  "Unsupported base type for OpTypeMatrix");
      assert(!glsl_type_is_error(val->type->type));
      val->type->length = columns;
      val->type->array_element = base;
      val->type->row_major = false;
      val->type->stride = 0;
      break;
   }

   case SpvOpTypeRuntimeArray:
   case SpvOpTypeArray: {
      struct vtn_type *array_element = vtn_get_type(b, w[2]);

      if (opcode == SpvOpTypeRuntimeArray) {
         /* A length of 0 is used to denote unsized arrays */
         val->type->length = 0;
      } else {
         val->type->length = vtn_constant_uint(b, w[3]);
      }

      val->type->base_type = vtn_base_type_array;
      val->type->array_element = array_element;

      vtn_foreach_decoration(b, val, array_stride_decoration_cb, NULL);
      val->type->type = glsl_array_type(array_element->type, val->type->length,
                                        val->type->stride);
      break;
   }

   case SpvOpTypeStruct: {
      unsigned num_fields = count - 2;
      val->type->base_type = vtn_base_type_struct;
      val->type->length = num_fields;
      val->type->members = ralloc_array(b, struct vtn_type *, num_fields);
      val->type->offsets = ralloc_array(b, unsigned, num_fields);
      val->type->packed = false;

      NIR_VLA(struct glsl_struct_field, fields, count);
      for (unsigned i = 0; i < num_fields; i++) {
         val->type->members[i] = vtn_get_type(b, w[i + 2]);
         const char *name = NULL;
         for (struct vtn_decoration *dec = val->decoration; dec; dec = dec->next) {
            if (dec->scope == VTN_DEC_STRUCT_MEMBER_NAME0 - i) {
               name = dec->member_name;
               break;
            }
         }
         if (!name)
            name = ralloc_asprintf(b, "field%d", i);

         fields[i] = (struct glsl_struct_field) {
            .type = val->type->members[i]->type,
            .name = name,
            .location = -1,
            .offset = -1,
         };
      }

      vtn_foreach_decoration(b, val, struct_packed_decoration_cb, NULL);

      struct member_decoration_ctx ctx = {
         .num_fields = num_fields,
         .fields = fields,
         .type = val->type
      };

      vtn_foreach_decoration(b, val, struct_member_decoration_cb, &ctx);

      /* Propagate access specifiers that are present on all members to the overall type */
      enum gl_access_qualifier overall_access = ACCESS_COHERENT | ACCESS_VOLATILE |
                                                ACCESS_NON_READABLE | ACCESS_NON_WRITEABLE;
      for (unsigned i = 0; i < num_fields; ++i)
         overall_access &= val->type->members[i]->access;
      val->type->access = overall_access;

      vtn_foreach_decoration(b, val, struct_member_matrix_stride_cb, &ctx);

      vtn_foreach_decoration(b, val, struct_block_decoration_cb, NULL);

      const char *name = val->name;

      if (val->type->block || val->type->buffer_block) {
         /* Packing will be ignored since types coming from SPIR-V are
          * explicitly laid out.
          */
         val->type->type = glsl_interface_type(fields, num_fields,
                                               /* packing */ 0, false,
                                               name ? name : "block");
      } else {
         val->type->type = glsl_struct_type(fields, num_fields,
                                            name ? name : "struct",
                                            val->type->packed);
      }
      break;
   }

   case SpvOpTypeFunction: {
      val->type->base_type = vtn_base_type_function;
      val->type->type = NULL;

      val->type->return_type = vtn_get_type(b, w[2]);

      const unsigned num_params = count - 3;
      val->type->length = num_params;
      val->type->params = ralloc_array(b, struct vtn_type *, num_params);
      for (unsigned i = 0; i < count - 3; i++) {
         val->type->params[i] = vtn_get_type(b, w[i + 3]);
      }
      break;
   }

   case SpvOpTypePointer:
   case SpvOpTypeForwardPointer: {
      /* We can't blindly push the value because it might be a forward
       * declaration.
       */
      val = vtn_untyped_value(b, w[1]);

      SpvStorageClass storage_class = w[2];

      vtn_fail_if(opcode == SpvOpTypeForwardPointer &&
                  b->shader->info.stage != MESA_SHADER_KERNEL &&
                  storage_class != SpvStorageClassPhysicalStorageBuffer,
                  "OpTypeForwardPointer is only allowed in Vulkan with "
                  "the PhysicalStorageBuffer storage class");

      struct vtn_type *deref_type = NULL;
      if (opcode == SpvOpTypePointer)
         deref_type = vtn_get_type(b, w[3]);

      bool has_forward_pointer = false;
      if (val->value_type == vtn_value_type_invalid) {
         val->value_type = vtn_value_type_type;
         val->type = rzalloc(b, struct vtn_type);
         val->type->id = w[1];
         val->type->base_type = vtn_base_type_pointer;
         val->type->storage_class = storage_class;

         /* These can actually be stored to nir_variables and used as SSA
          * values so they need a real glsl_type.
          */
         enum vtn_variable_mode mode = vtn_storage_class_to_mode(
            b, storage_class, deref_type, NULL);

         /* The deref type should only matter for the UniformConstant storage
          * class.  In particular, it should never matter for any storage
          * classes that are allowed in combination with OpTypeForwardPointer.
          */
         if (storage_class != SpvStorageClassUniform &&
             storage_class != SpvStorageClassUniformConstant) {
            assert(mode == vtn_storage_class_to_mode(b, storage_class,
                                                     NULL, NULL));
         }

         val->type->type = nir_address_format_to_glsl_type(
            vtn_mode_to_address_format(b, mode));
      } else {
         vtn_fail_if(val->type->storage_class != storage_class,
                     "The storage classes of an OpTypePointer and any "
                     "OpTypeForwardPointers that provide forward "
                     "declarations of it must match.");
         has_forward_pointer = true;
      }

      if (opcode == SpvOpTypePointer) {
         vtn_fail_if(val->type->deref != NULL,
                     "While OpTypeForwardPointer can be used to provide a "
                     "forward declaration of a pointer, OpTypePointer can "
                     "only be used once for a given id.");

         vtn_fail_if(has_forward_pointer &&
                     deref_type->base_type != vtn_base_type_struct,
                     "An OpTypePointer instruction must declare "
                     "Pointer Type to be a pointer to an OpTypeStruct.");

         val->type->deref = deref_type;

         /* Only certain storage classes use ArrayStride. */
         switch (storage_class) {
         case SpvStorageClassWorkgroup:
            if (!b->options->caps.workgroup_memory_explicit_layout)
               break;
            FALLTHROUGH;

         case SpvStorageClassUniform:
         case SpvStorageClassPushConstant:
         case SpvStorageClassStorageBuffer:
         case SpvStorageClassPhysicalStorageBuffer:
            vtn_foreach_decoration(b, val, array_stride_decoration_cb, NULL);
            break;

         default:
            /* Nothing to do. */
            break;
         }
      }
      break;
   }

   case SpvOpTypeImage: {
      val->type->base_type = vtn_base_type_image;

      /* Images are represented in NIR as a scalar SSA value that is the
       * result of a deref instruction.  An OpLoad on an OpTypeImage pointer
       * from UniformConstant memory just takes the NIR deref from the pointer
       * and turns it into an SSA value.
       */
      val->type->type = nir_address_format_to_glsl_type(
         vtn_mode_to_address_format(b, vtn_variable_mode_function));

      const struct vtn_type *sampled_type = vtn_get_type(b, w[2]);
      if (b->shader->info.stage == MESA_SHADER_KERNEL) {
         vtn_fail_if(sampled_type->base_type != vtn_base_type_void,
                     "Sampled type of OpTypeImage must be void for kernels");
      } else {
         vtn_fail_if(sampled_type->base_type != vtn_base_type_scalar,
                     "Sampled type of OpTypeImage must be a scalar");
         if (b->options->caps.image_atomic_int64) {
            vtn_fail_if(glsl_get_bit_size(sampled_type->type) != 32 &&
                        glsl_get_bit_size(sampled_type->type) != 64,
                        "Sampled type of OpTypeImage must be a 32 or 64-bit "
                        "scalar");
         } else {
            vtn_fail_if(glsl_get_bit_size(sampled_type->type) != 32,
                        "Sampled type of OpTypeImage must be a 32-bit scalar");
         }
      }

      enum glsl_sampler_dim dim;
      switch ((SpvDim)w[3]) {
      case SpvDim1D:       dim = GLSL_SAMPLER_DIM_1D;    break;
      case SpvDim2D:       dim = GLSL_SAMPLER_DIM_2D;    break;
      case SpvDim3D:       dim = GLSL_SAMPLER_DIM_3D;    break;
      case SpvDimCube:     dim = GLSL_SAMPLER_DIM_CUBE;  break;
      case SpvDimRect:     dim = GLSL_SAMPLER_DIM_RECT;  break;
      case SpvDimBuffer:   dim = GLSL_SAMPLER_DIM_BUF;   break;
      case SpvDimSubpassData: dim = GLSL_SAMPLER_DIM_SUBPASS; break;
      default:
         vtn_fail("Invalid SPIR-V image dimensionality: %s (%u)",
                  spirv_dim_to_string((SpvDim)w[3]), w[3]);
      }

      /* w[4]: as per Vulkan spec "Validation Rules within a Module",
       *       The “Depth” operand of OpTypeImage is ignored.
       */
      bool is_array = w[5];
      bool multisampled = w[6];
      unsigned sampled = w[7];
      SpvImageFormat format = w[8];

      if (count > 9)
         val->type->access_qualifier = w[9];
      else if (b->shader->info.stage == MESA_SHADER_KERNEL)
         /* Per the CL C spec: If no qualifier is provided, read_only is assumed. */
         val->type->access_qualifier = SpvAccessQualifierReadOnly;
      else
         val->type->access_qualifier = SpvAccessQualifierReadWrite;

      if (multisampled) {
         if (dim == GLSL_SAMPLER_DIM_2D)
            dim = GLSL_SAMPLER_DIM_MS;
         else if (dim == GLSL_SAMPLER_DIM_SUBPASS)
            dim = GLSL_SAMPLER_DIM_SUBPASS_MS;
         else
            vtn_fail("Unsupported multisampled image type");
      }

      val->type->image_format = translate_image_format(b, format);

      enum glsl_base_type sampled_base_type =
         glsl_get_base_type(sampled_type->type);
      if (sampled == 1) {
         val->type->glsl_image = glsl_texture_type(dim, is_array,
                                                   sampled_base_type);
      } else if (sampled == 2) {
         val->type->glsl_image = glsl_image_type(dim, is_array,
                                                 sampled_base_type);
      } else if (b->shader->info.stage == MESA_SHADER_KERNEL) {
         val->type->glsl_image = glsl_image_type(dim, is_array,
                                                 GLSL_TYPE_VOID);
      } else {
         vtn_fail("We need to know if the image will be sampled");
      }
      break;
   }

   case SpvOpTypeSampledImage: {
      val->type->base_type = vtn_base_type_sampled_image;
      val->type->image = vtn_get_type(b, w[2]);

      /* Sampled images are represented NIR as a vec2 SSA value where each
       * component is the result of a deref instruction.  The first component
       * is the image and the second is the sampler.  An OpLoad on an
       * OpTypeSampledImage pointer from UniformConstant memory just takes
       * the NIR deref from the pointer and duplicates it to both vector
       * components.
       */
      nir_address_format addr_format =
         vtn_mode_to_address_format(b, vtn_variable_mode_function);
      assert(nir_address_format_num_components(addr_format) == 1);
      unsigned bit_size = nir_address_format_bit_size(addr_format);
      assert(bit_size == 32 || bit_size == 64);

      enum glsl_base_type base_type =
         bit_size == 32 ? GLSL_TYPE_UINT : GLSL_TYPE_UINT64;
      val->type->type = glsl_vector_type(base_type, 2);
      break;
   }

   case SpvOpTypeSampler:
      val->type->base_type = vtn_base_type_sampler;

      /* Samplers are represented in NIR as a scalar SSA value that is the
       * result of a deref instruction.  An OpLoad on an OpTypeSampler pointer
       * from UniformConstant memory just takes the NIR deref from the pointer
       * and turns it into an SSA value.
       */
      val->type->type = nir_address_format_to_glsl_type(
         vtn_mode_to_address_format(b, vtn_variable_mode_function));
      break;

   case SpvOpTypeAccelerationStructureKHR:
      val->type->base_type = vtn_base_type_accel_struct;
      val->type->type = glsl_uint64_t_type();
      break;


   case SpvOpTypeOpaque: {
      val->type->base_type = vtn_base_type_struct;
      const char *name = vtn_string_literal(b, &w[2], count - 2, NULL);
      val->type->type = glsl_struct_type(NULL, 0, name, false);
      break;
   }

   case SpvOpTypeRayQueryKHR: {
      val->type->base_type = vtn_base_type_ray_query;
      val->type->type = glsl_uint64_t_type();
      /* We may need to run queries on helper invocations. Here the parser
       * doesn't go through a deeper analysis on whether the result of a query
       * will be used in derivative instructions.
       *
       * An implementation willing to optimize this would look through the IR
       * and check if any derivative instruction uses the result of a query
       * and drop this flag if not.
       */
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         val->type->access = ACCESS_INCLUDE_HELPERS;
      break;
   }

   case SpvOpTypeEvent:
      val->type->base_type = vtn_base_type_event;
      val->type->type = glsl_int_type();
      break;

   case SpvOpTypeDeviceEvent:
   case SpvOpTypeReserveId:
   case SpvOpTypeQueue:
   case SpvOpTypePipe:
   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }

   vtn_foreach_decoration(b, val, type_decoration_cb, NULL);

   if (val->type->base_type == vtn_base_type_struct &&
       (val->type->block || val->type->buffer_block)) {
      for (unsigned i = 0; i < val->type->length; i++) {
         vtn_fail_if(vtn_type_contains_block(b, val->type->members[i]),
                     "Block and BufferBlock decorations cannot decorate a "
                     "structure type that is nested at any level inside "
                     "another structure type decorated with Block or "
                     "BufferBlock.");
      }
   }
}

static nir_constant *
vtn_null_constant(struct vtn_builder *b, struct vtn_type *type)
{
   nir_constant *c = rzalloc(b, nir_constant);

   switch (type->base_type) {
   case vtn_base_type_scalar:
   case vtn_base_type_vector:
      /* Nothing to do here.  It's already initialized to zero */
      break;

   case vtn_base_type_pointer: {
      enum vtn_variable_mode mode = vtn_storage_class_to_mode(
         b, type->storage_class, type->deref, NULL);
      nir_address_format addr_format = vtn_mode_to_address_format(b, mode);

      const nir_const_value *null_value = nir_address_format_null_value(addr_format);
      memcpy(c->values, null_value,
             sizeof(nir_const_value) * nir_address_format_num_components(addr_format));
      break;
   }

   case vtn_base_type_void:
   case vtn_base_type_image:
   case vtn_base_type_sampler:
   case vtn_base_type_sampled_image:
   case vtn_base_type_function:
   case vtn_base_type_event:
      /* For those we have to return something but it doesn't matter what. */
      break;

   case vtn_base_type_matrix:
   case vtn_base_type_array:
      vtn_assert(type->length > 0);
      c->num_elements = type->length;
      c->elements = ralloc_array(b, nir_constant *, c->num_elements);

      c->elements[0] = vtn_null_constant(b, type->array_element);
      for (unsigned i = 1; i < c->num_elements; i++)
         c->elements[i] = c->elements[0];
      break;

   case vtn_base_type_struct:
      c->num_elements = type->length;
      c->elements = ralloc_array(b, nir_constant *, c->num_elements);
      for (unsigned i = 0; i < c->num_elements; i++)
         c->elements[i] = vtn_null_constant(b, type->members[i]);
      break;

   default:
      vtn_fail("Invalid type for null constant");
   }

   return c;
}

static void
spec_constant_decoration_cb(struct vtn_builder *b, struct vtn_value *val,
                            ASSERTED int member,
                            const struct vtn_decoration *dec, void *data)
{
   vtn_assert(member == -1);
   if (dec->decoration != SpvDecorationSpecId)
      return;

   val->is_sc = true;
   val->sc_id = dec->operands[0];
}

static void
handle_workgroup_size_decoration_cb(struct vtn_builder *b,
                                    struct vtn_value *val,
                                    ASSERTED int member,
                                    const struct vtn_decoration *dec,
                                    UNUSED void *data)
{
   vtn_assert(member == -1);
   if (dec->decoration != SpvDecorationBuiltIn ||
       dec->operands[0] != SpvBuiltInWorkgroupSize)
      return;

   vtn_assert(val->type->type == glsl_vector_type(GLSL_TYPE_UINT, 3));
   b->workgroup_size_builtin = val;
}

static void
vtn_handle_constant(struct vtn_builder *b, SpvOp opcode,
                    const uint32_t *w, unsigned count)
{
   struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_constant);
   if (opcode == SpvOpSpecConstantComposite || opcode == SpvOpSpecConstantOp) {
      val->value_type = vtn_value_type_ssa;
      val->ssa = NULL;
      return;
   }

   val->constant = rzalloc(b, nir_constant);
   switch (opcode) {
   case SpvOpConstantTrue:
   case SpvOpConstantFalse:
   case SpvOpSpecConstantTrue:
   case SpvOpSpecConstantFalse: {
      vtn_fail_if(val->type->type != glsl_bool_type(),
                  "Result type of %s must be OpTypeBool",
                  spirv_op_to_string(opcode));

      bool bval = (opcode == SpvOpConstantTrue ||
                   opcode == SpvOpSpecConstantTrue);

      nir_const_value u32val = nir_const_value_for_uint(bval, 32);

      if (opcode == SpvOpSpecConstantTrue ||
          opcode == SpvOpSpecConstantFalse)
         vtn_foreach_decoration(b, val, spec_constant_decoration_cb, NULL);

      val->constant->values[0].b = u32val.u32 != 0;
      break;
   }

   case SpvOpConstant:
   case SpvOpSpecConstant: {
      vtn_fail_if(val->type->base_type != vtn_base_type_scalar,
                  "Result type of %s must be a scalar",
                  spirv_op_to_string(opcode));
      int bit_size = glsl_get_bit_size(val->type->type);
      switch (bit_size) {
      case 64:
         val->constant->values[0].u64 = vtn_u64_literal(&w[3]);
         break;
      case 32:
         val->constant->values[0].u32 = w[3];
         break;
      case 16:
         val->constant->values[0].u16 = w[3];
         break;
      case 8:
         val->constant->values[0].u8 = w[3];
         break;
      default:
         vtn_fail("Unsupported SpvOpConstant bit size: %u", bit_size);
      }

      if (opcode == SpvOpSpecConstant)
         vtn_foreach_decoration(b, val, spec_constant_decoration_cb,
                                NULL);
      break;
   }

   case SpvOpConstantComposite: {
      unsigned elem_count = count - 3;
      vtn_fail_if(elem_count != val->type->length,
                  "%s has %u constituents, expected %u",
                  spirv_op_to_string(opcode), elem_count, val->type->length);

      nir_constant **elems = ralloc_array(b, nir_constant *, elem_count);
      val->is_undef_constant = true;
      for (unsigned i = 0; i < elem_count; i++) {
         struct vtn_value *elem_val = vtn_untyped_value(b, w[i + 3]);

         if (elem_val->value_type == vtn_value_type_constant) {
            elems[i] = elem_val->constant;
            val->is_undef_constant = val->is_undef_constant &&
                                     elem_val->is_undef_constant;
         } else {
            vtn_fail_if(elem_val->value_type != vtn_value_type_undef,
                        "only constants or undefs allowed for "
                        "SpvOpConstantComposite");
            /* to make it easier, just insert a NULL constant for now */
            elems[i] = vtn_null_constant(b, elem_val->type);
         }
      }

      switch (val->type->base_type) {
      case vtn_base_type_vector: {
         assert(glsl_type_is_vector(val->type->type));
         for (unsigned i = 0; i < elem_count; i++)
            val->constant->values[i] = elems[i]->values[0];
         break;
      }

      case vtn_base_type_matrix:
      case vtn_base_type_struct:
      case vtn_base_type_array:
         ralloc_steal(val->constant, elems);
         val->constant->num_elements = elem_count;
         val->constant->elements = elems;
         break;

      default:
         vtn_fail("Result type of %s must be a composite type",
                  spirv_op_to_string(opcode));
      }
      break;
   }

   case SpvOpConstantNull:
      val->constant = vtn_null_constant(b, val->type);
      val->is_null_constant = true;
      break;

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }

   /* Now that we have the value, update the workgroup size if needed */
   if (gl_shader_stage_uses_workgroup(b->entry_point_stage))
      vtn_foreach_decoration(b, val, handle_workgroup_size_decoration_cb,
                             NULL);
}

static void
vtn_split_barrier_semantics(struct vtn_builder *b,
                            SpvMemorySemanticsMask semantics,
                            SpvMemorySemanticsMask *before,
                            SpvMemorySemanticsMask *after)
{
   /* For memory semantics embedded in operations, we split them into up to
    * two barriers, to be added before and after the operation.  This is less
    * strict than if we propagated until the final backend stage, but still
    * result in correct execution.
    *
    * A further improvement could be pipe this information (and use!) into the
    * next compiler layers, at the expense of making the handling of barriers
    * more complicated.
    */

   *before = SpvMemorySemanticsMaskNone;
   *after = SpvMemorySemanticsMaskNone;

   SpvMemorySemanticsMask order_semantics =
      semantics & (SpvMemorySemanticsAcquireMask |
                   SpvMemorySemanticsReleaseMask |
                   SpvMemorySemanticsAcquireReleaseMask |
                   SpvMemorySemanticsSequentiallyConsistentMask);

   if (util_bitcount(order_semantics) > 1) {
      /* Old GLSLang versions incorrectly set all the ordering bits.  This was
       * fixed in c51287d744fb6e7e9ccc09f6f8451e6c64b1dad6 of glslang repo,
       * and it is in GLSLang since revision "SPIRV99.1321" (from Jul-2016).
       */
      vtn_warn("Multiple memory ordering semantics specified, "
               "assuming AcquireRelease.");
      order_semantics = SpvMemorySemanticsAcquireReleaseMask;
   }

   const SpvMemorySemanticsMask av_vis_semantics =
      semantics & (SpvMemorySemanticsMakeAvailableMask |
                   SpvMemorySemanticsMakeVisibleMask);

   const SpvMemorySemanticsMask storage_semantics =
      semantics & (SpvMemorySemanticsUniformMemoryMask |
                   SpvMemorySemanticsSubgroupMemoryMask |
                   SpvMemorySemanticsWorkgroupMemoryMask |
                   SpvMemorySemanticsCrossWorkgroupMemoryMask |
                   SpvMemorySemanticsAtomicCounterMemoryMask |
                   SpvMemorySemanticsImageMemoryMask |
                   SpvMemorySemanticsOutputMemoryMask);

   const SpvMemorySemanticsMask other_semantics =
      semantics & ~(order_semantics | av_vis_semantics | storage_semantics |
                    SpvMemorySemanticsVolatileMask);

   if (other_semantics)
      vtn_warn("Ignoring unhandled memory semantics: %u\n", other_semantics);

   /* SequentiallyConsistent is treated as AcquireRelease. */

   /* The RELEASE barrier happens BEFORE the operation, and it is usually
    * associated with a Store.  All the write operations with a matching
    * semantics will not be reordered after the Store.
    */
   if (order_semantics & (SpvMemorySemanticsReleaseMask |
                          SpvMemorySemanticsAcquireReleaseMask |
                          SpvMemorySemanticsSequentiallyConsistentMask)) {
      *before |= SpvMemorySemanticsReleaseMask | storage_semantics;
   }

   /* The ACQUIRE barrier happens AFTER the operation, and it is usually
    * associated with a Load.  All the operations with a matching semantics
    * will not be reordered before the Load.
    */
   if (order_semantics & (SpvMemorySemanticsAcquireMask |
                          SpvMemorySemanticsAcquireReleaseMask |
                          SpvMemorySemanticsSequentiallyConsistentMask)) {
      *after |= SpvMemorySemanticsAcquireMask | storage_semantics;
   }

   if (av_vis_semantics & SpvMemorySemanticsMakeVisibleMask)
      *before |= SpvMemorySemanticsMakeVisibleMask | storage_semantics;

   if (av_vis_semantics & SpvMemorySemanticsMakeAvailableMask)
      *after |= SpvMemorySemanticsMakeAvailableMask | storage_semantics;
}

static nir_memory_semantics
vtn_mem_semantics_to_nir_mem_semantics(struct vtn_builder *b,
                                       SpvMemorySemanticsMask semantics)
{
   nir_memory_semantics nir_semantics = 0;

   SpvMemorySemanticsMask order_semantics =
      semantics & (SpvMemorySemanticsAcquireMask |
                   SpvMemorySemanticsReleaseMask |
                   SpvMemorySemanticsAcquireReleaseMask |
                   SpvMemorySemanticsSequentiallyConsistentMask);

   if (util_bitcount(order_semantics) > 1) {
      /* Old GLSLang versions incorrectly set all the ordering bits.  This was
       * fixed in c51287d744fb6e7e9ccc09f6f8451e6c64b1dad6 of glslang repo,
       * and it is in GLSLang since revision "SPIRV99.1321" (from Jul-2016).
       */
      vtn_warn("Multiple memory ordering semantics bits specified, "
               "assuming AcquireRelease.");
      order_semantics = SpvMemorySemanticsAcquireReleaseMask;
   }

   switch (order_semantics) {
   case 0:
      /* Not an ordering barrier. */
      break;

   case SpvMemorySemanticsAcquireMask:
      nir_semantics = NIR_MEMORY_ACQUIRE;
      break;

   case SpvMemorySemanticsReleaseMask:
      nir_semantics = NIR_MEMORY_RELEASE;
      break;

   case SpvMemorySemanticsSequentiallyConsistentMask:
      FALLTHROUGH; /* Treated as AcquireRelease in Vulkan. */
   case SpvMemorySemanticsAcquireReleaseMask:
      nir_semantics = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE;
      break;

   default:
      unreachable("Invalid memory order semantics");
   }

   if (semantics & SpvMemorySemanticsMakeAvailableMask) {
      vtn_fail_if(!b->options->caps.vk_memory_model,
                  "To use MakeAvailable memory semantics the VulkanMemoryModel "
                  "capability must be declared.");
      nir_semantics |= NIR_MEMORY_MAKE_AVAILABLE;
   }

   if (semantics & SpvMemorySemanticsMakeVisibleMask) {
      vtn_fail_if(!b->options->caps.vk_memory_model,
                  "To use MakeVisible memory semantics the VulkanMemoryModel "
                  "capability must be declared.");
      nir_semantics |= NIR_MEMORY_MAKE_VISIBLE;
   }

   return nir_semantics;
}

static nir_variable_mode
vtn_mem_semantics_to_nir_var_modes(struct vtn_builder *b,
                                   SpvMemorySemanticsMask semantics)
{
   /* Vulkan Environment for SPIR-V says "SubgroupMemory, CrossWorkgroupMemory,
    * and AtomicCounterMemory are ignored".
    */
   if (b->options->environment == NIR_SPIRV_VULKAN) {
      semantics &= ~(SpvMemorySemanticsSubgroupMemoryMask |
                     SpvMemorySemanticsCrossWorkgroupMemoryMask |
                     SpvMemorySemanticsAtomicCounterMemoryMask);
   }

   nir_variable_mode modes = 0;
   if (semantics & SpvMemorySemanticsUniformMemoryMask) {
      modes |= nir_var_uniform |
               nir_var_mem_ubo |
               nir_var_mem_ssbo |
               nir_var_mem_global;
   }
   if (semantics & SpvMemorySemanticsImageMemoryMask)
      modes |= nir_var_image;
   if (semantics & SpvMemorySemanticsWorkgroupMemoryMask)
      modes |= nir_var_mem_shared;
   if (semantics & SpvMemorySemanticsCrossWorkgroupMemoryMask)
      modes |= nir_var_mem_global;
   if (semantics & SpvMemorySemanticsOutputMemoryMask) {
      modes |= nir_var_shader_out;

      if (b->shader->info.stage == MESA_SHADER_TASK)
         modes |= nir_var_mem_task_payload;
   }

   return modes;
}

static nir_scope
vtn_scope_to_nir_scope(struct vtn_builder *b, SpvScope scope)
{
   nir_scope nir_scope;
   switch (scope) {
   case SpvScopeDevice:
      vtn_fail_if(b->options->caps.vk_memory_model &&
                  !b->options->caps.vk_memory_model_device_scope,
                  "If the Vulkan memory model is declared and any instruction "
                  "uses Device scope, the VulkanMemoryModelDeviceScope "
                  "capability must be declared.");
      nir_scope = NIR_SCOPE_DEVICE;
      break;

   case SpvScopeQueueFamily:
      vtn_fail_if(!b->options->caps.vk_memory_model,
                  "To use Queue Family scope, the VulkanMemoryModel capability "
                  "must be declared.");
      nir_scope = NIR_SCOPE_QUEUE_FAMILY;
      break;

   case SpvScopeWorkgroup:
      nir_scope = NIR_SCOPE_WORKGROUP;
      break;

   case SpvScopeSubgroup:
      nir_scope = NIR_SCOPE_SUBGROUP;
      break;

   case SpvScopeInvocation:
      nir_scope = NIR_SCOPE_INVOCATION;
      break;

   case SpvScopeShaderCallKHR:
      nir_scope = NIR_SCOPE_SHADER_CALL;
      break;

   default:
      vtn_fail("Invalid memory scope");
   }

   return nir_scope;
}

static void
vtn_emit_scoped_control_barrier(struct vtn_builder *b, SpvScope exec_scope,
                                SpvScope mem_scope,
                                SpvMemorySemanticsMask semantics)
{
   nir_memory_semantics nir_semantics =
      vtn_mem_semantics_to_nir_mem_semantics(b, semantics);
   nir_variable_mode modes = vtn_mem_semantics_to_nir_var_modes(b, semantics);
   nir_scope nir_exec_scope = vtn_scope_to_nir_scope(b, exec_scope);

   /* Memory semantics is optional for OpControlBarrier. */
   nir_scope nir_mem_scope;
   if (nir_semantics == 0 || modes == 0)
      nir_mem_scope = NIR_SCOPE_NONE;
   else
      nir_mem_scope = vtn_scope_to_nir_scope(b, mem_scope);

   nir_scoped_barrier(&b->nb, .execution_scope=nir_exec_scope, .memory_scope=nir_mem_scope,
                              .memory_semantics=nir_semantics, .memory_modes=modes);
}

static void
vtn_emit_scoped_memory_barrier(struct vtn_builder *b, SpvScope scope,
                               SpvMemorySemanticsMask semantics)
{
   nir_variable_mode modes = vtn_mem_semantics_to_nir_var_modes(b, semantics);
   nir_memory_semantics nir_semantics =
      vtn_mem_semantics_to_nir_mem_semantics(b, semantics);

   /* No barrier to add. */
   if (nir_semantics == 0 || modes == 0)
      return;

   nir_scoped_barrier(&b->nb, .memory_scope=vtn_scope_to_nir_scope(b, scope),
                              .memory_semantics=nir_semantics,
                              .memory_modes=modes);
}

struct vtn_ssa_value *
vtn_create_ssa_value(struct vtn_builder *b, const struct glsl_type *type)
{
   /* Always use bare types for SSA values for a couple of reasons:
    *
    *  1. Code which emits deref chains should never listen to the explicit
    *     layout information on the SSA value if any exists.  If we've
    *     accidentally been relying on this, we want to find those bugs.
    *
    *  2. We want to be able to quickly check that an SSA value being assigned
    *     to a SPIR-V value has the right type.  Using bare types everywhere
    *     ensures that we can pointer-compare.
    */
   struct vtn_ssa_value *val = rzalloc(b, struct vtn_ssa_value);
   val->type = glsl_get_bare_type(type);


   if (!glsl_type_is_vector_or_scalar(type)) {
      unsigned elems = glsl_get_length(val->type);
      val->elems = ralloc_array(b, struct vtn_ssa_value *, elems);
      if (glsl_type_is_array_or_matrix(type)) {
         const struct glsl_type *elem_type = glsl_get_array_element(type);
         for (unsigned i = 0; i < elems; i++)
            val->elems[i] = vtn_create_ssa_value(b, elem_type);
      } else {
         vtn_assert(glsl_type_is_struct_or_ifc(type));
         for (unsigned i = 0; i < elems; i++) {
            const struct glsl_type *elem_type = glsl_get_struct_field(type, i);
            val->elems[i] = vtn_create_ssa_value(b, elem_type);
         }
      }
   }

   return val;
}

static nir_tex_src
vtn_tex_src(struct vtn_builder *b, unsigned index, nir_tex_src_type type)
{
   nir_tex_src src;
   src.src = nir_src_for_ssa(vtn_get_nir_ssa(b, index));
   src.src_type = type;
   return src;
}

static uint32_t
image_operand_arg(struct vtn_builder *b, const uint32_t *w, uint32_t count,
                  uint32_t mask_idx, SpvImageOperandsMask op)
{
   static const SpvImageOperandsMask ops_with_arg =
      SpvImageOperandsBiasMask |
      SpvImageOperandsLodMask |
      SpvImageOperandsGradMask |
      SpvImageOperandsConstOffsetMask |
      SpvImageOperandsOffsetMask |
      SpvImageOperandsConstOffsetsMask |
      SpvImageOperandsSampleMask |
      SpvImageOperandsMinLodMask |
      SpvImageOperandsMakeTexelAvailableMask |
      SpvImageOperandsMakeTexelVisibleMask;

   assert(util_bitcount(op) == 1);
   assert(w[mask_idx] & op);
   assert(op & ops_with_arg);

   uint32_t idx = util_bitcount(w[mask_idx] & (op - 1) & ops_with_arg) + 1;

   /* Adjust indices for operands with two arguments. */
   static const SpvImageOperandsMask ops_with_two_args =
      SpvImageOperandsGradMask;
   idx += util_bitcount(w[mask_idx] & (op - 1) & ops_with_two_args);

   idx += mask_idx;

   vtn_fail_if(idx + (op & ops_with_two_args ? 1 : 0) >= count,
               "Image op claims to have %s but does not enough "
               "following operands", spirv_imageoperands_to_string(op));

   return idx;
}

static void
non_uniform_decoration_cb(struct vtn_builder *b,
                          struct vtn_value *val, int member,
                          const struct vtn_decoration *dec, void *void_ctx)
{
   enum gl_access_qualifier *access = void_ctx;
   switch (dec->decoration) {
   case SpvDecorationNonUniformEXT:
      *access |= ACCESS_NON_UNIFORM;
      break;

   default:
      break;
   }
}

/* Apply SignExtend/ZeroExtend operands to get the actual result type for
 * image read/sample operations and source type for write operations.
 */
static nir_alu_type
get_image_type(struct vtn_builder *b, nir_alu_type type, unsigned operands)
{
   unsigned extend_operands =
      operands & (SpvImageOperandsSignExtendMask | SpvImageOperandsZeroExtendMask);
   vtn_fail_if(nir_alu_type_get_base_type(type) == nir_type_float && extend_operands,
               "SignExtend/ZeroExtend used on floating-point texel type");
   vtn_fail_if(extend_operands ==
               (SpvImageOperandsSignExtendMask | SpvImageOperandsZeroExtendMask),
               "SignExtend and ZeroExtend both specified");

   if (operands & SpvImageOperandsSignExtendMask)
      return nir_type_int | nir_alu_type_get_type_size(type);
   if (operands & SpvImageOperandsZeroExtendMask)
      return nir_type_uint | nir_alu_type_get_type_size(type);

   return type;
}

static void
vtn_handle_texture(struct vtn_builder *b, SpvOp opcode,
                   const uint32_t *w, unsigned count)
{
   if (opcode == SpvOpSampledImage) {
      struct vtn_sampled_image si = {
         .image = vtn_get_image(b, w[3], NULL),
         .sampler = vtn_get_sampler(b, w[4]),
      };

      enum gl_access_qualifier access = 0;
      vtn_foreach_decoration(b, vtn_untyped_value(b, w[3]),
                             non_uniform_decoration_cb, &access);
      vtn_foreach_decoration(b, vtn_untyped_value(b, w[4]),
                             non_uniform_decoration_cb, &access);

      vtn_push_sampled_image(b, w[2], si, access & ACCESS_NON_UNIFORM);
      return;
   } else if (opcode == SpvOpImage) {
      struct vtn_sampled_image si = vtn_get_sampled_image(b, w[3]);

      enum gl_access_qualifier access = 0;
      vtn_foreach_decoration(b, vtn_untyped_value(b, w[3]),
                             non_uniform_decoration_cb, &access);

      vtn_push_image(b, w[2], si.image, access & ACCESS_NON_UNIFORM);
      return;
   } else if (opcode == SpvOpImageSparseTexelsResident) {
      nir_ssa_def *code = vtn_get_nir_ssa(b, w[3]);
      vtn_push_nir_ssa(b, w[2], nir_is_sparse_texels_resident(&b->nb, 1, code));
      return;
   }

   nir_deref_instr *image = NULL, *sampler = NULL;
   struct vtn_value *sampled_val = vtn_untyped_value(b, w[3]);
   if (sampled_val->type->base_type == vtn_base_type_sampled_image) {
      struct vtn_sampled_image si = vtn_get_sampled_image(b, w[3]);
      image = si.image;
      sampler = si.sampler;
   } else {
      image = vtn_get_image(b, w[3], NULL);
   }

   const enum glsl_sampler_dim sampler_dim = glsl_get_sampler_dim(image->type);
   const bool is_array = glsl_sampler_type_is_array(image->type);
   nir_alu_type dest_type = nir_type_invalid;

   /* Figure out the base texture operation */
   nir_texop texop;
   switch (opcode) {
   case SpvOpImageSampleImplicitLod:
   case SpvOpImageSparseSampleImplicitLod:
   case SpvOpImageSampleDrefImplicitLod:
   case SpvOpImageSparseSampleDrefImplicitLod:
   case SpvOpImageSampleProjImplicitLod:
   case SpvOpImageSampleProjDrefImplicitLod:
      texop = nir_texop_tex;
      break;

   case SpvOpImageSampleExplicitLod:
   case SpvOpImageSparseSampleExplicitLod:
   case SpvOpImageSampleDrefExplicitLod:
   case SpvOpImageSparseSampleDrefExplicitLod:
   case SpvOpImageSampleProjExplicitLod:
   case SpvOpImageSampleProjDrefExplicitLod:
      texop = nir_texop_txl;
      break;

   case SpvOpImageFetch:
   case SpvOpImageSparseFetch:
      if (sampler_dim == GLSL_SAMPLER_DIM_MS) {
         texop = nir_texop_txf_ms;
      } else {
         texop = nir_texop_txf;
      }
      break;

   case SpvOpImageGather:
   case SpvOpImageSparseGather:
   case SpvOpImageDrefGather:
   case SpvOpImageSparseDrefGather:
      texop = nir_texop_tg4;
      break;

   case SpvOpImageQuerySizeLod:
   case SpvOpImageQuerySize:
      texop = nir_texop_txs;
      dest_type = nir_type_int32;
      break;

   case SpvOpImageQueryLod:
      texop = nir_texop_lod;
      dest_type = nir_type_float32;
      break;

   case SpvOpImageQueryLevels:
      texop = nir_texop_query_levels;
      dest_type = nir_type_int32;
      break;

   case SpvOpImageQuerySamples:
      texop = nir_texop_texture_samples;
      dest_type = nir_type_int32;
      break;

   case SpvOpFragmentFetchAMD:
      texop = nir_texop_fragment_fetch_amd;
      break;

   case SpvOpFragmentMaskFetchAMD:
      texop = nir_texop_fragment_mask_fetch_amd;
      dest_type = nir_type_uint32;
      break;

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }

   nir_tex_src srcs[10]; /* 10 should be enough */
   nir_tex_src *p = srcs;

   p->src = nir_src_for_ssa(&image->dest.ssa);
   p->src_type = nir_tex_src_texture_deref;
   p++;

   switch (texop) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_tg4:
   case nir_texop_lod:
      vtn_fail_if(sampler == NULL,
                  "%s requires an image of type OpTypeSampledImage",
                  spirv_op_to_string(opcode));
      p->src = nir_src_for_ssa(&sampler->dest.ssa);
      p->src_type = nir_tex_src_sampler_deref;
      p++;
      break;
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_txs:
   case nir_texop_query_levels:
   case nir_texop_texture_samples:
   case nir_texop_samples_identical:
   case nir_texop_fragment_fetch_amd:
   case nir_texop_fragment_mask_fetch_amd:
      /* These don't */
      break;
   case nir_texop_txf_ms_fb:
      vtn_fail("unexpected nir_texop_txf_ms_fb");
      break;
   case nir_texop_txf_ms_mcs_intel:
      vtn_fail("unexpected nir_texop_txf_ms_mcs");
      break;
   case nir_texop_tex_prefetch:
      vtn_fail("unexpected nir_texop_tex_prefetch");
      break;
   case nir_texop_descriptor_amd:
   case nir_texop_sampler_descriptor_amd:
      vtn_fail("unexpected nir_texop_*descriptor_amd");
      break;
   }

   unsigned idx = 4;

   struct nir_ssa_def *coord;
   unsigned coord_components;
   switch (opcode) {
   case SpvOpImageSampleImplicitLod:
   case SpvOpImageSparseSampleImplicitLod:
   case SpvOpImageSampleExplicitLod:
   case SpvOpImageSparseSampleExplicitLod:
   case SpvOpImageSampleDrefImplicitLod:
   case SpvOpImageSparseSampleDrefImplicitLod:
   case SpvOpImageSampleDrefExplicitLod:
   case SpvOpImageSparseSampleDrefExplicitLod:
   case SpvOpImageSampleProjImplicitLod:
   case SpvOpImageSampleProjExplicitLod:
   case SpvOpImageSampleProjDrefImplicitLod:
   case SpvOpImageSampleProjDrefExplicitLod:
   case SpvOpImageFetch:
   case SpvOpImageSparseFetch:
   case SpvOpImageGather:
   case SpvOpImageSparseGather:
   case SpvOpImageDrefGather:
   case SpvOpImageSparseDrefGather:
   case SpvOpImageQueryLod:
   case SpvOpFragmentFetchAMD:
   case SpvOpFragmentMaskFetchAMD: {
      /* All these types have the coordinate as their first real argument */
      coord_components = glsl_get_sampler_dim_coordinate_components(sampler_dim);

      if (is_array && texop != nir_texop_lod)
         coord_components++;

      struct vtn_ssa_value *coord_val = vtn_ssa_value(b, w[idx++]);
      coord = coord_val->def;
      /* From the SPIR-V spec verxion 1.5, rev. 5:
       *
       *    "Coordinate must be a scalar or vector of floating-point type. It
       *    contains (u[, v] ... [, array layer]) as needed by the definition
       *    of Sampled Image. It may be a vector larger than needed, but all
       *    unused components appear after all used components."
       */
      vtn_fail_if(coord->num_components < coord_components,
                  "Coordinate value passed has fewer components than sampler dimensionality.");
      p->src = nir_src_for_ssa(nir_trim_vector(&b->nb, coord, coord_components));

      /* OpenCL allows integer sampling coordinates */
      if (glsl_type_is_integer(coord_val->type) &&
          opcode == SpvOpImageSampleExplicitLod) {
         vtn_fail_if(b->shader->info.stage != MESA_SHADER_KERNEL,
                     "Unless the Kernel capability is being used, the coordinate parameter "
                     "OpImageSampleExplicitLod must be floating point.");

         nir_ssa_def *coords[4];
         nir_ssa_def *f0_5 = nir_imm_float(&b->nb, 0.5);
         for (unsigned i = 0; i < coord_components; i++) {
            coords[i] = nir_i2f32(&b->nb, nir_channel(&b->nb, p->src.ssa, i));

            if (!is_array || i != coord_components - 1)
               coords[i] = nir_fadd(&b->nb, coords[i], f0_5);
         }

         p->src = nir_src_for_ssa(nir_vec(&b->nb, coords, coord_components));
      }

      p->src_type = nir_tex_src_coord;
      p++;
      break;
   }

   default:
      coord = NULL;
      coord_components = 0;
      break;
   }

   switch (opcode) {
   case SpvOpImageSampleProjImplicitLod:
   case SpvOpImageSampleProjExplicitLod:
   case SpvOpImageSampleProjDrefImplicitLod:
   case SpvOpImageSampleProjDrefExplicitLod:
      /* These have the projector as the last coordinate component */
      p->src = nir_src_for_ssa(nir_channel(&b->nb, coord, coord_components));
      p->src_type = nir_tex_src_projector;
      p++;
      break;

   default:
      break;
   }

   bool is_shadow = false;
   unsigned gather_component = 0;
   switch (opcode) {
   case SpvOpImageSampleDrefImplicitLod:
   case SpvOpImageSparseSampleDrefImplicitLod:
   case SpvOpImageSampleDrefExplicitLod:
   case SpvOpImageSparseSampleDrefExplicitLod:
   case SpvOpImageSampleProjDrefImplicitLod:
   case SpvOpImageSampleProjDrefExplicitLod:
   case SpvOpImageDrefGather:
   case SpvOpImageSparseDrefGather:
      /* These all have an explicit depth value as their next source */
      is_shadow = true;
      (*p++) = vtn_tex_src(b, w[idx++], nir_tex_src_comparator);
      break;

   case SpvOpImageGather:
   case SpvOpImageSparseGather:
      /* This has a component as its next source */
      gather_component = vtn_constant_uint(b, w[idx++]);
      break;

   default:
      break;
   }

   bool is_sparse = false;
   switch (opcode) {
   case SpvOpImageSparseSampleImplicitLod:
   case SpvOpImageSparseSampleExplicitLod:
   case SpvOpImageSparseSampleDrefImplicitLod:
   case SpvOpImageSparseSampleDrefExplicitLod:
   case SpvOpImageSparseFetch:
   case SpvOpImageSparseGather:
   case SpvOpImageSparseDrefGather:
      is_sparse = true;
      break;
   default:
      break;
   }

   /* For OpImageQuerySizeLod, we always have an LOD */
   if (opcode == SpvOpImageQuerySizeLod)
      (*p++) = vtn_tex_src(b, w[idx++], nir_tex_src_lod);

   /* For OpFragmentFetchAMD, we always have a multisample index */
   if (opcode == SpvOpFragmentFetchAMD)
      (*p++) = vtn_tex_src(b, w[idx++], nir_tex_src_ms_index);

   /* Now we need to handle some number of optional arguments */
   struct vtn_value *gather_offsets = NULL;
   uint32_t operands = SpvImageOperandsMaskNone;
   if (idx < count) {
      operands = w[idx];

      if (operands & SpvImageOperandsBiasMask) {
         vtn_assert(texop == nir_texop_tex ||
                    texop == nir_texop_tg4);
         if (texop == nir_texop_tex)
            texop = nir_texop_txb;
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsBiasMask);
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_bias);
      }

      if (operands & SpvImageOperandsLodMask) {
         vtn_assert(texop == nir_texop_txl || texop == nir_texop_txf ||
                    texop == nir_texop_txs || texop == nir_texop_tg4);
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsLodMask);
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_lod);
      }

      if (operands & SpvImageOperandsGradMask) {
         vtn_assert(texop == nir_texop_txl);
         texop = nir_texop_txd;
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsGradMask);
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_ddx);
         (*p++) = vtn_tex_src(b, w[arg + 1], nir_tex_src_ddy);
      }

      vtn_fail_if(util_bitcount(operands & (SpvImageOperandsConstOffsetsMask |
                                            SpvImageOperandsOffsetMask |
                                            SpvImageOperandsConstOffsetMask)) > 1,
                  "At most one of the ConstOffset, Offset, and ConstOffsets "
                  "image operands can be used on a given instruction.");

      if (operands & SpvImageOperandsOffsetMask) {
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsOffsetMask);
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_offset);
      }

      if (operands & SpvImageOperandsConstOffsetMask) {
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsConstOffsetMask);
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_offset);
      }

      if (operands & SpvImageOperandsConstOffsetsMask) {
         vtn_assert(texop == nir_texop_tg4);
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsConstOffsetsMask);
         gather_offsets = vtn_value(b, w[arg], vtn_value_type_constant);
      }

      if (operands & SpvImageOperandsSampleMask) {
         vtn_assert(texop == nir_texop_txf_ms);
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsSampleMask);
         texop = nir_texop_txf_ms;
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_ms_index);
      }

      if (operands & SpvImageOperandsMinLodMask) {
         vtn_assert(texop == nir_texop_tex ||
                    texop == nir_texop_txb ||
                    texop == nir_texop_txd);
         uint32_t arg = image_operand_arg(b, w, count, idx,
                                          SpvImageOperandsMinLodMask);
         (*p++) = vtn_tex_src(b, w[arg], nir_tex_src_min_lod);
      }
   }

   struct vtn_type *ret_type = vtn_get_type(b, w[1]);
   struct vtn_type *struct_type = NULL;
   if (is_sparse) {
      vtn_assert(glsl_type_is_struct_or_ifc(ret_type->type));
      struct_type = ret_type;
      ret_type = struct_type->members[1];
   }

   nir_tex_instr *instr = nir_tex_instr_create(b->shader, p - srcs);
   instr->op = texop;

   memcpy(instr->src, srcs, instr->num_srcs * sizeof(*instr->src));

   instr->coord_components = coord_components;
   instr->sampler_dim = sampler_dim;
   instr->is_array = is_array;
   instr->is_shadow = is_shadow;
   instr->is_sparse = is_sparse;
   instr->is_new_style_shadow =
      is_shadow && glsl_get_components(ret_type->type) == 1;
   instr->component = gather_component;

   /* The Vulkan spec says:
    *
    *    "If an instruction loads from or stores to a resource (including
    *    atomics and image instructions) and the resource descriptor being
    *    accessed is not dynamically uniform, then the operand corresponding
    *    to that resource (e.g. the pointer or sampled image operand) must be
    *    decorated with NonUniform."
    *
    * It's very careful to specify that the exact operand must be decorated
    * NonUniform.  The SPIR-V parser is not expected to chase through long
    * chains to find the NonUniform decoration.  It's either right there or we
    * can assume it doesn't exist.
    */
   enum gl_access_qualifier access = 0;
   vtn_foreach_decoration(b, sampled_val, non_uniform_decoration_cb, &access);

   if (operands & SpvImageOperandsNontemporalMask)
      access |= ACCESS_STREAM_CACHE_POLICY;

   if (sampler && b->options->force_tex_non_uniform)
      access |= ACCESS_NON_UNIFORM;

   if (sampled_val->propagated_non_uniform)
      access |= ACCESS_NON_UNIFORM;

   if (image && (access & ACCESS_NON_UNIFORM))
      instr->texture_non_uniform = true;

   if (sampler && (access & ACCESS_NON_UNIFORM))
      instr->sampler_non_uniform = true;

   /* for non-query ops, get dest_type from SPIR-V return type */
   if (dest_type == nir_type_invalid) {
      /* the return type should match the image type, unless the image type is
       * VOID (CL image), in which case the return type dictates the sampler
       */
      enum glsl_base_type sampler_base =
         glsl_get_sampler_result_type(image->type);
      enum glsl_base_type ret_base = glsl_get_base_type(ret_type->type);
      vtn_fail_if(sampler_base != ret_base && sampler_base != GLSL_TYPE_VOID,
                  "SPIR-V return type mismatches image type. This is only valid "
                  "for untyped images (OpenCL).");
      dest_type = nir_get_nir_type_for_glsl_base_type(ret_base);
      dest_type = get_image_type(b, dest_type, operands);
   }

   instr->dest_type = dest_type;

   nir_ssa_dest_init(&instr->instr, &instr->dest,
                     nir_tex_instr_dest_size(instr), 32, NULL);

   vtn_assert(glsl_get_vector_elements(ret_type->type) ==
              nir_tex_instr_result_size(instr));

   if (gather_offsets) {
      vtn_fail_if(gather_offsets->type->base_type != vtn_base_type_array ||
                  gather_offsets->type->length != 4,
                  "ConstOffsets must be an array of size four of vectors "
                  "of two integer components");

      struct vtn_type *vec_type = gather_offsets->type->array_element;
      vtn_fail_if(vec_type->base_type != vtn_base_type_vector ||
                  vec_type->length != 2 ||
                  !glsl_type_is_integer(vec_type->type),
                  "ConstOffsets must be an array of size four of vectors "
                  "of two integer components");

      unsigned bit_size = glsl_get_bit_size(vec_type->type);
      for (uint32_t i = 0; i < 4; i++) {
         const nir_const_value *cvec =
            gather_offsets->constant->elements[i]->values;
         for (uint32_t j = 0; j < 2; j++) {
            switch (bit_size) {
            case 8:  instr->tg4_offsets[i][j] = cvec[j].i8;    break;
            case 16: instr->tg4_offsets[i][j] = cvec[j].i16;   break;
            case 32: instr->tg4_offsets[i][j] = cvec[j].i32;   break;
            case 64: instr->tg4_offsets[i][j] = cvec[j].i64;   break;
            default:
               vtn_fail("Unsupported bit size: %u", bit_size);
            }
         }
      }
   }

   nir_builder_instr_insert(&b->nb, &instr->instr);

   if (is_sparse) {
      struct vtn_ssa_value *dest = vtn_create_ssa_value(b, struct_type->type);
      unsigned result_size = glsl_get_vector_elements(ret_type->type);
      dest->elems[0]->def = nir_channel(&b->nb, &instr->dest.ssa, result_size);
      dest->elems[1]->def = nir_trim_vector(&b->nb, &instr->dest.ssa,
                                              result_size);
      vtn_push_ssa_value(b, w[2], dest);
   } else {
      vtn_push_nir_ssa(b, w[2], &instr->dest.ssa);
   }
}

static void
fill_common_atomic_sources(struct vtn_builder *b, SpvOp opcode,
                           const uint32_t *w, nir_src *src)
{
   const struct glsl_type *type = vtn_get_type(b, w[1])->type;
   unsigned bit_size = glsl_get_bit_size(type);

   switch (opcode) {
   case SpvOpAtomicIIncrement:
      src[0] = nir_src_for_ssa(nir_imm_intN_t(&b->nb, 1, bit_size));
      break;

   case SpvOpAtomicIDecrement:
      src[0] = nir_src_for_ssa(nir_imm_intN_t(&b->nb, -1, bit_size));
      break;

   case SpvOpAtomicISub:
      src[0] =
         nir_src_for_ssa(nir_ineg(&b->nb, vtn_get_nir_ssa(b, w[6])));
      break;

   case SpvOpAtomicCompareExchange:
   case SpvOpAtomicCompareExchangeWeak:
      src[0] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[8]));
      src[1] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[7]));
      break;

   case SpvOpAtomicExchange:
   case SpvOpAtomicIAdd:
   case SpvOpAtomicSMin:
   case SpvOpAtomicUMin:
   case SpvOpAtomicSMax:
   case SpvOpAtomicUMax:
   case SpvOpAtomicAnd:
   case SpvOpAtomicOr:
   case SpvOpAtomicXor:
   case SpvOpAtomicFAddEXT:
   case SpvOpAtomicFMinEXT:
   case SpvOpAtomicFMaxEXT:
      src[0] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[6]));
      break;

   default:
      vtn_fail_with_opcode("Invalid SPIR-V atomic", opcode);
   }
}

static nir_ssa_def *
get_image_coord(struct vtn_builder *b, uint32_t value)
{
   nir_ssa_def *coord = vtn_get_nir_ssa(b, value);
   /* The image_load_store intrinsics assume a 4-dim coordinate */
   return nir_pad_vec4(&b->nb, coord);
}

static void
vtn_handle_image(struct vtn_builder *b, SpvOp opcode,
                 const uint32_t *w, unsigned count)
{
   /* Just get this one out of the way */
   if (opcode == SpvOpImageTexelPointer) {
      struct vtn_value *val =
         vtn_push_value(b, w[2], vtn_value_type_image_pointer);
      val->image = ralloc(b, struct vtn_image_pointer);

      val->image->image = vtn_nir_deref(b, w[3]);
      val->image->coord = get_image_coord(b, w[4]);
      val->image->sample = vtn_get_nir_ssa(b, w[5]);
      val->image->lod = nir_imm_int(&b->nb, 0);
      return;
   }

   struct vtn_image_pointer image;
   SpvScope scope = SpvScopeInvocation;
   SpvMemorySemanticsMask semantics = 0;
   SpvImageOperandsMask operands = SpvImageOperandsMaskNone;

   enum gl_access_qualifier access = 0;

   struct vtn_value *res_val;
   switch (opcode) {
   case SpvOpAtomicExchange:
   case SpvOpAtomicCompareExchange:
   case SpvOpAtomicCompareExchangeWeak:
   case SpvOpAtomicIIncrement:
   case SpvOpAtomicIDecrement:
   case SpvOpAtomicIAdd:
   case SpvOpAtomicISub:
   case SpvOpAtomicLoad:
   case SpvOpAtomicSMin:
   case SpvOpAtomicUMin:
   case SpvOpAtomicSMax:
   case SpvOpAtomicUMax:
   case SpvOpAtomicAnd:
   case SpvOpAtomicOr:
   case SpvOpAtomicXor:
   case SpvOpAtomicFAddEXT:
   case SpvOpAtomicFMinEXT:
   case SpvOpAtomicFMaxEXT:
      res_val = vtn_value(b, w[3], vtn_value_type_image_pointer);
      image = *res_val->image;
      scope = vtn_constant_uint(b, w[4]);
      semantics = vtn_constant_uint(b, w[5]);
      access |= ACCESS_COHERENT;
      break;

   case SpvOpAtomicStore:
      res_val = vtn_value(b, w[1], vtn_value_type_image_pointer);
      image = *res_val->image;
      scope = vtn_constant_uint(b, w[2]);
      semantics = vtn_constant_uint(b, w[3]);
      access |= ACCESS_COHERENT;
      break;

   case SpvOpImageQuerySizeLod:
      res_val = vtn_untyped_value(b, w[3]);
      image.image = vtn_get_image(b, w[3], &access);
      image.coord = NULL;
      image.sample = NULL;
      image.lod = vtn_ssa_value(b, w[4])->def;
      break;

   case SpvOpImageQuerySize:
   case SpvOpImageQuerySamples:
      res_val = vtn_untyped_value(b, w[3]);
      image.image = vtn_get_image(b, w[3], &access);
      image.coord = NULL;
      image.sample = NULL;
      image.lod = NULL;
      break;

   case SpvOpImageQueryFormat:
   case SpvOpImageQueryOrder:
      res_val = vtn_untyped_value(b, w[3]);
      image.image = vtn_get_image(b, w[3], &access);
      image.coord = NULL;
      image.sample = NULL;
      image.lod = NULL;
      break;

   case SpvOpImageRead:
   case SpvOpImageSparseRead: {
      res_val = vtn_untyped_value(b, w[3]);
      image.image = vtn_get_image(b, w[3], &access);
      image.coord = get_image_coord(b, w[4]);

      operands = count > 5 ? w[5] : SpvImageOperandsMaskNone;

      if (operands & SpvImageOperandsSampleMask) {
         uint32_t arg = image_operand_arg(b, w, count, 5,
                                          SpvImageOperandsSampleMask);
         image.sample = vtn_get_nir_ssa(b, w[arg]);
      } else {
         image.sample = nir_ssa_undef(&b->nb, 1, 32);
      }

      if (operands & SpvImageOperandsMakeTexelVisibleMask) {
         vtn_fail_if((operands & SpvImageOperandsNonPrivateTexelMask) == 0,
                     "MakeTexelVisible requires NonPrivateTexel to also be set.");
         uint32_t arg = image_operand_arg(b, w, count, 5,
                                          SpvImageOperandsMakeTexelVisibleMask);
         semantics = SpvMemorySemanticsMakeVisibleMask;
         scope = vtn_constant_uint(b, w[arg]);
      }

      if (operands & SpvImageOperandsLodMask) {
         uint32_t arg = image_operand_arg(b, w, count, 5,
                                          SpvImageOperandsLodMask);
         image.lod = vtn_get_nir_ssa(b, w[arg]);
      } else {
         image.lod = nir_imm_int(&b->nb, 0);
      }

      if (operands & SpvImageOperandsVolatileTexelMask)
         access |= ACCESS_VOLATILE;
      if (operands & SpvImageOperandsNontemporalMask)
         access |= ACCESS_STREAM_CACHE_POLICY;

      break;
   }

   case SpvOpImageWrite: {
      res_val = vtn_untyped_value(b, w[1]);
      image.image = vtn_get_image(b, w[1], &access);
      image.coord = get_image_coord(b, w[2]);

      /* texel = w[3] */

      operands = count > 4 ? w[4] : SpvImageOperandsMaskNone;

      if (operands & SpvImageOperandsSampleMask) {
         uint32_t arg = image_operand_arg(b, w, count, 4,
                                          SpvImageOperandsSampleMask);
         image.sample = vtn_get_nir_ssa(b, w[arg]);
      } else {
         image.sample = nir_ssa_undef(&b->nb, 1, 32);
      }

      if (operands & SpvImageOperandsMakeTexelAvailableMask) {
         vtn_fail_if((operands & SpvImageOperandsNonPrivateTexelMask) == 0,
                     "MakeTexelAvailable requires NonPrivateTexel to also be set.");
         uint32_t arg = image_operand_arg(b, w, count, 4,
                                          SpvImageOperandsMakeTexelAvailableMask);
         semantics = SpvMemorySemanticsMakeAvailableMask;
         scope = vtn_constant_uint(b, w[arg]);
      }

      if (operands & SpvImageOperandsLodMask) {
         uint32_t arg = image_operand_arg(b, w, count, 4,
                                          SpvImageOperandsLodMask);
         image.lod = vtn_get_nir_ssa(b, w[arg]);
      } else {
         image.lod = nir_imm_int(&b->nb, 0);
      }

      if (operands & SpvImageOperandsVolatileTexelMask)
         access |= ACCESS_VOLATILE;
      if (operands & SpvImageOperandsNontemporalMask)
         access |= ACCESS_STREAM_CACHE_POLICY;

      break;
   }

   default:
      vtn_fail_with_opcode("Invalid image opcode", opcode);
   }

   if (semantics & SpvMemorySemanticsVolatileMask)
      access |= ACCESS_VOLATILE;

   nir_intrinsic_op op;
   switch (opcode) {
#define OP(S, N) case SpvOp##S: op = nir_intrinsic_image_deref_##N; break;
   OP(ImageQuerySize,            size)
   OP(ImageQuerySizeLod,         size)
   OP(ImageRead,                 load)
   OP(ImageSparseRead,           sparse_load)
   OP(ImageWrite,                store)
   OP(AtomicLoad,                load)
   OP(AtomicStore,               store)
   OP(AtomicExchange,            atomic_exchange)
   OP(AtomicCompareExchange,     atomic_comp_swap)
   OP(AtomicCompareExchangeWeak, atomic_comp_swap)
   OP(AtomicIIncrement,          atomic_add)
   OP(AtomicIDecrement,          atomic_add)
   OP(AtomicIAdd,                atomic_add)
   OP(AtomicISub,                atomic_add)
   OP(AtomicSMin,                atomic_imin)
   OP(AtomicUMin,                atomic_umin)
   OP(AtomicSMax,                atomic_imax)
   OP(AtomicUMax,                atomic_umax)
   OP(AtomicAnd,                 atomic_and)
   OP(AtomicOr,                  atomic_or)
   OP(AtomicXor,                 atomic_xor)
   OP(AtomicFAddEXT,             atomic_fadd)
   OP(AtomicFMinEXT,             atomic_fmin)
   OP(AtomicFMaxEXT,             atomic_fmax)
   OP(ImageQueryFormat,          format)
   OP(ImageQueryOrder,           order)
   OP(ImageQuerySamples,         samples)
#undef OP
   default:
      vtn_fail_with_opcode("Invalid image opcode", opcode);
   }

   nir_intrinsic_instr *intrin = nir_intrinsic_instr_create(b->shader, op);

   intrin->src[0] = nir_src_for_ssa(&image.image->dest.ssa);
   nir_intrinsic_set_image_dim(intrin, glsl_get_sampler_dim(image.image->type));
   nir_intrinsic_set_image_array(intrin,
      glsl_sampler_type_is_array(image.image->type));

   switch (opcode) {
   case SpvOpImageQuerySamples:
   case SpvOpImageQuerySize:
   case SpvOpImageQuerySizeLod:
   case SpvOpImageQueryFormat:
   case SpvOpImageQueryOrder:
      break;
   default:
      /* The image coordinate is always 4 components but we may not have that
       * many.  Swizzle to compensate.
       */
      intrin->src[1] = nir_src_for_ssa(nir_pad_vec4(&b->nb, image.coord));
      intrin->src[2] = nir_src_for_ssa(image.sample);
      break;
   }

   /* The Vulkan spec says:
    *
    *    "If an instruction loads from or stores to a resource (including
    *    atomics and image instructions) and the resource descriptor being
    *    accessed is not dynamically uniform, then the operand corresponding
    *    to that resource (e.g. the pointer or sampled image operand) must be
    *    decorated with NonUniform."
    *
    * It's very careful to specify that the exact operand must be decorated
    * NonUniform.  The SPIR-V parser is not expected to chase through long
    * chains to find the NonUniform decoration.  It's either right there or we
    * can assume it doesn't exist.
    */
   vtn_foreach_decoration(b, res_val, non_uniform_decoration_cb, &access);
   nir_intrinsic_set_access(intrin, access);

   switch (opcode) {
   case SpvOpImageQuerySamples:
   case SpvOpImageQueryFormat:
   case SpvOpImageQueryOrder:
      /* No additional sources */
      break;
   case SpvOpImageQuerySize:
      intrin->src[1] = nir_src_for_ssa(nir_imm_int(&b->nb, 0));
      break;
   case SpvOpImageQuerySizeLod:
      intrin->src[1] = nir_src_for_ssa(image.lod);
      break;
   case SpvOpAtomicLoad:
   case SpvOpImageRead:
   case SpvOpImageSparseRead:
      /* Only OpImageRead can support a lod parameter if
      * SPV_AMD_shader_image_load_store_lod is used but the current NIR
      * intrinsics definition for atomics requires us to set it for
      * OpAtomicLoad.
      */
      intrin->src[3] = nir_src_for_ssa(image.lod);
      break;
   case SpvOpAtomicStore:
   case SpvOpImageWrite: {
      const uint32_t value_id = opcode == SpvOpAtomicStore ? w[4] : w[3];
      struct vtn_ssa_value *value = vtn_ssa_value(b, value_id);
      /* nir_intrinsic_image_deref_store always takes a vec4 value */
      assert(op == nir_intrinsic_image_deref_store);
      intrin->num_components = 4;
      intrin->src[3] = nir_src_for_ssa(nir_pad_vec4(&b->nb, value->def));
      /* Only OpImageWrite can support a lod parameter if
       * SPV_AMD_shader_image_load_store_lod is used but the current NIR
       * intrinsics definition for atomics requires us to set it for
       * OpAtomicStore.
       */
      intrin->src[4] = nir_src_for_ssa(image.lod);

      nir_alu_type src_type =
         get_image_type(b, nir_get_nir_type_for_glsl_type(value->type), operands);
      nir_intrinsic_set_src_type(intrin, src_type);
      break;
   }

   case SpvOpAtomicCompareExchange:
   case SpvOpAtomicCompareExchangeWeak:
   case SpvOpAtomicIIncrement:
   case SpvOpAtomicIDecrement:
   case SpvOpAtomicExchange:
   case SpvOpAtomicIAdd:
   case SpvOpAtomicISub:
   case SpvOpAtomicSMin:
   case SpvOpAtomicUMin:
   case SpvOpAtomicSMax:
   case SpvOpAtomicUMax:
   case SpvOpAtomicAnd:
   case SpvOpAtomicOr:
   case SpvOpAtomicXor:
   case SpvOpAtomicFAddEXT:
   case SpvOpAtomicFMinEXT:
   case SpvOpAtomicFMaxEXT:
      fill_common_atomic_sources(b, opcode, w, &intrin->src[3]);
      break;

   default:
      vtn_fail_with_opcode("Invalid image opcode", opcode);
   }

   /* Image operations implicitly have the Image storage memory semantics. */
   semantics |= SpvMemorySemanticsImageMemoryMask;

   SpvMemorySemanticsMask before_semantics;
   SpvMemorySemanticsMask after_semantics;
   vtn_split_barrier_semantics(b, semantics, &before_semantics, &after_semantics);

   if (before_semantics)
      vtn_emit_memory_barrier(b, scope, before_semantics);

   if (opcode != SpvOpImageWrite && opcode != SpvOpAtomicStore) {
      struct vtn_type *type = vtn_get_type(b, w[1]);
      struct vtn_type *struct_type = NULL;
      if (opcode == SpvOpImageSparseRead) {
         vtn_assert(glsl_type_is_struct_or_ifc(type->type));
         struct_type = type;
         type = struct_type->members[1];
      }

      unsigned dest_components = glsl_get_vector_elements(type->type);
      if (opcode == SpvOpImageSparseRead)
         dest_components++;

      if (nir_intrinsic_infos[op].dest_components == 0)
         intrin->num_components = dest_components;

      unsigned bit_size = glsl_get_bit_size(type->type);
      if (opcode == SpvOpImageQuerySize ||
          opcode == SpvOpImageQuerySizeLod)
         bit_size = MIN2(bit_size, 32);

      nir_ssa_dest_init(&intrin->instr, &intrin->dest,
                        nir_intrinsic_dest_components(intrin),
                        bit_size, NULL);

      nir_builder_instr_insert(&b->nb, &intrin->instr);

      nir_ssa_def *result = nir_trim_vector(&b->nb, &intrin->dest.ssa,
                                              dest_components);

      if (opcode == SpvOpImageQuerySize ||
          opcode == SpvOpImageQuerySizeLod)
         result = nir_u2uN(&b->nb, result, glsl_get_bit_size(type->type));

      if (opcode == SpvOpImageSparseRead) {
         struct vtn_ssa_value *dest = vtn_create_ssa_value(b, struct_type->type);
         unsigned res_type_size = glsl_get_vector_elements(type->type);
         dest->elems[0]->def = nir_channel(&b->nb, result, res_type_size);
         if (intrin->dest.ssa.bit_size != 32)
            dest->elems[0]->def = nir_u2u32(&b->nb, dest->elems[0]->def);
         dest->elems[1]->def = nir_trim_vector(&b->nb, result, res_type_size);
         vtn_push_ssa_value(b, w[2], dest);
      } else {
         vtn_push_nir_ssa(b, w[2], result);
      }

      if (opcode == SpvOpImageRead || opcode == SpvOpImageSparseRead ||
          opcode == SpvOpAtomicLoad) {
         nir_alu_type dest_type =
            get_image_type(b, nir_get_nir_type_for_glsl_type(type->type), operands);
         nir_intrinsic_set_dest_type(intrin, dest_type);
      }
   } else {
      nir_builder_instr_insert(&b->nb, &intrin->instr);
   }

   if (after_semantics)
      vtn_emit_memory_barrier(b, scope, after_semantics);
}

static nir_intrinsic_op
get_uniform_nir_atomic_op(struct vtn_builder *b, SpvOp opcode)
{
   switch (opcode) {
#define OP(S, N) case SpvOp##S: return nir_intrinsic_atomic_counter_ ##N;
   OP(AtomicLoad,                read_deref)
   OP(AtomicExchange,            exchange)
   OP(AtomicCompareExchange,     comp_swap)
   OP(AtomicCompareExchangeWeak, comp_swap)
   OP(AtomicIIncrement,          inc_deref)
   OP(AtomicIDecrement,          post_dec_deref)
   OP(AtomicIAdd,                add_deref)
   OP(AtomicISub,                add_deref)
   OP(AtomicUMin,                min_deref)
   OP(AtomicUMax,                max_deref)
   OP(AtomicAnd,                 and_deref)
   OP(AtomicOr,                  or_deref)
   OP(AtomicXor,                 xor_deref)
#undef OP
   default:
      /* We left the following out: AtomicStore, AtomicSMin and
       * AtomicSmax. Right now there are not nir intrinsics for them. At this
       * moment Atomic Counter support is needed for ARB_spirv support, so is
       * only need to support GLSL Atomic Counters that are uints and don't
       * allow direct storage.
       */
      vtn_fail("Invalid uniform atomic");
   }
}

static nir_intrinsic_op
get_deref_nir_atomic_op(struct vtn_builder *b, SpvOp opcode)
{
   switch (opcode) {
   case SpvOpAtomicLoad:         return nir_intrinsic_load_deref;
   case SpvOpAtomicFlagClear:
   case SpvOpAtomicStore:        return nir_intrinsic_store_deref;
#define OP(S, N) case SpvOp##S: return nir_intrinsic_deref_##N;
   OP(AtomicExchange,            atomic_exchange)
   OP(AtomicCompareExchange,     atomic_comp_swap)
   OP(AtomicCompareExchangeWeak, atomic_comp_swap)
   OP(AtomicIIncrement,          atomic_add)
   OP(AtomicIDecrement,          atomic_add)
   OP(AtomicIAdd,                atomic_add)
   OP(AtomicISub,                atomic_add)
   OP(AtomicSMin,                atomic_imin)
   OP(AtomicUMin,                atomic_umin)
   OP(AtomicSMax,                atomic_imax)
   OP(AtomicUMax,                atomic_umax)
   OP(AtomicAnd,                 atomic_and)
   OP(AtomicOr,                  atomic_or)
   OP(AtomicXor,                 atomic_xor)
   OP(AtomicFAddEXT,             atomic_fadd)
   OP(AtomicFMinEXT,             atomic_fmin)
   OP(AtomicFMaxEXT,             atomic_fmax)
   OP(AtomicFlagTestAndSet,      atomic_comp_swap)
#undef OP
   default:
      vtn_fail_with_opcode("Invalid shared atomic", opcode);
   }
}

/*
 * Handles shared atomics, ssbo atomics and atomic counters.
 */
static void
vtn_handle_atomics(struct vtn_builder *b, SpvOp opcode,
                   const uint32_t *w, UNUSED unsigned count)
{
   struct vtn_pointer *ptr;
   nir_intrinsic_instr *atomic;

   SpvScope scope = SpvScopeInvocation;
   SpvMemorySemanticsMask semantics = 0;
   enum gl_access_qualifier access = 0;

   switch (opcode) {
   case SpvOpAtomicLoad:
   case SpvOpAtomicExchange:
   case SpvOpAtomicCompareExchange:
   case SpvOpAtomicCompareExchangeWeak:
   case SpvOpAtomicIIncrement:
   case SpvOpAtomicIDecrement:
   case SpvOpAtomicIAdd:
   case SpvOpAtomicISub:
   case SpvOpAtomicSMin:
   case SpvOpAtomicUMin:
   case SpvOpAtomicSMax:
   case SpvOpAtomicUMax:
   case SpvOpAtomicAnd:
   case SpvOpAtomicOr:
   case SpvOpAtomicXor:
   case SpvOpAtomicFAddEXT:
   case SpvOpAtomicFMinEXT:
   case SpvOpAtomicFMaxEXT:
   case SpvOpAtomicFlagTestAndSet:
      ptr = vtn_pointer(b, w[3]);
      scope = vtn_constant_uint(b, w[4]);
      semantics = vtn_constant_uint(b, w[5]);
      break;
   case SpvOpAtomicFlagClear:
   case SpvOpAtomicStore:
      ptr = vtn_pointer(b, w[1]);
      scope = vtn_constant_uint(b, w[2]);
      semantics = vtn_constant_uint(b, w[3]);
      break;

   default:
      vtn_fail_with_opcode("Invalid SPIR-V atomic", opcode);
   }

   if (semantics & SpvMemorySemanticsVolatileMask)
      access |= ACCESS_VOLATILE;

   /* uniform as "atomic counter uniform" */
   if (ptr->mode == vtn_variable_mode_atomic_counter) {
      nir_deref_instr *deref = vtn_pointer_to_deref(b, ptr);
      nir_intrinsic_op op = get_uniform_nir_atomic_op(b, opcode);
      atomic = nir_intrinsic_instr_create(b->nb.shader, op);
      atomic->src[0] = nir_src_for_ssa(&deref->dest.ssa);

      /* SSBO needs to initialize index/offset. In this case we don't need to,
       * as that info is already stored on the ptr->var->var nir_variable (see
       * vtn_create_variable)
       */

      switch (opcode) {
      case SpvOpAtomicLoad:
      case SpvOpAtomicExchange:
      case SpvOpAtomicCompareExchange:
      case SpvOpAtomicCompareExchangeWeak:
      case SpvOpAtomicIIncrement:
      case SpvOpAtomicIDecrement:
      case SpvOpAtomicIAdd:
      case SpvOpAtomicISub:
      case SpvOpAtomicSMin:
      case SpvOpAtomicUMin:
      case SpvOpAtomicSMax:
      case SpvOpAtomicUMax:
      case SpvOpAtomicAnd:
      case SpvOpAtomicOr:
      case SpvOpAtomicXor:
         /* Nothing: we don't need to call fill_common_atomic_sources here, as
          * atomic counter uniforms doesn't have sources
          */
         break;

      default:
         unreachable("Invalid SPIR-V atomic");

      }
   } else {
      nir_deref_instr *deref = vtn_pointer_to_deref(b, ptr);
      const struct glsl_type *deref_type = deref->type;
      nir_intrinsic_op op = get_deref_nir_atomic_op(b, opcode);
      atomic = nir_intrinsic_instr_create(b->nb.shader, op);
      atomic->src[0] = nir_src_for_ssa(&deref->dest.ssa);

      if (ptr->mode != vtn_variable_mode_workgroup)
         access |= ACCESS_COHERENT;

      nir_intrinsic_set_access(atomic, access);

      switch (opcode) {
      case SpvOpAtomicLoad:
         atomic->num_components = glsl_get_vector_elements(deref_type);
         break;

      case SpvOpAtomicStore:
         atomic->num_components = glsl_get_vector_elements(deref_type);
         nir_intrinsic_set_write_mask(atomic, (1 << atomic->num_components) - 1);
         atomic->src[1] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[4]));
         break;

      case SpvOpAtomicFlagClear:
         atomic->num_components = 1;
         nir_intrinsic_set_write_mask(atomic, 1);
         atomic->src[1] = nir_src_for_ssa(nir_imm_intN_t(&b->nb, 0, 32));
         break;
      case SpvOpAtomicFlagTestAndSet:
         atomic->src[1] = nir_src_for_ssa(nir_imm_intN_t(&b->nb, 0, 32));
         atomic->src[2] = nir_src_for_ssa(nir_imm_intN_t(&b->nb, -1, 32));
         break;
      case SpvOpAtomicExchange:
      case SpvOpAtomicCompareExchange:
      case SpvOpAtomicCompareExchangeWeak:
      case SpvOpAtomicIIncrement:
      case SpvOpAtomicIDecrement:
      case SpvOpAtomicIAdd:
      case SpvOpAtomicISub:
      case SpvOpAtomicSMin:
      case SpvOpAtomicUMin:
      case SpvOpAtomicSMax:
      case SpvOpAtomicUMax:
      case SpvOpAtomicAnd:
      case SpvOpAtomicOr:
      case SpvOpAtomicXor:
      case SpvOpAtomicFAddEXT:
      case SpvOpAtomicFMinEXT:
      case SpvOpAtomicFMaxEXT:
         fill_common_atomic_sources(b, opcode, w, &atomic->src[1]);
         break;

      default:
         vtn_fail_with_opcode("Invalid SPIR-V atomic", opcode);
      }
   }

   /* Atomic ordering operations will implicitly apply to the atomic operation
    * storage class, so include that too.
    */
   semantics |= vtn_mode_to_memory_semantics(ptr->mode);

   SpvMemorySemanticsMask before_semantics;
   SpvMemorySemanticsMask after_semantics;
   vtn_split_barrier_semantics(b, semantics, &before_semantics, &after_semantics);

   if (before_semantics)
      vtn_emit_memory_barrier(b, scope, before_semantics);

   if (opcode != SpvOpAtomicStore && opcode != SpvOpAtomicFlagClear) {
      struct vtn_type *type = vtn_get_type(b, w[1]);

      if (opcode == SpvOpAtomicFlagTestAndSet) {
         /* map atomic flag to a 32-bit atomic integer. */
         nir_ssa_dest_init(&atomic->instr, &atomic->dest,
                           1, 32, NULL);
      } else {
         nir_ssa_dest_init(&atomic->instr, &atomic->dest,
                           glsl_get_vector_elements(type->type),
                           glsl_get_bit_size(type->type), NULL);

         vtn_push_nir_ssa(b, w[2], &atomic->dest.ssa);
      }
   }

   nir_builder_instr_insert(&b->nb, &atomic->instr);

   if (opcode == SpvOpAtomicFlagTestAndSet) {
      vtn_push_nir_ssa(b, w[2], nir_i2b(&b->nb, &atomic->dest.ssa));
   }
   if (after_semantics)
      vtn_emit_memory_barrier(b, scope, after_semantics);
}

static nir_alu_instr *
create_vec(struct vtn_builder *b, unsigned num_components, unsigned bit_size)
{
   nir_op op = nir_op_vec(num_components);
   nir_alu_instr *vec = nir_alu_instr_create(b->shader, op);
   nir_ssa_dest_init(&vec->instr, &vec->dest.dest, num_components,
                     bit_size, NULL);
   vec->dest.write_mask = (1 << num_components) - 1;

   return vec;
}

struct vtn_ssa_value *
vtn_ssa_transpose(struct vtn_builder *b, struct vtn_ssa_value *src)
{
   if (src->transposed)
      return src->transposed;

   struct vtn_ssa_value *dest =
      vtn_create_ssa_value(b, glsl_transposed_type(src->type));

   for (unsigned i = 0; i < glsl_get_matrix_columns(dest->type); i++) {
      if (glsl_type_is_vector_or_scalar(src->type)) {
         dest->elems[i]->def = nir_channel(&b->nb, src->def, i);
      } else {
         unsigned cols = glsl_get_matrix_columns(src->type);
         nir_ssa_scalar srcs[NIR_MAX_MATRIX_COLUMNS];
         for (unsigned j = 0; j < cols; j++) {
            srcs[j] = nir_get_ssa_scalar(src->elems[j]->def, i);
         }
         dest->elems[i]->def = nir_vec_scalars(&b->nb, srcs, cols);
      }
   }

   dest->transposed = src;

   return dest;
}

static nir_ssa_def *
vtn_vector_shuffle(struct vtn_builder *b, unsigned num_components,
                   nir_ssa_def *src0, nir_ssa_def *src1,
                   const uint32_t *indices)
{
   nir_alu_instr *vec = create_vec(b, num_components, src0->bit_size);

   for (unsigned i = 0; i < num_components; i++) {
      uint32_t index = indices[i];
      unsigned total_components = src0->num_components + src1->num_components;
      vtn_fail_if(index != 0xffffffff && index >= total_components,
                  "OpVectorShuffle: All Component literals must either be "
                  "FFFFFFFF or in [0, N - 1] (inclusive)");

      if (index == 0xffffffff) {
         vec->src[i].src =
            nir_src_for_ssa(nir_ssa_undef(&b->nb, 1, src0->bit_size));
      } else if (index < src0->num_components) {
         vec->src[i].src = nir_src_for_ssa(src0);
         vec->src[i].swizzle[0] = index;
      } else {
         vec->src[i].src = nir_src_for_ssa(src1);
         vec->src[i].swizzle[0] = index - src0->num_components;
      }
   }

   nir_builder_instr_insert(&b->nb, &vec->instr);

   return &vec->dest.dest.ssa;
}

/*
 * Concatentates a number of vectors/scalars together to produce a vector
 */
static nir_ssa_def *
vtn_vector_construct(struct vtn_builder *b, unsigned num_components,
                     unsigned num_srcs, nir_ssa_def **srcs)
{
   nir_alu_instr *vec = create_vec(b, num_components, srcs[0]->bit_size);

   /* From the SPIR-V 1.1 spec for OpCompositeConstruct:
    *
    *    "When constructing a vector, there must be at least two Constituent
    *    operands."
    */
   vtn_assert(num_srcs >= 2);

   unsigned dest_idx = 0;
   for (unsigned i = 0; i < num_srcs; i++) {
      nir_ssa_def *src = srcs[i];
      vtn_assert(dest_idx + src->num_components <= num_components);
      for (unsigned j = 0; j < src->num_components; j++) {
         vec->src[dest_idx].src = nir_src_for_ssa(src);
         vec->src[dest_idx].swizzle[0] = j;
         dest_idx++;
      }
   }

   /* From the SPIR-V 1.1 spec for OpCompositeConstruct:
    *
    *    "When constructing a vector, the total number of components in all
    *    the operands must equal the number of components in Result Type."
    */
   vtn_assert(dest_idx == num_components);

   nir_builder_instr_insert(&b->nb, &vec->instr);

   return &vec->dest.dest.ssa;
}

static struct vtn_ssa_value *
vtn_composite_copy(void *mem_ctx, struct vtn_ssa_value *src)
{
   struct vtn_ssa_value *dest = rzalloc(mem_ctx, struct vtn_ssa_value);
   dest->type = src->type;

   if (glsl_type_is_vector_or_scalar(src->type)) {
      dest->def = src->def;
   } else {
      unsigned elems = glsl_get_length(src->type);

      dest->elems = ralloc_array(mem_ctx, struct vtn_ssa_value *, elems);
      for (unsigned i = 0; i < elems; i++)
         dest->elems[i] = vtn_composite_copy(mem_ctx, src->elems[i]);
   }

   return dest;
}

static struct vtn_ssa_value *
vtn_composite_insert(struct vtn_builder *b, struct vtn_ssa_value *src,
                     struct vtn_ssa_value *insert, const uint32_t *indices,
                     unsigned num_indices)
{
   struct vtn_ssa_value *dest = vtn_composite_copy(b, src);

   struct vtn_ssa_value *cur = dest;
   unsigned i;
   for (i = 0; i < num_indices - 1; i++) {
      /* If we got a vector here, that means the next index will be trying to
       * dereference a scalar.
       */
      vtn_fail_if(glsl_type_is_vector_or_scalar(cur->type),
                  "OpCompositeInsert has too many indices.");
      vtn_fail_if(indices[i] >= glsl_get_length(cur->type),
                  "All indices in an OpCompositeInsert must be in-bounds");
      cur = cur->elems[indices[i]];
   }

   if (glsl_type_is_vector_or_scalar(cur->type)) {
      vtn_fail_if(indices[i] >= glsl_get_vector_elements(cur->type),
                  "All indices in an OpCompositeInsert must be in-bounds");

      /* According to the SPIR-V spec, OpCompositeInsert may work down to
       * the component granularity. In that case, the last index will be
       * the index to insert the scalar into the vector.
       */

      cur->def = nir_vector_insert_imm(&b->nb, cur->def, insert->def, indices[i]);
   } else {
      vtn_fail_if(indices[i] >= glsl_get_length(cur->type),
                  "All indices in an OpCompositeInsert must be in-bounds");
      cur->elems[indices[i]] = insert;
   }

   return dest;
}

static struct vtn_ssa_value *
vtn_composite_extract(struct vtn_builder *b, struct vtn_ssa_value *src,
                      const uint32_t *indices, unsigned num_indices)
{
   struct vtn_ssa_value *cur = src;
   for (unsigned i = 0; i < num_indices; i++) {
      if (glsl_type_is_vector_or_scalar(cur->type)) {
         vtn_assert(i == num_indices - 1);
         vtn_fail_if(indices[i] >= glsl_get_vector_elements(cur->type),
                     "All indices in an OpCompositeExtract must be in-bounds");

         /* According to the SPIR-V spec, OpCompositeExtract may work down to
          * the component granularity. The last index will be the index of the
          * vector to extract.
          */

         const struct glsl_type *scalar_type =
            glsl_scalar_type(glsl_get_base_type(cur->type));
         struct vtn_ssa_value *ret = vtn_create_ssa_value(b, scalar_type);
         ret->def = nir_channel(&b->nb, cur->def, indices[i]);
         return ret;
      } else {
         vtn_fail_if(indices[i] >= glsl_get_length(cur->type),
                     "All indices in an OpCompositeExtract must be in-bounds");
         cur = cur->elems[indices[i]];
      }
   }

   return cur;
}

static void
vtn_handle_composite(struct vtn_builder *b, SpvOp opcode,
                     const uint32_t *w, unsigned count)
{
   struct vtn_type *type = vtn_get_type(b, w[1]);
   struct vtn_ssa_value *ssa = vtn_create_ssa_value(b, type->type);

   switch (opcode) {
   case SpvOpVectorExtractDynamic:
      ssa->def = nir_vector_extract(&b->nb, vtn_get_nir_ssa(b, w[3]),
                                    vtn_get_nir_ssa(b, w[4]));
      break;

   case SpvOpVectorInsertDynamic:
      ssa->def = nir_vector_insert(&b->nb, vtn_get_nir_ssa(b, w[3]),
                                   vtn_get_nir_ssa(b, w[4]),
                                   vtn_get_nir_ssa(b, w[5]));
      break;

   case SpvOpVectorShuffle:
      ssa->def = vtn_vector_shuffle(b, glsl_get_vector_elements(type->type),
                                    vtn_get_nir_ssa(b, w[3]),
                                    vtn_get_nir_ssa(b, w[4]),
                                    w + 5);
      break;

   case SpvOpCompositeConstruct: {
      unsigned elems = count - 3;
      assume(elems >= 1);
      if (glsl_type_is_vector_or_scalar(type->type)) {
         nir_ssa_def *srcs[NIR_MAX_VEC_COMPONENTS];
         for (unsigned i = 0; i < elems; i++) {
            srcs[i] = vtn_get_nir_ssa(b, w[3 + i]);
            vtn_assert(glsl_get_bit_size(type->type) == srcs[i]->bit_size);
         }
         ssa->def =
            vtn_vector_construct(b, glsl_get_vector_elements(type->type),
                                 elems, srcs);
      } else {
         ssa->elems = ralloc_array(b, struct vtn_ssa_value *, elems);
         for (unsigned i = 0; i < elems; i++)
            ssa->elems[i] = vtn_ssa_value(b, w[3 + i]);
      }
      break;
   }
   case SpvOpCompositeExtract:
      ssa = vtn_composite_extract(b, vtn_ssa_value(b, w[3]),
                                  w + 4, count - 4);
      break;

   case SpvOpCompositeInsert:
      ssa = vtn_composite_insert(b, vtn_ssa_value(b, w[4]),
                                 vtn_ssa_value(b, w[3]),
                                 w + 5, count - 5);
      break;

   case SpvOpCopyLogical:
      ssa = vtn_composite_copy(b, vtn_ssa_value(b, w[3]));
      break;
   case SpvOpCopyObject:
      vtn_copy_value(b, w[3], w[2]);
      return;

   default:
      vtn_fail_with_opcode("unknown composite operation", opcode);
   }

   vtn_push_ssa_value(b, w[2], ssa);
}

void
vtn_emit_memory_barrier(struct vtn_builder *b, SpvScope scope,
                        SpvMemorySemanticsMask semantics)
{
   if (b->shader->options->use_scoped_barrier) {
      vtn_emit_scoped_memory_barrier(b, scope, semantics);
      return;
   }

   static const SpvMemorySemanticsMask all_memory_semantics =
      SpvMemorySemanticsUniformMemoryMask |
      SpvMemorySemanticsWorkgroupMemoryMask |
      SpvMemorySemanticsAtomicCounterMemoryMask |
      SpvMemorySemanticsImageMemoryMask |
      SpvMemorySemanticsOutputMemoryMask;

   /* If we're not actually doing a memory barrier, bail */
   if (!(semantics & all_memory_semantics))
      return;

   /* GL and Vulkan don't have these */
   vtn_assert(scope != SpvScopeCrossDevice);

   if (scope == SpvScopeSubgroup)
      return; /* Nothing to do here */

   if (scope == SpvScopeWorkgroup) {
      nir_group_memory_barrier(&b->nb);
      return;
   }

   /* There's only three scopes left */
   vtn_assert(scope == SpvScopeInvocation || scope == SpvScopeDevice || scope == SpvScopeQueueFamily);

   /* Map the GLSL memoryBarrier() construct and any barriers with more than one
    * semantic to the corresponding NIR one.
    */
   if (util_bitcount(semantics & all_memory_semantics) > 1) {
      nir_memory_barrier(&b->nb);
      if (semantics & SpvMemorySemanticsOutputMemoryMask) {
         /* GLSL memoryBarrier() (and the corresponding NIR one) doesn't include
          * TCS outputs, so we have to emit it's own intrinsic for that. We
          * then need to emit another memory_barrier to prevent moving
          * non-output operations to before the tcs_patch barrier.
          */
         nir_memory_barrier_tcs_patch(&b->nb);
         nir_memory_barrier(&b->nb);
      }
      return;
   }

   /* Issue a more specific barrier */
   switch (semantics & all_memory_semantics) {
   case SpvMemorySemanticsUniformMemoryMask:
      nir_memory_barrier_buffer(&b->nb);
      break;
   case SpvMemorySemanticsWorkgroupMemoryMask:
      nir_memory_barrier_shared(&b->nb);
      break;
   case SpvMemorySemanticsAtomicCounterMemoryMask:
      nir_memory_barrier_atomic_counter(&b->nb);
      break;
   case SpvMemorySemanticsImageMemoryMask:
      nir_memory_barrier_image(&b->nb);
      break;
   case SpvMemorySemanticsOutputMemoryMask:
      if (b->nb.shader->info.stage == MESA_SHADER_TESS_CTRL)
         nir_memory_barrier_tcs_patch(&b->nb);
      break;
   default:
      break;
   }
}

static void
vtn_handle_barrier(struct vtn_builder *b, SpvOp opcode,
                   const uint32_t *w, UNUSED unsigned count)
{
   switch (opcode) {
   case SpvOpEmitVertex:
   case SpvOpEmitStreamVertex:
   case SpvOpEndPrimitive:
   case SpvOpEndStreamPrimitive: {
      unsigned stream = 0;
      if (opcode == SpvOpEmitStreamVertex || opcode == SpvOpEndStreamPrimitive)
         stream = vtn_constant_uint(b, w[1]);

      switch (opcode) {
      case SpvOpEmitStreamVertex:
      case SpvOpEmitVertex:
         nir_emit_vertex(&b->nb, stream);
         break;
      case SpvOpEndPrimitive:
      case SpvOpEndStreamPrimitive:
         nir_end_primitive(&b->nb, stream);
         break;
      default:
         unreachable("Invalid opcode");
      }
      break;
   }

   case SpvOpMemoryBarrier: {
      SpvScope scope = vtn_constant_uint(b, w[1]);
      SpvMemorySemanticsMask semantics = vtn_constant_uint(b, w[2]);
      vtn_emit_memory_barrier(b, scope, semantics);
      return;
   }

   case SpvOpControlBarrier: {
      SpvScope execution_scope = vtn_constant_uint(b, w[1]);
      SpvScope memory_scope = vtn_constant_uint(b, w[2]);
      SpvMemorySemanticsMask memory_semantics = vtn_constant_uint(b, w[3]);

      /* GLSLang, prior to commit 8297936dd6eb3, emitted OpControlBarrier with
       * memory semantics of None for GLSL barrier().
       * And before that, prior to c3f1cdfa, emitted the OpControlBarrier with
       * Device instead of Workgroup for execution scope.
       */
      if (b->wa_glslang_cs_barrier &&
          b->nb.shader->info.stage == MESA_SHADER_COMPUTE &&
          (execution_scope == SpvScopeWorkgroup ||
           execution_scope == SpvScopeDevice) &&
          memory_semantics == SpvMemorySemanticsMaskNone) {
         execution_scope = SpvScopeWorkgroup;
         memory_scope = SpvScopeWorkgroup;
         memory_semantics = SpvMemorySemanticsAcquireReleaseMask |
                            SpvMemorySemanticsWorkgroupMemoryMask;
      }

      /* From the SPIR-V spec:
       *
       *    "When used with the TessellationControl execution model, it also
       *    implicitly synchronizes the Output Storage Class: Writes to Output
       *    variables performed by any invocation executed prior to a
       *    OpControlBarrier will be visible to any other invocation after
       *    return from that OpControlBarrier."
       *
       * The same applies to VK_NV_mesh_shader.
       */
      if (b->nb.shader->info.stage == MESA_SHADER_TESS_CTRL ||
          b->nb.shader->info.stage == MESA_SHADER_TASK ||
          b->nb.shader->info.stage == MESA_SHADER_MESH) {
         memory_semantics &= ~(SpvMemorySemanticsAcquireMask |
                               SpvMemorySemanticsReleaseMask |
                               SpvMemorySemanticsAcquireReleaseMask |
                               SpvMemorySemanticsSequentiallyConsistentMask);
         memory_semantics |= SpvMemorySemanticsAcquireReleaseMask |
                             SpvMemorySemanticsOutputMemoryMask;
      }

      if (b->shader->options->use_scoped_barrier) {
         vtn_emit_scoped_control_barrier(b, execution_scope, memory_scope,
                                         memory_semantics);
      } else {
         vtn_emit_memory_barrier(b, memory_scope, memory_semantics);

         if (execution_scope == SpvScopeWorkgroup)
            nir_control_barrier(&b->nb);
      }
      break;
   }

   default:
      unreachable("unknown barrier instruction");
   }
}

static enum tess_primitive_mode
tess_primitive_mode_from_spv_execution_mode(struct vtn_builder *b,
                                            SpvExecutionMode mode)
{
   switch (mode) {
   case SpvExecutionModeTriangles:
      return TESS_PRIMITIVE_TRIANGLES;
   case SpvExecutionModeQuads:
      return TESS_PRIMITIVE_QUADS;
   case SpvExecutionModeIsolines:
      return TESS_PRIMITIVE_ISOLINES;
   default:
      vtn_fail("Invalid tess primitive type: %s (%u)",
               spirv_executionmode_to_string(mode), mode);
   }
}

static enum shader_prim
primitive_from_spv_execution_mode(struct vtn_builder *b,
                                  SpvExecutionMode mode)
{
   switch (mode) {
   case SpvExecutionModeInputPoints:
   case SpvExecutionModeOutputPoints:
      return SHADER_PRIM_POINTS;
   case SpvExecutionModeInputLines:
   case SpvExecutionModeOutputLinesNV:
      return SHADER_PRIM_LINES;
   case SpvExecutionModeInputLinesAdjacency:
      return SHADER_PRIM_LINES_ADJACENCY;
   case SpvExecutionModeTriangles:
   case SpvExecutionModeOutputTrianglesNV:
      return SHADER_PRIM_TRIANGLES;
   case SpvExecutionModeInputTrianglesAdjacency:
      return SHADER_PRIM_TRIANGLES_ADJACENCY;
   case SpvExecutionModeQuads:
      return SHADER_PRIM_QUADS;
   case SpvExecutionModeOutputLineStrip:
      return SHADER_PRIM_LINE_STRIP;
   case SpvExecutionModeOutputTriangleStrip:
      return SHADER_PRIM_TRIANGLE_STRIP;
   default:
      vtn_fail("Invalid primitive type: %s (%u)",
               spirv_executionmode_to_string(mode), mode);
   }
}

static unsigned
vertices_in_from_spv_execution_mode(struct vtn_builder *b,
                                    SpvExecutionMode mode)
{
   switch (mode) {
   case SpvExecutionModeInputPoints:
      return 1;
   case SpvExecutionModeInputLines:
      return 2;
   case SpvExecutionModeInputLinesAdjacency:
      return 4;
   case SpvExecutionModeTriangles:
      return 3;
   case SpvExecutionModeInputTrianglesAdjacency:
      return 6;
   default:
      vtn_fail("Invalid GS input mode: %s (%u)",
               spirv_executionmode_to_string(mode), mode);
   }
}

static gl_shader_stage
stage_for_execution_model(struct vtn_builder *b, SpvExecutionModel model)
{
   switch (model) {
   case SpvExecutionModelVertex:
      return MESA_SHADER_VERTEX;
   case SpvExecutionModelTessellationControl:
      return MESA_SHADER_TESS_CTRL;
   case SpvExecutionModelTessellationEvaluation:
      return MESA_SHADER_TESS_EVAL;
   case SpvExecutionModelGeometry:
      return MESA_SHADER_GEOMETRY;
   case SpvExecutionModelFragment:
      return MESA_SHADER_FRAGMENT;
   case SpvExecutionModelGLCompute:
      return MESA_SHADER_COMPUTE;
   case SpvExecutionModelKernel:
      return MESA_SHADER_KERNEL;
   case SpvExecutionModelRayGenerationKHR:
      return MESA_SHADER_RAYGEN;
   case SpvExecutionModelAnyHitKHR:
      return MESA_SHADER_ANY_HIT;
   case SpvExecutionModelClosestHitKHR:
      return MESA_SHADER_CLOSEST_HIT;
   case SpvExecutionModelMissKHR:
      return MESA_SHADER_MISS;
   case SpvExecutionModelIntersectionKHR:
      return MESA_SHADER_INTERSECTION;
   case SpvExecutionModelCallableKHR:
       return MESA_SHADER_CALLABLE;
   case SpvExecutionModelTaskNV:
   case SpvExecutionModelTaskEXT:
      return MESA_SHADER_TASK;
   case SpvExecutionModelMeshNV:
   case SpvExecutionModelMeshEXT:
      return MESA_SHADER_MESH;
   default:
      vtn_fail("Unsupported execution model: %s (%u)",
               spirv_executionmodel_to_string(model), model);
   }
}

#define spv_check_supported(name, cap) do {                 \
      if (!(b->options && b->options->caps.name))           \
         vtn_warn("Unsupported SPIR-V capability: %s (%u)", \
                  spirv_capability_to_string(cap), cap);    \
   } while(0)


void
vtn_handle_entry_point(struct vtn_builder *b, const uint32_t *w,
                       unsigned count)
{
   struct vtn_value *entry_point = &b->values[w[2]];
   /* Let this be a name label regardless */
   unsigned name_words;
   entry_point->name = vtn_string_literal(b, &w[3], count - 3, &name_words);

   if (strcmp(entry_point->name, b->entry_point_name) != 0 ||
       stage_for_execution_model(b, w[1]) != b->entry_point_stage)
      return;

   vtn_assert(b->entry_point == NULL);
   b->entry_point = entry_point;

   /* Entry points enumerate which global variables are used. */
   size_t start = 3 + name_words;
   b->interface_ids_count = count - start;
   b->interface_ids = ralloc_array(b, uint32_t, b->interface_ids_count);
   memcpy(b->interface_ids, &w[start], b->interface_ids_count * 4);
   qsort(b->interface_ids, b->interface_ids_count, 4, cmp_uint32_t);
}

static bool
vtn_handle_preamble_instruction(struct vtn_builder *b, SpvOp opcode,
                                const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpSource: {
      const char *lang;
      switch (w[1]) {
      default:
      case SpvSourceLanguageUnknown:      lang = "unknown";    break;
      case SpvSourceLanguageESSL:         lang = "ESSL";       break;
      case SpvSourceLanguageGLSL:         lang = "GLSL";       break;
      case SpvSourceLanguageOpenCL_C:     lang = "OpenCL C";   break;
      case SpvSourceLanguageOpenCL_CPP:   lang = "OpenCL C++"; break;
      case SpvSourceLanguageHLSL:         lang = "HLSL";       break;
      }

      uint32_t version = w[2];

      const char *file =
         (count > 3) ? vtn_value(b, w[3], vtn_value_type_string)->str : "";

      vtn_info("Parsing SPIR-V from %s %u source file %s", lang, version, file);

      b->source_lang = w[1];
      break;
   }

   case SpvOpExtension: {
      /* Implementing both NV_mesh_shader and EXT_mesh_shader
       * is difficult without knowing which we're dealing with.
       * TODO: remove this when we stop supporting NV_mesh_shader.
       */
      const char *ext_name = (const char *)&w[1];
      if (strcmp(ext_name, "SPV_NV_mesh_shader") == 0)
         b->shader->info.mesh.nv = true;
      break;
   }
   case SpvOpSourceExtension:
   case SpvOpSourceContinued:
   case SpvOpModuleProcessed:
      /* Unhandled, but these are for debug so that's ok. */
      break;

   case SpvOpCapability: {
      SpvCapability cap = w[1];
      switch (cap) {
      case SpvCapabilityMatrix:
      case SpvCapabilityShader:
      case SpvCapabilityGeometry:
      case SpvCapabilityGeometryPointSize:
      case SpvCapabilityUniformBufferArrayDynamicIndexing:
      case SpvCapabilitySampledImageArrayDynamicIndexing:
      case SpvCapabilityStorageBufferArrayDynamicIndexing:
      case SpvCapabilityStorageImageArrayDynamicIndexing:
      case SpvCapabilityImageRect:
      case SpvCapabilitySampledRect:
      case SpvCapabilitySampled1D:
      case SpvCapabilityImage1D:
      case SpvCapabilitySampledCubeArray:
      case SpvCapabilityImageCubeArray:
      case SpvCapabilitySampledBuffer:
      case SpvCapabilityImageBuffer:
      case SpvCapabilityImageQuery:
      case SpvCapabilityDerivativeControl:
      case SpvCapabilityInterpolationFunction:
      case SpvCapabilityMultiViewport:
      case SpvCapabilitySampleRateShading:
      case SpvCapabilityClipDistance:
      case SpvCapabilityCullDistance:
      case SpvCapabilityInputAttachment:
      case SpvCapabilityImageGatherExtended:
      case SpvCapabilityStorageImageExtendedFormats:
      case SpvCapabilityVector16:
      case SpvCapabilityDotProduct:
      case SpvCapabilityDotProductInputAll:
      case SpvCapabilityDotProductInput4x8Bit:
      case SpvCapabilityDotProductInput4x8BitPacked:
         break;

      case SpvCapabilityLinkage:
         if (!b->options->create_library)
            vtn_warn("Unsupported SPIR-V capability: %s",
                     spirv_capability_to_string(cap));
         spv_check_supported(linkage, cap);
         break;

      case SpvCapabilitySparseResidency:
         spv_check_supported(sparse_residency, cap);
         break;

      case SpvCapabilityMinLod:
         spv_check_supported(min_lod, cap);
         break;

      case SpvCapabilityAtomicStorage:
         spv_check_supported(atomic_storage, cap);
         break;

      case SpvCapabilityFloat64:
         spv_check_supported(float64, cap);
         break;
      case SpvCapabilityInt64:
         spv_check_supported(int64, cap);
         break;
      case SpvCapabilityInt16:
         spv_check_supported(int16, cap);
         break;
      case SpvCapabilityInt8:
         spv_check_supported(int8, cap);
         break;

      case SpvCapabilityTransformFeedback:
         spv_check_supported(transform_feedback, cap);
         break;

      case SpvCapabilityGeometryStreams:
         spv_check_supported(geometry_streams, cap);
         break;

      case SpvCapabilityInt64Atomics:
         spv_check_supported(int64_atomics, cap);
         break;

      case SpvCapabilityStorageImageMultisample:
         spv_check_supported(storage_image_ms, cap);
         break;

      case SpvCapabilityAddresses:
         spv_check_supported(address, cap);
         break;

      case SpvCapabilityKernel:
      case SpvCapabilityFloat16Buffer:
         spv_check_supported(kernel, cap);
         break;

      case SpvCapabilityGenericPointer:
         spv_check_supported(generic_pointers, cap);
         break;

      case SpvCapabilityImageBasic:
         spv_check_supported(kernel_image, cap);
         break;

      case SpvCapabilityImageReadWrite:
         spv_check_supported(kernel_image_read_write, cap);
         break;

      case SpvCapabilityLiteralSampler:
         spv_check_supported(literal_sampler, cap);
         break;

      case SpvCapabilityImageMipmap:
      case SpvCapabilityPipes:
      case SpvCapabilityDeviceEnqueue:
         vtn_warn("Unsupported OpenCL-style SPIR-V capability: %s",
                  spirv_capability_to_string(cap));
         break;

      case SpvCapabilityImageMSArray:
         spv_check_supported(image_ms_array, cap);
         break;

      case SpvCapabilityTessellation:
      case SpvCapabilityTessellationPointSize:
         spv_check_supported(tessellation, cap);
         break;

      case SpvCapabilityDrawParameters:
         spv_check_supported(draw_parameters, cap);
         break;

      case SpvCapabilityStorageImageReadWithoutFormat:
         spv_check_supported(image_read_without_format, cap);
         break;

      case SpvCapabilityStorageImageWriteWithoutFormat:
         spv_check_supported(image_write_without_format, cap);
         break;

      case SpvCapabilityDeviceGroup:
         spv_check_supported(device_group, cap);
         break;

      case SpvCapabilityMultiView:
         spv_check_supported(multiview, cap);
         break;

      case SpvCapabilityGroupNonUniform:
         spv_check_supported(subgroup_basic, cap);
         break;

      case SpvCapabilitySubgroupVoteKHR:
      case SpvCapabilityGroupNonUniformVote:
         spv_check_supported(subgroup_vote, cap);
         break;

      case SpvCapabilitySubgroupBallotKHR:
      case SpvCapabilityGroupNonUniformBallot:
         spv_check_supported(subgroup_ballot, cap);
         break;

      case SpvCapabilityGroupNonUniformShuffle:
      case SpvCapabilityGroupNonUniformShuffleRelative:
         spv_check_supported(subgroup_shuffle, cap);
         break;

      case SpvCapabilityGroupNonUniformQuad:
         spv_check_supported(subgroup_quad, cap);
         break;

      case SpvCapabilityGroupNonUniformArithmetic:
      case SpvCapabilityGroupNonUniformClustered:
         spv_check_supported(subgroup_arithmetic, cap);
         break;

      case SpvCapabilityGroups:
         spv_check_supported(groups, cap);
         break;

      case SpvCapabilitySubgroupDispatch:
         spv_check_supported(subgroup_dispatch, cap);
         /* Missing :
          *   - SpvOpGetKernelLocalSizeForSubgroupCount
          *   - SpvOpGetKernelMaxNumSubgroups
          *   - SpvExecutionModeSubgroupsPerWorkgroup
          *   - SpvExecutionModeSubgroupsPerWorkgroupId
          */
         vtn_warn("Not fully supported capability: %s",
                  spirv_capability_to_string(cap));
         break;

      case SpvCapabilityVariablePointersStorageBuffer:
      case SpvCapabilityVariablePointers:
         spv_check_supported(variable_pointers, cap);
         b->variable_pointers = true;
         break;

      case SpvCapabilityStorageUniformBufferBlock16:
      case SpvCapabilityStorageUniform16:
      case SpvCapabilityStoragePushConstant16:
      case SpvCapabilityStorageInputOutput16:
         spv_check_supported(storage_16bit, cap);
         break;

      case SpvCapabilityShaderLayer:
      case SpvCapabilityShaderViewportIndex:
      case SpvCapabilityShaderViewportIndexLayerEXT:
         spv_check_supported(shader_viewport_index_layer, cap);
         break;

      case SpvCapabilityStorageBuffer8BitAccess:
      case SpvCapabilityUniformAndStorageBuffer8BitAccess:
      case SpvCapabilityStoragePushConstant8:
         spv_check_supported(storage_8bit, cap);
         break;

      case SpvCapabilityShaderNonUniformEXT:
         spv_check_supported(descriptor_indexing, cap);
         break;

      case SpvCapabilityInputAttachmentArrayDynamicIndexingEXT:
      case SpvCapabilityUniformTexelBufferArrayDynamicIndexingEXT:
      case SpvCapabilityStorageTexelBufferArrayDynamicIndexingEXT:
         spv_check_supported(descriptor_array_dynamic_indexing, cap);
         break;

      case SpvCapabilityUniformBufferArrayNonUniformIndexingEXT:
      case SpvCapabilitySampledImageArrayNonUniformIndexingEXT:
      case SpvCapabilityStorageBufferArrayNonUniformIndexingEXT:
      case SpvCapabilityStorageImageArrayNonUniformIndexingEXT:
      case SpvCapabilityInputAttachmentArrayNonUniformIndexingEXT:
      case SpvCapabilityUniformTexelBufferArrayNonUniformIndexingEXT:
      case SpvCapabilityStorageTexelBufferArrayNonUniformIndexingEXT:
         spv_check_supported(descriptor_array_non_uniform_indexing, cap);
         break;

      case SpvCapabilityRuntimeDescriptorArrayEXT:
         spv_check_supported(runtime_descriptor_array, cap);
         break;

      case SpvCapabilityStencilExportEXT:
         spv_check_supported(stencil_export, cap);
         break;

      case SpvCapabilitySampleMaskPostDepthCoverage:
         spv_check_supported(post_depth_coverage, cap);
         break;

      case SpvCapabilityDenormFlushToZero:
      case SpvCapabilityDenormPreserve:
      case SpvCapabilitySignedZeroInfNanPreserve:
      case SpvCapabilityRoundingModeRTE:
      case SpvCapabilityRoundingModeRTZ:
         spv_check_supported(float_controls, cap);
         break;

      case SpvCapabilityPhysicalStorageBufferAddresses:
         spv_check_supported(physical_storage_buffer_address, cap);
         break;

      case SpvCapabilityComputeDerivativeGroupQuadsNV:
      case SpvCapabilityComputeDerivativeGroupLinearNV:
         spv_check_supported(derivative_group, cap);
         break;

      case SpvCapabilityFloat16:
         spv_check_supported(float16, cap);
         break;

      case SpvCapabilityFragmentShaderSampleInterlockEXT:
         spv_check_supported(fragment_shader_sample_interlock, cap);
         break;

      case SpvCapabilityFragmentShaderPixelInterlockEXT:
         spv_check_supported(fragment_shader_pixel_interlock, cap);
         break;

      case SpvCapabilityDemoteToHelperInvocation:
         spv_check_supported(demote_to_helper_invocation, cap);
         b->uses_demote_to_helper_invocation = true;
         break;

      case SpvCapabilityShaderClockKHR:
         spv_check_supported(shader_clock, cap);
	 break;

      case SpvCapabilityVulkanMemoryModel:
         spv_check_supported(vk_memory_model, cap);
         break;

      case SpvCapabilityVulkanMemoryModelDeviceScope:
         spv_check_supported(vk_memory_model_device_scope, cap);
         break;

      case SpvCapabilityImageReadWriteLodAMD:
         spv_check_supported(amd_image_read_write_lod, cap);
         break;

      case SpvCapabilityIntegerFunctions2INTEL:
         spv_check_supported(integer_functions2, cap);
         break;

      case SpvCapabilityFragmentMaskAMD:
         spv_check_supported(amd_fragment_mask, cap);
         break;

      case SpvCapabilityImageGatherBiasLodAMD:
         spv_check_supported(amd_image_gather_bias_lod, cap);
         break;

      case SpvCapabilityAtomicFloat16AddEXT:
         spv_check_supported(float16_atomic_add, cap);
         break;

      case SpvCapabilityAtomicFloat32AddEXT:
         spv_check_supported(float32_atomic_add, cap);
         break;

      case SpvCapabilityAtomicFloat64AddEXT:
         spv_check_supported(float64_atomic_add, cap);
         break;

      case SpvCapabilitySubgroupShuffleINTEL:
         spv_check_supported(intel_subgroup_shuffle, cap);
         break;

      case SpvCapabilitySubgroupBufferBlockIOINTEL:
         spv_check_supported(intel_subgroup_buffer_block_io, cap);
         break;

      case SpvCapabilityRayCullMaskKHR:
         spv_check_supported(ray_cull_mask, cap);
         break;

      case SpvCapabilityRayTracingKHR:
         spv_check_supported(ray_tracing, cap);
         break;

      case SpvCapabilityRayQueryKHR:
         spv_check_supported(ray_query, cap);
         break;

      case SpvCapabilityRayTraversalPrimitiveCullingKHR:
         spv_check_supported(ray_traversal_primitive_culling, cap);
         break;

      case SpvCapabilityInt64ImageEXT:
         spv_check_supported(image_atomic_int64, cap);
         break;

      case SpvCapabilityFragmentShadingRateKHR:
         spv_check_supported(fragment_shading_rate, cap);
         break;

      case SpvCapabilityWorkgroupMemoryExplicitLayoutKHR:
         spv_check_supported(workgroup_memory_explicit_layout, cap);
         break;

      case SpvCapabilityWorkgroupMemoryExplicitLayout8BitAccessKHR:
         spv_check_supported(workgroup_memory_explicit_layout, cap);
         spv_check_supported(storage_8bit, cap);
         break;

      case SpvCapabilityWorkgroupMemoryExplicitLayout16BitAccessKHR:
         spv_check_supported(workgroup_memory_explicit_layout, cap);
         spv_check_supported(storage_16bit, cap);
         break;

      case SpvCapabilityAtomicFloat16MinMaxEXT:
         spv_check_supported(float16_atomic_min_max, cap);
         break;

      case SpvCapabilityAtomicFloat32MinMaxEXT:
         spv_check_supported(float32_atomic_min_max, cap);
         break;

      case SpvCapabilityAtomicFloat64MinMaxEXT:
         spv_check_supported(float64_atomic_min_max, cap);
         break;

      case SpvCapabilityMeshShadingEXT:
         spv_check_supported(mesh_shading, cap);
         break;

      case SpvCapabilityMeshShadingNV:
         spv_check_supported(mesh_shading_nv, cap);
         break;

      case SpvCapabilityPerViewAttributesNV:
         spv_check_supported(per_view_attributes_nv, cap);
         break;

      case SpvCapabilityShaderViewportMaskNV:
         spv_check_supported(shader_viewport_mask_nv, cap);
         break;

      default:
         vtn_fail("Unhandled capability: %s (%u)",
                  spirv_capability_to_string(cap), cap);
      }
      break;
   }

   case SpvOpExtInstImport:
      vtn_handle_extension(b, opcode, w, count);
      break;

   case SpvOpMemoryModel:
      switch (w[1]) {
      case SpvAddressingModelPhysical32:
         vtn_fail_if(b->shader->info.stage != MESA_SHADER_KERNEL,
                     "AddressingModelPhysical32 only supported for kernels");
         b->shader->info.cs.ptr_size = 32;
         b->physical_ptrs = true;
         assert(nir_address_format_bit_size(b->options->global_addr_format) == 32);
         assert(nir_address_format_num_components(b->options->global_addr_format) == 1);
         assert(nir_address_format_bit_size(b->options->shared_addr_format) == 32);
         assert(nir_address_format_num_components(b->options->shared_addr_format) == 1);
         assert(nir_address_format_bit_size(b->options->constant_addr_format) == 32);
         assert(nir_address_format_num_components(b->options->constant_addr_format) == 1);
         break;
      case SpvAddressingModelPhysical64:
         vtn_fail_if(b->shader->info.stage != MESA_SHADER_KERNEL,
                     "AddressingModelPhysical64 only supported for kernels");
         b->shader->info.cs.ptr_size = 64;
         b->physical_ptrs = true;
         assert(nir_address_format_bit_size(b->options->global_addr_format) == 64);
         assert(nir_address_format_num_components(b->options->global_addr_format) == 1);
         assert(nir_address_format_bit_size(b->options->shared_addr_format) == 64);
         assert(nir_address_format_num_components(b->options->shared_addr_format) == 1);
         assert(nir_address_format_bit_size(b->options->constant_addr_format) == 64);
         assert(nir_address_format_num_components(b->options->constant_addr_format) == 1);
         break;
      case SpvAddressingModelLogical:
         vtn_fail_if(b->shader->info.stage == MESA_SHADER_KERNEL,
                     "AddressingModelLogical only supported for shaders");
         b->physical_ptrs = false;
         break;
      case SpvAddressingModelPhysicalStorageBuffer64:
         vtn_fail_if(!b->options ||
                     !b->options->caps.physical_storage_buffer_address,
                     "AddressingModelPhysicalStorageBuffer64 not supported");
         break;
      default:
         vtn_fail("Unknown addressing model: %s (%u)",
                  spirv_addressingmodel_to_string(w[1]), w[1]);
         break;
      }

      b->mem_model = w[2];
      switch (w[2]) {
      case SpvMemoryModelSimple:
      case SpvMemoryModelGLSL450:
      case SpvMemoryModelOpenCL:
         break;
      case SpvMemoryModelVulkan:
         vtn_fail_if(!b->options->caps.vk_memory_model,
                     "Vulkan memory model is unsupported by this driver");
         break;
      default:
         vtn_fail("Unsupported memory model: %s",
                  spirv_memorymodel_to_string(w[2]));
         break;
      }
      break;

   case SpvOpEntryPoint:
      vtn_handle_entry_point(b, w, count);
      break;

   case SpvOpString:
      vtn_push_value(b, w[1], vtn_value_type_string)->str =
         vtn_string_literal(b, &w[2], count - 2, NULL);
      break;

   case SpvOpName:
      b->values[w[1]].name = vtn_string_literal(b, &w[2], count - 2, NULL);
      break;

   case SpvOpMemberName:
   case SpvOpExecutionMode:
   case SpvOpExecutionModeId:
   case SpvOpDecorationGroup:
   case SpvOpDecorate:
   case SpvOpDecorateId:
   case SpvOpMemberDecorate:
   case SpvOpGroupDecorate:
   case SpvOpGroupMemberDecorate:
   case SpvOpDecorateString:
   case SpvOpMemberDecorateString:
      vtn_handle_decoration(b, opcode, w, count);
      break;

   case SpvOpExtInst: {
      struct vtn_value *val = vtn_value(b, w[3], vtn_value_type_extension);
      if (val->ext_handler == vtn_handle_non_semantic_instruction) {
         /* NonSemantic extended instructions are acceptable in preamble. */
         vtn_handle_non_semantic_instruction(b, w[4], w, count);
         return true;
      } else {
         return false; /* End of preamble. */
      }
   }

   default:
      return false; /* End of preamble */
   }

   return true;
}

static void
vtn_handle_execution_mode(struct vtn_builder *b, struct vtn_value *entry_point,
                          const struct vtn_decoration *mode, UNUSED void *data)
{
   vtn_assert(b->entry_point == entry_point);

   switch(mode->exec_mode) {
   case SpvExecutionModeOriginUpperLeft:
   case SpvExecutionModeOriginLowerLeft:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.origin_upper_left =
         (mode->exec_mode == SpvExecutionModeOriginUpperLeft);
      break;

   case SpvExecutionModeEarlyFragmentTests:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.early_fragment_tests = true;
      break;

   case SpvExecutionModePostDepthCoverage:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.post_depth_coverage = true;
      break;

   case SpvExecutionModeInvocations:
      vtn_assert(b->shader->info.stage == MESA_SHADER_GEOMETRY);
      b->shader->info.gs.invocations = MAX2(1, mode->operands[0]);
      break;

   case SpvExecutionModeDepthReplacing:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_ANY;
      break;
   case SpvExecutionModeDepthGreater:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_GREATER;
      break;
   case SpvExecutionModeDepthLess:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_LESS;
      break;
   case SpvExecutionModeDepthUnchanged:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_UNCHANGED;
      break;

   case SpvExecutionModeLocalSizeHint:
      vtn_assert(b->shader->info.stage == MESA_SHADER_KERNEL);
      b->shader->info.cs.workgroup_size_hint[0] = mode->operands[0];
      b->shader->info.cs.workgroup_size_hint[1] = mode->operands[1];
      b->shader->info.cs.workgroup_size_hint[2] = mode->operands[2];
      break;

   case SpvExecutionModeLocalSize:
      if (gl_shader_stage_uses_workgroup(b->shader->info.stage)) {
         b->shader->info.workgroup_size[0] = mode->operands[0];
         b->shader->info.workgroup_size[1] = mode->operands[1];
         b->shader->info.workgroup_size[2] = mode->operands[2];
      } else {
         vtn_fail("Execution mode LocalSize not supported in stage %s",
                  _mesa_shader_stage_to_string(b->shader->info.stage));
      }
      break;

   case SpvExecutionModeOutputVertices:
      switch (b->shader->info.stage) {
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
         b->shader->info.tess.tcs_vertices_out = mode->operands[0];
         break;
      case MESA_SHADER_GEOMETRY:
         b->shader->info.gs.vertices_out = mode->operands[0];
         break;
      case MESA_SHADER_MESH:
         b->shader->info.mesh.max_vertices_out = mode->operands[0];
         break;
      default:
         vtn_fail("Execution mode OutputVertices not supported in stage %s",
                  _mesa_shader_stage_to_string(b->shader->info.stage));
         break;
      }
      break;

   case SpvExecutionModeInputPoints:
   case SpvExecutionModeInputLines:
   case SpvExecutionModeInputLinesAdjacency:
   case SpvExecutionModeTriangles:
   case SpvExecutionModeInputTrianglesAdjacency:
   case SpvExecutionModeQuads:
   case SpvExecutionModeIsolines:
      if (b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
          b->shader->info.stage == MESA_SHADER_TESS_EVAL) {
         b->shader->info.tess._primitive_mode =
            tess_primitive_mode_from_spv_execution_mode(b, mode->exec_mode);
      } else {
         vtn_assert(b->shader->info.stage == MESA_SHADER_GEOMETRY);
         b->shader->info.gs.vertices_in =
            vertices_in_from_spv_execution_mode(b, mode->exec_mode);
         b->shader->info.gs.input_primitive =
            primitive_from_spv_execution_mode(b, mode->exec_mode);
      }
      break;

   case SpvExecutionModeOutputPrimitivesNV:
      vtn_assert(b->shader->info.stage == MESA_SHADER_MESH);
      b->shader->info.mesh.max_primitives_out = mode->operands[0];
      break;

   case SpvExecutionModeOutputLinesNV:
   case SpvExecutionModeOutputTrianglesNV:
      vtn_assert(b->shader->info.stage == MESA_SHADER_MESH);
      b->shader->info.mesh.primitive_type =
         primitive_from_spv_execution_mode(b, mode->exec_mode);
      break;

   case SpvExecutionModeOutputPoints: {
      const unsigned primitive =
         primitive_from_spv_execution_mode(b, mode->exec_mode);

      switch (b->shader->info.stage) {
      case MESA_SHADER_GEOMETRY:
         b->shader->info.gs.output_primitive = primitive;
         break;
      case MESA_SHADER_MESH:
         b->shader->info.mesh.primitive_type = primitive;
         break;
      default:
         vtn_fail("Execution mode OutputPoints not supported in stage %s",
                  _mesa_shader_stage_to_string(b->shader->info.stage));
         break;
      }
      break;
   }

   case SpvExecutionModeOutputLineStrip:
   case SpvExecutionModeOutputTriangleStrip:
      vtn_assert(b->shader->info.stage == MESA_SHADER_GEOMETRY);
      b->shader->info.gs.output_primitive =
         primitive_from_spv_execution_mode(b, mode->exec_mode);
      break;

   case SpvExecutionModeSpacingEqual:
      vtn_assert(b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL);
      b->shader->info.tess.spacing = TESS_SPACING_EQUAL;
      break;
   case SpvExecutionModeSpacingFractionalEven:
      vtn_assert(b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL);
      b->shader->info.tess.spacing = TESS_SPACING_FRACTIONAL_EVEN;
      break;
   case SpvExecutionModeSpacingFractionalOdd:
      vtn_assert(b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL);
      b->shader->info.tess.spacing = TESS_SPACING_FRACTIONAL_ODD;
      break;
   case SpvExecutionModeVertexOrderCw:
      vtn_assert(b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL);
      b->shader->info.tess.ccw = false;
      break;
   case SpvExecutionModeVertexOrderCcw:
      vtn_assert(b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL);
      b->shader->info.tess.ccw = true;
      break;
   case SpvExecutionModePointMode:
      vtn_assert(b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL);
      b->shader->info.tess.point_mode = true;
      break;

   case SpvExecutionModePixelCenterInteger:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.pixel_center_integer = true;
      break;

   case SpvExecutionModeXfb:
      b->shader->info.has_transform_feedback_varyings = true;
      break;

   case SpvExecutionModeVecTypeHint:
      break; /* OpenCL */

   case SpvExecutionModeContractionOff:
      if (b->shader->info.stage != MESA_SHADER_KERNEL)
         vtn_warn("ExectionMode only allowed for CL-style kernels: %s",
                  spirv_executionmode_to_string(mode->exec_mode));
      else
         b->exact = true;
      break;

   case SpvExecutionModeStencilRefReplacingEXT:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      break;

   case SpvExecutionModeDerivativeGroupQuadsNV:
      vtn_assert(b->shader->info.stage == MESA_SHADER_COMPUTE);
      b->shader->info.cs.derivative_group = DERIVATIVE_GROUP_QUADS;
      break;

   case SpvExecutionModeDerivativeGroupLinearNV:
      vtn_assert(b->shader->info.stage == MESA_SHADER_COMPUTE);
      b->shader->info.cs.derivative_group = DERIVATIVE_GROUP_LINEAR;
      break;

   case SpvExecutionModePixelInterlockOrderedEXT:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.pixel_interlock_ordered = true;
      break;

   case SpvExecutionModePixelInterlockUnorderedEXT:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.pixel_interlock_unordered = true;
      break;

   case SpvExecutionModeSampleInterlockOrderedEXT:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.sample_interlock_ordered = true;
      break;

   case SpvExecutionModeSampleInterlockUnorderedEXT:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.sample_interlock_unordered = true;
      break;

   case SpvExecutionModeDenormPreserve:
   case SpvExecutionModeDenormFlushToZero:
   case SpvExecutionModeSignedZeroInfNanPreserve:
   case SpvExecutionModeRoundingModeRTE:
   case SpvExecutionModeRoundingModeRTZ: {
      unsigned execution_mode = 0;
      switch (mode->exec_mode) {
      case SpvExecutionModeDenormPreserve:
         switch (mode->operands[0]) {
         case 16: execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP16; break;
         case 32: execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32; break;
         case 64: execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP64; break;
         default: vtn_fail("Floating point type not supported");
         }
         break;
      case SpvExecutionModeDenormFlushToZero:
         switch (mode->operands[0]) {
         case 16: execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16; break;
         case 32: execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32; break;
         case 64: execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP64; break;
         default: vtn_fail("Floating point type not supported");
         }
         break;
      case SpvExecutionModeSignedZeroInfNanPreserve:
         switch (mode->operands[0]) {
         case 16: execution_mode = FLOAT_CONTROLS_SIGNED_ZERO_INF_NAN_PRESERVE_FP16; break;
         case 32: execution_mode = FLOAT_CONTROLS_SIGNED_ZERO_INF_NAN_PRESERVE_FP32; break;
         case 64: execution_mode = FLOAT_CONTROLS_SIGNED_ZERO_INF_NAN_PRESERVE_FP64; break;
         default: vtn_fail("Floating point type not supported");
         }
         break;
      case SpvExecutionModeRoundingModeRTE:
         switch (mode->operands[0]) {
         case 16: execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP16; break;
         case 32: execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32; break;
         case 64: execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP64; break;
         default: vtn_fail("Floating point type not supported");
         }
         break;
      case SpvExecutionModeRoundingModeRTZ:
         switch (mode->operands[0]) {
         case 16: execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16; break;
         case 32: execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32; break;
         case 64: execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP64; break;
         default: vtn_fail("Floating point type not supported");
         }
         break;
      default:
         break;
      }

      b->shader->info.float_controls_execution_mode |= execution_mode;

      for (unsigned bit_size = 16; bit_size <= 64; bit_size *= 2) {
         vtn_fail_if(nir_is_denorm_flush_to_zero(b->shader->info.float_controls_execution_mode, bit_size) &&
                     nir_is_denorm_preserve(b->shader->info.float_controls_execution_mode, bit_size),
                     "Cannot flush to zero and preserve denorms for the same bit size.");
         vtn_fail_if(nir_is_rounding_mode_rtne(b->shader->info.float_controls_execution_mode, bit_size) &&
                     nir_is_rounding_mode_rtz(b->shader->info.float_controls_execution_mode, bit_size),
                     "Cannot set rounding mode to RTNE and RTZ for the same bit size.");
      }
      break;
   }

   case SpvExecutionModeLocalSizeId:
   case SpvExecutionModeLocalSizeHintId:
      /* Handled later by vtn_handle_execution_mode_id(). */
      break;

   case SpvExecutionModeSubgroupSize:
      vtn_assert(b->shader->info.stage == MESA_SHADER_KERNEL);
      vtn_assert(b->shader->info.subgroup_size == SUBGROUP_SIZE_VARYING);
      b->shader->info.subgroup_size = mode->operands[0];
      break;

   case SpvExecutionModeSubgroupUniformControlFlowKHR:
      /* There's no corresponding SPIR-V capability, so check here. */
      vtn_fail_if(!b->options->caps.subgroup_uniform_control_flow,
                  "SpvExecutionModeSubgroupUniformControlFlowKHR not supported.");
      break;

   case SpvExecutionModeEarlyAndLateFragmentTestsAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.early_and_late_fragment_tests = true;
      break;

   case SpvExecutionModeStencilRefGreaterFrontAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.stencil_front_layout = FRAG_STENCIL_LAYOUT_GREATER;
      break;

   case SpvExecutionModeStencilRefLessFrontAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.stencil_front_layout = FRAG_STENCIL_LAYOUT_LESS;
      break;

   case SpvExecutionModeStencilRefUnchangedFrontAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.stencil_front_layout = FRAG_STENCIL_LAYOUT_UNCHANGED;
      break;

   case SpvExecutionModeStencilRefGreaterBackAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.stencil_back_layout = FRAG_STENCIL_LAYOUT_GREATER;
      break;

   case SpvExecutionModeStencilRefLessBackAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.stencil_back_layout = FRAG_STENCIL_LAYOUT_LESS;
      break;

   case SpvExecutionModeStencilRefUnchangedBackAMD:
      vtn_assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      b->shader->info.fs.stencil_back_layout = FRAG_STENCIL_LAYOUT_UNCHANGED;
      break;

   default:
      vtn_fail("Unhandled execution mode: %s (%u)",
               spirv_executionmode_to_string(mode->exec_mode),
               mode->exec_mode);
   }
}

static void
vtn_handle_execution_mode_id(struct vtn_builder *b, struct vtn_value *entry_point,
                             const struct vtn_decoration *mode, UNUSED void *data)
{

   vtn_assert(b->entry_point == entry_point);

   switch (mode->exec_mode) {
   case SpvExecutionModeLocalSizeId:
      if (gl_shader_stage_uses_workgroup(b->shader->info.stage)) {
         b->shader->info.workgroup_size[0] = vtn_constant_uint(b, mode->operands[0]);
         b->shader->info.workgroup_size[1] = vtn_constant_uint(b, mode->operands[1]);
         b->shader->info.workgroup_size[2] = vtn_constant_uint(b, mode->operands[2]);
      } else {
         vtn_fail("Execution mode LocalSizeId not supported in stage %s",
                  _mesa_shader_stage_to_string(b->shader->info.stage));
      }
      break;

   case SpvExecutionModeLocalSizeHintId:
      vtn_assert(b->shader->info.stage == MESA_SHADER_KERNEL);
      b->shader->info.cs.workgroup_size_hint[0] = vtn_constant_uint(b, mode->operands[0]);
      b->shader->info.cs.workgroup_size_hint[1] = vtn_constant_uint(b, mode->operands[1]);
      b->shader->info.cs.workgroup_size_hint[2] = vtn_constant_uint(b, mode->operands[2]);
      break;

   default:
      /* Nothing to do.  Literal execution modes already handled by
       * vtn_handle_execution_mode(). */
      break;
   }
}

static bool
vtn_handle_variable_or_type_instruction(struct vtn_builder *b, SpvOp opcode,
                                        const uint32_t *w, unsigned count)
{
   vtn_set_instruction_result_type(b, opcode, w, count);

   switch (opcode) {
   case SpvOpSource:
   case SpvOpSourceContinued:
   case SpvOpSourceExtension:
   case SpvOpExtension:
   case SpvOpCapability:
   case SpvOpExtInstImport:
   case SpvOpMemoryModel:
   case SpvOpEntryPoint:
   case SpvOpExecutionMode:
   case SpvOpString:
   case SpvOpName:
   case SpvOpMemberName:
   case SpvOpDecorationGroup:
   case SpvOpDecorate:
   case SpvOpDecorateId:
   case SpvOpMemberDecorate:
   case SpvOpGroupDecorate:
   case SpvOpGroupMemberDecorate:
   case SpvOpDecorateString:
   case SpvOpMemberDecorateString:
      vtn_fail("Invalid opcode types and variables section");
      break;

   case SpvOpTypeVoid:
   case SpvOpTypeBool:
   case SpvOpTypeInt:
   case SpvOpTypeFloat:
   case SpvOpTypeVector:
   case SpvOpTypeMatrix:
   case SpvOpTypeImage:
   case SpvOpTypeSampler:
   case SpvOpTypeSampledImage:
   case SpvOpTypeArray:
   case SpvOpTypeRuntimeArray:
   case SpvOpTypeStruct:
   case SpvOpTypeOpaque:
   case SpvOpTypePointer:
   case SpvOpTypeForwardPointer:
   case SpvOpTypeFunction:
   case SpvOpTypeEvent:
   case SpvOpTypeDeviceEvent:
   case SpvOpTypeReserveId:
   case SpvOpTypeQueue:
   case SpvOpTypePipe:
   case SpvOpTypeAccelerationStructureKHR:
   case SpvOpTypeRayQueryKHR:
      vtn_handle_type(b, opcode, w, count);
      break;

   case SpvOpConstantTrue:
   case SpvOpConstantFalse:
   case SpvOpConstant:
   case SpvOpConstantComposite:
   case SpvOpConstantNull:
   case SpvOpSpecConstantTrue:
   case SpvOpSpecConstantFalse:
   case SpvOpSpecConstant:
   case SpvOpSpecConstantComposite:
   case SpvOpSpecConstantOp:
      vtn_handle_constant(b, opcode, w, count);
      break;

   case SpvOpUndef:
   case SpvOpVariable:
   case SpvOpConstantSampler:
      vtn_handle_variables(b, opcode, w, count);
      break;

   case SpvOpExtInst: {
      struct vtn_value *val = vtn_value(b, w[3], vtn_value_type_extension);
      /* NonSemantic extended instructions are acceptable in preamble, others
       * will indicate the end of preamble.
       */
      return val->ext_handler == vtn_handle_non_semantic_instruction;
   }

   default:
      return false; /* End of preamble */
   }

   return true;
}

static struct vtn_ssa_value *
vtn_nir_select(struct vtn_builder *b, struct vtn_ssa_value *src0,
               struct vtn_ssa_value *src1, struct vtn_ssa_value *src2)
{
   struct vtn_ssa_value *dest = rzalloc(b, struct vtn_ssa_value);
   dest->type = src1->type;

   if (glsl_type_is_vector_or_scalar(src1->type)) {
      dest->def = nir_bcsel(&b->nb, src0->def, src1->def, src2->def);
   } else {
      unsigned elems = glsl_get_length(src1->type);

      dest->elems = ralloc_array(b, struct vtn_ssa_value *, elems);
      for (unsigned i = 0; i < elems; i++) {
         dest->elems[i] = vtn_nir_select(b, src0,
                                         src1->elems[i], src2->elems[i]);
      }
   }

   return dest;
}

static void
vtn_handle_select(struct vtn_builder *b, SpvOp opcode,
                  const uint32_t *w, unsigned count)
{
   /* Handle OpSelect up-front here because it needs to be able to handle
    * pointers and not just regular vectors and scalars.
    */
   struct vtn_value *res_val = vtn_untyped_value(b, w[2]);
   struct vtn_value *cond_val = vtn_untyped_value(b, w[3]);
   struct vtn_value *obj1_val = vtn_untyped_value(b, w[4]);
   struct vtn_value *obj2_val = vtn_untyped_value(b, w[5]);

   vtn_fail_if(obj1_val->type != res_val->type ||
               obj2_val->type != res_val->type,
               "Object types must match the result type in OpSelect");

   vtn_fail_if((cond_val->type->base_type != vtn_base_type_scalar &&
                cond_val->type->base_type != vtn_base_type_vector) ||
               !glsl_type_is_boolean(cond_val->type->type),
               "OpSelect must have either a vector of booleans or "
               "a boolean as Condition type");

   vtn_fail_if(cond_val->type->base_type == vtn_base_type_vector &&
               (res_val->type->base_type != vtn_base_type_vector ||
                res_val->type->length != cond_val->type->length),
               "When Condition type in OpSelect is a vector, the Result "
               "type must be a vector of the same length");

   switch (res_val->type->base_type) {
   case vtn_base_type_scalar:
   case vtn_base_type_vector:
   case vtn_base_type_matrix:
   case vtn_base_type_array:
   case vtn_base_type_struct:
      /* OK. */
      break;
   case vtn_base_type_pointer:
      /* We need to have actual storage for pointer types. */
      vtn_fail_if(res_val->type->type == NULL,
                  "Invalid pointer result type for OpSelect");
      break;
   default:
      vtn_fail("Result type of OpSelect must be a scalar, composite, or pointer");
   }

   vtn_push_ssa_value(b, w[2],
      vtn_nir_select(b, vtn_ssa_value(b, w[3]),
                        vtn_ssa_value(b, w[4]),
                        vtn_ssa_value(b, w[5])));
}

static void
vtn_handle_ptr(struct vtn_builder *b, SpvOp opcode,
               const uint32_t *w, unsigned count)
{
   struct vtn_type *type1 = vtn_get_value_type(b, w[3]);
   struct vtn_type *type2 = vtn_get_value_type(b, w[4]);
   vtn_fail_if(type1->base_type != vtn_base_type_pointer ||
               type2->base_type != vtn_base_type_pointer,
               "%s operands must have pointer types",
               spirv_op_to_string(opcode));
   vtn_fail_if(type1->storage_class != type2->storage_class,
               "%s operands must have the same storage class",
               spirv_op_to_string(opcode));

   struct vtn_type *vtn_type = vtn_get_type(b, w[1]);
   const struct glsl_type *type = vtn_type->type;

   nir_address_format addr_format = vtn_mode_to_address_format(
      b, vtn_storage_class_to_mode(b, type1->storage_class, NULL, NULL));

   nir_ssa_def *def;

   switch (opcode) {
   case SpvOpPtrDiff: {
      /* OpPtrDiff returns the difference in number of elements (not byte offset). */
      unsigned elem_size, elem_align;
      glsl_get_natural_size_align_bytes(type1->deref->type,
                                        &elem_size, &elem_align);

      def = nir_build_addr_isub(&b->nb,
                                vtn_get_nir_ssa(b, w[3]),
                                vtn_get_nir_ssa(b, w[4]),
                                addr_format);
      def = nir_idiv(&b->nb, def, nir_imm_intN_t(&b->nb, elem_size, def->bit_size));
      def = nir_i2iN(&b->nb, def, glsl_get_bit_size(type));
      break;
   }

   case SpvOpPtrEqual:
   case SpvOpPtrNotEqual: {
      def = nir_build_addr_ieq(&b->nb,
                               vtn_get_nir_ssa(b, w[3]),
                               vtn_get_nir_ssa(b, w[4]),
                               addr_format);
      if (opcode == SpvOpPtrNotEqual)
         def = nir_inot(&b->nb, def);
      break;
   }

   default:
      unreachable("Invalid ptr operation");
   }

   vtn_push_nir_ssa(b, w[2], def);
}

static void
vtn_handle_ray_intrinsic(struct vtn_builder *b, SpvOp opcode,
                         const uint32_t *w, unsigned count)
{
   nir_intrinsic_instr *intrin;

   switch (opcode) {
   case SpvOpTraceNV:
   case SpvOpTraceRayKHR: {
      intrin = nir_intrinsic_instr_create(b->nb.shader,
                                          nir_intrinsic_trace_ray);

      /* The sources are in the same order in the NIR intrinsic */
      for (unsigned i = 0; i < 10; i++)
         intrin->src[i] = nir_src_for_ssa(vtn_ssa_value(b, w[i + 1])->def);

      nir_deref_instr *payload;
      if (opcode == SpvOpTraceNV)
         payload = vtn_get_call_payload_for_location(b, w[11]);
      else
         payload = vtn_nir_deref(b, w[11]);
      intrin->src[10] = nir_src_for_ssa(&payload->dest.ssa);
      nir_builder_instr_insert(&b->nb, &intrin->instr);
      break;
   }

   case SpvOpReportIntersectionKHR: {
      intrin = nir_intrinsic_instr_create(b->nb.shader,
                                          nir_intrinsic_report_ray_intersection);
      intrin->src[0] = nir_src_for_ssa(vtn_ssa_value(b, w[3])->def);
      intrin->src[1] = nir_src_for_ssa(vtn_ssa_value(b, w[4])->def);
      nir_ssa_dest_init(&intrin->instr, &intrin->dest, 1, 1, NULL);
      nir_builder_instr_insert(&b->nb, &intrin->instr);
      vtn_push_nir_ssa(b, w[2], &intrin->dest.ssa);
      break;
   }

   case SpvOpIgnoreIntersectionNV:
      intrin = nir_intrinsic_instr_create(b->nb.shader,
                                          nir_intrinsic_ignore_ray_intersection);
      nir_builder_instr_insert(&b->nb, &intrin->instr);
      break;

   case SpvOpTerminateRayNV:
      intrin = nir_intrinsic_instr_create(b->nb.shader,
                                          nir_intrinsic_terminate_ray);
      nir_builder_instr_insert(&b->nb, &intrin->instr);
      break;

   case SpvOpExecuteCallableNV:
   case SpvOpExecuteCallableKHR: {
      intrin = nir_intrinsic_instr_create(b->nb.shader,
                                          nir_intrinsic_execute_callable);
      intrin->src[0] = nir_src_for_ssa(vtn_ssa_value(b, w[1])->def);
      nir_deref_instr *payload;
      if (opcode == SpvOpExecuteCallableNV)
         payload = vtn_get_call_payload_for_location(b, w[2]);
      else
         payload = vtn_nir_deref(b, w[2]);
      intrin->src[1] = nir_src_for_ssa(&payload->dest.ssa);
      nir_builder_instr_insert(&b->nb, &intrin->instr);
      break;
   }

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }
}

static void
vtn_handle_write_packed_primitive_indices(struct vtn_builder *b, SpvOp opcode,
                                          const uint32_t *w, unsigned count)
{
   vtn_assert(opcode == SpvOpWritePackedPrimitiveIndices4x8NV);

   /* TODO(mesh): Use or create a primitive that allow the unpacking to
    * happen in the backend.  What we have here is functional but too
    * blunt.
    */

   struct vtn_type *offset_type = vtn_get_value_type(b, w[1]);
   vtn_fail_if(offset_type->base_type != vtn_base_type_scalar ||
               offset_type->type != glsl_uint_type(),
               "Index Offset type of OpWritePackedPrimitiveIndices4x8NV "
               "must be an OpTypeInt with 32-bit Width and 0 Signedness.");

   struct vtn_type *packed_type = vtn_get_value_type(b, w[2]);
   vtn_fail_if(packed_type->base_type != vtn_base_type_scalar ||
               packed_type->type != glsl_uint_type(),
               "Packed Indices type of OpWritePackedPrimitiveIndices4x8NV "
               "must be an OpTypeInt with 32-bit Width and 0 Signedness.");

   nir_deref_instr *indices = NULL;
   nir_foreach_variable_with_modes(var, b->nb.shader, nir_var_shader_out) {
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES) {
         indices = nir_build_deref_var(&b->nb, var);
         break;
      }
   }

   /* It may be the case that the variable is not present in the
    * entry point interface list.
    *
    * See https://github.com/KhronosGroup/SPIRV-Registry/issues/104.
    */

   if (!indices) {
      unsigned vertices_per_prim =
         num_mesh_vertices_per_primitive(b->shader->info.mesh.primitive_type);
      unsigned max_prim_indices =
         vertices_per_prim * b->shader->info.mesh.max_primitives_out;
      const struct glsl_type *t =
         glsl_array_type(glsl_uint_type(), max_prim_indices, 0);
      nir_variable *var =
         nir_variable_create(b->shader, nir_var_shader_out, t,
                             "gl_PrimitiveIndicesNV");

      var->data.location = VARYING_SLOT_PRIMITIVE_INDICES;
      var->data.interpolation = INTERP_MODE_NONE;
      indices = nir_build_deref_var(&b->nb, var);
   }

   nir_ssa_def *offset = vtn_get_nir_ssa(b, w[1]);
   nir_ssa_def *packed = vtn_get_nir_ssa(b, w[2]);
   nir_ssa_def *unpacked = nir_unpack_bits(&b->nb, packed, 8);
   for (int i = 0; i < 4; i++) {
      nir_deref_instr *offset_deref =
         nir_build_deref_array(&b->nb, indices,
                               nir_iadd_imm(&b->nb, offset, i));
      nir_ssa_def *val = nir_u2u32(&b->nb, nir_channel(&b->nb, unpacked, i));

      nir_store_deref(&b->nb, offset_deref, val, 0x1);
   }
}

struct ray_query_value {
   nir_ray_query_value     nir_value;
   const struct glsl_type *glsl_type;
};

static struct ray_query_value
spirv_to_nir_type_ray_query_intrinsic(struct vtn_builder *b,
                                      SpvOp opcode)
{
   switch (opcode) {
#define CASE(_spv, _nir, _type) case SpvOpRayQueryGet##_spv:            \
      return (struct ray_query_value) { .nir_value = nir_ray_query_value_##_nir, .glsl_type = _type }
      CASE(RayTMinKHR,                                            tmin,                               glsl_floatN_t_type(32));
      CASE(RayFlagsKHR,                                           flags,                              glsl_uint_type());
      CASE(WorldRayDirectionKHR,                                  world_ray_direction,                glsl_vec_type(3));
      CASE(WorldRayOriginKHR,                                     world_ray_origin,                   glsl_vec_type(3));
      CASE(IntersectionTypeKHR,                                   intersection_type,                  glsl_uint_type());
      CASE(IntersectionTKHR,                                      intersection_t,                     glsl_floatN_t_type(32));
      CASE(IntersectionInstanceCustomIndexKHR,                    intersection_instance_custom_index, glsl_int_type());
      CASE(IntersectionInstanceIdKHR,                             intersection_instance_id,           glsl_int_type());
      CASE(IntersectionInstanceShaderBindingTableRecordOffsetKHR, intersection_instance_sbt_index,    glsl_uint_type());
      CASE(IntersectionGeometryIndexKHR,                          intersection_geometry_index,        glsl_int_type());
      CASE(IntersectionPrimitiveIndexKHR,                         intersection_primitive_index,       glsl_int_type());
      CASE(IntersectionBarycentricsKHR,                           intersection_barycentrics,          glsl_vec_type(2));
      CASE(IntersectionFrontFaceKHR,                              intersection_front_face,            glsl_bool_type());
      CASE(IntersectionCandidateAABBOpaqueKHR,                    intersection_candidate_aabb_opaque, glsl_bool_type());
      CASE(IntersectionObjectToWorldKHR,                          intersection_object_to_world,       glsl_matrix_type(glsl_get_base_type(glsl_float_type()), 3, 4));
      CASE(IntersectionWorldToObjectKHR,                          intersection_world_to_object,       glsl_matrix_type(glsl_get_base_type(glsl_float_type()), 3, 4));
      CASE(IntersectionObjectRayOriginKHR,                        intersection_object_ray_origin,     glsl_vec_type(3));
      CASE(IntersectionObjectRayDirectionKHR,                     intersection_object_ray_direction,  glsl_vec_type(3));
#undef CASE
   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }
}

static void
ray_query_load_intrinsic_create(struct vtn_builder *b, SpvOp opcode,
                                const uint32_t *w, nir_ssa_def *src0,
                                nir_ssa_def *src1)
{
   struct ray_query_value value =
      spirv_to_nir_type_ray_query_intrinsic(b, opcode);

   if (glsl_type_is_matrix(value.glsl_type)) {
      const struct glsl_type *elem_type = glsl_get_array_element(value.glsl_type);
      const unsigned elems = glsl_get_length(value.glsl_type);

      struct vtn_ssa_value *ssa = vtn_create_ssa_value(b, value.glsl_type);
      for (unsigned i = 0; i < elems; i++) {
         ssa->elems[i]->def =
            nir_build_rq_load(&b->nb,
                              glsl_get_vector_elements(elem_type),
                              glsl_get_bit_size(elem_type),
                              src0, src1,
                              .ray_query_value = value.nir_value,
                              .column = i);
      }

      vtn_push_ssa_value(b, w[2], ssa);
   } else {
      assert(glsl_type_is_vector_or_scalar(value.glsl_type));

      vtn_push_nir_ssa(b, w[2],
                       nir_rq_load(&b->nb,
                                   glsl_get_vector_elements(value.glsl_type),
                                   glsl_get_bit_size(value.glsl_type),
                                   src0, src1,
                                   .ray_query_value = value.nir_value));
   }
}

static void
vtn_handle_ray_query_intrinsic(struct vtn_builder *b, SpvOp opcode,
                               const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpRayQueryInitializeKHR: {
      nir_intrinsic_instr *intrin =
         nir_intrinsic_instr_create(b->nb.shader,
                                    nir_intrinsic_rq_initialize);
      /* The sources are in the same order in the NIR intrinsic */
      for (unsigned i = 0; i < 8; i++)
         intrin->src[i] = nir_src_for_ssa(vtn_ssa_value(b, w[i + 1])->def);
      nir_builder_instr_insert(&b->nb, &intrin->instr);
      break;
   }

   case SpvOpRayQueryTerminateKHR:
      nir_rq_terminate(&b->nb, vtn_ssa_value(b, w[1])->def);
      break;

   case SpvOpRayQueryProceedKHR:
      vtn_push_nir_ssa(b, w[2],
                       nir_rq_proceed(&b->nb, 1, vtn_ssa_value(b, w[3])->def));
      break;

   case SpvOpRayQueryGenerateIntersectionKHR:
      nir_rq_generate_intersection(&b->nb,
                                   vtn_ssa_value(b, w[1])->def,
                                   vtn_ssa_value(b, w[2])->def);
      break;

   case SpvOpRayQueryConfirmIntersectionKHR:
      nir_rq_confirm_intersection(&b->nb, vtn_ssa_value(b, w[1])->def);
      break;

   case SpvOpRayQueryGetIntersectionTKHR:
   case SpvOpRayQueryGetIntersectionTypeKHR:
   case SpvOpRayQueryGetIntersectionInstanceCustomIndexKHR:
   case SpvOpRayQueryGetIntersectionInstanceIdKHR:
   case SpvOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
   case SpvOpRayQueryGetIntersectionGeometryIndexKHR:
   case SpvOpRayQueryGetIntersectionPrimitiveIndexKHR:
   case SpvOpRayQueryGetIntersectionBarycentricsKHR:
   case SpvOpRayQueryGetIntersectionFrontFaceKHR:
   case SpvOpRayQueryGetIntersectionObjectRayDirectionKHR:
   case SpvOpRayQueryGetIntersectionObjectRayOriginKHR:
   case SpvOpRayQueryGetIntersectionObjectToWorldKHR:
   case SpvOpRayQueryGetIntersectionWorldToObjectKHR:
      ray_query_load_intrinsic_create(b, opcode, w,
                                      vtn_ssa_value(b, w[3])->def,
                                      nir_i2b(&b->nb, vtn_ssa_value(b, w[4])->def));
      break;

   case SpvOpRayQueryGetRayTMinKHR:
   case SpvOpRayQueryGetRayFlagsKHR:
   case SpvOpRayQueryGetWorldRayDirectionKHR:
   case SpvOpRayQueryGetWorldRayOriginKHR:
   case SpvOpRayQueryGetIntersectionCandidateAABBOpaqueKHR:
      ray_query_load_intrinsic_create(b, opcode, w,
                                      vtn_ssa_value(b, w[3])->def,
                                      /* Committed value is ignored for these */
                                      nir_imm_bool(&b->nb, false));
      break;

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }
}

static bool
vtn_handle_body_instruction(struct vtn_builder *b, SpvOp opcode,
                            const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpLabel:
      break;

   case SpvOpLoopMerge:
   case SpvOpSelectionMerge:
      /* This is handled by cfg pre-pass and walk_blocks */
      break;

   case SpvOpUndef: {
      struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_undef);
      val->type = vtn_get_type(b, w[1]);
      break;
   }

   case SpvOpExtInst:
      vtn_handle_extension(b, opcode, w, count);
      break;

   case SpvOpVariable:
   case SpvOpLoad:
   case SpvOpStore:
   case SpvOpCopyMemory:
   case SpvOpCopyMemorySized:
   case SpvOpAccessChain:
   case SpvOpPtrAccessChain:
   case SpvOpInBoundsAccessChain:
   case SpvOpInBoundsPtrAccessChain:
   case SpvOpArrayLength:
   case SpvOpConvertPtrToU:
   case SpvOpConvertUToPtr:
   case SpvOpGenericCastToPtrExplicit:
   case SpvOpGenericPtrMemSemantics:
   case SpvOpSubgroupBlockReadINTEL:
   case SpvOpSubgroupBlockWriteINTEL:
   case SpvOpConvertUToAccelerationStructureKHR:
      vtn_handle_variables(b, opcode, w, count);
      break;

   case SpvOpFunctionCall:
      vtn_handle_function_call(b, opcode, w, count);
      break;

   case SpvOpSampledImage:
   case SpvOpImage:
   case SpvOpImageSparseTexelsResident:
   case SpvOpImageSampleImplicitLod:
   case SpvOpImageSparseSampleImplicitLod:
   case SpvOpImageSampleExplicitLod:
   case SpvOpImageSparseSampleExplicitLod:
   case SpvOpImageSampleDrefImplicitLod:
   case SpvOpImageSparseSampleDrefImplicitLod:
   case SpvOpImageSampleDrefExplicitLod:
   case SpvOpImageSparseSampleDrefExplicitLod:
   case SpvOpImageSampleProjImplicitLod:
   case SpvOpImageSampleProjExplicitLod:
   case SpvOpImageSampleProjDrefImplicitLod:
   case SpvOpImageSampleProjDrefExplicitLod:
   case SpvOpImageFetch:
   case SpvOpImageSparseFetch:
   case SpvOpImageGather:
   case SpvOpImageSparseGather:
   case SpvOpImageDrefGather:
   case SpvOpImageSparseDrefGather:
   case SpvOpImageQueryLod:
   case SpvOpImageQueryLevels:
      vtn_handle_texture(b, opcode, w, count);
      break;

   case SpvOpImageRead:
   case SpvOpImageSparseRead:
   case SpvOpImageWrite:
   case SpvOpImageTexelPointer:
   case SpvOpImageQueryFormat:
   case SpvOpImageQueryOrder:
      vtn_handle_image(b, opcode, w, count);
      break;

   case SpvOpImageQuerySamples:
   case SpvOpImageQuerySizeLod:
   case SpvOpImageQuerySize: {
      struct vtn_type *image_type = vtn_get_value_type(b, w[3]);
      vtn_assert(image_type->base_type == vtn_base_type_image);
      if (glsl_type_is_image(image_type->glsl_image)) {
         vtn_handle_image(b, opcode, w, count);
      } else {
         vtn_assert(glsl_type_is_texture(image_type->glsl_image));
         vtn_handle_texture(b, opcode, w, count);
      }
      break;
   }

   case SpvOpFragmentMaskFetchAMD:
   case SpvOpFragmentFetchAMD:
      vtn_handle_texture(b, opcode, w, count);
      break;

   case SpvOpAtomicLoad:
   case SpvOpAtomicExchange:
   case SpvOpAtomicCompareExchange:
   case SpvOpAtomicCompareExchangeWeak:
   case SpvOpAtomicIIncrement:
   case SpvOpAtomicIDecrement:
   case SpvOpAtomicIAdd:
   case SpvOpAtomicISub:
   case SpvOpAtomicSMin:
   case SpvOpAtomicUMin:
   case SpvOpAtomicSMax:
   case SpvOpAtomicUMax:
   case SpvOpAtomicAnd:
   case SpvOpAtomicOr:
   case SpvOpAtomicXor:
   case SpvOpAtomicFAddEXT:
   case SpvOpAtomicFMinEXT:
   case SpvOpAtomicFMaxEXT:
   case SpvOpAtomicFlagTestAndSet: {
      struct vtn_value *pointer = vtn_untyped_value(b, w[3]);
      if (pointer->value_type == vtn_value_type_image_pointer) {
         vtn_handle_image(b, opcode, w, count);
      } else {
         vtn_assert(pointer->value_type == vtn_value_type_pointer);
         vtn_handle_atomics(b, opcode, w, count);
      }
      break;
   }

   case SpvOpAtomicStore:
   case SpvOpAtomicFlagClear: {
      struct vtn_value *pointer = vtn_untyped_value(b, w[1]);
      if (pointer->value_type == vtn_value_type_image_pointer) {
         vtn_handle_image(b, opcode, w, count);
      } else {
         vtn_assert(pointer->value_type == vtn_value_type_pointer);
         vtn_handle_atomics(b, opcode, w, count);
      }
      break;
   }

   case SpvOpSelect:
      vtn_handle_select(b, opcode, w, count);
      break;

   case SpvOpSNegate:
   case SpvOpFNegate:
   case SpvOpNot:
   case SpvOpAny:
   case SpvOpAll:
   case SpvOpConvertFToU:
   case SpvOpConvertFToS:
   case SpvOpConvertSToF:
   case SpvOpConvertUToF:
   case SpvOpUConvert:
   case SpvOpSConvert:
   case SpvOpFConvert:
   case SpvOpQuantizeToF16:
   case SpvOpSatConvertSToU:
   case SpvOpSatConvertUToS:
   case SpvOpPtrCastToGeneric:
   case SpvOpGenericCastToPtr:
   case SpvOpIsNan:
   case SpvOpIsInf:
   case SpvOpIsFinite:
   case SpvOpIsNormal:
   case SpvOpSignBitSet:
   case SpvOpLessOrGreater:
   case SpvOpOrdered:
   case SpvOpUnordered:
   case SpvOpIAdd:
   case SpvOpFAdd:
   case SpvOpISub:
   case SpvOpFSub:
   case SpvOpIMul:
   case SpvOpFMul:
   case SpvOpUDiv:
   case SpvOpSDiv:
   case SpvOpFDiv:
   case SpvOpUMod:
   case SpvOpSRem:
   case SpvOpSMod:
   case SpvOpFRem:
   case SpvOpFMod:
   case SpvOpVectorTimesScalar:
   case SpvOpDot:
   case SpvOpIAddCarry:
   case SpvOpISubBorrow:
   case SpvOpUMulExtended:
   case SpvOpSMulExtended:
   case SpvOpShiftRightLogical:
   case SpvOpShiftRightArithmetic:
   case SpvOpShiftLeftLogical:
   case SpvOpLogicalEqual:
   case SpvOpLogicalNotEqual:
   case SpvOpLogicalOr:
   case SpvOpLogicalAnd:
   case SpvOpLogicalNot:
   case SpvOpBitwiseOr:
   case SpvOpBitwiseXor:
   case SpvOpBitwiseAnd:
   case SpvOpIEqual:
   case SpvOpFOrdEqual:
   case SpvOpFUnordEqual:
   case SpvOpINotEqual:
   case SpvOpFOrdNotEqual:
   case SpvOpFUnordNotEqual:
   case SpvOpULessThan:
   case SpvOpSLessThan:
   case SpvOpFOrdLessThan:
   case SpvOpFUnordLessThan:
   case SpvOpUGreaterThan:
   case SpvOpSGreaterThan:
   case SpvOpFOrdGreaterThan:
   case SpvOpFUnordGreaterThan:
   case SpvOpULessThanEqual:
   case SpvOpSLessThanEqual:
   case SpvOpFOrdLessThanEqual:
   case SpvOpFUnordLessThanEqual:
   case SpvOpUGreaterThanEqual:
   case SpvOpSGreaterThanEqual:
   case SpvOpFOrdGreaterThanEqual:
   case SpvOpFUnordGreaterThanEqual:
   case SpvOpDPdx:
   case SpvOpDPdy:
   case SpvOpFwidth:
   case SpvOpDPdxFine:
   case SpvOpDPdyFine:
   case SpvOpFwidthFine:
   case SpvOpDPdxCoarse:
   case SpvOpDPdyCoarse:
   case SpvOpFwidthCoarse:
   case SpvOpBitFieldInsert:
   case SpvOpBitFieldSExtract:
   case SpvOpBitFieldUExtract:
   case SpvOpBitReverse:
   case SpvOpBitCount:
   case SpvOpTranspose:
   case SpvOpOuterProduct:
   case SpvOpMatrixTimesScalar:
   case SpvOpVectorTimesMatrix:
   case SpvOpMatrixTimesVector:
   case SpvOpMatrixTimesMatrix:
   case SpvOpUCountLeadingZerosINTEL:
   case SpvOpUCountTrailingZerosINTEL:
   case SpvOpAbsISubINTEL:
   case SpvOpAbsUSubINTEL:
   case SpvOpIAddSatINTEL:
   case SpvOpUAddSatINTEL:
   case SpvOpIAverageINTEL:
   case SpvOpUAverageINTEL:
   case SpvOpIAverageRoundedINTEL:
   case SpvOpUAverageRoundedINTEL:
   case SpvOpISubSatINTEL:
   case SpvOpUSubSatINTEL:
   case SpvOpIMul32x16INTEL:
   case SpvOpUMul32x16INTEL:
      vtn_handle_alu(b, opcode, w, count);
      break;

   case SpvOpSDotKHR:
   case SpvOpUDotKHR:
   case SpvOpSUDotKHR:
   case SpvOpSDotAccSatKHR:
   case SpvOpUDotAccSatKHR:
   case SpvOpSUDotAccSatKHR:
      vtn_handle_integer_dot(b, opcode, w, count);
      break;

   case SpvOpBitcast:
      vtn_handle_bitcast(b, w, count);
      break;

   case SpvOpVectorExtractDynamic:
   case SpvOpVectorInsertDynamic:
   case SpvOpVectorShuffle:
   case SpvOpCompositeConstruct:
   case SpvOpCompositeExtract:
   case SpvOpCompositeInsert:
   case SpvOpCopyLogical:
   case SpvOpCopyObject:
      vtn_handle_composite(b, opcode, w, count);
      break;

   case SpvOpEmitVertex:
   case SpvOpEndPrimitive:
   case SpvOpEmitStreamVertex:
   case SpvOpEndStreamPrimitive:
   case SpvOpControlBarrier:
   case SpvOpMemoryBarrier:
      vtn_handle_barrier(b, opcode, w, count);
      break;

   case SpvOpGroupNonUniformElect:
   case SpvOpGroupNonUniformAll:
   case SpvOpGroupNonUniformAny:
   case SpvOpGroupNonUniformAllEqual:
   case SpvOpGroupNonUniformBroadcast:
   case SpvOpGroupNonUniformBroadcastFirst:
   case SpvOpGroupNonUniformBallot:
   case SpvOpGroupNonUniformInverseBallot:
   case SpvOpGroupNonUniformBallotBitExtract:
   case SpvOpGroupNonUniformBallotBitCount:
   case SpvOpGroupNonUniformBallotFindLSB:
   case SpvOpGroupNonUniformBallotFindMSB:
   case SpvOpGroupNonUniformShuffle:
   case SpvOpGroupNonUniformShuffleXor:
   case SpvOpGroupNonUniformShuffleUp:
   case SpvOpGroupNonUniformShuffleDown:
   case SpvOpGroupNonUniformIAdd:
   case SpvOpGroupNonUniformFAdd:
   case SpvOpGroupNonUniformIMul:
   case SpvOpGroupNonUniformFMul:
   case SpvOpGroupNonUniformSMin:
   case SpvOpGroupNonUniformUMin:
   case SpvOpGroupNonUniformFMin:
   case SpvOpGroupNonUniformSMax:
   case SpvOpGroupNonUniformUMax:
   case SpvOpGroupNonUniformFMax:
   case SpvOpGroupNonUniformBitwiseAnd:
   case SpvOpGroupNonUniformBitwiseOr:
   case SpvOpGroupNonUniformBitwiseXor:
   case SpvOpGroupNonUniformLogicalAnd:
   case SpvOpGroupNonUniformLogicalOr:
   case SpvOpGroupNonUniformLogicalXor:
   case SpvOpGroupNonUniformQuadBroadcast:
   case SpvOpGroupNonUniformQuadSwap:
   case SpvOpGroupAll:
   case SpvOpGroupAny:
   case SpvOpGroupBroadcast:
   case SpvOpGroupIAdd:
   case SpvOpGroupFAdd:
   case SpvOpGroupFMin:
   case SpvOpGroupUMin:
   case SpvOpGroupSMin:
   case SpvOpGroupFMax:
   case SpvOpGroupUMax:
   case SpvOpGroupSMax:
   case SpvOpSubgroupBallotKHR:
   case SpvOpSubgroupFirstInvocationKHR:
   case SpvOpSubgroupReadInvocationKHR:
   case SpvOpSubgroupAllKHR:
   case SpvOpSubgroupAnyKHR:
   case SpvOpSubgroupAllEqualKHR:
   case SpvOpGroupIAddNonUniformAMD:
   case SpvOpGroupFAddNonUniformAMD:
   case SpvOpGroupFMinNonUniformAMD:
   case SpvOpGroupUMinNonUniformAMD:
   case SpvOpGroupSMinNonUniformAMD:
   case SpvOpGroupFMaxNonUniformAMD:
   case SpvOpGroupUMaxNonUniformAMD:
   case SpvOpGroupSMaxNonUniformAMD:
   case SpvOpSubgroupShuffleINTEL:
   case SpvOpSubgroupShuffleDownINTEL:
   case SpvOpSubgroupShuffleUpINTEL:
   case SpvOpSubgroupShuffleXorINTEL:
      vtn_handle_subgroup(b, opcode, w, count);
      break;

   case SpvOpPtrDiff:
   case SpvOpPtrEqual:
   case SpvOpPtrNotEqual:
      vtn_handle_ptr(b, opcode, w, count);
      break;

   case SpvOpBeginInvocationInterlockEXT:
      nir_begin_invocation_interlock(&b->nb);
      break;

   case SpvOpEndInvocationInterlockEXT:
      nir_end_invocation_interlock(&b->nb);
      break;

   case SpvOpDemoteToHelperInvocation: {
      nir_demote(&b->nb);
      break;
   }

   case SpvOpIsHelperInvocationEXT: {
      vtn_push_nir_ssa(b, w[2], nir_is_helper_invocation(&b->nb, 1));
      break;
   }

   case SpvOpReadClockKHR: {
      SpvScope scope = vtn_constant_uint(b, w[3]);
      nir_scope nir_scope;

      switch (scope) {
      case SpvScopeDevice:
         nir_scope = NIR_SCOPE_DEVICE;
         break;
      case SpvScopeSubgroup:
         nir_scope = NIR_SCOPE_SUBGROUP;
         break;
      default:
         vtn_fail("invalid read clock scope");
      }

      /* Operation supports two result types: uvec2 and uint64_t.  The NIR
       * intrinsic gives uvec2, so pack the result for the other case.
       */
      nir_ssa_def *result = nir_shader_clock(&b->nb, nir_scope);

      struct vtn_type *type = vtn_get_type(b, w[1]);
      const struct glsl_type *dest_type = type->type;

      if (glsl_type_is_vector(dest_type)) {
         assert(dest_type == glsl_vector_type(GLSL_TYPE_UINT, 2));
      } else {
         assert(glsl_type_is_scalar(dest_type));
         assert(glsl_get_base_type(dest_type) == GLSL_TYPE_UINT64);
         result = nir_pack_64_2x32(&b->nb, result);
      }

      vtn_push_nir_ssa(b, w[2], result);
      break;
   }

   case SpvOpTraceNV:
   case SpvOpTraceRayKHR:
   case SpvOpReportIntersectionKHR:
   case SpvOpIgnoreIntersectionNV:
   case SpvOpTerminateRayNV:
   case SpvOpExecuteCallableNV:
   case SpvOpExecuteCallableKHR:
      vtn_handle_ray_intrinsic(b, opcode, w, count);
      break;

   case SpvOpRayQueryInitializeKHR:
   case SpvOpRayQueryTerminateKHR:
   case SpvOpRayQueryGenerateIntersectionKHR:
   case SpvOpRayQueryConfirmIntersectionKHR:
   case SpvOpRayQueryProceedKHR:
   case SpvOpRayQueryGetIntersectionTypeKHR:
   case SpvOpRayQueryGetRayTMinKHR:
   case SpvOpRayQueryGetRayFlagsKHR:
   case SpvOpRayQueryGetIntersectionTKHR:
   case SpvOpRayQueryGetIntersectionInstanceCustomIndexKHR:
   case SpvOpRayQueryGetIntersectionInstanceIdKHR:
   case SpvOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
   case SpvOpRayQueryGetIntersectionGeometryIndexKHR:
   case SpvOpRayQueryGetIntersectionPrimitiveIndexKHR:
   case SpvOpRayQueryGetIntersectionBarycentricsKHR:
   case SpvOpRayQueryGetIntersectionFrontFaceKHR:
   case SpvOpRayQueryGetIntersectionCandidateAABBOpaqueKHR:
   case SpvOpRayQueryGetIntersectionObjectRayDirectionKHR:
   case SpvOpRayQueryGetIntersectionObjectRayOriginKHR:
   case SpvOpRayQueryGetWorldRayDirectionKHR:
   case SpvOpRayQueryGetWorldRayOriginKHR:
   case SpvOpRayQueryGetIntersectionObjectToWorldKHR:
   case SpvOpRayQueryGetIntersectionWorldToObjectKHR:
      vtn_handle_ray_query_intrinsic(b, opcode, w, count);
      break;

   case SpvOpLifetimeStart:
   case SpvOpLifetimeStop:
      break;

   case SpvOpGroupAsyncCopy:
   case SpvOpGroupWaitEvents:
      vtn_handle_opencl_core_instruction(b, opcode, w, count);
      break;

   case SpvOpWritePackedPrimitiveIndices4x8NV:
      vtn_handle_write_packed_primitive_indices(b, opcode, w, count);
      break;

   case SpvOpSetMeshOutputsEXT:
      nir_set_vertex_and_primitive_count(
         &b->nb, vtn_get_nir_ssa(b, w[1]), vtn_get_nir_ssa(b, w[2]));
      break;

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }

   return true;
}

static bool
vtn_handle_spec_constant_instructions(struct vtn_builder* b, SpvOp opcode,
                                      const uint32_t* w, unsigned count)
{
   switch (opcode) {
   case SpvOpSpecConstantTrue:
   case SpvOpSpecConstantFalse:
   case SpvOpSpecConstant:
   case SpvOpSpecConstantComposite:
   case SpvOpSpecConstantOp:
      break;
   default:
      return true;
   }

   struct vtn_value* val = vtn_untyped_value(b, w[2]);

   switch (opcode) {
   case SpvOpSpecConstantTrue:
   case SpvOpSpecConstantFalse:
   case SpvOpSpecConstant: {
      vtn_assert(val->is_sc);
      vtn_assert(val->value_type == vtn_value_type_constant || val->value_type == vtn_value_type_ssa);

      val->value_type = vtn_value_type_ssa;
      val->ssa = vtn_create_ssa_value(b, val->type->type);

      nir_ssa_def *sc_imm = nir_imm_int(&b->nb, GODOT_NIR_SC_SENTINEL_MAGIC | val->sc_id);
      nir_ssa_def *non_opt_const = nir_build_load_constant_non_opt(&b->nb, 1, 32, sc_imm);

      vtn_assert(b->nb.cursor.option == nir_cursor_after_instr);
      vtn_assert(b->nb.cursor.instr->type == nir_instr_type_intrinsic);

      nir_ssa_def *temp = nir_build_alu(
         &b->nb,
         nir_op_iand,
         non_opt_const,
         nir_imm_int(&b->nb, 0xbfffffff),
         NULL,
         NULL);
      val->ssa = vtn_create_ssa_value(b, val->type->type);
      if (val->type->type == glsl_uint_type()) {
         val->ssa->def = temp;
      } else if (val->type->type == glsl_bool_type()) {
         val->ssa->def = nir_build_alu(
            &b->nb,
            nir_op_ine,
            temp,
            nir_imm_int(&b->nb, 0),
            NULL,
            NULL);
      } else if (val->type->type == glsl_float_type()) {
         val->ssa->def = nir_build_alu(
            &b->nb,
            nir_op_ishl,
            temp,
            nir_imm_int(&b->nb, 1),
            NULL,
            NULL);
      } else {
         vtn_assert(false);
      }
   } break;

   case SpvOpSpecConstantComposite:
      abort(); // Unimplemented
      break;

   case SpvOpSpecConstantOp: {
      vtn_assert(val->value_type == vtn_value_type_ssa);
      val->value_type = vtn_value_type_invalid;

      unsigned count = (w[0] >> SpvWordCountShift) - 1;
      uint32_t* sub_w = (uint32_t*)alloca(4 * count);
      sub_w[0] = 0; /* Doesn't really matter */
      sub_w[1] = val->type->id;
      sub_w[2] = w[2];
      SpvOp sub_opcode = w[3];
      for (unsigned i = 0; i < count - 3; ++i)
         sub_w[3 + i] = w[4 + i];
      vtn_handle_body_instruction(b, sub_opcode, sub_w, count);
   } break;

   default:
      return false; /* End of preamble */
   }

   return true;
}

static bool
is_glslang(const struct vtn_builder *b)
{
   return b->generator_id == vtn_generator_glslang_reference_front_end ||
          b->generator_id == vtn_generator_shaderc_over_glslang;
}

struct vtn_builder*
vtn_create_builder(const uint32_t *words, size_t word_count,
                   gl_shader_stage stage, const char *entry_point_name,
                   const struct spirv_to_nir_options *options)
{
   /* Initialize the vtn_builder object */
   struct vtn_builder *b = rzalloc(NULL, struct vtn_builder);
   struct spirv_to_nir_options *dup_options =
      ralloc(b, struct spirv_to_nir_options);
   *dup_options = *options;

   b->spirv = words;
   b->spirv_word_count = word_count;
   b->file = NULL;
   b->line = -1;
   b->col = -1;
   list_inithead(&b->functions);
   b->entry_point_stage = stage;
   b->entry_point_name = entry_point_name;
   b->options = dup_options;

   /*
    * Handle the SPIR-V header (first 5 dwords).
    * Can't use vtx_assert() as the setjmp(3) target isn't initialized yet.
    */
   if (word_count <= 5)
      goto fail;

   if (words[0] != SpvMagicNumber) {
      vtn_err("words[0] was 0x%x, want 0x%x", words[0], SpvMagicNumber);
      goto fail;
   }

   b->version = words[1];
   if (b->version < 0x10000) {
      vtn_err("version was 0x%x, want >= 0x10000", b->version);
      goto fail;
   }

   b->generator_id = words[2] >> 16;
   uint16_t generator_version = words[2];

   /* In GLSLang commit 8297936dd6eb3, their handling of barrier() was fixed
    * to provide correct memory semantics on compute shader barrier()
    * commands.  Prior to that, we need to fix them up ourselves.  This
    * GLSLang fix caused them to bump to generator version 3.
    */
   b->wa_glslang_cs_barrier = is_glslang(b) && generator_version < 3;

   /* Identifying the LLVM-SPIRV translator:
    *
    * The LLVM-SPIRV translator currently doesn't store any generator ID [1].
    * Our use case involving the SPIRV-Tools linker also mean we want to check
    * for that tool instead. Finally the SPIRV-Tools linker also stores its
    * generator ID in the wrong location [2].
    *
    * [1] : https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/1223
    * [2] : https://github.com/KhronosGroup/SPIRV-Tools/pull/4549
    */
   const bool is_llvm_spirv_translator =
      (b->generator_id == 0 &&
       generator_version == vtn_generator_spirv_tools_linker) ||
      b->generator_id == vtn_generator_spirv_tools_linker;

   /* The LLVM-SPIRV translator generates Undef initializers for _local
    * variables [1].
    *
    * [1] : https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/1224
    */
   b->wa_llvm_spirv_ignore_workgroup_initializer =
      b->options->environment == NIR_SPIRV_OPENCL && is_llvm_spirv_translator;

   /* Older versions of GLSLang would incorrectly emit OpReturn after
    * OpEmitMeshTasksEXT. This is incorrect since the latter is already
    * a terminator instruction.
    *
    * See https://github.com/KhronosGroup/glslang/issues/3020 for details.
    *
    * Clay Shader Compiler (used by GravityMark) is also affected.
    */
   b->wa_ignore_return_after_emit_mesh_tasks =
      (is_glslang(b) && generator_version < 11) ||
      (b->generator_id == vtn_generator_clay_shader_compiler &&
       generator_version < 18);

   /* words[2] == generator magic */
   unsigned value_id_bound = words[3];
   if (words[4] != 0) {
      vtn_err("words[4] was %u, want 0", words[4]);
      goto fail;
   }

   b->value_id_bound = value_id_bound;
   b->values = rzalloc_array(b, struct vtn_value, value_id_bound);

   if (b->options->environment == NIR_SPIRV_VULKAN && b->version < 0x10400)
      b->vars_used_indirectly = _mesa_pointer_set_create(b);

   return b;
 fail:
   ralloc_free(b);
   return NULL;
}

static nir_function *
vtn_emit_kernel_entry_point_wrapper(struct vtn_builder *b,
                                    nir_function *entry_point)
{
   vtn_assert(entry_point == b->entry_point->func->nir_func);
   vtn_fail_if(!entry_point->name, "entry points are required to have a name");
   const char *func_name =
      ralloc_asprintf(b->shader, "__wrapped_%s", entry_point->name);

   vtn_assert(b->shader->info.stage == MESA_SHADER_KERNEL);

   nir_function *main_entry_point = nir_function_create(b->shader, func_name);
   main_entry_point->impl = nir_function_impl_create(main_entry_point);
   nir_builder_init(&b->nb, main_entry_point->impl);
   b->nb.cursor = nir_after_cf_list(&main_entry_point->impl->body);
   b->func_param_idx = 0;

   nir_call_instr *call = nir_call_instr_create(b->nb.shader, entry_point);

   for (unsigned i = 0; i < entry_point->num_params; ++i) {
      struct vtn_type *param_type = b->entry_point->func->type->params[i];

      /* consider all pointers to function memory to be parameters passed
       * by value
       */
      bool is_by_val = param_type->base_type == vtn_base_type_pointer &&
         param_type->storage_class == SpvStorageClassFunction;

      /* input variable */
      nir_variable *in_var = rzalloc(b->nb.shader, nir_variable);

      if (is_by_val) {
         in_var->data.mode = nir_var_uniform;
         in_var->type = param_type->deref->type;
      } else if (param_type->base_type == vtn_base_type_image) {
         in_var->data.mode = nir_var_image;
         in_var->type = param_type->glsl_image;
         in_var->data.access =
            spirv_to_gl_access_qualifier(b, param_type->access_qualifier);
      } else if (param_type->base_type == vtn_base_type_sampler) {
         in_var->data.mode = nir_var_uniform;
         in_var->type = glsl_bare_sampler_type();
      } else {
         in_var->data.mode = nir_var_uniform;
         in_var->type = param_type->type;
      }

      in_var->data.read_only = true;
      in_var->data.location = i;

      nir_shader_add_variable(b->nb.shader, in_var);

      /* we have to copy the entire variable into function memory */
      if (is_by_val) {
         nir_variable *copy_var =
            nir_local_variable_create(main_entry_point->impl, in_var->type,
                                      "copy_in");
         nir_copy_var(&b->nb, copy_var, in_var);
         call->params[i] =
            nir_src_for_ssa(&nir_build_deref_var(&b->nb, copy_var)->dest.ssa);
      } else if (param_type->base_type == vtn_base_type_image ||
                 param_type->base_type == vtn_base_type_sampler) {
         /* Don't load the var, just pass a deref of it */
         call->params[i] = nir_src_for_ssa(&nir_build_deref_var(&b->nb, in_var)->dest.ssa);
      } else {
         call->params[i] = nir_src_for_ssa(nir_load_var(&b->nb, in_var));
      }
   }

   nir_builder_instr_insert(&b->nb, &call->instr);

   return main_entry_point;
}

static bool
can_remove(nir_variable *var, void *data)
{
   const struct set *vars_used_indirectly = data;
   return !_mesa_set_search(vars_used_indirectly, var);
}

#ifndef NDEBUG
static void
initialize_mesa_spirv_debug(void)
{
   mesa_spirv_debug = debug_get_option_mesa_spirv_debug();
}
#endif

nir_shader *
spirv_to_nir(const uint32_t *words, size_t word_count,
             struct nir_spirv_specialization *spec, unsigned num_spec,
             gl_shader_stage stage, const char *entry_point_name,
             const struct spirv_to_nir_options *options,
             const nir_shader_compiler_options *nir_options)

{
#ifndef NDEBUG
   static once_flag initialized_debug_flag = ONCE_FLAG_INIT;
   call_once(&initialized_debug_flag, initialize_mesa_spirv_debug);
#endif

   const uint32_t *word_end = words + word_count;

   struct vtn_builder *b = vtn_create_builder(words, word_count,
                                              stage, entry_point_name,
                                              options);

   if (b == NULL)
      return NULL;

   /* See also _vtn_fail() */
   if (vtn_setjmp(b->fail_jump)) {
      ralloc_free(b);
      return NULL;
   }

   /* Skip the SPIR-V header, handled at vtn_create_builder */
   words+= 5;

   b->shader = nir_shader_create(b, stage, nir_options, NULL);
   b->shader->info.subgroup_size = options->subgroup_size;
   b->shader->info.float_controls_execution_mode = options->float_controls_execution_mode;

   const uint32_t *preamble_words = words;

   /* Handle all the preamble instructions */
   words = vtn_foreach_instruction(b, words, word_end,
                                   vtn_handle_preamble_instruction);

   /* DirectXShaderCompiler and glslang/shaderc both create OpKill from HLSL's
    * discard/clip, which uses demote semantics. DirectXShaderCompiler will use
    * demote if the extension is enabled, so we disable this workaround in that
    * case.
    *
    * Related glslang issue: https://github.com/KhronosGroup/glslang/issues/2416
    */
   bool dxsc = b->generator_id == vtn_generator_spiregg;
   b->convert_discard_to_demote = ((dxsc && !b->uses_demote_to_helper_invocation) ||
                                   (is_glslang(b) && b->source_lang == SpvSourceLanguageHLSL)) &&
                                  options->caps.demote_to_helper_invocation;

   if (!options->create_library && b->entry_point == NULL) {
      vtn_fail("Entry point not found for %s shader \"%s\"",
               _mesa_shader_stage_to_string(stage), entry_point_name);
      ralloc_free(b);
      return NULL;
   }

   /* Ensure a sane address mode is being used for function temps */
   assert(nir_address_format_bit_size(b->options->temp_addr_format) == nir_get_ptr_bitsize(b->shader));
   assert(nir_address_format_num_components(b->options->temp_addr_format) == 1);

   /* Set shader info defaults */
   if (stage == MESA_SHADER_GEOMETRY)
      b->shader->info.gs.invocations = 1;

   /* Parse execution modes. */
   if (!options->create_library)
      vtn_foreach_execution_mode(b, b->entry_point,
                                 vtn_handle_execution_mode, NULL);

   b->specializations = spec;
   b->num_specializations = num_spec;

   /* Handle all variable, type, and constant instructions */
   words = vtn_foreach_instruction(b, words, word_end,
                                   vtn_handle_variable_or_type_instruction);

   /* Parse execution modes that depend on IDs. Must happen after we have
    * constants parsed.
    */
   if (!options->create_library)
      vtn_foreach_execution_mode(b, b->entry_point,
                                 vtn_handle_execution_mode_id, NULL);

   if (b->workgroup_size_builtin) {
      vtn_assert(gl_shader_stage_uses_workgroup(stage));
      vtn_assert(b->workgroup_size_builtin->type->type ==
                 glsl_vector_type(GLSL_TYPE_UINT, 3));

      nir_const_value *const_size =
         b->workgroup_size_builtin->constant->values;

      b->shader->info.workgroup_size[0] = const_size[0].u32;
      b->shader->info.workgroup_size[1] = const_size[1].u32;
      b->shader->info.workgroup_size[2] = const_size[2].u32;
   }

   /* Set types on all vtn_values */
   vtn_foreach_instruction(b, words, word_end, vtn_set_instruction_result_type);

   vtn_build_cfg(b, words, word_end);

   if (!options->create_library) {
      assert(b->entry_point->value_type == vtn_value_type_function);
      b->entry_point->func->referenced = true;
   }

   bool progress;
   do {
      progress = false;
      vtn_foreach_cf_node(node, &b->functions) {
         struct vtn_function *func = vtn_cf_node_as_function(node);
         if ((options->create_library || func->referenced) && !func->emitted) {
            b->const_table = _mesa_pointer_hash_table_create(b);

            vtn_function_emit(b, func, vtn_handle_spec_constant_instructions, preamble_words, vtn_handle_body_instruction);
            progress = true;
         }
      }
   } while (progress);

   if (!options->create_library) {
      vtn_assert(b->entry_point->value_type == vtn_value_type_function);
      nir_function *entry_point = b->entry_point->func->nir_func;
      vtn_assert(entry_point);

      /* post process entry_points with input params */
      if (entry_point->num_params && b->shader->info.stage == MESA_SHADER_KERNEL)
         entry_point = vtn_emit_kernel_entry_point_wrapper(b, entry_point);

      entry_point->is_entrypoint = true;
   }

   /* structurize the CFG */
   nir_lower_goto_ifs(b->shader);

   /* A SPIR-V module can have multiple shaders stages and also multiple
    * shaders of the same stage.  Global variables are declared per-module.
    *
    * Starting in SPIR-V 1.4 the list of global variables is part of
    * OpEntryPoint, so only valid ones will be created.  Previous versions
    * only have Input and Output variables listed, so remove dead variables to
    * clean up the remaining ones.
    */
   if (!options->create_library && b->version < 0x10400) {
      const nir_remove_dead_variables_options dead_opts = {
         .can_remove_var = can_remove,
         .can_remove_var_data = b->vars_used_indirectly,
      };
      nir_remove_dead_variables(b->shader, ~(nir_var_function_temp |
                                             nir_var_shader_out |
                                             nir_var_shader_in |
                                             nir_var_system_value),
                                b->vars_used_indirectly ? &dead_opts : NULL);
   }

   nir_foreach_variable_in_shader(var, b->shader) {
      switch (var->data.mode) {
      case nir_var_mem_ubo:
         b->shader->info.num_ubos++;
         break;
      case nir_var_mem_ssbo:
         b->shader->info.num_ssbos++;
         break;
      case nir_var_mem_push_const:
         vtn_assert(b->shader->num_uniforms == 0);
         b->shader->num_uniforms =
            glsl_get_explicit_size(glsl_without_array(var->type), false);
         break;
      }
   }

   /* We sometimes generate bogus derefs that, while never used, give the
    * validator a bit of heartburn.  Run dead code to get rid of them.
    */
   nir_opt_dce(b->shader);

   /* Per SPV_KHR_workgroup_storage_explicit_layout, if one shared variable is
    * a Block, all of them will be and Blocks are explicitly laid out.
    */
   nir_foreach_variable_with_modes(var, b->shader, nir_var_mem_shared) {
      if (glsl_type_is_interface(var->type)) {
         assert(b->options->caps.workgroup_memory_explicit_layout);
         b->shader->info.shared_memory_explicit_layout = true;
         break;
      }
   }
   if (b->shader->info.shared_memory_explicit_layout) {
      unsigned size = 0;
      nir_foreach_variable_with_modes(var, b->shader, nir_var_mem_shared) {
         assert(glsl_type_is_interface(var->type));
         const bool align_to_stride = false;
         size = MAX2(size, glsl_get_explicit_size(var->type, align_to_stride));
      }
      b->shader->info.shared_size = size;
   }

   if (stage == MESA_SHADER_FRAGMENT) {
      /* From the Vulkan 1.2.199 spec:
       *
       *    "If a fragment shader entry point’s interface includes an input
       *    variable decorated with SamplePosition, Sample Shading is
       *    considered enabled with a minSampleShading value of 1.0."
       *
       * Similar text exists for SampleId.  Regarding the Sample decoration,
       * the Vulkan 1.2.199 spec says:
       *
       *    "If a fragment shader input is decorated with Sample, a separate
       *    value must be assigned to that variable for each covered sample in
       *    the fragment, and that value must be sampled at the location of
       *    the individual sample. When rasterizationSamples is
       *    VK_SAMPLE_COUNT_1_BIT, the fragment center must be used for
       *    Centroid, Sample, and undecorated attribute interpolation."
       *
       * Unfortunately, this isn't quite as clear about static use and the
       * interface but the static use check should be valid.
       *
       * For OpenGL, similar language exists but it's all more wishy-washy.
       * We'll assume the same behavior across APIs.
       */
      nir_foreach_variable_with_modes(var, b->shader,
                                      nir_var_shader_in |
                                      nir_var_system_value) {
         struct nir_variable_data *members =
            var->members ? var->members : &var->data;
         uint16_t num_members = var->members ? var->num_members : 1;
         for (uint16_t i = 0; i < num_members; i++) {
            if (members[i].mode == nir_var_system_value &&
                (members[i].location == SYSTEM_VALUE_SAMPLE_ID ||
                 members[i].location == SYSTEM_VALUE_SAMPLE_POS))
               b->shader->info.fs.uses_sample_shading = true;

            if (members[i].mode == nir_var_shader_in && members[i].sample)
               b->shader->info.fs.uses_sample_shading = true;
         }
      }
   }

   /* Unparent the shader from the vtn_builder before we delete the builder */
   ralloc_steal(NULL, b->shader);

   nir_shader *shader = b->shader;
   ralloc_free(b);

   return shader;
}

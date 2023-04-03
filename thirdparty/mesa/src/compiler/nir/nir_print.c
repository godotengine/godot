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

#include "nir.h"
#include "compiler/shader_enums.h"
#include "util/half_float.h"
#include "util/memstream.h"
#include "util/mesa-sha1.h"
#include "vulkan/vulkan_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> /* for PRIx64 macro */

static void
print_tabs(unsigned num_tabs, FILE *fp)
{
   for (unsigned i = 0; i < num_tabs; i++)
      fprintf(fp, "\t");
}

typedef struct {
   FILE *fp;
   nir_shader *shader;
   /** map from nir_variable -> printable name */
   struct hash_table *ht;

   /** set of names used so far for nir_variables */
   struct set *syms;

   /* an index used to make new non-conflicting names */
   unsigned index;

   /**
    * Optional table of annotations mapping nir object
    * (such as instr or var) to message to print.
    */
   struct hash_table *annotations;
} print_state;

static void
print_annotation(print_state *state, void *obj)
{
   FILE *fp = state->fp;

   if (!state->annotations)
      return;

   struct hash_entry *entry = _mesa_hash_table_search(state->annotations, obj);
   if (!entry)
      return;

   const char *note = entry->data;
   _mesa_hash_table_remove(state->annotations, entry);

   fprintf(fp, "%s\n\n", note);
}

static void
print_register(nir_register *reg, print_state *state)
{
   FILE *fp = state->fp;
   fprintf(fp, "r%u", reg->index);
}

static const char *sizes[] = { "error", "vec1", "vec2", "vec3", "vec4",
                               "vec5", "error", "error", "vec8",
                               "error", "error", "error", "error",
                               "error", "error", "error", "vec16"};

static void
print_register_decl(nir_register *reg, print_state *state)
{
   FILE *fp = state->fp;
   fprintf(fp, "decl_reg %s %u ", sizes[reg->num_components], reg->bit_size);
   print_register(reg, state);
   if (reg->num_array_elems != 0)
      fprintf(fp, "[%u]", reg->num_array_elems);
   fprintf(fp, "\n");
}

static void
print_ssa_def(nir_ssa_def *def, print_state *state)
{
   FILE *fp = state->fp;

   const char *divergence = "";
   if (state->shader->info.divergence_analysis_run)
      divergence = def->divergent ? "div " : "con ";

   fprintf(fp, "%s %2u %sssa_%u", sizes[def->num_components], def->bit_size,
           divergence, def->index);
}

static void
print_const_from_load(nir_load_const_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   /*
    * we don't really know the type of the constant (if it will be used as a
    * float or an int), so just print the raw constant in hex for fidelity
    * and then print in float again for readability.
    */

   fprintf(fp, "(");

   for (unsigned i = 0; i < instr->def.num_components; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      switch (instr->def.bit_size) {
      case 64:
         fprintf(fp, "0x%016" PRIx64, instr->value[i].u64);
         break;
      case 32:
         fprintf(fp, "0x%08x", instr->value[i].u32);
         break;
      case 16:
         fprintf(fp, "0x%04x", instr->value[i].u16);
         break;
      case 8:
         fprintf(fp, "0x%02x", instr->value[i].u8);
         break;
      case 1:
         fprintf(fp, "%s", instr->value[i].b ? "true" : "false");
         break;
      }
   }

   if (instr->def.bit_size > 8) {
      if (instr->def.num_components > 1)
         fprintf(fp, ") = (");
      else
         fprintf(fp, " = ");

      for (unsigned i = 0; i < instr->def.num_components; i++) {
         if (i != 0)
            fprintf(fp, ", ");

         switch (instr->def.bit_size) {
         case 64:
            fprintf(fp, "%f", instr->value[i].f64);
            break;
         case 32:
            fprintf(fp, "%f", instr->value[i].f32);
            break;
         case 16:
            fprintf(fp, "%f", _mesa_half_to_float(instr->value[i].u16));
            break;
         default:
            unreachable("unhandled bit size");
         }
      }
   }

   fprintf(fp, ")");
}

static void
print_load_const_instr(nir_load_const_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_ssa_def(&instr->def, state);

   fprintf(fp, " = load_const ");

   print_const_from_load(instr, state);
}

static void
print_ssa_use(nir_ssa_def *def, print_state *state)
{
   FILE *fp = state->fp;
   fprintf(fp, "ssa_%u", def->index);
   nir_instr *instr = def->parent_instr;
   if (instr->type == nir_instr_type_load_const && NIR_DEBUG(PRINT_CONSTS)) {
      fprintf(fp, " /*");
      print_const_from_load(nir_instr_as_load_const(instr), state);
      fprintf(fp, "*/");
   }
}

static void print_src(const nir_src *src, print_state *state);

static void
print_reg_src(const nir_reg_src *src, print_state *state)
{
   FILE *fp = state->fp;
   print_register(src->reg, state);
   if (src->reg->num_array_elems != 0) {
      fprintf(fp, "[%u", src->base_offset);
      if (src->indirect != NULL) {
         fprintf(fp, " + ");
         print_src(src->indirect, state);
      }
      fprintf(fp, "]");
   }
}

static void
print_reg_dest(nir_reg_dest *dest, print_state *state)
{
   FILE *fp = state->fp;
   print_register(dest->reg, state);
   if (dest->reg->num_array_elems != 0) {
      fprintf(fp, "[%u", dest->base_offset);
      if (dest->indirect != NULL) {
         fprintf(fp, " + ");
         print_src(dest->indirect, state);
      }
      fprintf(fp, "]");
   }
}

static void
print_src(const nir_src *src, print_state *state)
{
   if (src->is_ssa)
      print_ssa_use(src->ssa, state);
   else
      print_reg_src(&src->reg, state);
}

static void
print_dest(nir_dest *dest, print_state *state)
{
   if (dest->is_ssa)
      print_ssa_def(&dest->ssa, state);
   else
      print_reg_dest(&dest->reg, state);
}

static const char *
comp_mask_string(unsigned num_components)
{
   return (num_components > 4) ? "abcdefghijklmnop" : "xyzw";
}

static void
print_alu_src(nir_alu_instr *instr, unsigned src, print_state *state)
{
   FILE *fp = state->fp;

   if (instr->src[src].negate)
      fprintf(fp, "-");
   if (instr->src[src].abs)
      fprintf(fp, "abs(");

   print_src(&instr->src[src].src, state);

   bool print_swizzle = false;
   nir_component_mask_t used_channels = 0;

   for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) {
      if (!nir_alu_instr_channel_used(instr, src, i))
         continue;

      used_channels++;

      if (instr->src[src].swizzle[i] != i) {
         print_swizzle = true;
         break;
      }
   }

   unsigned live_channels = nir_src_num_components(instr->src[src].src);

   if (print_swizzle || used_channels != live_channels) {
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) {
         if (!nir_alu_instr_channel_used(instr, src, i))
            continue;

         fprintf(fp, "%c", comp_mask_string(live_channels)[instr->src[src].swizzle[i]]);
      }
   }

   if (instr->src[src].abs)
      fprintf(fp, ")");
}

static void
print_alu_dest(nir_alu_dest *dest, print_state *state)
{
   FILE *fp = state->fp;
   /* we're going to print the saturate modifier later, after the opcode */

   print_dest(&dest->dest, state);

   if (!dest->dest.is_ssa &&
       dest->write_mask != (1 << dest->dest.reg.reg->num_components) - 1) {
      unsigned live_channels = dest->dest.reg.reg->num_components;
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
         if ((dest->write_mask >> i) & 1)
            fprintf(fp, "%c", comp_mask_string(live_channels)[i]);
   }
}

static void
print_alu_instr(nir_alu_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_alu_dest(&instr->dest, state);

   fprintf(fp, " = %s", nir_op_infos[instr->op].name);
   if (instr->exact)
      fprintf(fp, "!");
   if (instr->dest.saturate)
      fprintf(fp, ".sat");
   if (instr->no_signed_wrap)
      fprintf(fp, ".nsw");
   if (instr->no_unsigned_wrap)
      fprintf(fp, ".nuw");
   fprintf(fp, " ");

   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_alu_src(instr, i, state);
   }
}

static const char *
get_var_name(nir_variable *var, print_state *state)
{
   if (state->ht == NULL)
      return var->name ? var->name : "unnamed";

   assert(state->syms);

   struct hash_entry *entry = _mesa_hash_table_search(state->ht, var);
   if (entry)
      return entry->data;

   char *name;
   if (var->name == NULL) {
      name = ralloc_asprintf(state->syms, "@%u", state->index++);
   } else {
      struct set_entry *set_entry = _mesa_set_search(state->syms, var->name);
      if (set_entry != NULL) {
         /* we have a collision with another name, append an @ + a unique
          * index */
         name = ralloc_asprintf(state->syms, "%s@%u", var->name,
                                state->index++);
      } else {
         /* Mark this one as seen */
         _mesa_set_add(state->syms, var->name);
         name = var->name;
      }
   }

   _mesa_hash_table_insert(state->ht, var, name);

   return name;
}

static const char *
get_constant_sampler_addressing_mode(enum cl_sampler_addressing_mode mode)
{
   switch (mode) {
   case SAMPLER_ADDRESSING_MODE_NONE: return "none";
   case SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE: return "clamp_to_edge";
   case SAMPLER_ADDRESSING_MODE_CLAMP: return "clamp";
   case SAMPLER_ADDRESSING_MODE_REPEAT: return "repeat";
   case SAMPLER_ADDRESSING_MODE_REPEAT_MIRRORED: return "repeat_mirrored";
   default: unreachable("Invalid addressing mode");
   }
}

static const char *
get_constant_sampler_filter_mode(enum cl_sampler_filter_mode mode)
{
   switch (mode) {
   case SAMPLER_FILTER_MODE_NEAREST: return "nearest";
   case SAMPLER_FILTER_MODE_LINEAR: return "linear";
   default: unreachable("Invalid filter mode");
   }
}

static void
print_constant(nir_constant *c, const struct glsl_type *type, print_state *state)
{
   FILE *fp = state->fp;
   const unsigned rows = glsl_get_vector_elements(type);
   const unsigned cols = glsl_get_matrix_columns(type);
   unsigned i;

   switch (glsl_get_base_type(type)) {
   case GLSL_TYPE_BOOL:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "%s", c->values[i].b ? "true" : "false");
      }
      break;

   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%02x", c->values[i].u8);
      }
      break;

   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%04x", c->values[i].u16);
      }
      break;

   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%08x", c->values[i].u32);
      }
      break;

   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_DOUBLE:
      if (cols > 1) {
         for (i = 0; i < cols; i++) {
            if (i > 0) fprintf(fp, ", ");
            print_constant(c->elements[i], glsl_get_column_type(type), state);
         }
      } else {
         switch (glsl_get_base_type(type)) {
         case GLSL_TYPE_FLOAT16:
            for (i = 0; i < rows; i++) {
               if (i > 0) fprintf(fp, ", ");
               fprintf(fp, "%f", _mesa_half_to_float(c->values[i].u16));
            }
            break;

         case GLSL_TYPE_FLOAT:
            for (i = 0; i < rows; i++) {
               if (i > 0) fprintf(fp, ", ");
               fprintf(fp, "%f", c->values[i].f32);
            }
            break;

         case GLSL_TYPE_DOUBLE:
            for (i = 0; i < rows; i++) {
               if (i > 0) fprintf(fp, ", ");
               fprintf(fp, "%f", c->values[i].f64);
            }
            break;

         default:
            unreachable("Cannot get here from the first level switch");
         }
      }
      break;

   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < cols; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%08" PRIx64, c->values[i].u64);
      }
      break;

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
      for (i = 0; i < c->num_elements; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "{ ");
         print_constant(c->elements[i], glsl_get_struct_field(type, i), state);
         fprintf(fp, " }");
      }
      break;

   case GLSL_TYPE_ARRAY:
      for (i = 0; i < c->num_elements; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "{ ");
         print_constant(c->elements[i], glsl_get_array_element(type), state);
         fprintf(fp, " }");
      }
      break;

   default:
      unreachable("not reached");
   }
}

static const char *
get_variable_mode_str(nir_variable_mode mode, bool want_local_global_mode)
{
   switch (mode) {
   case nir_var_shader_in:
      return "shader_in";
   case nir_var_shader_out:
      return "shader_out";
   case nir_var_uniform:
      return "uniform";
   case nir_var_mem_ubo:
      return "ubo";
   case nir_var_system_value:
      return "system";
   case nir_var_mem_ssbo:
      return "ssbo";
   case nir_var_mem_shared:
      return "shared";
   case nir_var_mem_global:
      return "global";
   case nir_var_mem_push_const:
      return "push_const";
   case nir_var_mem_constant:
      return "constant";
   case nir_var_image:
      return "image";
   case nir_var_shader_temp:
      return want_local_global_mode ? "shader_temp" : "";
   case nir_var_function_temp:
      return want_local_global_mode ? "function_temp" : "";
   case nir_var_shader_call_data:
      return "shader_call_data";
   case nir_var_ray_hit_attrib:
      return "ray_hit_attrib";
   case nir_var_mem_task_payload:
      return "task_payload";
   default:
      if (mode && (mode & nir_var_mem_generic) == mode)
         return "generic";
      return "";
   }
}

static void
print_var_decl(nir_variable *var, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "decl_var ");

   const char *const bindless = (var->data.bindless) ? "bindless " : "";
   const char *const cent = (var->data.centroid) ? "centroid " : "";
   const char *const samp = (var->data.sample) ? "sample " : "";
   const char *const patch = (var->data.patch) ? "patch " : "";
   const char *const inv = (var->data.invariant) ? "invariant " : "";
   const char *const per_view = (var->data.per_view) ? "per_view " : "";
   const char *const per_primitive = (var->data.per_primitive) ? "per_primitive " : "";
   const char *const ray_query = (var->data.ray_query) ? "ray_query " : "";
   fprintf(fp, "%s%s%s%s%s%s%s%s%s %s ",
           bindless, cent, samp, patch, inv, per_view, per_primitive, ray_query,
           get_variable_mode_str(var->data.mode, false),
           glsl_interp_mode_name(var->data.interpolation));

   enum gl_access_qualifier access = var->data.access;
   const char *const coher = (access & ACCESS_COHERENT) ? "coherent " : "";
   const char *const volat = (access & ACCESS_VOLATILE) ? "volatile " : "";
   const char *const restr = (access & ACCESS_RESTRICT) ? "restrict " : "";
   const char *const ronly = (access & ACCESS_NON_WRITEABLE) ? "readonly " : "";
   const char *const wonly = (access & ACCESS_NON_READABLE) ? "writeonly " : "";
   const char *const reorder = (access & ACCESS_CAN_REORDER) ? "reorderable " : "";
   const char *const stream_cache_policy = (access & ACCESS_STREAM_CACHE_POLICY) ?
                                           "stream-cache-policy " : "";
   const char *const include_helpers = (access & ACCESS_INCLUDE_HELPERS) ?
                                       "include-helpers " : "";
   fprintf(fp, "%s%s%s%s%s%s%s%s", coher, volat, restr, ronly, wonly, reorder,
           stream_cache_policy, include_helpers);

   if (glsl_get_base_type(glsl_without_array(var->type)) == GLSL_TYPE_IMAGE) {
      fprintf(fp, "%s ", util_format_short_name(var->data.image.format));
   }

   if (var->data.precision) {
      const char *precisions[] = {
         "",
         "highp",
         "mediump",
         "lowp",
      };
      fprintf(fp, "%s ", precisions[var->data.precision]);
   }

   fprintf(fp, "%s %s", glsl_get_type_name(var->type),
           get_var_name(var, state));

   if (var->data.mode & (nir_var_shader_in |
                         nir_var_shader_out |
                         nir_var_uniform |
                         nir_var_mem_ubo |
                         nir_var_mem_ssbo |
                         nir_var_image)) {
      const char *loc = NULL;
      char buf[4];

      switch (state->shader->info.stage) {
      case MESA_SHADER_VERTEX:
         if (var->data.mode == nir_var_shader_in)
            loc = gl_vert_attrib_name(var->data.location);
         else if (var->data.mode == nir_var_shader_out)
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         break;
      case MESA_SHADER_TASK:
      case MESA_SHADER_MESH:
      case MESA_SHADER_GEOMETRY:
         if ((var->data.mode == nir_var_shader_in) ||
             (var->data.mode == nir_var_shader_out)) {
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         }
         break;
      case MESA_SHADER_FRAGMENT:
         if (var->data.mode == nir_var_shader_in) {
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         } else if (var->data.mode == nir_var_shader_out) {
            loc = gl_frag_result_name(var->data.location);
         }
         break;
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
      case MESA_SHADER_COMPUTE:
      case MESA_SHADER_KERNEL:
      default:
         /* TODO */
         break;
      }

      if (!loc) {
         if (var->data.location == ~0) {
            loc = "~0";
         } else {
            snprintf(buf, sizeof(buf), "%u", var->data.location);
            loc = buf;
         }
      }

      /* For shader I/O vars that have been split to components or packed,
       * print the fractional location within the input/output.
       */
      unsigned int num_components =
         glsl_get_components(glsl_without_array(var->type));
      const char *components = NULL;
      char components_local[18] = {'.' /* the rest is 0-filled */};
      switch (var->data.mode) {
      case nir_var_shader_in:
      case nir_var_shader_out:
         if (num_components < 16 && num_components != 0) {
            const char *xyzw = comp_mask_string(num_components);
            for (int i = 0; i < num_components; i++)
               components_local[i + 1] = xyzw[i + var->data.location_frac];

            components = components_local;
         }
         break;
      default:
         break;
      }

      fprintf(fp, " (%s%s, %u, %u)%s", loc,
              components ? components : "",
              var->data.driver_location, var->data.binding,
              var->data.compact ? " compact" : "");
   }

   if (var->constant_initializer) {
      fprintf(fp, " = { ");
      print_constant(var->constant_initializer, var->type, state);
      fprintf(fp, " }");
   }
   if (glsl_type_is_sampler(var->type) && var->data.sampler.is_inline_sampler) {
      fprintf(fp, " = { %s, %s, %s }",
              get_constant_sampler_addressing_mode(var->data.sampler.addressing_mode),
              var->data.sampler.normalized_coordinates ? "true" : "false",
              get_constant_sampler_filter_mode(var->data.sampler.filter_mode));
   }
   if (var->pointer_initializer)
      fprintf(fp, " = &%s", get_var_name(var->pointer_initializer, state));

   fprintf(fp, "\n");
   print_annotation(state, var);
}

static void
print_deref_link(const nir_deref_instr *instr, bool whole_chain, print_state *state)
{
   FILE *fp = state->fp;

   if (instr->deref_type == nir_deref_type_var) {
      fprintf(fp, "%s", get_var_name(instr->var, state));
      return;
   } else if (instr->deref_type == nir_deref_type_cast) {
      fprintf(fp, "(%s *)", glsl_get_type_name(instr->type));
      print_src(&instr->parent, state);
      return;
   }

   assert(instr->parent.is_ssa);
   nir_deref_instr *parent =
      nir_instr_as_deref(instr->parent.ssa->parent_instr);

   /* Is the parent we're going to print a bare cast? */
   const bool is_parent_cast =
      whole_chain && parent->deref_type == nir_deref_type_cast;

   /* If we're not printing the whole chain, the parent we print will be a SSA
    * value that represents a pointer.  The only deref type that naturally
    * gives a pointer is a cast.
    */
   const bool is_parent_pointer =
      !whole_chain || parent->deref_type == nir_deref_type_cast;

   /* Struct derefs have a nice syntax that works on pointers, arrays derefs
    * do not.
    */
   const bool need_deref =
      is_parent_pointer && instr->deref_type != nir_deref_type_struct;

   /* Cast need extra parens and so * dereferences */
   if (is_parent_cast || need_deref)
      fprintf(fp, "(");

   if (need_deref)
      fprintf(fp, "*");

   if (whole_chain) {
      print_deref_link(parent, whole_chain, state);
   } else {
      print_src(&instr->parent, state);
   }

   if (is_parent_cast || need_deref)
      fprintf(fp, ")");

   switch (instr->deref_type) {
   case nir_deref_type_struct:
      fprintf(fp, "%s%s", is_parent_pointer ? "->" : ".",
              glsl_get_struct_elem_name(parent->type, instr->strct.index));
      break;

   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array: {
      if (nir_src_is_const(instr->arr.index)) {
         fprintf(fp, "[%"PRId64"]", nir_src_as_int(instr->arr.index));
      } else {
         fprintf(fp, "[");
         print_src(&instr->arr.index, state);
         fprintf(fp, "]");
      }
      break;
   }

   case nir_deref_type_array_wildcard:
      fprintf(fp, "[*]");
      break;

   default:
      unreachable("Invalid deref instruction type");
   }
}

static void
print_deref_instr(nir_deref_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_dest(&instr->dest, state);

   switch (instr->deref_type) {
   case nir_deref_type_var:
      fprintf(fp, " = deref_var ");
      break;
   case nir_deref_type_array:
   case nir_deref_type_array_wildcard:
      fprintf(fp, " = deref_array ");
      break;
   case nir_deref_type_struct:
      fprintf(fp, " = deref_struct ");
      break;
   case nir_deref_type_cast:
      fprintf(fp, " = deref_cast ");
      break;
   case nir_deref_type_ptr_as_array:
      fprintf(fp, " = deref_ptr_as_array ");
      break;
   default:
      unreachable("Invalid deref instruction type");
   }

   /* Only casts naturally return a pointer type */
   if (instr->deref_type != nir_deref_type_cast)
      fprintf(fp, "&");

   print_deref_link(instr, false, state);

   fprintf(fp, " (");
   unsigned modes = instr->modes;
   while (modes) {
      int m = u_bit_scan(&modes);
      fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true),
                          modes ? "|" : "");
   }
   fprintf(fp, " %s) ", glsl_get_type_name(instr->type));

   if (instr->deref_type != nir_deref_type_var &&
       instr->deref_type != nir_deref_type_cast) {
      /* Print the entire chain as a comment */
      fprintf(fp, "/* &");
      print_deref_link(instr, true, state);
      fprintf(fp, " */");
   }

   if (instr->deref_type == nir_deref_type_cast) {
      fprintf(fp, " /* ptr_stride=%u, align_mul=%u, align_offset=%u */",
              instr->cast.ptr_stride,
              instr->cast.align_mul, instr->cast.align_offset);
   }
}

static const char *
vulkan_descriptor_type_name(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER: return "sampler";
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "texture+sampler";
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "texture";
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "image";
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: return "texture-buffer";
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return "image-buffer";
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UBO";
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "SSBO";
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: return "UBO";
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return "SSBO";
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return "input-att";
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: return "inline-UBO";
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return "accel-struct";
   default: return "unknown";
   }
}

static void
print_alu_type(nir_alu_type type, print_state *state)
{
   FILE *fp = state->fp;
   unsigned size = nir_alu_type_get_type_size(type);
   const char *name;

   switch (nir_alu_type_get_base_type(type)) {
   case nir_type_int: name = "int"; break;
   case nir_type_uint: name = "uint"; break;
   case nir_type_bool: name = "bool"; break;
   case nir_type_float: name = "float"; break;
   default: name = "invalid";
   }
   if (size)
      fprintf(fp, "%s%u", name, size);
   else
      fprintf(fp, "%s", name);
}

static void
print_intrinsic_instr(nir_intrinsic_instr *instr, print_state *state)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[instr->intrinsic];
   unsigned num_srcs = info->num_srcs;
   FILE *fp = state->fp;

   if (info->has_dest) {
      print_dest(&instr->dest, state);
      fprintf(fp, " = ");
   }

   fprintf(fp, "intrinsic %s (", info->name);

   for (unsigned i = 0; i < num_srcs; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_src(&instr->src[i], state);
   }

   fprintf(fp, ") (");

   for (unsigned i = 0; i < info->num_indices; i++) {
      unsigned idx = info->indices[i];
      bool print_raw = true;
      if (i != 0)
         fprintf(fp, ", ");
      switch (idx) {
      case NIR_INTRINSIC_WRITE_MASK: {
         /* special case wrmask to show it as a writemask.. */
         unsigned wrmask = nir_intrinsic_write_mask(instr);
         fprintf(fp, "wrmask=");
         for (unsigned i = 0; i < instr->num_components; i++)
            if ((wrmask >> i) & 1)
               fprintf(fp, "%c", comp_mask_string(instr->num_components)[i]);
         break;
      }

      case NIR_INTRINSIC_REDUCTION_OP: {
         nir_op reduction_op = nir_intrinsic_reduction_op(instr);
         fprintf(fp, "reduction_op=%s", nir_op_infos[reduction_op].name);
         break;
      }

      case NIR_INTRINSIC_IMAGE_DIM: {
         static const char *dim_name[] = {
            [GLSL_SAMPLER_DIM_1D] = "1D",
            [GLSL_SAMPLER_DIM_2D] = "2D",
            [GLSL_SAMPLER_DIM_3D] = "3D",
            [GLSL_SAMPLER_DIM_CUBE] = "Cube",
            [GLSL_SAMPLER_DIM_RECT] = "Rect",
            [GLSL_SAMPLER_DIM_BUF] = "Buf",
            [GLSL_SAMPLER_DIM_MS] = "2D-MSAA",
            [GLSL_SAMPLER_DIM_SUBPASS] = "Subpass",
            [GLSL_SAMPLER_DIM_SUBPASS_MS] = "Subpass-MSAA",
         };
         enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
         assert(dim < ARRAY_SIZE(dim_name) && dim_name[dim]);
         fprintf(fp, "image_dim=%s", dim_name[dim]);
         break;
      }

      case NIR_INTRINSIC_IMAGE_ARRAY: {
         bool array = nir_intrinsic_image_array(instr);
         fprintf(fp, "image_array=%s", array ? "true" : "false");
         break;
      }

      case NIR_INTRINSIC_FORMAT: {
         enum pipe_format format = nir_intrinsic_format(instr);
         fprintf(fp, "format=%s", util_format_short_name(format));
         break;
      }

      case NIR_INTRINSIC_DESC_TYPE: {
         VkDescriptorType desc_type = nir_intrinsic_desc_type(instr);
         fprintf(fp, "desc_type=%s", vulkan_descriptor_type_name(desc_type));
         break;
      }

      case NIR_INTRINSIC_SRC_TYPE: {
         fprintf(fp, "src_type=");
         print_alu_type(nir_intrinsic_src_type(instr), state);
         break;
      }

      case NIR_INTRINSIC_DEST_TYPE: {
         fprintf(fp, "dest_type=");
         print_alu_type(nir_intrinsic_dest_type(instr), state);
         break;
      }

      case NIR_INTRINSIC_SWIZZLE_MASK: {
         fprintf(fp, "swizzle_mask=");
         unsigned mask = nir_intrinsic_swizzle_mask(instr);
         if (instr->intrinsic == nir_intrinsic_quad_swizzle_amd) {
            for (unsigned i = 0; i < 4; i++)
               fprintf(fp, "%d", (mask >> (i * 2) & 3));
         } else if (instr->intrinsic == nir_intrinsic_masked_swizzle_amd) {
            fprintf(fp, "((id & %d) | %d) ^ %d", mask & 0x1F,
                                                (mask >> 5) & 0x1F,
                                                (mask >> 10) & 0x1F);
         } else {
            fprintf(fp, "%d", mask);
         }
         break;
      }

      case NIR_INTRINSIC_MEMORY_SEMANTICS: {
         nir_memory_semantics semantics = nir_intrinsic_memory_semantics(instr);
         fprintf(fp, "mem_semantics=");
         switch (semantics & (NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE)) {
         case 0:                  fprintf(fp, "NONE");    break;
         case NIR_MEMORY_ACQUIRE: fprintf(fp, "ACQ");     break;
         case NIR_MEMORY_RELEASE: fprintf(fp, "REL");     break;
         default:                 fprintf(fp, "ACQ|REL"); break;
         }
         if (semantics & (NIR_MEMORY_MAKE_AVAILABLE)) fprintf(fp, "|AVAILABLE");
         if (semantics & (NIR_MEMORY_MAKE_VISIBLE))   fprintf(fp, "|VISIBLE");
         break;
      }

      case NIR_INTRINSIC_MEMORY_MODES: {
         fprintf(fp, "mem_modes=");
         unsigned int modes = nir_intrinsic_memory_modes(instr);
         while (modes) {
            nir_variable_mode m = u_bit_scan(&modes);
            fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true), modes ? "|" : "");
         }
         break;
      }

      case NIR_INTRINSIC_EXECUTION_SCOPE:
      case NIR_INTRINSIC_MEMORY_SCOPE: {
         fprintf(fp, "%s=", nir_intrinsic_index_names[idx]);
         nir_scope scope =
            idx == NIR_INTRINSIC_MEMORY_SCOPE ? nir_intrinsic_memory_scope(instr)
                                              : nir_intrinsic_execution_scope(instr);
         switch (scope) {
         case NIR_SCOPE_NONE:         fprintf(fp, "NONE");         break;
         case NIR_SCOPE_DEVICE:       fprintf(fp, "DEVICE");       break;
         case NIR_SCOPE_QUEUE_FAMILY: fprintf(fp, "QUEUE_FAMILY"); break;
         case NIR_SCOPE_WORKGROUP:    fprintf(fp, "WORKGROUP");    break;
         case NIR_SCOPE_SHADER_CALL:  fprintf(fp, "SHADER_CALL");  break;
         case NIR_SCOPE_SUBGROUP:     fprintf(fp, "SUBGROUP");     break;
         case NIR_SCOPE_INVOCATION:   fprintf(fp, "INVOCATION");   break;
         }
         break;
      }

      case NIR_INTRINSIC_IO_SEMANTICS: {
         struct nir_io_semantics io = nir_intrinsic_io_semantics(instr);
         fprintf(fp, "io location=%u slots=%u", io.location, io.num_slots);

         if (io.dual_source_blend_index)
            fprintf(fp, " dualsrc");

         if (io.fb_fetch_output)
            fprintf(fp, " fbfetch");

         if (io.per_view)
            fprintf(fp, " perview");

         if (io.medium_precision)
            fprintf(fp, " mediump");

         if (io.high_16bits)
            fprintf(fp, " high_16bits");

         if (io.no_varying)
            fprintf(fp, " no_varying");

         if (io.no_sysval_output)
            fprintf(fp, " no_sysval_output");

         if (state->shader &&
               state->shader->info.stage == MESA_SHADER_GEOMETRY &&
               (instr->intrinsic == nir_intrinsic_store_output ||
                instr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                instr->intrinsic == nir_intrinsic_store_per_vertex_output)) {
            unsigned gs_streams = io.gs_streams;
            fprintf(fp, " gs_streams(");
            for (unsigned i = 0; i < 4; i++) {
               fprintf(fp, "%s%c=%u", i ? " " : "", "xyzw"[i],
                       (gs_streams >> (i * 2)) & 0x3);
            }
            fprintf(fp, ")");
         }

         break;
      }

      case NIR_INTRINSIC_IO_XFB:
      case NIR_INTRINSIC_IO_XFB2: {
         /* This prints both IO_XFB and IO_XFB2. */
         fprintf(fp, "xfb%s(", idx == NIR_INTRINSIC_IO_XFB ? "" : "2");
         bool first = true;
         for (unsigned i = 0; i < 2; i++) {
            unsigned start_comp = (idx == NIR_INTRINSIC_IO_XFB ? 0 : 2) + i;
            nir_io_xfb xfb = start_comp < 2 ? nir_intrinsic_io_xfb(instr) :
                                              nir_intrinsic_io_xfb2(instr);

            if (!xfb.out[i].num_components)
               continue;

            if (!first)
               fprintf(fp, ", ");
            first = false;

            if (xfb.out[i].num_components > 1) {
               fprintf(fp, "components=%u..%u",
                       start_comp, start_comp + xfb.out[i].num_components - 1);
            } else {
               fprintf(fp, "component=%u", start_comp);
            }
            fprintf(fp, " buffer=%u offset=%u",
                    xfb.out[i].buffer, (uint32_t)xfb.out[i].offset * 4);
         }
         fprintf(fp, ")");
         break;
      }

      case NIR_INTRINSIC_ROUNDING_MODE: {
         fprintf(fp, "rounding_mode=");
         switch (nir_intrinsic_rounding_mode(instr)) {
         case nir_rounding_mode_undef: fprintf(fp, "undef");   break;
         case nir_rounding_mode_rtne:  fprintf(fp, "rtne");    break;
         case nir_rounding_mode_ru:    fprintf(fp, "ru");      break;
         case nir_rounding_mode_rd:    fprintf(fp, "rd");      break;
         case nir_rounding_mode_rtz:   fprintf(fp, "rtz");     break;
         default:                      fprintf(fp, "unkown");  break;
         }
         break;
      }

      case NIR_INTRINSIC_RAY_QUERY_VALUE: {
         fprintf(fp, "ray_query_value=");
         switch (nir_intrinsic_ray_query_value(instr)) {
#define VAL(_name) case nir_ray_query_value_##_name: fprintf(fp, #_name); break
         VAL(intersection_type);
         VAL(intersection_t);
         VAL(intersection_instance_custom_index);
         VAL(intersection_instance_id);
         VAL(intersection_instance_sbt_index);
         VAL(intersection_geometry_index);
         VAL(intersection_primitive_index);
         VAL(intersection_barycentrics);
         VAL(intersection_front_face);
         VAL(intersection_object_ray_direction);
         VAL(intersection_object_ray_origin);
         VAL(intersection_object_to_world);
         VAL(intersection_world_to_object);
         VAL(intersection_candidate_aabb_opaque);
         VAL(tmin);
         VAL(flags);
         VAL(world_ray_direction);
         VAL(world_ray_origin);
#undef VAL
         default: fprintf(fp, "unknown"); break;
         }
         break;
      }

      default: {
         unsigned off = info->index_map[idx] - 1;
         fprintf(fp, "%s=%d", nir_intrinsic_index_names[idx], instr->const_index[off]);
         print_raw = false;
         break;
      }
      }
      if (print_raw)
         fprintf(fp, " /*%d*/", instr->const_index[i]);
   }
   fprintf(fp, ")");

   if (!state->shader)
      return;

   nir_variable_mode var_mode;
   switch (instr->intrinsic) {
   case nir_intrinsic_load_uniform:
      var_mode = nir_var_uniform;
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_vertex_input:
      var_mode = nir_var_shader_in;
      break;
   case nir_intrinsic_load_output:
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      var_mode = nir_var_shader_out;
      break;
   default:
      return;
   }

   nir_foreach_variable_with_modes(var, state->shader, var_mode) {
      if ((var->data.driver_location == nir_intrinsic_base(instr)) &&
          (instr->intrinsic == nir_intrinsic_load_uniform ||
           (nir_intrinsic_component(instr) >= var->data.location_frac  &&
            nir_intrinsic_component(instr) <
            (var->data.location_frac + glsl_get_components(var->type)))) &&
           var->name) {
         fprintf(fp, "\t/* %s */", var->name);
         break;
      }
   }
}

static void
print_tex_instr(nir_tex_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_dest(&instr->dest, state);

   fprintf(fp, " = (");
   print_alu_type(instr->dest_type, state);
   fprintf(fp, ")");

   switch (instr->op) {
   case nir_texop_tex:
      fprintf(fp, "tex ");
      break;
   case nir_texop_txb:
      fprintf(fp, "txb ");
      break;
   case nir_texop_txl:
      fprintf(fp, "txl ");
      break;
   case nir_texop_txd:
      fprintf(fp, "txd ");
      break;
   case nir_texop_txf:
      fprintf(fp, "txf ");
      break;
   case nir_texop_txf_ms:
      fprintf(fp, "txf_ms ");
      break;
   case nir_texop_txf_ms_fb:
      fprintf(fp, "txf_ms_fb ");
      break;
   case nir_texop_txf_ms_mcs_intel:
      fprintf(fp, "txf_ms_mcs_intel ");
      break;
   case nir_texop_txs:
      fprintf(fp, "txs ");
      break;
   case nir_texop_lod:
      fprintf(fp, "lod ");
      break;
   case nir_texop_tg4:
      fprintf(fp, "tg4 ");
      break;
   case nir_texop_query_levels:
      fprintf(fp, "query_levels ");
      break;
   case nir_texop_texture_samples:
      fprintf(fp, "texture_samples ");
      break;
   case nir_texop_samples_identical:
      fprintf(fp, "samples_identical ");
      break;
   case nir_texop_tex_prefetch:
      fprintf(fp, "tex (pre-dispatchable) ");
      break;
   case nir_texop_fragment_fetch_amd:
      fprintf(fp, "fragment_fetch_amd ");
      break;
   case nir_texop_fragment_mask_fetch_amd:
      fprintf(fp, "fragment_mask_fetch_amd ");
      break;
   case nir_texop_descriptor_amd:
      fprintf(fp, "descriptor_amd ");
      break;
   case nir_texop_sampler_descriptor_amd:
      fprintf(fp, "sampler_descriptor_amd ");
      break;
   default:
      unreachable("Invalid texture operation");
      break;
   }

   bool has_texture_deref = false, has_sampler_deref = false;
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      if (i > 0) {
         fprintf(fp, ", ");
      }

      print_src(&instr->src[i].src, state);
      fprintf(fp, " ");

      switch(instr->src[i].src_type) {
      case nir_tex_src_backend1:
         fprintf(fp, "(backend1)");
         break;
      case nir_tex_src_backend2:
         fprintf(fp, "(backend2)");
         break;
      case nir_tex_src_coord:
         fprintf(fp, "(coord)");
         break;
      case nir_tex_src_projector:
         fprintf(fp, "(projector)");
         break;
      case nir_tex_src_comparator:
         fprintf(fp, "(comparator)");
         break;
      case nir_tex_src_offset:
         fprintf(fp, "(offset)");
         break;
      case nir_tex_src_bias:
         fprintf(fp, "(bias)");
         break;
      case nir_tex_src_lod:
         fprintf(fp, "(lod)");
         break;
      case nir_tex_src_min_lod:
         fprintf(fp, "(min_lod)");
         break;
      case nir_tex_src_ms_index:
         fprintf(fp, "(ms_index)");
         break;
      case nir_tex_src_ms_mcs_intel:
         fprintf(fp, "(ms_mcs_intel)");
         break;
      case nir_tex_src_ddx:
         fprintf(fp, "(ddx)");
         break;
      case nir_tex_src_ddy:
         fprintf(fp, "(ddy)");
         break;
      case nir_tex_src_texture_deref:
         has_texture_deref = true;
         fprintf(fp, "(texture_deref)");
         break;
      case nir_tex_src_sampler_deref:
         has_sampler_deref = true;
         fprintf(fp, "(sampler_deref)");
         break;
      case nir_tex_src_texture_offset:
         fprintf(fp, "(texture_offset)");
         break;
      case nir_tex_src_sampler_offset:
         fprintf(fp, "(sampler_offset)");
         break;
      case nir_tex_src_texture_handle:
         fprintf(fp, "(texture_handle)");
         break;
      case nir_tex_src_sampler_handle:
         fprintf(fp, "(sampler_handle)");
         break;
      case nir_tex_src_plane:
         fprintf(fp, "(plane)");
         break;

      default:
         unreachable("Invalid texture source type");
         break;
      }
   }

   if (instr->op == nir_texop_tg4) {
      fprintf(fp, ", %u (gather_component)", instr->component);
   }

   if (nir_tex_instr_has_explicit_tg4_offsets(instr)) {
      fprintf(fp, ", { (%i, %i)", instr->tg4_offsets[0][0], instr->tg4_offsets[0][1]);
      for (unsigned i = 1; i < 4; ++i)
         fprintf(fp, ", (%i, %i)", instr->tg4_offsets[i][0],
                 instr->tg4_offsets[i][1]);
      fprintf(fp, " } (offsets)");
   }

   if (instr->op != nir_texop_txf_ms_fb) {
      if (!has_texture_deref) {
         fprintf(fp, ", %u (texture)", instr->texture_index);
      }

      if (!has_sampler_deref) {
         fprintf(fp, ", %u (sampler)", instr->sampler_index);
      }
   }

   if (instr->texture_non_uniform) {
      fprintf(fp, ", texture non-uniform");
   }

   if (instr->sampler_non_uniform) {
      fprintf(fp, ", sampler non-uniform");
   }

   if (instr->is_sparse) {
      fprintf(fp, ", sparse");
   }
}

static void
print_call_instr(nir_call_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "call %s ", instr->callee->name);

   for (unsigned i = 0; i < instr->num_params; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_src(&instr->params[i], state);
   }
}

static void
print_jump_instr(nir_jump_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   switch (instr->type) {
   case nir_jump_break:
      fprintf(fp, "break");
      break;

   case nir_jump_continue:
      fprintf(fp, "continue");
      break;

   case nir_jump_return:
      fprintf(fp, "return");
      break;

   case nir_jump_halt:
      fprintf(fp, "halt");
      break;

   case nir_jump_goto:
      fprintf(fp, "goto block_%u",
              instr->target ? instr->target->index : -1);
      break;

   case nir_jump_goto_if:
      fprintf(fp, "goto block_%u if ",
              instr->target ? instr->target->index : -1);
      print_src(&instr->condition, state);
      fprintf(fp, " else block_%u",
              instr->else_target ? instr->else_target->index : -1);
      break;
   }
}

static void
print_ssa_undef_instr(nir_ssa_undef_instr* instr, print_state *state)
{
   FILE *fp = state->fp;
   print_ssa_def(&instr->def, state);
   fprintf(fp, " = undefined");
}

static void
print_phi_instr(nir_phi_instr *instr, print_state *state)
{
   FILE *fp = state->fp;
   print_dest(&instr->dest, state);
   fprintf(fp, " = phi ");
   nir_foreach_phi_src(src, instr) {
      if (&src->node != exec_list_get_head(&instr->srcs))
         fprintf(fp, ", ");

      fprintf(fp, "block_%u: ", src->pred->index);
      print_src(&src->src, state);
   }
}

static void
print_parallel_copy_instr(nir_parallel_copy_instr *instr, print_state *state)
{
   FILE *fp = state->fp;
   nir_foreach_parallel_copy_entry(entry, instr) {
      if (&entry->node != exec_list_get_head(&instr->entries))
         fprintf(fp, "; ");

      print_dest(&entry->dest, state);
      fprintf(fp, " = ");
      print_src(&entry->src, state);
   }
}

static void
print_instr(const nir_instr *instr, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;
   print_tabs(tabs, fp);

   switch (instr->type) {
   case nir_instr_type_alu:
      print_alu_instr(nir_instr_as_alu(instr), state);
      break;

   case nir_instr_type_deref:
      print_deref_instr(nir_instr_as_deref(instr), state);
      break;

   case nir_instr_type_call:
      print_call_instr(nir_instr_as_call(instr), state);
      break;

   case nir_instr_type_intrinsic:
      print_intrinsic_instr(nir_instr_as_intrinsic(instr), state);
      break;

   case nir_instr_type_tex:
      print_tex_instr(nir_instr_as_tex(instr), state);
      break;

   case nir_instr_type_load_const:
      print_load_const_instr(nir_instr_as_load_const(instr), state);
      break;

   case nir_instr_type_jump:
      print_jump_instr(nir_instr_as_jump(instr), state);
      break;

   case nir_instr_type_ssa_undef:
      print_ssa_undef_instr(nir_instr_as_ssa_undef(instr), state);
      break;

   case nir_instr_type_phi:
      print_phi_instr(nir_instr_as_phi(instr), state);
      break;

   case nir_instr_type_parallel_copy:
      print_parallel_copy_instr(nir_instr_as_parallel_copy(instr), state);
      break;

   default:
      unreachable("Invalid instruction type");
      break;
   }
}

static void print_cf_node(nir_cf_node *node, print_state *state,
                          unsigned tabs);

static void
print_block(nir_block *block, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "block block_%u:\n", block->index);

   nir_block **preds = nir_block_get_predecessors_sorted(block, NULL);

   print_tabs(tabs, fp);
   fprintf(fp, "/* preds: ");
   for (unsigned i = 0; i < block->predecessors->entries; i++) {
      fprintf(fp, "block_%u ", preds[i]->index);
   }
   fprintf(fp, "*/\n");

   ralloc_free(preds);

   nir_foreach_instr(instr, block) {
      print_instr(instr, state, tabs);
      fprintf(fp, "\n");
      print_annotation(state, instr);
   }

   print_tabs(tabs, fp);
   fprintf(fp, "/* succs: ");
   for (unsigned i = 0; i < 2; i++)
      if (block->successors[i]) {
         fprintf(fp, "block_%u ", block->successors[i]->index);
      }
   fprintf(fp, "*/\n");
}

static void
print_if(nir_if *if_stmt, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "if ");
   print_src(&if_stmt->condition, state);
   switch (if_stmt->control) {
   case nir_selection_control_flatten:
      fprintf(fp, " /* flatten */");
      break;
   case nir_selection_control_dont_flatten:
      fprintf(fp, " /* don't flatten */");
      break;
   case nir_selection_control_divergent_always_taken:
      fprintf(fp, " /* divergent always taken */");
      break;
   case nir_selection_control_none:
   default:
      break;
   }
   fprintf(fp, " {\n");
   foreach_list_typed(nir_cf_node, node, node, &if_stmt->then_list) {
      print_cf_node(node, state, tabs + 1);
   }
   print_tabs(tabs, fp);
   fprintf(fp, "} else {\n");
   foreach_list_typed(nir_cf_node, node, node, &if_stmt->else_list) {
      print_cf_node(node, state, tabs + 1);
   }
   print_tabs(tabs, fp);
   fprintf(fp, "}\n");
}

static void
print_loop(nir_loop *loop, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "loop {\n");
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      print_cf_node(node, state, tabs + 1);
   }
   print_tabs(tabs, fp);
   fprintf(fp, "}\n");
}

static void
print_cf_node(nir_cf_node *node, print_state *state, unsigned int tabs)
{
   switch (node->type) {
   case nir_cf_node_block:
      print_block(nir_cf_node_as_block(node), state, tabs);
      break;

   case nir_cf_node_if:
      print_if(nir_cf_node_as_if(node), state, tabs);
      break;

   case nir_cf_node_loop:
      print_loop(nir_cf_node_as_loop(node), state, tabs);
      break;

   default:
      unreachable("Invalid CFG node type");
   }
}

static void
print_function_impl(nir_function_impl *impl, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "\nimpl %s ", impl->function->name);

   fprintf(fp, "{\n");

   if (impl->preamble) {
      fprintf(fp, "\tpreamble %s\n", impl->preamble->name);
   }

   nir_foreach_function_temp_variable(var, impl) {
      fprintf(fp, "\t");
      print_var_decl(var, state);
   }

   foreach_list_typed(nir_register, reg, node, &impl->registers) {
      fprintf(fp, "\t");
      print_register_decl(reg, state);
   }

   nir_index_blocks(impl);

   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      print_cf_node(node, state, 1);
   }

   fprintf(fp, "\tblock block_%u:\n}\n\n", impl->end_block->index);
}

static void
print_function(nir_function *function, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "decl_function %s (%d params)", function->name,
           function->num_params);

   fprintf(fp, "\n");

   if (function->impl != NULL) {
      print_function_impl(function->impl, state);
      return;
   }
}

static void
init_print_state(print_state *state, nir_shader *shader, FILE *fp)
{
   state->fp = fp;
   state->shader = shader;
   state->ht = _mesa_pointer_hash_table_create(NULL);
   state->syms = _mesa_set_create(NULL, _mesa_hash_string,
                                  _mesa_key_string_equal);
   state->index = 0;
}

static void
destroy_print_state(print_state *state)
{
   _mesa_hash_table_destroy(state->ht, NULL);
   _mesa_set_destroy(state->syms, NULL);
}

static const char *
primitive_name(unsigned primitive)
{
#define PRIM(X) case SHADER_PRIM_ ## X : return #X
   switch (primitive) {
   PRIM(POINTS);
   PRIM(LINES);
   PRIM(LINE_LOOP);
   PRIM(LINE_STRIP);
   PRIM(TRIANGLES);
   PRIM(TRIANGLE_STRIP);
   PRIM(TRIANGLE_FAN);
   PRIM(QUADS);
   PRIM(QUAD_STRIP);
   PRIM(POLYGON);
   default:
      return "UNKNOWN";
   }
}

static void
print_bitset(FILE *fp, const char *label, const unsigned *words, int size)
{
   fprintf(fp, "%s: ", label);
   /* Iterate back-to-front to get proper digit order (most significant first). */
   for (int i = size - 1; i >= 0; --i) {
      fprintf(fp, (i == size - 1) ? "0x%08x" : "'%08x", words[i]);
   }
   fprintf(fp, "\n");
}

/* Print bitset, only if some bits are set */
static void
print_nz_bitset(FILE *fp, const char *label, const unsigned *words, int size)
{
   bool is_all_zero = true;
   for (int i = 0; i < size; ++i) {
      if (words[i]) {
         is_all_zero = false;
         break;
      }
   }

   if (!is_all_zero)
      print_bitset(fp, label, words, size);
}

/* Print uint64_t value, only if non-zero.
 * The value is printed by enumerating the ranges of bits that are set.
 * E.g. inputs_read: 0,15-17
 */
static void
print_nz_x64(FILE *fp, const char *label, uint64_t value)
{
   if (value) {
      char acc[256] = {0};
      char buf[32];
      int start = 0;
      int count = 0;
      while (value) {
         u_bit_scan_consecutive_range64(&value, &start, &count);
         assert(count > 0);
         bool is_first = !acc[0];
         if (count > 1) {
            snprintf(buf, sizeof(buf), is_first ? "%d-%d" : ",%d-%d", start, start + count - 1);
         } else {
            snprintf(buf, sizeof(buf), is_first ? "%d" : ",%d", start);
         }
         assert(strlen(acc) + strlen(buf) + 1 < sizeof(acc));
         strcat(acc, buf);
      }
      fprintf(fp, "%s: %s\n", label, acc);
   }
}

/* Print uint32_t value in hex, only if non-zero */
static void
print_nz_x32(FILE *fp, const char *label, uint32_t value)
{
   if (value)
      fprintf(fp, "%s: 0x%08" PRIx32 "\n", label, value);
}

/* Print uint16_t value in hex, only if non-zero */
static void
print_nz_x16(FILE *fp, const char *label, uint16_t value)
{
   if (value)
      fprintf(fp, "%s: 0x%04x\n", label, value);
}

/* Print uint8_t value in hex, only if non-zero */
static void
print_nz_x8(FILE *fp, const char *label, uint8_t value)
{
   if (value)
      fprintf(fp, "%s: 0x%02x\n", label, value);
}

/* Print unsigned value in decimal, only if non-zero */
static void
print_nz_unsigned(FILE *fp, const char *label, unsigned value)
{
   if (value)
      fprintf(fp, "%s: %u\n", label, value);
}

/* Print bool only if set */
static void
print_nz_bool(FILE *fp, const char *label, bool value)
{
   if (value)
      fprintf(fp, "%s: true\n", label);
}

static void
print_shader_info(const struct shader_info *info, FILE *fp)
{
   fprintf(fp, "shader: %s\n", gl_shader_stage_name(info->stage));

   fprintf(fp, "source_sha1: {");
   _mesa_sha1_print(fp, info->source_sha1);
   fprintf(fp, "}\n");

   if (info->name)
      fprintf(fp, "name: %s\n", info->name);

   if (info->label)
      fprintf(fp, "label: %s\n", info->label);

   if (gl_shader_stage_uses_workgroup(info->stage)) {
      fprintf(fp, "workgroup-size: %u, %u, %u%s\n",
              info->workgroup_size[0],
              info->workgroup_size[1],
              info->workgroup_size[2],
              info->workgroup_size_variable ? " (variable)" : "");
      fprintf(fp, "shared-size: %u\n", info->shared_size);
   }

   fprintf(fp, "stage: %d\n"
               "next_stage: %d\n",
           info->stage, info->next_stage);

   print_nz_unsigned(fp, "num_textures", info->num_textures);
   print_nz_unsigned(fp, "num_ubos", info->num_ubos);
   print_nz_unsigned(fp, "num_abos", info->num_abos);
   print_nz_unsigned(fp, "num_ssbos", info->num_ssbos);
   print_nz_unsigned(fp, "num_images", info->num_images);

   print_nz_x64(fp, "inputs_read", info->inputs_read);
   print_nz_x64(fp, "outputs_written", info->outputs_written);
   print_nz_x64(fp, "outputs_read", info->outputs_read);

   print_nz_bitset(fp, "system_values_read", info->system_values_read, ARRAY_SIZE(info->system_values_read));

   print_nz_x64(fp, "per_primitive_inputs", info->per_primitive_inputs);
   print_nz_x64(fp, "per_primitive_outputs", info->per_primitive_outputs);
   print_nz_x64(fp, "per_view_outputs", info->per_view_outputs);

   print_nz_x16(fp, "inputs_read_16bit", info->inputs_read_16bit);
   print_nz_x16(fp, "outputs_written_16bit", info->outputs_written_16bit);
   print_nz_x16(fp, "outputs_read_16bit", info->outputs_read_16bit);
   print_nz_x16(fp, "inputs_read_indirectly_16bit", info->inputs_read_indirectly_16bit);
   print_nz_x16(fp, "outputs_accessed_indirectly_16bit", info->outputs_accessed_indirectly_16bit);

   print_nz_x32(fp, "patch_inputs_read", info->patch_inputs_read);
   print_nz_x32(fp, "patch_outputs_written", info->patch_outputs_written);
   print_nz_x32(fp, "patch_outputs_read", info->patch_outputs_read);

   print_nz_x64(fp, "inputs_read_indirectly", info->inputs_read_indirectly);
   print_nz_x64(fp, "outputs_accessed_indirectly", info->outputs_accessed_indirectly);
   print_nz_x64(fp, "patch_inputs_read_indirectly", info->patch_inputs_read_indirectly);
   print_nz_x64(fp, "patch_outputs_accessed_indirectly", info->patch_outputs_accessed_indirectly);

   print_nz_bitset(fp, "textures_used", info->textures_used, ARRAY_SIZE(info->textures_used));
   print_nz_bitset(fp, "textures_used_by_txf", info->textures_used_by_txf, ARRAY_SIZE(info->textures_used_by_txf));
   print_nz_bitset(fp, "samplers_used", info->samplers_used, ARRAY_SIZE(info->samplers_used));
   print_nz_bitset(fp, "images_used", info->images_used, ARRAY_SIZE(info->images_used));
   print_nz_bitset(fp, "image_buffers", info->image_buffers, ARRAY_SIZE(info->image_buffers));
   print_nz_bitset(fp, "msaa_images", info->msaa_images, ARRAY_SIZE(info->msaa_images));

   print_nz_x16(fp, "float_controls_execution_mode", info->float_controls_execution_mode);

   print_nz_unsigned(fp, "shared_size", info->shared_size);

   if (info->stage == MESA_SHADER_MESH || info->stage == MESA_SHADER_TASK) {
      fprintf(fp, "task_payload_size: %u\n", info->task_payload_size);
   }

   print_nz_unsigned(fp, "ray queries", info->ray_queries);

   fprintf(fp, "subgroup_size: %u\n", info->subgroup_size);

   print_nz_bool(fp, "uses_wide_subgroup_intrinsics", info->uses_wide_subgroup_intrinsics);

   bool has_xfb_stride = info->xfb_stride[0] || info->xfb_stride[1] || info->xfb_stride[2] || info->xfb_stride[3];
   if (has_xfb_stride)
      fprintf(fp, "xfb_stride: {%u, %u, %u, %u}\n",
              info->xfb_stride[0],
              info->xfb_stride[1],
              info->xfb_stride[2],
              info->xfb_stride[3]);

   bool has_inlinable_uniform_dw_offsets = info->inlinable_uniform_dw_offsets[0]
                                           || info->inlinable_uniform_dw_offsets[1]
                                           || info->inlinable_uniform_dw_offsets[2]
                                           || info->inlinable_uniform_dw_offsets[3];
   if (has_inlinable_uniform_dw_offsets)
      fprintf(fp, "inlinable_uniform_dw_offsets: {%u, %u, %u, %u}\n",
              info->inlinable_uniform_dw_offsets[0],
              info->inlinable_uniform_dw_offsets[1],
              info->inlinable_uniform_dw_offsets[2],
              info->inlinable_uniform_dw_offsets[3]);

   print_nz_unsigned(fp, "num_inlinable_uniforms", info->num_inlinable_uniforms);
   print_nz_unsigned(fp, "clip_distance_array_size", info->clip_distance_array_size);
   print_nz_unsigned(fp, "cull_distance_array_size", info->cull_distance_array_size);

   print_nz_bool(fp, "uses_texture_gather", info->uses_texture_gather);
   print_nz_bool(fp, "uses_resource_info_query", info->uses_resource_info_query);
   print_nz_bool(fp, "uses_fddx_fddy", info->uses_fddx_fddy);
   print_nz_bool(fp, "divergence_analysis_run", info->divergence_analysis_run);

   print_nz_x8(fp, "bit_sizes_float", info->bit_sizes_float);
   print_nz_x8(fp, "bit_sizes_int", info->bit_sizes_int);

   print_nz_bool(fp, "first_ubo_is_default_ubo", info->first_ubo_is_default_ubo);
   print_nz_bool(fp, "separate_shader", info->separate_shader);
   print_nz_bool(fp, "has_transform_feedback_varyings", info->has_transform_feedback_varyings);
   print_nz_bool(fp, "flrp_lowered", info->flrp_lowered);
   print_nz_bool(fp, "io_lowered", info->io_lowered);
   print_nz_bool(fp, "writes_memory", info->writes_memory);

   switch (info->stage) {
   case MESA_SHADER_VERTEX:
      print_nz_x64(fp, "double_inputs", info->vs.double_inputs);
      print_nz_unsigned(fp, "blit_sgprs_amd", info->vs.blit_sgprs_amd);
      print_nz_bool(fp, "window_space_position", info->vs.window_space_position);
      print_nz_bool(fp, "needs_edge_flag", info->vs.needs_edge_flag);
      break;

   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
      fprintf(fp, "primitive_mode: %u\n", info->tess._primitive_mode);
      fprintf(fp, "tcs_vertices_out: %u\n", info->tess.tcs_vertices_out);
      fprintf(fp, "spacing: %u\n", info->tess.spacing);

      print_nz_bool(fp, "ccw", info->tess.ccw);
      print_nz_bool(fp, "point_mode", info->tess.point_mode);
      print_nz_x64(fp, "tcs_cross_invocation_inputs_read", info->tess.tcs_cross_invocation_inputs_read);
      print_nz_x64(fp, "tcs_cross_invocation_outputs_read", info->tess.tcs_cross_invocation_outputs_read);
      break;

   case MESA_SHADER_GEOMETRY:
      fprintf(fp, "output_primitive: %s\n", primitive_name(info->gs.output_primitive));
      fprintf(fp, "input_primitive: %s\n", primitive_name(info->gs.input_primitive));
      fprintf(fp, "vertices_out: %u\n", info->gs.vertices_out);
      fprintf(fp, "invocations: %u\n", info->gs.invocations);
      fprintf(fp, "vertices_in: %u\n", info->gs.vertices_in);
      print_nz_bool(fp, "uses_end_primitive", info->gs.uses_end_primitive);
      fprintf(fp, "active_stream_mask: 0x%02x\n", info->gs.active_stream_mask);
      break;

   case MESA_SHADER_FRAGMENT:
      print_nz_bool(fp, "uses_discard", info->fs.uses_discard);
      print_nz_bool(fp, "uses_demote", info->fs.uses_demote);
      print_nz_bool(fp, "uses_fbfetch_output", info->fs.uses_fbfetch_output);
      print_nz_bool(fp, "color_is_dual_source", info->fs.color_is_dual_source);

      print_nz_bool(fp, "needs_quad_helper_invocations", info->fs.needs_quad_helper_invocations);
      print_nz_bool(fp, "needs_all_helper_invocations", info->fs.needs_all_helper_invocations);
      print_nz_bool(fp, "uses_sample_qualifier", info->fs.uses_sample_qualifier);
      print_nz_bool(fp, "uses_sample_shading", info->fs.uses_sample_shading);
      print_nz_bool(fp, "early_fragment_tests", info->fs.early_fragment_tests);
      print_nz_bool(fp, "inner_coverage", info->fs.inner_coverage);
      print_nz_bool(fp, "post_depth_coverage", info->fs.post_depth_coverage);

      print_nz_bool(fp, "pixel_center_integer", info->fs.pixel_center_integer);
      print_nz_bool(fp, "origin_upper_left", info->fs.origin_upper_left);
      print_nz_bool(fp, "pixel_interlock_ordered", info->fs.pixel_interlock_ordered);
      print_nz_bool(fp, "pixel_interlock_unordered", info->fs.pixel_interlock_unordered);
      print_nz_bool(fp, "sample_interlock_ordered", info->fs.sample_interlock_ordered);
      print_nz_bool(fp, "sample_interlock_unordered", info->fs.sample_interlock_unordered);
      print_nz_bool(fp, "untyped_color_outputs", info->fs.untyped_color_outputs);

      print_nz_unsigned(fp, "depth_layout", info->fs.depth_layout);

      fprintf(fp, "color0_interp: %u\n", info->fs.color0_interp);      
      print_nz_bool(fp, "color0_sample", info->fs.color0_sample);
      print_nz_bool(fp, "color0_centroid", info->fs.color0_centroid);

      fprintf(fp, "color1_interp: %u\n", info->fs.color1_interp);
      print_nz_bool(fp, "color1_sample", info->fs.color1_sample);
      print_nz_bool(fp, "color1_centroid", info->fs.color1_centroid);

      print_nz_x32(fp, "advanced_blend_modes", info->fs.advanced_blend_modes);
      break;

   case MESA_SHADER_COMPUTE:
      if (info->cs.workgroup_size_hint[0]
          || info->cs.workgroup_size_hint[1]
          || info->cs.workgroup_size_hint[2])
         fprintf(fp, "workgroup_size_hint: {%u, %u, %u}\n",
                 info->cs.workgroup_size_hint[0],
                 info->cs.workgroup_size_hint[1],
                 info->cs.workgroup_size_hint[2]);
      print_nz_unsigned(fp, "user_data_components_amd", info->cs.user_data_components_amd);
      print_nz_unsigned(fp, "derivative_group", info->cs.derivative_group);
      fprintf(fp, "ptr_size: %u\n", info->cs.ptr_size);
      break;

   case MESA_SHADER_MESH:
      print_nz_x64(fp, "ms_cross_invocation_output_access", info->mesh.ms_cross_invocation_output_access);
      fprintf(fp, "max_vertices_out: %u\n", info->mesh.max_vertices_out);
      fprintf(fp, "max_primitives_out: %u\n", info->mesh.max_primitives_out);
      fprintf(fp, "primitive_type: %s\n", primitive_name(info->mesh.primitive_type));
      print_nz_bool(fp, "nv", info->mesh.nv);
      break;

   default:
      fprintf(fp, "Unhandled stage %d\n", info->stage);
   }
}

void
nir_print_shader_annotated(nir_shader *shader, FILE *fp,
                           struct hash_table *annotations)
{
   print_state state;
   init_print_state(&state, shader, fp);
   state.annotations = annotations;

   print_shader_info(&shader->info, fp);

   fprintf(fp, "inputs: %u\n", shader->num_inputs);
   fprintf(fp, "outputs: %u\n", shader->num_outputs);
   fprintf(fp, "uniforms: %u\n", shader->num_uniforms);
   if (shader->scratch_size)
      fprintf(fp, "scratch: %u\n", shader->scratch_size);
   if (shader->constant_data_size)
      fprintf(fp, "constants: %u\n", shader->constant_data_size);

   nir_foreach_variable_in_shader(var, shader)
      print_var_decl(var, &state);

   foreach_list_typed(nir_function, func, node, &shader->functions) {
      print_function(func, &state);
   }

   destroy_print_state(&state);
}

void
nir_print_shader(nir_shader *shader, FILE *fp)
{
   nir_print_shader_annotated(shader, fp, NULL);
   fflush(fp);
}

char *
nir_shader_as_str_annotated(nir_shader *nir, struct hash_table *annotations, void *mem_ctx)
{
   char *stream_data = NULL;
   size_t stream_size = 0;
   struct u_memstream mem;
   if (u_memstream_open(&mem, &stream_data, &stream_size)) {
      FILE *const stream = u_memstream_get(&mem);
      nir_print_shader_annotated(nir, stream, annotations);
      u_memstream_close(&mem);
   }

   char *str = ralloc_size(mem_ctx, stream_size + 1);
   memcpy(str, stream_data, stream_size);
   str[stream_size] = '\0';

   free(stream_data);

   return str;
}

char *
nir_shader_as_str(nir_shader *nir, void *mem_ctx)
{
   return nir_shader_as_str_annotated(nir, NULL, mem_ctx);
}

void
nir_print_instr(const nir_instr *instr, FILE *fp)
{
   print_state state = {
      .fp = fp,
   };
   if (instr->block) {
      nir_function_impl *impl = nir_cf_node_get_function(&instr->block->cf_node);
      state.shader = impl->function->shader;
   }

   print_instr(instr, &state, 0);
}

char *
nir_instr_as_str(const nir_instr *instr, void *mem_ctx)
{
   char *stream_data = NULL;
   size_t stream_size = 0;
   struct u_memstream mem;
   if (u_memstream_open(&mem, &stream_data, &stream_size)) {
      FILE *const stream = u_memstream_get(&mem);
      nir_print_instr(instr, stream);
      u_memstream_close(&mem);
   }

   char *str = ralloc_size(mem_ctx, stream_size + 1);
   memcpy(str, stream_data, stream_size);
   str[stream_size] = '\0';

   free(stream_data);

   return str;
}

void
nir_print_deref(const nir_deref_instr *deref, FILE *fp)
{
   print_state state = {
      .fp = fp,
   };
   print_deref_link(deref, true, &state);
}

void nir_log_shader_annotated_tagged(enum mesa_log_level level, const char *tag,
                                     nir_shader *shader, struct hash_table *annotations)
{
   char *str = nir_shader_as_str_annotated(shader, annotations, NULL);
   _mesa_log_multiline(level, tag, str);
   ralloc_free(str);
}

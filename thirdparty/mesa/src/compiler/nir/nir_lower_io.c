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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

/*
 * This lowering pass converts references to input/output variables with
 * loads/stores to actual input/output intrinsics.
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "nir_xfb_info.h"

#include "util/u_math.h"

struct lower_io_state {
   void *dead_ctx;
   nir_builder builder;
   int (*type_size)(const struct glsl_type *type, bool);
   nir_variable_mode modes;
   nir_lower_io_options options;
};

static nir_intrinsic_op
ssbo_atomic_for_deref(nir_intrinsic_op deref_op)
{
   switch (deref_op) {
#define OP(O) case nir_intrinsic_deref_##O: return nir_intrinsic_ssbo_##O;
   OP(atomic_exchange)
   OP(atomic_comp_swap)
   OP(atomic_add)
   OP(atomic_imin)
   OP(atomic_umin)
   OP(atomic_imax)
   OP(atomic_umax)
   OP(atomic_and)
   OP(atomic_or)
   OP(atomic_xor)
   OP(atomic_fadd)
   OP(atomic_fmin)
   OP(atomic_fmax)
   OP(atomic_fcomp_swap)
#undef OP
   default:
      unreachable("Invalid SSBO atomic");
   }
}

static nir_intrinsic_op
global_atomic_for_deref(nir_address_format addr_format,
                        nir_intrinsic_op deref_op)
{
   switch (deref_op) {
#define OP(O) case nir_intrinsic_deref_##O:              \
   if (addr_format != nir_address_format_2x32bit_global) \
      return nir_intrinsic_global_##O;                   \
   else                                                  \
      return nir_intrinsic_global_##O##_2x32;
   OP(atomic_exchange)
   OP(atomic_comp_swap)
   OP(atomic_add)
   OP(atomic_imin)
   OP(atomic_umin)
   OP(atomic_imax)
   OP(atomic_umax)
   OP(atomic_and)
   OP(atomic_or)
   OP(atomic_xor)
   OP(atomic_fadd)
   OP(atomic_fmin)
   OP(atomic_fmax)
   OP(atomic_fcomp_swap)
#undef OP
   default:
      unreachable("Invalid SSBO atomic");
   }
}

static nir_intrinsic_op
shared_atomic_for_deref(nir_intrinsic_op deref_op)
{
   switch (deref_op) {
#define OP(O) case nir_intrinsic_deref_##O: return nir_intrinsic_shared_##O;
   OP(atomic_exchange)
   OP(atomic_comp_swap)
   OP(atomic_add)
   OP(atomic_imin)
   OP(atomic_umin)
   OP(atomic_imax)
   OP(atomic_umax)
   OP(atomic_and)
   OP(atomic_or)
   OP(atomic_xor)
   OP(atomic_fadd)
   OP(atomic_fmin)
   OP(atomic_fmax)
   OP(atomic_fcomp_swap)
#undef OP
   default:
      unreachable("Invalid shared atomic");
   }
}

static nir_intrinsic_op
task_payload_atomic_for_deref(nir_intrinsic_op deref_op)
{
   switch (deref_op) {
#define OP(O) case nir_intrinsic_deref_##O: return nir_intrinsic_task_payload_##O;
   OP(atomic_exchange)
   OP(atomic_comp_swap)
   OP(atomic_add)
   OP(atomic_imin)
   OP(atomic_umin)
   OP(atomic_imax)
   OP(atomic_umax)
   OP(atomic_and)
   OP(atomic_or)
   OP(atomic_xor)
   OP(atomic_fadd)
   OP(atomic_fmin)
   OP(atomic_fmax)
   OP(atomic_fcomp_swap)
#undef OP
   default:
      unreachable("Invalid task payload atomic");
   }
}

void
nir_assign_var_locations(nir_shader *shader, nir_variable_mode mode,
                         unsigned *size,
                         int (*type_size)(const struct glsl_type *, bool))
{
   unsigned location = 0;

   nir_foreach_variable_with_modes(var, shader, mode) {
      var->data.driver_location = location;
      bool bindless_type_size = var->data.mode == nir_var_shader_in ||
                                var->data.mode == nir_var_shader_out ||
                                var->data.bindless;
      location += type_size(var->type, bindless_type_size);
   }

   *size = location;
}

/**
 * Some inputs and outputs are arrayed, meaning that there is an extra level
 * of array indexing to handle mismatches between the shader interface and the
 * dispatch pattern of the shader.  For instance, geometry shaders are
 * executed per-primitive while their inputs and outputs are specified
 * per-vertex so all inputs and outputs have to be additionally indexed with
 * the vertex index within the primitive.
 */
bool
nir_is_arrayed_io(const nir_variable *var, gl_shader_stage stage)
{
   if (var->data.patch || !glsl_type_is_array(var->type))
      return false;

   if (stage == MESA_SHADER_MESH) {
      /* NV_mesh_shader: this is flat array for the whole workgroup. */
      if (var->data.location == VARYING_SLOT_PRIMITIVE_INDICES)
         return var->data.per_primitive;
   }

   if (var->data.mode == nir_var_shader_in)
      return stage == MESA_SHADER_GEOMETRY ||
             stage == MESA_SHADER_TESS_CTRL ||
             stage == MESA_SHADER_TESS_EVAL;

   if (var->data.mode == nir_var_shader_out)
      return stage == MESA_SHADER_TESS_CTRL ||
             stage == MESA_SHADER_MESH;

   return false;
}

static unsigned get_number_of_slots(struct lower_io_state *state,
                                    const nir_variable *var)
{
   const struct glsl_type *type = var->type;

   if (nir_is_arrayed_io(var, state->builder.shader->info.stage)) {
      assert(glsl_type_is_array(type));
      type = glsl_get_array_element(type);
   }

   /* NV_mesh_shader:
    * PRIMITIVE_INDICES is a flat array, not a proper arrayed output,
    * as opposed to D3D-style mesh shaders where it's addressed by
    * the primitive index.
    * Prevent assigning several slots to primitive indices,
    * to avoid some issues.
    */
   if (state->builder.shader->info.stage == MESA_SHADER_MESH &&
       var->data.location == VARYING_SLOT_PRIMITIVE_INDICES &&
       !nir_is_arrayed_io(var, state->builder.shader->info.stage))
      return 1;

   return state->type_size(type, var->data.bindless);
}

static nir_ssa_def *
get_io_offset(nir_builder *b, nir_deref_instr *deref,
              nir_ssa_def **array_index,
              int (*type_size)(const struct glsl_type *, bool),
              unsigned *component, bool bts)
{
   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);

   assert(path.path[0]->deref_type == nir_deref_type_var);
   nir_deref_instr **p = &path.path[1];

   /* For arrayed I/O (e.g., per-vertex input arrays in geometry shader
    * inputs), skip the outermost array index.  Process the rest normally.
    */
   if (array_index != NULL) {
      assert((*p)->deref_type == nir_deref_type_array);
      *array_index = nir_ssa_for_src(b, (*p)->arr.index, 1);
      p++;
   }

   if (path.path[0]->var->data.compact) {
      assert((*p)->deref_type == nir_deref_type_array);
      assert(glsl_type_is_scalar((*p)->type));

      /* We always lower indirect dereferences for "compact" array vars. */
      const unsigned index = nir_src_as_uint((*p)->arr.index);
      const unsigned total_offset = *component + index;
      const unsigned slot_offset = total_offset / 4;
      *component = total_offset % 4;
      return nir_imm_int(b, type_size(glsl_vec4_type(), bts) * slot_offset);
   }

   /* Just emit code and let constant-folding go to town */
   nir_ssa_def *offset = nir_imm_int(b, 0);

   for (; *p; p++) {
      if ((*p)->deref_type == nir_deref_type_array) {
         unsigned size = type_size((*p)->type, bts);

         nir_ssa_def *mul =
            nir_amul_imm(b, nir_ssa_for_src(b, (*p)->arr.index, 1), size);

         offset = nir_iadd(b, offset, mul);
      } else if ((*p)->deref_type == nir_deref_type_struct) {
         /* p starts at path[1], so this is safe */
         nir_deref_instr *parent = *(p - 1);

         unsigned field_offset = 0;
         for (unsigned i = 0; i < (*p)->strct.index; i++) {
            field_offset += type_size(glsl_get_struct_field(parent->type, i), bts);
         }
         offset = nir_iadd_imm(b, offset, field_offset);
      } else {
         unreachable("Unsupported deref type");
      }
   }

   nir_deref_path_finish(&path);

   return offset;
}

static nir_ssa_def *
emit_load(struct lower_io_state *state,
          nir_ssa_def *array_index, nir_variable *var, nir_ssa_def *offset,
          unsigned component, unsigned num_components, unsigned bit_size,
          nir_alu_type dest_type)
{
   nir_builder *b = &state->builder;
   const nir_shader *nir = b->shader;
   nir_variable_mode mode = var->data.mode;
   nir_ssa_def *barycentric = NULL;

   nir_intrinsic_op op;
   switch (mode) {
   case nir_var_shader_in:
      if (nir->info.stage == MESA_SHADER_FRAGMENT &&
          nir->options->use_interpolated_input_intrinsics &&
          var->data.interpolation != INTERP_MODE_FLAT &&
          !var->data.per_primitive) {
         if (var->data.interpolation == INTERP_MODE_EXPLICIT) {
            assert(array_index != NULL);
            op = nir_intrinsic_load_input_vertex;
         } else {
            assert(array_index == NULL);

            nir_intrinsic_op bary_op;
            if (var->data.sample ||
                (state->options & nir_lower_io_force_sample_interpolation))
               bary_op = nir_intrinsic_load_barycentric_sample;
            else if (var->data.centroid)
               bary_op = nir_intrinsic_load_barycentric_centroid;
            else
               bary_op = nir_intrinsic_load_barycentric_pixel;

            barycentric = nir_load_barycentric(&state->builder, bary_op,
                                               var->data.interpolation);
            op = nir_intrinsic_load_interpolated_input;
         }
      } else {
         op = array_index ? nir_intrinsic_load_per_vertex_input :
                            nir_intrinsic_load_input;
      }
      break;
   case nir_var_shader_out:
      op = !array_index            ? nir_intrinsic_load_output :
           var->data.per_primitive ? nir_intrinsic_load_per_primitive_output :
                                     nir_intrinsic_load_per_vertex_output;
      break;
   case nir_var_uniform:
      op = nir_intrinsic_load_uniform;
      break;
   default:
      unreachable("Unknown variable mode");
   }

   nir_intrinsic_instr *load =
      nir_intrinsic_instr_create(state->builder.shader, op);
   load->num_components = num_components;

   nir_intrinsic_set_base(load, var->data.driver_location);
   if (mode == nir_var_shader_in || mode == nir_var_shader_out)
      nir_intrinsic_set_component(load, component);

   if (load->intrinsic == nir_intrinsic_load_uniform)
      nir_intrinsic_set_range(load,
                              state->type_size(var->type, var->data.bindless));

   if (nir_intrinsic_has_access(load))
      nir_intrinsic_set_access(load, var->data.access);

   nir_intrinsic_set_dest_type(load, dest_type);

   if (load->intrinsic != nir_intrinsic_load_uniform) {
      nir_io_semantics semantics = {0};
      semantics.location = var->data.location;
      semantics.num_slots = get_number_of_slots(state, var);
      semantics.fb_fetch_output = var->data.fb_fetch_output;
      semantics.medium_precision =
         var->data.precision == GLSL_PRECISION_MEDIUM ||
         var->data.precision == GLSL_PRECISION_LOW;
      nir_intrinsic_set_io_semantics(load, semantics);
   }

   if (array_index) {
      load->src[0] = nir_src_for_ssa(array_index);
      load->src[1] = nir_src_for_ssa(offset);
   } else if (barycentric) {
      load->src[0] = nir_src_for_ssa(barycentric);
      load->src[1] = nir_src_for_ssa(offset);
   } else {
      load->src[0] = nir_src_for_ssa(offset);
   }

   nir_ssa_dest_init(&load->instr, &load->dest,
                     num_components, bit_size, NULL);
   nir_builder_instr_insert(b, &load->instr);

   return &load->dest.ssa;
}

static nir_ssa_def *
lower_load(nir_intrinsic_instr *intrin, struct lower_io_state *state,
           nir_ssa_def *array_index, nir_variable *var, nir_ssa_def *offset,
           unsigned component, const struct glsl_type *type)
{
   assert(intrin->dest.is_ssa);
   if (intrin->dest.ssa.bit_size == 64 &&
       (state->options & nir_lower_io_lower_64bit_to_32)) {
      nir_builder *b = &state->builder;

      const unsigned slot_size = state->type_size(glsl_dvec_type(2), false);

      nir_ssa_def *comp64[4];
      assert(component == 0 || component == 2);
      unsigned dest_comp = 0;
      while (dest_comp < intrin->dest.ssa.num_components) {
         const unsigned num_comps =
            MIN2(intrin->dest.ssa.num_components - dest_comp,
                 (4 - component) / 2);

         nir_ssa_def *data32 =
            emit_load(state, array_index, var, offset, component,
                      num_comps * 2, 32, nir_type_uint32);
         for (unsigned i = 0; i < num_comps; i++) {
            comp64[dest_comp + i] =
               nir_pack_64_2x32(b, nir_channels(b, data32, 3 << (i * 2)));
         }

         /* Only the first store has a component offset */
         component = 0;
         dest_comp += num_comps;
         offset = nir_iadd_imm(b, offset, slot_size);
      }

      return nir_vec(b, comp64, intrin->dest.ssa.num_components);
   } else if (intrin->dest.ssa.bit_size == 1) {
      /* Booleans are 32-bit */
      assert(glsl_type_is_boolean(type));
      return nir_b2b1(&state->builder,
                      emit_load(state, array_index, var, offset, component,
                                intrin->dest.ssa.num_components, 32,
                                nir_type_bool32));
   } else {
      return emit_load(state, array_index, var, offset, component,
                       intrin->dest.ssa.num_components,
                       intrin->dest.ssa.bit_size,
                       nir_get_nir_type_for_glsl_type(type));
   }
}

static void
emit_store(struct lower_io_state *state, nir_ssa_def *data,
           nir_ssa_def *array_index, nir_variable *var, nir_ssa_def *offset,
           unsigned component, unsigned num_components,
           nir_component_mask_t write_mask, nir_alu_type src_type)
{
   nir_builder *b = &state->builder;

   assert(var->data.mode == nir_var_shader_out);
   nir_intrinsic_op op =
      !array_index            ? nir_intrinsic_store_output :
      var->data.per_primitive ? nir_intrinsic_store_per_primitive_output :
                                nir_intrinsic_store_per_vertex_output;

   nir_intrinsic_instr *store =
      nir_intrinsic_instr_create(state->builder.shader, op);
   store->num_components = num_components;

   store->src[0] = nir_src_for_ssa(data);

   nir_intrinsic_set_base(store, var->data.driver_location);
   nir_intrinsic_set_component(store, component);
   nir_intrinsic_set_src_type(store, src_type);

   nir_intrinsic_set_write_mask(store, write_mask);

   if (nir_intrinsic_has_access(store))
      nir_intrinsic_set_access(store, var->data.access);

   if (array_index)
      store->src[1] = nir_src_for_ssa(array_index);

   store->src[array_index ? 2 : 1] = nir_src_for_ssa(offset);

   unsigned gs_streams = 0;
   if (state->builder.shader->info.stage == MESA_SHADER_GEOMETRY) {
      if (var->data.stream & NIR_STREAM_PACKED) {
         gs_streams = var->data.stream & ~NIR_STREAM_PACKED;
      } else {
         assert(var->data.stream < 4);
         gs_streams = 0;
         for (unsigned i = 0; i < num_components; ++i)
            gs_streams |= var->data.stream << (2 * i);
      }
   }

   nir_io_semantics semantics = {0};
   semantics.location = var->data.location;
   semantics.num_slots = get_number_of_slots(state, var);
   semantics.dual_source_blend_index = var->data.index;
   semantics.gs_streams = gs_streams;
   semantics.medium_precision =
      var->data.precision == GLSL_PRECISION_MEDIUM ||
      var->data.precision == GLSL_PRECISION_LOW;
   semantics.per_view = var->data.per_view;
   semantics.invariant = var->data.invariant;

   nir_intrinsic_set_io_semantics(store, semantics);

   nir_builder_instr_insert(b, &store->instr);
}

static void
lower_store(nir_intrinsic_instr *intrin, struct lower_io_state *state,
            nir_ssa_def *array_index, nir_variable *var, nir_ssa_def *offset,
            unsigned component, const struct glsl_type *type)
{
   assert(intrin->src[1].is_ssa);
   if (intrin->src[1].ssa->bit_size == 64 &&
       (state->options & nir_lower_io_lower_64bit_to_32)) {
      nir_builder *b = &state->builder;

      const unsigned slot_size = state->type_size(glsl_dvec_type(2), false);

      assert(component == 0 || component == 2);
      unsigned src_comp = 0;
      nir_component_mask_t write_mask = nir_intrinsic_write_mask(intrin);
      while (src_comp < intrin->num_components) {
         const unsigned num_comps =
            MIN2(intrin->num_components - src_comp,
                 (4 - component) / 2);

         if (write_mask & BITFIELD_MASK(num_comps)) {
            nir_ssa_def *data =
               nir_channels(b, intrin->src[1].ssa,
                            BITFIELD_RANGE(src_comp, num_comps));
            nir_ssa_def *data32 = nir_bitcast_vector(b, data, 32);

            nir_component_mask_t write_mask32 = 0;
            for (unsigned i = 0; i < num_comps; i++) {
               if (write_mask & BITFIELD_MASK(num_comps) & (1 << i))
                  write_mask32 |= 3 << (i * 2);
            }

            emit_store(state, data32, array_index, var, offset,
                       component, data32->num_components, write_mask32,
                       nir_type_uint32);
         }

         /* Only the first store has a component offset */
         component = 0;
         src_comp += num_comps;
         write_mask >>= num_comps;
         offset = nir_iadd_imm(b, offset, slot_size);
      }
   } else if (intrin->dest.ssa.bit_size == 1) {
      /* Booleans are 32-bit */
      assert(glsl_type_is_boolean(type));
      nir_ssa_def *b32_val = nir_b2b32(&state->builder, intrin->src[1].ssa);
      emit_store(state, b32_val, array_index, var, offset,
                 component, intrin->num_components,
                 nir_intrinsic_write_mask(intrin),
                 nir_type_bool32);
   } else {
      emit_store(state, intrin->src[1].ssa, array_index, var, offset,
                 component, intrin->num_components,
                 nir_intrinsic_write_mask(intrin),
                 nir_get_nir_type_for_glsl_type(type));
   }
}

static nir_ssa_def *
lower_interpolate_at(nir_intrinsic_instr *intrin, struct lower_io_state *state,
                     nir_variable *var, nir_ssa_def *offset, unsigned component,
                     const struct glsl_type *type)
{
   nir_builder *b = &state->builder;
   assert(var->data.mode == nir_var_shader_in);

   /* Ignore interpolateAt() for flat variables - flat is flat. Lower
    * interpolateAtVertex() for explicit variables.
    */
   if (var->data.interpolation == INTERP_MODE_FLAT ||
       var->data.interpolation == INTERP_MODE_EXPLICIT) {
      nir_ssa_def *vertex_index = NULL;

      if (var->data.interpolation == INTERP_MODE_EXPLICIT) {
         assert(intrin->intrinsic == nir_intrinsic_interp_deref_at_vertex);
         vertex_index = intrin->src[1].ssa;
      }

      return lower_load(intrin, state, vertex_index, var, offset, component, type);
   }

   /* None of the supported APIs allow interpolation on 64-bit things */
   assert(intrin->dest.is_ssa && intrin->dest.ssa.bit_size <= 32);

   nir_intrinsic_op bary_op;
   switch (intrin->intrinsic) {
   case nir_intrinsic_interp_deref_at_centroid:
      bary_op = (state->options & nir_lower_io_force_sample_interpolation) ?
                nir_intrinsic_load_barycentric_sample :
                nir_intrinsic_load_barycentric_centroid;
      break;
   case nir_intrinsic_interp_deref_at_sample:
      bary_op = nir_intrinsic_load_barycentric_at_sample;
      break;
   case nir_intrinsic_interp_deref_at_offset:
      bary_op = nir_intrinsic_load_barycentric_at_offset;
      break;
   default:
      unreachable("Bogus interpolateAt() intrinsic.");
   }

   nir_intrinsic_instr *bary_setup =
      nir_intrinsic_instr_create(state->builder.shader, bary_op);

   nir_ssa_dest_init(&bary_setup->instr, &bary_setup->dest, 2, 32, NULL);
   nir_intrinsic_set_interp_mode(bary_setup, var->data.interpolation);

   if (intrin->intrinsic == nir_intrinsic_interp_deref_at_sample ||
       intrin->intrinsic == nir_intrinsic_interp_deref_at_offset ||
       intrin->intrinsic == nir_intrinsic_interp_deref_at_vertex)
      nir_src_copy(&bary_setup->src[0], &intrin->src[1], &bary_setup->instr);

   nir_builder_instr_insert(b, &bary_setup->instr);

   nir_io_semantics semantics = {0};
   semantics.location = var->data.location;
   semantics.num_slots = get_number_of_slots(state, var);
   semantics.medium_precision =
      var->data.precision == GLSL_PRECISION_MEDIUM ||
      var->data.precision == GLSL_PRECISION_LOW;

   assert(intrin->dest.is_ssa);
   nir_ssa_def *load =
      nir_load_interpolated_input(&state->builder,
                                  intrin->dest.ssa.num_components,
                                  intrin->dest.ssa.bit_size,
                                  &bary_setup->dest.ssa,
                                  offset,
                                  .base = var->data.driver_location,
                                  .component = component,
                                  .io_semantics = semantics,
                                  .dest_type = nir_type_float | intrin->dest.ssa.bit_size);

   return load;
}

static bool
nir_lower_io_block(nir_block *block,
                   struct lower_io_state *state)
{
   nir_builder *b = &state->builder;
   const nir_shader_compiler_options *options = b->shader->options;
   bool progress = false;

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      switch (intrin->intrinsic) {
      case nir_intrinsic_load_deref:
      case nir_intrinsic_store_deref:
         /* We can lower the io for this nir instrinsic */
         break;
      case nir_intrinsic_interp_deref_at_centroid:
      case nir_intrinsic_interp_deref_at_sample:
      case nir_intrinsic_interp_deref_at_offset:
      case nir_intrinsic_interp_deref_at_vertex:
         /* We can optionally lower these to load_interpolated_input */
         if (options->use_interpolated_input_intrinsics ||
             options->lower_interpolate_at)
            break;
         FALLTHROUGH;
      default:
         /* We can't lower the io for this nir instrinsic, so skip it */
         continue;
      }

      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
      if (!nir_deref_mode_is_one_of(deref, state->modes))
         continue;

      nir_variable *var = nir_deref_instr_get_variable(deref);

      b->cursor = nir_before_instr(instr);

      const bool is_arrayed = nir_is_arrayed_io(var, b->shader->info.stage);

      nir_ssa_def *offset;
      nir_ssa_def *array_index = NULL;
      unsigned component_offset = var->data.location_frac;
      bool bindless_type_size = var->data.mode == nir_var_shader_in ||
                                var->data.mode == nir_var_shader_out ||
                                var->data.bindless;

     if (nir_deref_instr_is_known_out_of_bounds(deref)) {
        /* Section 5.11 (Out-of-Bounds Accesses) of the GLSL 4.60 spec says:
         *
         *    In the subsections described above for array, vector, matrix and
         *    structure accesses, any out-of-bounds access produced undefined
         *    behavior....
         *    Out-of-bounds reads return undefined values, which
         *    include values from other variables of the active program or zero.
         *    Out-of-bounds writes may be discarded or overwrite
         *    other variables of the active program.
         *
         * GL_KHR_robustness and GL_ARB_robustness encourage us to return zero
         * for reads.
         *
         * Otherwise get_io_offset would return out-of-bound offset which may
         * result in out-of-bound loading/storing of inputs/outputs,
         * that could cause issues in drivers down the line.
         */
         if (intrin->intrinsic != nir_intrinsic_store_deref) {
            nir_ssa_def *zero =
               nir_imm_zero(b, intrin->dest.ssa.num_components,
                             intrin->dest.ssa.bit_size);
            nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                  zero);
         }

         nir_instr_remove(&intrin->instr);
         progress = true;
         continue;
      }

      offset = get_io_offset(b, deref, is_arrayed ? &array_index : NULL,
                             state->type_size, &component_offset,
                             bindless_type_size);

      nir_ssa_def *replacement = NULL;

      switch (intrin->intrinsic) {
      case nir_intrinsic_load_deref:
         replacement = lower_load(intrin, state, array_index, var, offset,
                                  component_offset, deref->type);
         break;

      case nir_intrinsic_store_deref:
         lower_store(intrin, state, array_index, var, offset,
                     component_offset, deref->type);
         break;

      case nir_intrinsic_interp_deref_at_centroid:
      case nir_intrinsic_interp_deref_at_sample:
      case nir_intrinsic_interp_deref_at_offset:
      case nir_intrinsic_interp_deref_at_vertex:
         assert(array_index == NULL);
         replacement = lower_interpolate_at(intrin, state, var, offset,
                                            component_offset, deref->type);
         break;

      default:
         continue;
      }

      if (replacement) {
         nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                  replacement);
      }
      nir_instr_remove(&intrin->instr);
      progress = true;
   }

   return progress;
}

static bool
nir_lower_io_impl(nir_function_impl *impl,
                  nir_variable_mode modes,
                  int (*type_size)(const struct glsl_type *, bool),
                  nir_lower_io_options options)
{
   struct lower_io_state state;
   bool progress = false;

   nir_builder_init(&state.builder, impl);
   state.dead_ctx = ralloc_context(NULL);
   state.modes = modes;
   state.type_size = type_size;
   state.options = options;

   ASSERTED nir_variable_mode supported_modes =
      nir_var_shader_in | nir_var_shader_out | nir_var_uniform;
   assert(!(modes & ~supported_modes));

   nir_foreach_block(block, impl) {
      progress |= nir_lower_io_block(block, &state);
   }

   ralloc_free(state.dead_ctx);

   nir_metadata_preserve(impl, nir_metadata_none);

   return progress;
}

/** Lower load/store_deref intrinsics on I/O variables to offset-based intrinsics
 *
 * This pass is intended to be used for cross-stage shader I/O and driver-
 * managed uniforms to turn deref-based access into a simpler model using
 * locations or offsets.  For fragment shader inputs, it can optionally turn
 * load_deref into an explicit interpolation using barycentrics coming from
 * one of the load_barycentric_* intrinsics.  This pass requires that all
 * deref chains are complete and contain no casts.
 */
bool
nir_lower_io(nir_shader *shader, nir_variable_mode modes,
             int (*type_size)(const struct glsl_type *, bool),
             nir_lower_io_options options)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl) {
         progress |= nir_lower_io_impl(function->impl, modes,
                                       type_size, options);
      }
   }

   return progress;
}

static unsigned
type_scalar_size_bytes(const struct glsl_type *type)
{
   assert(glsl_type_is_vector_or_scalar(type) ||
          glsl_type_is_matrix(type));
   return glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
}

static nir_ssa_def *
build_addr_iadd(nir_builder *b, nir_ssa_def *addr,
                nir_address_format addr_format,
                nir_variable_mode modes,
                nir_ssa_def *offset)
{
   assert(offset->num_components == 1);

   switch (addr_format) {
   case nir_address_format_32bit_global:
   case nir_address_format_64bit_global:
   case nir_address_format_32bit_offset:
      assert(addr->bit_size == offset->bit_size);
      assert(addr->num_components == 1);
      return nir_iadd(b, addr, offset);

   case nir_address_format_2x32bit_global: {
      assert(addr->num_components == 2);
      nir_ssa_def *lo = nir_channel(b, addr, 0);
      nir_ssa_def *hi = nir_channel(b, addr, 1);
      nir_ssa_def *res_lo = nir_iadd(b, lo, offset);
      nir_ssa_def *carry = nir_b2i32(b, nir_ult(b, res_lo, lo));
      nir_ssa_def *res_hi = nir_iadd(b, hi, carry);
      return nir_vec2(b, res_lo, res_hi);
   }

   case nir_address_format_32bit_offset_as_64bit:
      assert(addr->num_components == 1);
      assert(offset->bit_size == 32);
      return nir_u2u64(b, nir_iadd(b, nir_u2u32(b, addr), offset));

   case nir_address_format_64bit_global_32bit_offset:
   case nir_address_format_64bit_bounded_global:
      assert(addr->num_components == 4);
      assert(addr->bit_size == offset->bit_size);
      return nir_vector_insert_imm(b, addr, nir_iadd(b, nir_channel(b, addr, 3), offset), 3);

   case nir_address_format_32bit_index_offset:
      assert(addr->num_components == 2);
      assert(addr->bit_size == offset->bit_size);
      return nir_vector_insert_imm(b, addr, nir_iadd(b, nir_channel(b, addr, 1), offset), 1);

   case nir_address_format_32bit_index_offset_pack64:
      assert(addr->num_components == 1);
      assert(offset->bit_size == 32);
      return nir_pack_64_2x32_split(b,
                                    nir_iadd(b, nir_unpack_64_2x32_split_x(b, addr), offset),
                                    nir_unpack_64_2x32_split_y(b, addr));

   case nir_address_format_vec2_index_32bit_offset:
      assert(addr->num_components == 3);
      assert(offset->bit_size == 32);
      return nir_vector_insert_imm(b, addr, nir_iadd(b, nir_channel(b, addr, 2), offset), 2);

   case nir_address_format_62bit_generic:
      assert(addr->num_components == 1);
      assert(addr->bit_size == 64);
      assert(offset->bit_size == 64);
      if (!(modes & ~(nir_var_function_temp |
                      nir_var_shader_temp |
                      nir_var_mem_shared))) {
         /* If we're sure it's one of these modes, we can do an easy 32-bit
          * addition and don't need to bother with 64-bit math.
          */
         nir_ssa_def *addr32 = nir_unpack_64_2x32_split_x(b, addr);
         nir_ssa_def *type = nir_unpack_64_2x32_split_y(b, addr);
         addr32 = nir_iadd(b, addr32, nir_u2u32(b, offset));
         return nir_pack_64_2x32_split(b, addr32, type);
      } else {
         return nir_iadd(b, addr, offset);
      }

   case nir_address_format_logical:
      unreachable("Unsupported address format");
   }
   unreachable("Invalid address format");
}

static unsigned
addr_get_offset_bit_size(nir_ssa_def *addr, nir_address_format addr_format)
{
   if (addr_format == nir_address_format_32bit_offset_as_64bit ||
       addr_format == nir_address_format_32bit_index_offset_pack64)
      return 32;
   return addr->bit_size;
}

static nir_ssa_def *
build_addr_iadd_imm(nir_builder *b, nir_ssa_def *addr,
                    nir_address_format addr_format,
                    nir_variable_mode modes,
                    int64_t offset)
{
   return build_addr_iadd(b, addr, addr_format, modes,
                             nir_imm_intN_t(b, offset,
                                            addr_get_offset_bit_size(addr, addr_format)));
}

static nir_ssa_def *
build_addr_for_var(nir_builder *b, nir_variable *var,
                   nir_address_format addr_format)
{
   assert(var->data.mode & (nir_var_uniform | nir_var_mem_shared |
                            nir_var_mem_task_payload |
                            nir_var_mem_global |
                            nir_var_shader_temp | nir_var_function_temp |
                            nir_var_mem_push_const | nir_var_mem_constant));

   const unsigned num_comps = nir_address_format_num_components(addr_format);
   const unsigned bit_size = nir_address_format_bit_size(addr_format);

   switch (addr_format) {
   case nir_address_format_2x32bit_global:
   case nir_address_format_32bit_global:
   case nir_address_format_64bit_global: {
      nir_ssa_def *base_addr;
      switch (var->data.mode) {
      case nir_var_shader_temp:
         base_addr = nir_load_scratch_base_ptr(b, num_comps, bit_size, 0);
         break;

      case nir_var_function_temp:
         base_addr = nir_load_scratch_base_ptr(b, num_comps, bit_size, 1);
         break;

      case nir_var_mem_constant:
         base_addr = nir_load_constant_base_ptr(b, num_comps, bit_size);
         break;

      case nir_var_mem_shared:
         base_addr = nir_load_shared_base_ptr(b, num_comps, bit_size);
         break;

      case nir_var_mem_global:
         base_addr = nir_load_global_base_ptr(b, num_comps, bit_size);
         break;

      default:
         unreachable("Unsupported variable mode");
      }

      return build_addr_iadd_imm(b, base_addr, addr_format, var->data.mode,
                                    var->data.driver_location);
   }

   case nir_address_format_32bit_offset:
      assert(var->data.driver_location <= UINT32_MAX);
      return nir_imm_int(b, var->data.driver_location);

   case nir_address_format_32bit_offset_as_64bit:
      assert(var->data.driver_location <= UINT32_MAX);
      return nir_imm_int64(b, var->data.driver_location);

   case nir_address_format_62bit_generic:
      switch (var->data.mode) {
      case nir_var_shader_temp:
      case nir_var_function_temp:
         assert(var->data.driver_location <= UINT32_MAX);
         return nir_imm_intN_t(b, var->data.driver_location | 2ull << 62, 64);

      case nir_var_mem_shared:
         assert(var->data.driver_location <= UINT32_MAX);
         return nir_imm_intN_t(b, var->data.driver_location | 1ull << 62, 64);

      case nir_var_mem_global:
         return nir_iadd_imm(b, nir_load_global_base_ptr(b, num_comps, bit_size),
                                var->data.driver_location);

      default:
         unreachable("Unsupported variable mode");
      }

   default:
      unreachable("Unsupported address format");
   }
}

static nir_ssa_def *
build_runtime_addr_mode_check(nir_builder *b, nir_ssa_def *addr,
                              nir_address_format addr_format,
                              nir_variable_mode mode)
{
   /* The compile-time check failed; do a run-time check */
   switch (addr_format) {
   case nir_address_format_62bit_generic: {
      assert(addr->num_components == 1);
      assert(addr->bit_size == 64);
      nir_ssa_def *mode_enum = nir_ushr(b, addr, nir_imm_int(b, 62));
      switch (mode) {
      case nir_var_function_temp:
      case nir_var_shader_temp:
         return nir_ieq_imm(b, mode_enum, 0x2);

      case nir_var_mem_shared:
         return nir_ieq_imm(b, mode_enum, 0x1);

      case nir_var_mem_global:
         return nir_ior(b, nir_ieq_imm(b, mode_enum, 0x0),
                           nir_ieq_imm(b, mode_enum, 0x3));

      default:
         unreachable("Invalid mode check intrinsic");
      }
   }

   default:
      unreachable("Unsupported address mode");
   }
}

unsigned
nir_address_format_bit_size(nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:              return 32;
   case nir_address_format_2x32bit_global:            return 32;
   case nir_address_format_64bit_global:              return 64;
   case nir_address_format_64bit_global_32bit_offset: return 32;
   case nir_address_format_64bit_bounded_global:      return 32;
   case nir_address_format_32bit_index_offset:        return 32;
   case nir_address_format_32bit_index_offset_pack64: return 64;
   case nir_address_format_vec2_index_32bit_offset:   return 32;
   case nir_address_format_62bit_generic:             return 64;
   case nir_address_format_32bit_offset:              return 32;
   case nir_address_format_32bit_offset_as_64bit:     return 64;
   case nir_address_format_logical:                   return 32;
   }
   unreachable("Invalid address format");
}

unsigned
nir_address_format_num_components(nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:              return 1;
   case nir_address_format_2x32bit_global:            return 2;
   case nir_address_format_64bit_global:              return 1;
   case nir_address_format_64bit_global_32bit_offset: return 4;
   case nir_address_format_64bit_bounded_global:      return 4;
   case nir_address_format_32bit_index_offset:        return 2;
   case nir_address_format_32bit_index_offset_pack64: return 1;
   case nir_address_format_vec2_index_32bit_offset:   return 3;
   case nir_address_format_62bit_generic:             return 1;
   case nir_address_format_32bit_offset:              return 1;
   case nir_address_format_32bit_offset_as_64bit:     return 1;
   case nir_address_format_logical:                   return 1;
   }
   unreachable("Invalid address format");
}

static nir_ssa_def *
addr_to_index(nir_builder *b, nir_ssa_def *addr,
              nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_index_offset:
      assert(addr->num_components == 2);
      return nir_channel(b, addr, 0);
   case nir_address_format_32bit_index_offset_pack64:
      return nir_unpack_64_2x32_split_y(b, addr);
   case nir_address_format_vec2_index_32bit_offset:
      assert(addr->num_components == 3);
      return nir_channels(b, addr, 0x3);
   default: unreachable("Invalid address format");
   }
}

static nir_ssa_def *
addr_to_offset(nir_builder *b, nir_ssa_def *addr,
               nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_index_offset:
      assert(addr->num_components == 2);
      return nir_channel(b, addr, 1);
   case nir_address_format_32bit_index_offset_pack64:
      return nir_unpack_64_2x32_split_x(b, addr);
   case nir_address_format_vec2_index_32bit_offset:
      assert(addr->num_components == 3);
      return nir_channel(b, addr, 2);
   case nir_address_format_32bit_offset:
      return addr;
   case nir_address_format_32bit_offset_as_64bit:
   case nir_address_format_62bit_generic:
      return nir_u2u32(b, addr);
   default:
      unreachable("Invalid address format");
   }
}

/** Returns true if the given address format resolves to a global address */
static bool
addr_format_is_global(nir_address_format addr_format,
                      nir_variable_mode mode)
{
   if (addr_format == nir_address_format_62bit_generic)
      return mode == nir_var_mem_global;

   return addr_format == nir_address_format_32bit_global ||
          addr_format == nir_address_format_2x32bit_global ||
          addr_format == nir_address_format_64bit_global ||
          addr_format == nir_address_format_64bit_global_32bit_offset ||
          addr_format == nir_address_format_64bit_bounded_global;
}

static bool
addr_format_is_offset(nir_address_format addr_format,
                      nir_variable_mode mode)
{
   if (addr_format == nir_address_format_62bit_generic)
      return mode != nir_var_mem_global;

   return addr_format == nir_address_format_32bit_offset ||
          addr_format == nir_address_format_32bit_offset_as_64bit;
}

static nir_ssa_def *
addr_to_global(nir_builder *b, nir_ssa_def *addr,
               nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:
   case nir_address_format_64bit_global:
   case nir_address_format_62bit_generic:
      assert(addr->num_components == 1);
      return addr;

   case nir_address_format_2x32bit_global:
      assert(addr->num_components == 2);
      return addr;

   case nir_address_format_64bit_global_32bit_offset:
   case nir_address_format_64bit_bounded_global:
      assert(addr->num_components == 4);
      return nir_iadd(b, nir_pack_64_2x32(b, nir_channels(b, addr, 0x3)),
                         nir_u2u64(b, nir_channel(b, addr, 3)));

   case nir_address_format_32bit_index_offset:
   case nir_address_format_32bit_index_offset_pack64:
   case nir_address_format_vec2_index_32bit_offset:
   case nir_address_format_32bit_offset:
   case nir_address_format_32bit_offset_as_64bit:
   case nir_address_format_logical:
      unreachable("Cannot get a 64-bit address with this address format");
   }

   unreachable("Invalid address format");
}

static bool
addr_format_needs_bounds_check(nir_address_format addr_format)
{
   return addr_format == nir_address_format_64bit_bounded_global;
}

static nir_ssa_def *
addr_is_in_bounds(nir_builder *b, nir_ssa_def *addr,
                  nir_address_format addr_format, unsigned size)
{
   assert(addr_format == nir_address_format_64bit_bounded_global);
   assert(addr->num_components == 4);
   assert(size > 0);
   return nir_ult(b, nir_iadd_imm(b, nir_channel(b, addr, 3), size - 1),
                     nir_channel(b, addr, 2));
}

static void
nir_get_explicit_deref_range(nir_deref_instr *deref,
                             nir_address_format addr_format,
                             uint32_t *out_base,
                             uint32_t *out_range)
{
   uint32_t base = 0;
   uint32_t range = glsl_get_explicit_size(deref->type, false);

   while (true) {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);

      switch (deref->deref_type) {
      case nir_deref_type_array:
      case nir_deref_type_array_wildcard:
      case nir_deref_type_ptr_as_array: {
         const unsigned stride = nir_deref_instr_array_stride(deref);
         if (stride == 0)
            goto fail;

         if (!parent)
            goto fail;

         if (deref->deref_type != nir_deref_type_array_wildcard &&
             nir_src_is_const(deref->arr.index)) {
            base += stride * nir_src_as_uint(deref->arr.index);
         } else {
            if (glsl_get_length(parent->type) == 0)
               goto fail;
            range += stride * (glsl_get_length(parent->type) - 1);
         }
         break;
      }

      case nir_deref_type_struct: {
         if (!parent)
            goto fail;

         base += glsl_get_struct_field_offset(parent->type, deref->strct.index);
         break;
      }

      case nir_deref_type_cast: {
         nir_instr *parent_instr = deref->parent.ssa->parent_instr;

         switch (parent_instr->type) {
         case nir_instr_type_load_const: {
            nir_load_const_instr *load = nir_instr_as_load_const(parent_instr);

            switch (addr_format) {
            case nir_address_format_32bit_offset:
               base += load->value[1].u32;
               break;
            case nir_address_format_32bit_index_offset:
               base += load->value[1].u32;
               break;
            case nir_address_format_vec2_index_32bit_offset:
               base += load->value[2].u32;
               break;
            default:
               goto fail;
            }

            *out_base = base;
            *out_range = range;
            return;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent_instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_vulkan_descriptor:
               /* Assume that a load_vulkan_descriptor won't contribute to an
                * offset within the resource.
                */
               break;
            default:
               goto fail;
            }

            *out_base = base;
            *out_range = range;
            return;
         }

         default:
            goto fail;
         }
      }

      default:
         goto fail;
      }

      deref = parent;
   }

fail:
   *out_base = 0;
   *out_range = ~0;
}

static nir_variable_mode
canonicalize_generic_modes(nir_variable_mode modes)
{
   assert(modes != 0);
   if (util_bitcount(modes) == 1)
      return modes;

   assert(!(modes & ~(nir_var_function_temp | nir_var_shader_temp |
                      nir_var_mem_shared | nir_var_mem_global)));

   /* Canonicalize by converting shader_temp to function_temp */
   if (modes & nir_var_shader_temp) {
      modes &= ~nir_var_shader_temp;
      modes |= nir_var_function_temp;
   }

   return modes;
}

static nir_intrinsic_op
get_store_global_op_from_addr_format(nir_address_format addr_format)
{
   if (addr_format != nir_address_format_2x32bit_global)
      return nir_intrinsic_store_global;
   else
      return nir_intrinsic_store_global_2x32;
}

static nir_intrinsic_op
get_load_global_op_from_addr_format(nir_address_format addr_format)
{
   if (addr_format != nir_address_format_2x32bit_global)
      return nir_intrinsic_load_global;
   else
      return nir_intrinsic_load_global_2x32;
}

static nir_ssa_def *
build_explicit_io_load(nir_builder *b, nir_intrinsic_instr *intrin,
                       nir_ssa_def *addr, nir_address_format addr_format,
                       nir_variable_mode modes,
                       uint32_t align_mul, uint32_t align_offset,
                       unsigned num_components)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   modes = canonicalize_generic_modes(modes);

   if (util_bitcount(modes) > 1) {
      if (addr_format_is_global(addr_format, modes)) {
         return build_explicit_io_load(b, intrin, addr, addr_format,
                                       nir_var_mem_global,
                                       align_mul, align_offset,
                                       num_components);
      } else if (modes & nir_var_function_temp) {
         nir_push_if(b, build_runtime_addr_mode_check(b, addr, addr_format,
                                                      nir_var_function_temp));
         nir_ssa_def *res1 =
            build_explicit_io_load(b, intrin, addr, addr_format,
                                   nir_var_function_temp,
                                   align_mul, align_offset,
                                   num_components);
         nir_push_else(b, NULL);
         nir_ssa_def *res2 =
            build_explicit_io_load(b, intrin, addr, addr_format,
                                   modes & ~nir_var_function_temp,
                                   align_mul, align_offset,
                                   num_components);
         nir_pop_if(b, NULL);
         return nir_if_phi(b, res1, res2);
      } else {
         nir_push_if(b, build_runtime_addr_mode_check(b, addr, addr_format,
                                                      nir_var_mem_shared));
         assert(modes & nir_var_mem_shared);
         nir_ssa_def *res1 =
            build_explicit_io_load(b, intrin, addr, addr_format,
                                   nir_var_mem_shared,
                                   align_mul, align_offset,
                                   num_components);
         nir_push_else(b, NULL);
         assert(modes & nir_var_mem_global);
         nir_ssa_def *res2 =
            build_explicit_io_load(b, intrin, addr, addr_format,
                                   nir_var_mem_global,
                                   align_mul, align_offset,
                                   num_components);
         nir_pop_if(b, NULL);
         return nir_if_phi(b, res1, res2);
      }
   }

   assert(util_bitcount(modes) == 1);
   const nir_variable_mode mode = modes;

   nir_intrinsic_op op;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_deref:
      switch (mode) {
      case nir_var_mem_ubo:
         if (addr_format == nir_address_format_64bit_global_32bit_offset)
            op = nir_intrinsic_load_global_constant_offset;
         else if (addr_format == nir_address_format_64bit_bounded_global)
            op = nir_intrinsic_load_global_constant_bounded;
         else if (addr_format_is_global(addr_format, mode))
            op = nir_intrinsic_load_global_constant;
         else
            op = nir_intrinsic_load_ubo;
         break;
      case nir_var_mem_ssbo:
         if (addr_format_is_global(addr_format, mode))
            op = nir_intrinsic_load_global;
         else
            op = nir_intrinsic_load_ssbo;
         break;
      case nir_var_mem_global:
         assert(addr_format_is_global(addr_format, mode));
         op = get_load_global_op_from_addr_format(addr_format);
         break;
      case nir_var_uniform:
         assert(addr_format_is_offset(addr_format, mode));
         assert(b->shader->info.stage == MESA_SHADER_KERNEL);
         op = nir_intrinsic_load_kernel_input;
         break;
      case nir_var_mem_shared:
         assert(addr_format_is_offset(addr_format, mode));
         op = nir_intrinsic_load_shared;
         break;
      case nir_var_mem_task_payload:
         assert(addr_format_is_offset(addr_format, mode));
         op = nir_intrinsic_load_task_payload;
         break;
      case nir_var_shader_temp:
      case nir_var_function_temp:
         if (addr_format_is_offset(addr_format, mode)) {
            op = nir_intrinsic_load_scratch;
         } else {
            assert(addr_format_is_global(addr_format, mode));
            op = get_load_global_op_from_addr_format(addr_format);
         }
         break;
      case nir_var_mem_push_const:
         assert(addr_format == nir_address_format_32bit_offset);
         op = nir_intrinsic_load_push_constant;
         break;
      case nir_var_mem_constant:
         if (addr_format_is_offset(addr_format, mode)) {
            op = nir_intrinsic_load_constant;
         } else {
            assert(addr_format_is_global(addr_format, mode));
            op = get_load_global_op_from_addr_format(addr_format);
         }
         break;
      default:
         unreachable("Unsupported explicit IO variable mode");
      }
      break;

   case nir_intrinsic_load_deref_block_intel:
      switch (mode) {
      case nir_var_mem_ssbo:
         if (addr_format_is_global(addr_format, mode))
            op = nir_intrinsic_load_global_block_intel;
         else
            op = nir_intrinsic_load_ssbo_block_intel;
         break;
      case nir_var_mem_global:
         op = nir_intrinsic_load_global_block_intel;
         break;
      case nir_var_mem_shared:
         op = nir_intrinsic_load_shared_block_intel;
         break;
      default:
         unreachable("Unsupported explicit IO variable mode");
      }
      break;

   default:
      unreachable("Invalid intrinsic");
   }

   nir_intrinsic_instr *load = nir_intrinsic_instr_create(b->shader, op);

   if (op == nir_intrinsic_load_global_constant_offset) {
      assert(addr_format == nir_address_format_64bit_global_32bit_offset);
      load->src[0] = nir_src_for_ssa(
         nir_pack_64_2x32(b, nir_channels(b, addr, 0x3)));
      load->src[1] = nir_src_for_ssa(nir_channel(b, addr, 3));
   } else if (op == nir_intrinsic_load_global_constant_bounded) {
      assert(addr_format == nir_address_format_64bit_bounded_global);
      load->src[0] = nir_src_for_ssa(
         nir_pack_64_2x32(b, nir_channels(b, addr, 0x3)));
      load->src[1] = nir_src_for_ssa(nir_channel(b, addr, 3));
      load->src[2] = nir_src_for_ssa(nir_channel(b, addr, 2));
   } else if (addr_format_is_global(addr_format, mode)) {
      load->src[0] = nir_src_for_ssa(addr_to_global(b, addr, addr_format));
   } else if (addr_format_is_offset(addr_format, mode)) {
      assert(addr->num_components == 1);
      load->src[0] = nir_src_for_ssa(addr_to_offset(b, addr, addr_format));
   } else {
      load->src[0] = nir_src_for_ssa(addr_to_index(b, addr, addr_format));
      load->src[1] = nir_src_for_ssa(addr_to_offset(b, addr, addr_format));
   }

   if (nir_intrinsic_has_access(load))
      nir_intrinsic_set_access(load, nir_intrinsic_access(intrin));

   if (op == nir_intrinsic_load_constant) {
      nir_intrinsic_set_base(load, 0);
      nir_intrinsic_set_range(load, b->shader->constant_data_size);
   } else if (op == nir_intrinsic_load_kernel_input) {
      nir_intrinsic_set_base(load, 0);
      nir_intrinsic_set_range(load, b->shader->num_uniforms);
   } else if (mode == nir_var_mem_push_const) {
      /* Push constants are required to be able to be chased back to the
       * variable so we can provide a base/range.
       */
      nir_variable *var = nir_deref_instr_get_variable(deref);
      nir_intrinsic_set_base(load, 0);
      nir_intrinsic_set_range(load, glsl_get_explicit_size(var->type, false));
   }

   unsigned bit_size = intrin->dest.ssa.bit_size;
   if (bit_size == 1) {
      /* TODO: Make the native bool bit_size an option. */
      bit_size = 32;
   }

   if (nir_intrinsic_has_align(load))
      nir_intrinsic_set_align(load, align_mul, align_offset);

   if (nir_intrinsic_has_range_base(load)) {
      unsigned base, range;
      nir_get_explicit_deref_range(deref, addr_format, &base, &range);
      nir_intrinsic_set_range_base(load, base);
      nir_intrinsic_set_range(load, range);
   }

   assert(intrin->dest.is_ssa);
   load->num_components = num_components;
   nir_ssa_dest_init(&load->instr, &load->dest, num_components,
                     bit_size, NULL);

   assert(bit_size % 8 == 0);

   nir_ssa_def *result;
   if (addr_format_needs_bounds_check(addr_format) &&
       op != nir_intrinsic_load_global_constant_bounded) {
      /* We don't need to bounds-check global_constant_bounded because bounds
       * checking is handled by the intrinsic itself.
       *
       * The Vulkan spec for robustBufferAccess gives us quite a few options
       * as to what we can do with an OOB read.  Unfortunately, returning
       * undefined values isn't one of them so we return an actual zero.
       */
      nir_ssa_def *zero = nir_imm_zero(b, load->num_components, bit_size);

      /* TODO: Better handle block_intel. */
      const unsigned load_size = (bit_size / 8) * load->num_components;
      nir_push_if(b, addr_is_in_bounds(b, addr, addr_format, load_size));

      nir_builder_instr_insert(b, &load->instr);

      nir_pop_if(b, NULL);

      result = nir_if_phi(b, &load->dest.ssa, zero);
   } else {
      nir_builder_instr_insert(b, &load->instr);
      result = &load->dest.ssa;
   }

   if (intrin->dest.ssa.bit_size == 1) {
      /* For shared, we can go ahead and use NIR's and/or the back-end's
       * standard encoding for booleans rather than forcing a 0/1 boolean.
       * This should save an instruction or two.
       */
      if (mode == nir_var_mem_shared ||
          mode == nir_var_shader_temp ||
          mode == nir_var_function_temp)
         result = nir_b2b1(b, result);
      else
         result = nir_i2b(b, result);
   }

   return result;
}

static void
build_explicit_io_store(nir_builder *b, nir_intrinsic_instr *intrin,
                        nir_ssa_def *addr, nir_address_format addr_format,
                        nir_variable_mode modes,
                        uint32_t align_mul, uint32_t align_offset,
                        nir_ssa_def *value, nir_component_mask_t write_mask)
{
   modes = canonicalize_generic_modes(modes);

   if (util_bitcount(modes) > 1) {
      if (addr_format_is_global(addr_format, modes)) {
         build_explicit_io_store(b, intrin, addr, addr_format,
                                 nir_var_mem_global,
                                 align_mul, align_offset,
                                 value, write_mask);
      } else if (modes & nir_var_function_temp) {
         nir_push_if(b, build_runtime_addr_mode_check(b, addr, addr_format,
                                                      nir_var_function_temp));
         build_explicit_io_store(b, intrin, addr, addr_format,
                                 nir_var_function_temp,
                                 align_mul, align_offset,
                                 value, write_mask);
         nir_push_else(b, NULL);
         build_explicit_io_store(b, intrin, addr, addr_format,
                                 modes & ~nir_var_function_temp,
                                 align_mul, align_offset,
                                 value, write_mask);
         nir_pop_if(b, NULL);
      } else {
         nir_push_if(b, build_runtime_addr_mode_check(b, addr, addr_format,
                                                      nir_var_mem_shared));
         assert(modes & nir_var_mem_shared);
         build_explicit_io_store(b, intrin, addr, addr_format,
                                 nir_var_mem_shared,
                                 align_mul, align_offset,
                                 value, write_mask);
         nir_push_else(b, NULL);
         assert(modes & nir_var_mem_global);
         build_explicit_io_store(b, intrin, addr, addr_format,
                                 nir_var_mem_global,
                                 align_mul, align_offset,
                                 value, write_mask);
         nir_pop_if(b, NULL);
      }
      return;
   }

   assert(util_bitcount(modes) == 1);
   const nir_variable_mode mode = modes;

   nir_intrinsic_op op;
   switch (intrin->intrinsic) {
   case nir_intrinsic_store_deref:
      assert(write_mask != 0);

      switch (mode) {
      case nir_var_mem_ssbo:
         if (addr_format_is_global(addr_format, mode))
            op = get_store_global_op_from_addr_format(addr_format);
         else
            op = nir_intrinsic_store_ssbo;
         break;
      case nir_var_mem_global:
         assert(addr_format_is_global(addr_format, mode));
         op = get_store_global_op_from_addr_format(addr_format);
         break;
      case nir_var_mem_shared:
         assert(addr_format_is_offset(addr_format, mode));
         op = nir_intrinsic_store_shared;
         break;
      case nir_var_mem_task_payload:
         assert(addr_format_is_offset(addr_format, mode));
         op = nir_intrinsic_store_task_payload;
         break;
      case nir_var_shader_temp:
      case nir_var_function_temp:
         if (addr_format_is_offset(addr_format, mode)) {
            op = nir_intrinsic_store_scratch;
         } else {
            assert(addr_format_is_global(addr_format, mode));
            op = get_store_global_op_from_addr_format(addr_format);
         }
         break;
      default:
         unreachable("Unsupported explicit IO variable mode");
      }
      break;

   case nir_intrinsic_store_deref_block_intel:
      assert(write_mask == 0);

      switch (mode) {
      case nir_var_mem_ssbo:
         if (addr_format_is_global(addr_format, mode))
            op = nir_intrinsic_store_global_block_intel;
         else
            op = nir_intrinsic_store_ssbo_block_intel;
         break;
      case nir_var_mem_global:
         op = nir_intrinsic_store_global_block_intel;
         break;
      case nir_var_mem_shared:
         op = nir_intrinsic_store_shared_block_intel;
         break;
      default:
         unreachable("Unsupported explicit IO variable mode");
      }
      break;

   default:
      unreachable("Invalid intrinsic");
   }

   nir_intrinsic_instr *store = nir_intrinsic_instr_create(b->shader, op);

   if (value->bit_size == 1) {
      /* For shared, we can go ahead and use NIR's and/or the back-end's
       * standard encoding for booleans rather than forcing a 0/1 boolean.
       * This should save an instruction or two.
       *
       * TODO: Make the native bool bit_size an option.
       */
      if (mode == nir_var_mem_shared ||
          mode == nir_var_shader_temp ||
          mode == nir_var_function_temp)
         value = nir_b2b32(b, value);
      else
         value = nir_b2iN(b, value, 32);
   }

   store->src[0] = nir_src_for_ssa(value);
   if (addr_format_is_global(addr_format, mode)) {
      store->src[1] = nir_src_for_ssa(addr_to_global(b, addr, addr_format));
   } else if (addr_format_is_offset(addr_format, mode)) {
      assert(addr->num_components == 1);
      store->src[1] = nir_src_for_ssa(addr_to_offset(b, addr, addr_format));
   } else {
      store->src[1] = nir_src_for_ssa(addr_to_index(b, addr, addr_format));
      store->src[2] = nir_src_for_ssa(addr_to_offset(b, addr, addr_format));
   }

   nir_intrinsic_set_write_mask(store, write_mask);

   if (nir_intrinsic_has_access(store))
      nir_intrinsic_set_access(store, nir_intrinsic_access(intrin));

   nir_intrinsic_set_align(store, align_mul, align_offset);

   assert(value->num_components == 1 ||
          value->num_components == intrin->num_components);
   store->num_components = value->num_components;

   assert(value->bit_size % 8 == 0);

   if (addr_format_needs_bounds_check(addr_format)) {
      /* TODO: Better handle block_intel. */
      const unsigned store_size = (value->bit_size / 8) * store->num_components;
      nir_push_if(b, addr_is_in_bounds(b, addr, addr_format, store_size));

      nir_builder_instr_insert(b, &store->instr);

      nir_pop_if(b, NULL);
   } else {
      nir_builder_instr_insert(b, &store->instr);
   }
}

static nir_ssa_def *
build_explicit_io_atomic(nir_builder *b, nir_intrinsic_instr *intrin,
                         nir_ssa_def *addr, nir_address_format addr_format,
                         nir_variable_mode modes)
{
   modes = canonicalize_generic_modes(modes);

   if (util_bitcount(modes) > 1) {
      if (addr_format_is_global(addr_format, modes)) {
         return build_explicit_io_atomic(b, intrin, addr, addr_format,
                                         nir_var_mem_global);
      } else if (modes & nir_var_function_temp) {
         nir_push_if(b, build_runtime_addr_mode_check(b, addr, addr_format,
                                                      nir_var_function_temp));
         nir_ssa_def *res1 =
            build_explicit_io_atomic(b, intrin, addr, addr_format,
                                     nir_var_function_temp);
         nir_push_else(b, NULL);
         nir_ssa_def *res2 =
            build_explicit_io_atomic(b, intrin, addr, addr_format,
                                     modes & ~nir_var_function_temp);
         nir_pop_if(b, NULL);
         return nir_if_phi(b, res1, res2);
      } else {
         nir_push_if(b, build_runtime_addr_mode_check(b, addr, addr_format,
                                                      nir_var_mem_shared));
         assert(modes & nir_var_mem_shared);
         nir_ssa_def *res1 =
            build_explicit_io_atomic(b, intrin, addr, addr_format,
                                     nir_var_mem_shared);
         nir_push_else(b, NULL);
         assert(modes & nir_var_mem_global);
         nir_ssa_def *res2 =
            build_explicit_io_atomic(b, intrin, addr, addr_format,
                                     nir_var_mem_global);
         nir_pop_if(b, NULL);
         return nir_if_phi(b, res1, res2);
      }
   }

   assert(util_bitcount(modes) == 1);
   const nir_variable_mode mode = modes;

   const unsigned num_data_srcs =
      nir_intrinsic_infos[intrin->intrinsic].num_srcs - 1;

   nir_intrinsic_op op;
   switch (mode) {
   case nir_var_mem_ssbo:
      if (addr_format_is_global(addr_format, mode))
         op = global_atomic_for_deref(addr_format, intrin->intrinsic);
      else
         op = ssbo_atomic_for_deref(intrin->intrinsic);
      break;
   case nir_var_mem_global:
      assert(addr_format_is_global(addr_format, mode));
      op = global_atomic_for_deref(addr_format, intrin->intrinsic);
      break;
   case nir_var_mem_shared:
      assert(addr_format_is_offset(addr_format, mode));
      op = shared_atomic_for_deref(intrin->intrinsic);
      break;
   case nir_var_mem_task_payload:
      assert(addr_format_is_offset(addr_format, mode));
      op = task_payload_atomic_for_deref(intrin->intrinsic);
      break;
   default:
      unreachable("Unsupported explicit IO variable mode");
   }

   nir_intrinsic_instr *atomic = nir_intrinsic_instr_create(b->shader, op);

   unsigned src = 0;
   if (addr_format_is_global(addr_format, mode)) {
      atomic->src[src++] = nir_src_for_ssa(addr_to_global(b, addr, addr_format));
   } else if (addr_format_is_offset(addr_format, mode)) {
      assert(addr->num_components == 1);
      atomic->src[src++] = nir_src_for_ssa(addr_to_offset(b, addr, addr_format));
   } else {
      atomic->src[src++] = nir_src_for_ssa(addr_to_index(b, addr, addr_format));
      atomic->src[src++] = nir_src_for_ssa(addr_to_offset(b, addr, addr_format));
   }
   for (unsigned i = 0; i < num_data_srcs; i++) {
      atomic->src[src++] = nir_src_for_ssa(intrin->src[1 + i].ssa);
   }

   /* Global atomics don't have access flags because they assume that the
    * address may be non-uniform.
    */
   if (nir_intrinsic_has_access(atomic))
      nir_intrinsic_set_access(atomic, nir_intrinsic_access(intrin));

   assert(intrin->dest.ssa.num_components == 1);
   nir_ssa_dest_init(&atomic->instr, &atomic->dest,
                     1, intrin->dest.ssa.bit_size, NULL);

   assert(atomic->dest.ssa.bit_size % 8 == 0);

   if (addr_format_needs_bounds_check(addr_format)) {
      const unsigned atomic_size = atomic->dest.ssa.bit_size / 8;
      nir_push_if(b, addr_is_in_bounds(b, addr, addr_format, atomic_size));

      nir_builder_instr_insert(b, &atomic->instr);

      nir_pop_if(b, NULL);
      return nir_if_phi(b, &atomic->dest.ssa,
                           nir_ssa_undef(b, 1, atomic->dest.ssa.bit_size));
   } else {
      nir_builder_instr_insert(b, &atomic->instr);
      return &atomic->dest.ssa;
   }
}

nir_ssa_def *
nir_explicit_io_address_from_deref(nir_builder *b, nir_deref_instr *deref,
                                   nir_ssa_def *base_addr,
                                   nir_address_format addr_format)
{
   assert(deref->dest.is_ssa);
   switch (deref->deref_type) {
   case nir_deref_type_var:
      return build_addr_for_var(b, deref->var, addr_format);

   case nir_deref_type_ptr_as_array:
   case nir_deref_type_array: {
      unsigned stride = nir_deref_instr_array_stride(deref);
      assert(stride > 0);

      unsigned offset_bit_size = addr_get_offset_bit_size(base_addr, addr_format);
      nir_ssa_def *index = nir_ssa_for_src(b, deref->arr.index, 1);
      nir_ssa_def *offset;

      /* If the access chain has been declared in-bounds, then we know it doesn't
       * overflow the type.  For nir_deref_type_array, this implies it cannot be
       * negative. Also, since types in NIR have a maximum 32-bit size, we know the
       * final result will fit in a 32-bit value so we can convert the index to
       * 32-bit before multiplying and save ourselves from a 64-bit multiply.
       */
      if (deref->arr.in_bounds && deref->deref_type == nir_deref_type_array) {
         index = nir_u2u32(b, index);
         offset = nir_u2uN(b, nir_amul_imm(b, index, stride), offset_bit_size);
      } else {
         index = nir_i2iN(b, index, offset_bit_size);
         offset = nir_amul_imm(b, index, stride);
      }

      return build_addr_iadd(b, base_addr, addr_format, deref->modes, offset);
   }

   case nir_deref_type_array_wildcard:
      unreachable("Wildcards should be lowered by now");
      break;

   case nir_deref_type_struct: {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      int offset = glsl_get_struct_field_offset(parent->type,
                                                deref->strct.index);
      assert(offset >= 0);
      return build_addr_iadd_imm(b, base_addr, addr_format,
                                 deref->modes, offset);
   }

   case nir_deref_type_cast:
      /* Nothing to do here */
      return base_addr;
   }

   unreachable("Invalid NIR deref type");
}

void
nir_lower_explicit_io_instr(nir_builder *b,
                            nir_intrinsic_instr *intrin,
                            nir_ssa_def *addr,
                            nir_address_format addr_format)
{
   b->cursor = nir_after_instr(&intrin->instr);

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   unsigned vec_stride = glsl_get_explicit_stride(deref->type);
   unsigned scalar_size = type_scalar_size_bytes(deref->type);
   assert(vec_stride == 0 || glsl_type_is_vector(deref->type));
   assert(vec_stride == 0 || vec_stride >= scalar_size);

   uint32_t align_mul, align_offset;
   if (!nir_get_explicit_deref_align(deref, true, &align_mul, &align_offset)) {
      /* If we don't have an alignment from the deref, assume scalar */
      align_mul = scalar_size;
      align_offset = 0;
   }

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_deref: {
      nir_ssa_def *value;
      if (vec_stride > scalar_size) {
         nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS] = { NULL, };
         for (unsigned i = 0; i < intrin->num_components; i++) {
            unsigned comp_offset = i * vec_stride;
            nir_ssa_def *comp_addr = build_addr_iadd_imm(b, addr, addr_format,
                                                         deref->modes,
                                                         comp_offset);
            comps[i] = build_explicit_io_load(b, intrin, comp_addr,
                                              addr_format, deref->modes,
                                              align_mul,
                                              (align_offset + comp_offset) %
                                                 align_mul,
                                              1);
         }
         value = nir_vec(b, comps, intrin->num_components);
      } else {
         value = build_explicit_io_load(b, intrin, addr, addr_format,
                                        deref->modes, align_mul, align_offset,
                                        intrin->num_components);
      }
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, value);
      break;
   }

   case nir_intrinsic_store_deref: {
      assert(intrin->src[1].is_ssa);
      nir_ssa_def *value = intrin->src[1].ssa;
      nir_component_mask_t write_mask = nir_intrinsic_write_mask(intrin);
      if (vec_stride > scalar_size) {
         for (unsigned i = 0; i < intrin->num_components; i++) {
            if (!(write_mask & (1 << i)))
               continue;

            unsigned comp_offset = i * vec_stride;
            nir_ssa_def *comp_addr = build_addr_iadd_imm(b, addr, addr_format,
                                                         deref->modes,
                                                         comp_offset);
            build_explicit_io_store(b, intrin, comp_addr, addr_format,
                                    deref->modes, align_mul,
                                    (align_offset + comp_offset) % align_mul,
                                    nir_channel(b, value, i), 1);
         }
      } else {
         build_explicit_io_store(b, intrin, addr, addr_format,
                                 deref->modes, align_mul, align_offset,
                                 value, write_mask);
      }
      break;
   }

   case nir_intrinsic_load_deref_block_intel: {
      nir_ssa_def *value = build_explicit_io_load(b, intrin, addr, addr_format,
                                                  deref->modes,
                                                  align_mul, align_offset,
                                                  intrin->num_components);
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, value);
      break;
   }

   case nir_intrinsic_store_deref_block_intel: {
      assert(intrin->src[1].is_ssa);
      nir_ssa_def *value = intrin->src[1].ssa;
      const nir_component_mask_t write_mask = 0;
      build_explicit_io_store(b, intrin, addr, addr_format,
                              deref->modes, align_mul, align_offset,
                              value, write_mask);
      break;
   }

   default: {
      nir_ssa_def *value =
         build_explicit_io_atomic(b, intrin, addr, addr_format, deref->modes);
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, value);
      break;
   }
   }

   nir_instr_remove(&intrin->instr);
}

bool
nir_get_explicit_deref_align(nir_deref_instr *deref,
                             bool default_to_type_align,
                             uint32_t *align_mul,
                             uint32_t *align_offset)
{
   if (deref->deref_type == nir_deref_type_var) {
      /* If we see a variable, align_mul is effectively infinite because we
       * know the offset exactly (up to the offset of the base pointer for the
       * given variable mode).   We have to pick something so we choose 256B
       * as an arbitrary alignment which seems high enough for any reasonable
       * wide-load use-case.  Back-ends should clamp alignments down if 256B
       * is too large for some reason.
       */
      *align_mul = 256;
      *align_offset = deref->var->data.driver_location % 256;
      return true;
   }

   /* If we're a cast deref that has an alignment, use that. */
   if (deref->deref_type == nir_deref_type_cast && deref->cast.align_mul > 0) {
      *align_mul = deref->cast.align_mul;
      *align_offset = deref->cast.align_offset;
      return true;
   }

   /* Otherwise, we need to compute the alignment based on the parent */
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (parent == NULL) {
      assert(deref->deref_type == nir_deref_type_cast);
      if (default_to_type_align) {
         /* If we don't have a parent, assume the type's alignment, if any. */
         unsigned type_align = glsl_get_explicit_alignment(deref->type);
         if (type_align == 0)
            return false;

         *align_mul = type_align;
         *align_offset = 0;
         return true;
      } else {
         return false;
      }
   }

   uint32_t parent_mul, parent_offset;
   if (!nir_get_explicit_deref_align(parent, default_to_type_align,
                                     &parent_mul, &parent_offset))
      return false;

   switch (deref->deref_type) {
   case nir_deref_type_var:
      unreachable("Handled above");

   case nir_deref_type_array:
   case nir_deref_type_array_wildcard:
   case nir_deref_type_ptr_as_array: {
      const unsigned stride = nir_deref_instr_array_stride(deref);
      if (stride == 0)
         return false;

      if (deref->deref_type != nir_deref_type_array_wildcard &&
          nir_src_is_const(deref->arr.index)) {
         unsigned offset = nir_src_as_uint(deref->arr.index) * stride;
         *align_mul = parent_mul;
         *align_offset = (parent_offset + offset) % parent_mul;
      } else {
         /* If this is a wildcard or an indirect deref, we have to go with the
          * power-of-two gcd.
          */
         *align_mul = MIN2(parent_mul, 1 << (ffs(stride) - 1));
         *align_offset = parent_offset % *align_mul;
      }
      return true;
   }

   case nir_deref_type_struct: {
      const int offset = glsl_get_struct_field_offset(parent->type,
                                                      deref->strct.index);
      if (offset < 0)
         return false;

      *align_mul = parent_mul;
      *align_offset = (parent_offset + offset) % parent_mul;
      return true;
   }

   case nir_deref_type_cast:
      /* We handled the explicit alignment case above. */
      assert(deref->cast.align_mul == 0);
      *align_mul = parent_mul;
      *align_offset = parent_offset;
      return true;
   }

   unreachable("Invalid deref_instr_type");
}

static void
lower_explicit_io_deref(nir_builder *b, nir_deref_instr *deref,
                        nir_address_format addr_format)
{
   /* Just delete the deref if it's not used.  We can't use
    * nir_deref_instr_remove_if_unused here because it may remove more than
    * one deref which could break our list walking since we walk the list
    * backwards.
    */
   assert(list_is_empty(&deref->dest.ssa.if_uses));
   if (list_is_empty(&deref->dest.ssa.uses)) {
      nir_instr_remove(&deref->instr);
      return;
   }

   b->cursor = nir_after_instr(&deref->instr);

   nir_ssa_def *base_addr = NULL;
   if (deref->deref_type != nir_deref_type_var) {
      assert(deref->parent.is_ssa);
      base_addr = deref->parent.ssa;
   }

   nir_ssa_def *addr = nir_explicit_io_address_from_deref(b, deref, base_addr,
                                                          addr_format);
   assert(addr->bit_size == deref->dest.ssa.bit_size);
   assert(addr->num_components == deref->dest.ssa.num_components);

   nir_instr_remove(&deref->instr);
   nir_ssa_def_rewrite_uses(&deref->dest.ssa, addr);
}

static void
lower_explicit_io_access(nir_builder *b, nir_intrinsic_instr *intrin,
                         nir_address_format addr_format)
{
   assert(intrin->src[0].is_ssa);
   nir_lower_explicit_io_instr(b, intrin, intrin->src[0].ssa, addr_format);
}

static void
lower_explicit_io_array_length(nir_builder *b, nir_intrinsic_instr *intrin,
                               nir_address_format addr_format)
{
   b->cursor = nir_after_instr(&intrin->instr);

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);

   assert(glsl_type_is_array(deref->type));
   assert(glsl_get_length(deref->type) == 0);
   assert(nir_deref_mode_is(deref, nir_var_mem_ssbo));
   unsigned stride = glsl_get_explicit_stride(deref->type);
   assert(stride > 0);

   nir_ssa_def *addr = &deref->dest.ssa;
   nir_ssa_def *index = addr_to_index(b, addr, addr_format);
   nir_ssa_def *offset = addr_to_offset(b, addr, addr_format);
   unsigned access = nir_intrinsic_access(intrin);

   nir_ssa_def *arr_size = nir_get_ssbo_size(b, index, .access=access);
   arr_size = nir_usub_sat(b, arr_size, offset);
   arr_size = nir_udiv_imm(b, arr_size, stride);

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, arr_size);
   nir_instr_remove(&intrin->instr);
}

static void
lower_explicit_io_mode_check(nir_builder *b, nir_intrinsic_instr *intrin,
                             nir_address_format addr_format)
{
   if (addr_format_is_global(addr_format, 0)) {
      /* If the address format is always global, then the driver can use
       * global addresses regardless of the mode.  In that case, don't create
       * a check, just whack the intrinsic to addr_mode_is and delegate to the
       * driver lowering.
       */
      intrin->intrinsic = nir_intrinsic_addr_mode_is;
      return;
   }

   assert(intrin->src[0].is_ssa);
   nir_ssa_def *addr = intrin->src[0].ssa;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_ssa_def *is_mode =
      build_runtime_addr_mode_check(b, addr, addr_format,
                                    nir_intrinsic_memory_modes(intrin));

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, is_mode);
}

static bool
nir_lower_explicit_io_impl(nir_function_impl *impl, nir_variable_mode modes,
                           nir_address_format addr_format)
{
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   /* Walk in reverse order so that we can see the full deref chain when we
    * lower the access operations.  We lower them assuming that the derefs
    * will be turned into address calculations later.
    */
   nir_foreach_block_reverse(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (nir_deref_mode_is_in_set(deref, modes)) {
               lower_explicit_io_deref(&b, deref, addr_format);
               progress = true;
            }
            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
            case nir_intrinsic_load_deref_block_intel:
            case nir_intrinsic_store_deref_block_intel:
            case nir_intrinsic_deref_atomic_add:
            case nir_intrinsic_deref_atomic_imin:
            case nir_intrinsic_deref_atomic_umin:
            case nir_intrinsic_deref_atomic_imax:
            case nir_intrinsic_deref_atomic_umax:
            case nir_intrinsic_deref_atomic_and:
            case nir_intrinsic_deref_atomic_or:
            case nir_intrinsic_deref_atomic_xor:
            case nir_intrinsic_deref_atomic_exchange:
            case nir_intrinsic_deref_atomic_comp_swap:
            case nir_intrinsic_deref_atomic_fadd:
            case nir_intrinsic_deref_atomic_fmin:
            case nir_intrinsic_deref_atomic_fmax:
            case nir_intrinsic_deref_atomic_fcomp_swap: {
               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               if (nir_deref_mode_is_in_set(deref, modes)) {
                  lower_explicit_io_access(&b, intrin, addr_format);
                  progress = true;
               }
               break;
            }

            case nir_intrinsic_deref_buffer_array_length: {
               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               if (nir_deref_mode_is_in_set(deref, modes)) {
                  lower_explicit_io_array_length(&b, intrin, addr_format);
                  progress = true;
               }
               break;
            }

            case nir_intrinsic_deref_mode_is: {
               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               if (nir_deref_mode_is_in_set(deref, modes)) {
                  lower_explicit_io_mode_check(&b, intrin, addr_format);
                  progress = true;
               }
               break;
            }

            case nir_intrinsic_launch_mesh_workgroups_with_payload_deref: {
               if (modes & nir_var_mem_task_payload) {
                  /* Get address and size of the payload variable. */
                  nir_deref_instr *deref = nir_src_as_deref(intrin->src[1]);
                  assert(deref->deref_type == nir_deref_type_var);
                  unsigned base = deref->var->data.explicit_location;
                  unsigned size = glsl_get_explicit_size(deref->var->type, false);

                  /* Replace the current instruction with the explicit intrinsic. */
                  nir_ssa_def *dispatch_3d = intrin->src[0].ssa;
                  b.cursor = nir_instr_remove(instr);
                  nir_launch_mesh_workgroups(&b, dispatch_3d, .base = base, .range = size);
                  progress = true;
               }

               break;
            }

            default:
               break;
            }
            break;
         }

         default:
            /* Nothing to do */
            break;
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_none);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

/** Lower explicitly laid out I/O access to byte offset/address intrinsics
 *
 * This pass is intended to be used for any I/O which touches memory external
 * to the shader or which is directly visible to the client.  It requires that
 * all data types in the given modes have a explicit stride/offset decorations
 * to tell it exactly how to calculate the offset/address for the given load,
 * store, or atomic operation.  If the offset/stride information does not come
 * from the client explicitly (as with shared variables in GL or Vulkan),
 * nir_lower_vars_to_explicit_types() can be used to add them.
 *
 * Unlike nir_lower_io, this pass is fully capable of handling incomplete
 * pointer chains which may contain cast derefs.  It does so by walking the
 * deref chain backwards and simply replacing each deref, one at a time, with
 * the appropriate address calculation.  The pass takes a nir_address_format
 * parameter which describes how the offset or address is to be represented
 * during calculations.  By ensuring that the address is always in a
 * consistent format, pointers can safely be conjured from thin air by the
 * driver, stored to variables, passed through phis, etc.
 *
 * The one exception to the simple algorithm described above is for handling
 * row-major matrices in which case we may look down one additional level of
 * the deref chain.
 *
 * This pass is also capable of handling OpenCL generic pointers.  If the
 * address mode is global, it will lower any ambiguous (more than one mode)
 * access to global and pass through the deref_mode_is run-time checks as
 * addr_mode_is.  This assumes the driver has somehow mapped shared and
 * scratch memory to the global address space.  For other modes such as
 * 62bit_generic, there is an enum embedded in the address and we lower
 * ambiguous access to an if-ladder and deref_mode_is to a check against the
 * embedded enum.  If nir_lower_explicit_io is called on any shader that
 * contains generic pointers, it must either be used on all of the generic
 * modes or none.
 */
bool
nir_lower_explicit_io(nir_shader *shader, nir_variable_mode modes,
                      nir_address_format addr_format)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl &&
          nir_lower_explicit_io_impl(function->impl, modes, addr_format))
         progress = true;
   }

   return progress;
}

static bool
nir_lower_vars_to_explicit_types_impl(nir_function_impl *impl,
                                      nir_variable_mode modes,
                                      glsl_type_size_align_func type_info)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is_in_set(deref, modes))
            continue;

         unsigned size, alignment;
         const struct glsl_type *new_type =
            glsl_get_explicit_type_for_size_align(deref->type, type_info, &size, &alignment);
         if (new_type != deref->type) {
            progress = true;
            deref->type = new_type;
         }
         if (deref->deref_type == nir_deref_type_cast) {
            /* See also glsl_type::get_explicit_type_for_size_align() */
            unsigned new_stride = align(size, alignment);
            if (new_stride != deref->cast.ptr_stride) {
               deref->cast.ptr_stride = new_stride;
               progress = true;
            }
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance |
                                  nir_metadata_live_ssa_defs |
                                  nir_metadata_loop_analysis);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

static bool
lower_vars_to_explicit(nir_shader *shader,
                       struct exec_list *vars, nir_variable_mode mode,
                       glsl_type_size_align_func type_info)
{
   bool progress = false;
   unsigned offset;
   switch (mode) {
   case nir_var_uniform:
      assert(shader->info.stage == MESA_SHADER_KERNEL);
      offset = 0;
      break;
   case nir_var_function_temp:
   case nir_var_shader_temp:
      offset = shader->scratch_size;
      break;
   case nir_var_mem_shared:
      offset = shader->info.shared_size;
      break;
   case nir_var_mem_task_payload:
      offset = shader->info.task_payload_size;
      break;
   case nir_var_mem_global:
      offset = shader->global_mem_size;
      break;
   case nir_var_mem_constant:
      offset = shader->constant_data_size;
      break;
   case nir_var_shader_call_data:
   case nir_var_ray_hit_attrib:
      offset = 0;
      break;
   default:
      unreachable("Unsupported mode");
   }
   nir_foreach_variable_in_list(var, vars) {
      if (var->data.mode != mode)
         continue;

      unsigned size, align;
      const struct glsl_type *explicit_type =
         glsl_get_explicit_type_for_size_align(var->type, type_info, &size, &align);

      if (explicit_type != var->type)
         var->type = explicit_type;

      UNUSED bool is_empty_struct =
         glsl_type_is_struct_or_ifc(explicit_type) &&
         glsl_get_length(explicit_type) == 0;

      assert(util_is_power_of_two_nonzero(align) || is_empty_struct);
      var->data.driver_location = ALIGN_POT(offset, align);
      offset = var->data.driver_location + size;
      progress = true;
   }

   switch (mode) {
   case nir_var_uniform:
      assert(shader->info.stage == MESA_SHADER_KERNEL);
      shader->num_uniforms = offset;
      break;
   case nir_var_shader_temp:
   case nir_var_function_temp:
      shader->scratch_size = offset;
      break;
   case nir_var_mem_shared:
      shader->info.shared_size = offset;
      break;
   case nir_var_mem_task_payload:
      shader->info.task_payload_size = offset;
      break;
   case nir_var_mem_global:
      shader->global_mem_size = offset;
      break;
   case nir_var_mem_constant:
      shader->constant_data_size = offset;
      break;
   case nir_var_shader_call_data:
   case nir_var_ray_hit_attrib:
      break;
   default:
      unreachable("Unsupported mode");
   }

   return progress;
}

/* If nir_lower_vars_to_explicit_types is called on any shader that contains
 * generic pointers, it must either be used on all of the generic modes or
 * none.
 */
bool
nir_lower_vars_to_explicit_types(nir_shader *shader,
                                 nir_variable_mode modes,
                                 glsl_type_size_align_func type_info)
{
   /* TODO: Situations which need to be handled to support more modes:
    * - row-major matrices
    * - compact shader inputs/outputs
    * - interface types
    */
   ASSERTED nir_variable_mode supported =
      nir_var_mem_shared | nir_var_mem_global | nir_var_mem_constant |
      nir_var_shader_temp | nir_var_function_temp | nir_var_uniform |
      nir_var_shader_call_data | nir_var_ray_hit_attrib |
      nir_var_mem_task_payload;
   assert(!(modes & ~supported) && "unsupported");

   bool progress = false;

   if (modes & nir_var_uniform)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_uniform, type_info);
   if (modes & nir_var_mem_global)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_mem_global, type_info);

   if (modes & nir_var_mem_shared) {
      assert(!shader->info.shared_memory_explicit_layout);
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_mem_shared, type_info);
   }

   if (modes & nir_var_shader_temp)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_shader_temp, type_info);
   if (modes & nir_var_mem_constant)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_mem_constant, type_info);
   if (modes & nir_var_shader_call_data)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_shader_call_data, type_info);
   if (modes & nir_var_ray_hit_attrib)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_ray_hit_attrib, type_info);
   if (modes & nir_var_mem_task_payload)
      progress |= lower_vars_to_explicit(shader, &shader->variables, nir_var_mem_task_payload, type_info);

   nir_foreach_function(function, shader) {
      if (function->impl) {
         if (modes & nir_var_function_temp)
            progress |= lower_vars_to_explicit(shader, &function->impl->locals, nir_var_function_temp, type_info);

         progress |= nir_lower_vars_to_explicit_types_impl(function->impl, modes, type_info);
      }
   }

   return progress;
}

static void
write_constant(void *dst, size_t dst_size,
               const nir_constant *c, const struct glsl_type *type)
{
   if (glsl_type_is_vector_or_scalar(type)) {
      const unsigned num_components = glsl_get_vector_elements(type);
      const unsigned bit_size = glsl_get_bit_size(type);
      if (bit_size == 1) {
         /* Booleans are special-cased to be 32-bit
          *
          * TODO: Make the native bool bit_size an option.
          */
         assert(num_components * 4 <= dst_size);
         for (unsigned i = 0; i < num_components; i++) {
            int32_t b32 = -(int)c->values[i].b;
            memcpy((char *)dst + i * 4, &b32, 4);
         }
      } else {
         assert(bit_size >= 8 && bit_size % 8 == 0);
         const unsigned byte_size = bit_size / 8;
         assert(num_components * byte_size <= dst_size);
         for (unsigned i = 0; i < num_components; i++) {
            /* Annoyingly, thanks to packed structs, we can't make any
             * assumptions about the alignment of dst.  To avoid any strange
             * issues with unaligned writes, we always use memcpy.
             */
            memcpy((char *)dst + i * byte_size, &c->values[i], byte_size);
         }
      }
   } else if (glsl_type_is_array_or_matrix(type)) {
      const unsigned array_len = glsl_get_length(type);
      const unsigned stride = glsl_get_explicit_stride(type);
      assert(stride > 0);
      const struct glsl_type *elem_type = glsl_get_array_element(type);
      for (unsigned i = 0; i < array_len; i++) {
         unsigned elem_offset = i * stride;
         assert(elem_offset < dst_size);
         write_constant((char *)dst + elem_offset, dst_size - elem_offset,
                        c->elements[i], elem_type);
      }
   } else {
      assert(glsl_type_is_struct_or_ifc(type));
      const unsigned num_fields = glsl_get_length(type);
      for (unsigned i = 0; i < num_fields; i++) {
         const int field_offset = glsl_get_struct_field_offset(type, i);
         assert(field_offset >= 0 && field_offset < dst_size);
         const struct glsl_type *field_type = glsl_get_struct_field(type, i);
         write_constant((char *)dst + field_offset, dst_size - field_offset,
                        c->elements[i], field_type);
      }
   }
}

void
nir_gather_explicit_io_initializers(nir_shader *shader,
                                    void *dst, size_t dst_size,
                                    nir_variable_mode mode)
{
   /* It doesn't really make sense to gather initializers for more than one
    * mode at a time.  If this ever becomes well-defined, we can drop the
    * assert then.
    */
   assert(util_bitcount(mode) == 1);

   nir_foreach_variable_with_modes(var, shader, mode) {
      assert(var->data.driver_location < dst_size);
      write_constant((char *)dst + var->data.driver_location,
                     dst_size - var->data.driver_location,
                     var->constant_initializer, var->type);
   }
}

/**
 * Return the offset source for a load/store intrinsic.
 */
nir_src *
nir_get_io_offset_src(nir_intrinsic_instr *instr)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_task_payload:
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_kernel_input:
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_2x32:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_scratch:
   case nir_intrinsic_load_fs_input_interp_deltas:
   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_comp_swap:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_fadd:
   case nir_intrinsic_shared_atomic_fcomp_swap:
   case nir_intrinsic_shared_atomic_fmax:
   case nir_intrinsic_shared_atomic_fmin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_task_payload_atomic_add:
   case nir_intrinsic_task_payload_atomic_imin:
   case nir_intrinsic_task_payload_atomic_umin:
   case nir_intrinsic_task_payload_atomic_imax:
   case nir_intrinsic_task_payload_atomic_umax:
   case nir_intrinsic_task_payload_atomic_and:
   case nir_intrinsic_task_payload_atomic_or:
   case nir_intrinsic_task_payload_atomic_xor:
   case nir_intrinsic_task_payload_atomic_exchange:
   case nir_intrinsic_task_payload_atomic_comp_swap:
   case nir_intrinsic_task_payload_atomic_fadd:
   case nir_intrinsic_task_payload_atomic_fmin:
   case nir_intrinsic_task_payload_atomic_fmax:
   case nir_intrinsic_task_payload_atomic_fcomp_swap:
   case nir_intrinsic_global_atomic_add:
   case nir_intrinsic_global_atomic_and:
   case nir_intrinsic_global_atomic_comp_swap:
   case nir_intrinsic_global_atomic_exchange:
   case nir_intrinsic_global_atomic_fadd:
   case nir_intrinsic_global_atomic_fcomp_swap:
   case nir_intrinsic_global_atomic_fmax:
   case nir_intrinsic_global_atomic_fmin:
   case nir_intrinsic_global_atomic_imax:
   case nir_intrinsic_global_atomic_imin:
   case nir_intrinsic_global_atomic_or:
   case nir_intrinsic_global_atomic_umax:
   case nir_intrinsic_global_atomic_umin:
   case nir_intrinsic_global_atomic_xor:
      return &instr->src[0];
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_input_vertex:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_task_payload:
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_global_2x32:
   case nir_intrinsic_store_scratch:
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
   case nir_intrinsic_ssbo_atomic_fadd:
   case nir_intrinsic_ssbo_atomic_fmin:
   case nir_intrinsic_ssbo_atomic_fmax:
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
      return &instr->src[1];
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
      return &instr->src[2];
   default:
      return NULL;
   }
}

/**
 * Return the vertex index source for a load/store per_vertex intrinsic.
 */
nir_src *
nir_get_io_arrayed_index_src(nir_intrinsic_instr *instr)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output:
      return &instr->src[0];
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
      return &instr->src[1];
   default:
      return NULL;
   }
}

/**
 * Return the numeric constant that identify a NULL pointer for each address
 * format.
 */
const nir_const_value *
nir_address_format_null_value(nir_address_format addr_format)
{
   const static nir_const_value null_values[][NIR_MAX_VEC_COMPONENTS] = {
      [nir_address_format_32bit_global] = {{0}},
      [nir_address_format_2x32bit_global] = {{0}},
      [nir_address_format_64bit_global] = {{0}},
      [nir_address_format_64bit_global_32bit_offset] = {{0}},
      [nir_address_format_64bit_bounded_global] = {{0}},
      [nir_address_format_32bit_index_offset] = {{.u32 = ~0}, {.u32 = ~0}},
      [nir_address_format_32bit_index_offset_pack64] = {{.u64 = ~0ull}},
      [nir_address_format_vec2_index_32bit_offset] = {{.u32 = ~0}, {.u32 = ~0}, {.u32 = ~0}},
      [nir_address_format_32bit_offset] = {{.u32 = ~0}},
      [nir_address_format_32bit_offset_as_64bit] = {{.u64 = ~0ull}},
      [nir_address_format_62bit_generic] = {{.u64 = 0}},
      [nir_address_format_logical] = {{.u32 = ~0}},
   };

   assert(addr_format < ARRAY_SIZE(null_values));
   return null_values[addr_format];
}

nir_ssa_def *
nir_build_addr_ieq(nir_builder *b, nir_ssa_def *addr0, nir_ssa_def *addr1,
                   nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:
   case nir_address_format_2x32bit_global:
   case nir_address_format_64bit_global:
   case nir_address_format_64bit_bounded_global:
   case nir_address_format_32bit_index_offset:
   case nir_address_format_vec2_index_32bit_offset:
   case nir_address_format_32bit_offset:
   case nir_address_format_62bit_generic:
      return nir_ball_iequal(b, addr0, addr1);

   case nir_address_format_64bit_global_32bit_offset:
      return nir_ball_iequal(b, nir_channels(b, addr0, 0xb),
                                nir_channels(b, addr1, 0xb));

   case nir_address_format_32bit_offset_as_64bit:
      assert(addr0->num_components == 1 && addr1->num_components == 1);
      return nir_ieq(b, nir_u2u32(b, addr0), nir_u2u32(b, addr1));

   case nir_address_format_32bit_index_offset_pack64:
      assert(addr0->num_components == 1 && addr1->num_components == 1);
      return nir_ball_iequal(b, nir_unpack_64_2x32(b, addr0), nir_unpack_64_2x32(b, addr1));

   case nir_address_format_logical:
      unreachable("Unsupported address format");
   }

   unreachable("Invalid address format");
}

nir_ssa_def *
nir_build_addr_isub(nir_builder *b, nir_ssa_def *addr0, nir_ssa_def *addr1,
                    nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:
   case nir_address_format_64bit_global:
   case nir_address_format_32bit_offset:
   case nir_address_format_32bit_index_offset_pack64:
   case nir_address_format_62bit_generic:
      assert(addr0->num_components == 1);
      assert(addr1->num_components == 1);
      return nir_isub(b, addr0, addr1);

   case nir_address_format_2x32bit_global:
      return nir_isub(b, addr_to_global(b, addr0, addr_format),
                         addr_to_global(b, addr1, addr_format));

   case nir_address_format_32bit_offset_as_64bit:
      assert(addr0->num_components == 1);
      assert(addr1->num_components == 1);
      return nir_u2u64(b, nir_isub(b, nir_u2u32(b, addr0), nir_u2u32(b, addr1)));

   case nir_address_format_64bit_global_32bit_offset:
   case nir_address_format_64bit_bounded_global:
      return nir_isub(b, addr_to_global(b, addr0, addr_format),
                         addr_to_global(b, addr1, addr_format));

   case nir_address_format_32bit_index_offset:
      assert(addr0->num_components == 2);
      assert(addr1->num_components == 2);
      /* Assume the same buffer index. */
      return nir_isub(b, nir_channel(b, addr0, 1), nir_channel(b, addr1, 1));

   case nir_address_format_vec2_index_32bit_offset:
      assert(addr0->num_components == 3);
      assert(addr1->num_components == 3);
      /* Assume the same buffer index. */
      return nir_isub(b, nir_channel(b, addr0, 2), nir_channel(b, addr1, 2));

   case nir_address_format_logical:
      unreachable("Unsupported address format");
   }

   unreachable("Invalid address format");
}

static bool
is_input(nir_intrinsic_instr *intrin)
{
   return intrin->intrinsic == nir_intrinsic_load_input ||
          intrin->intrinsic == nir_intrinsic_load_per_vertex_input ||
          intrin->intrinsic == nir_intrinsic_load_interpolated_input ||
          intrin->intrinsic == nir_intrinsic_load_fs_input_interp_deltas;
}

static bool
is_output(nir_intrinsic_instr *intrin)
{
   return intrin->intrinsic == nir_intrinsic_load_output ||
          intrin->intrinsic == nir_intrinsic_load_per_vertex_output ||
          intrin->intrinsic == nir_intrinsic_load_per_primitive_output ||
          intrin->intrinsic == nir_intrinsic_store_output ||
          intrin->intrinsic == nir_intrinsic_store_per_vertex_output ||
          intrin->intrinsic == nir_intrinsic_store_per_primitive_output;
}

static bool is_dual_slot(nir_intrinsic_instr *intrin)
{
   if (intrin->intrinsic == nir_intrinsic_store_output ||
       intrin->intrinsic == nir_intrinsic_store_per_vertex_output ||
       intrin->intrinsic == nir_intrinsic_store_per_primitive_output) {
      return nir_src_bit_size(intrin->src[0]) == 64 &&
             nir_src_num_components(intrin->src[0]) >= 3;
   }

   return nir_dest_bit_size(intrin->dest) == 64 &&
          nir_dest_num_components(intrin->dest) >= 3;
}

/**
 * This pass adds constant offsets to instr->const_index[0] for input/output
 * intrinsics, and resets the offset source to 0.  Non-constant offsets remain
 * unchanged - since we don't know what part of a compound variable is
 * accessed, we allocate storage for the entire thing. For drivers that use
 * nir_lower_io_to_temporaries() before nir_lower_io(), this guarantees that
 * the offset source will be 0, so that they don't have to add it in manually.
 */

static bool
add_const_offset_to_base_block(nir_block *block, nir_builder *b,
                               nir_variable_mode modes)
{
   bool progress = false;
   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      if (((modes & nir_var_shader_in) && is_input(intrin)) ||
          ((modes & nir_var_shader_out) && is_output(intrin))) {
         nir_io_semantics sem = nir_intrinsic_io_semantics(intrin);

         /* NV_mesh_shader: ignore MS primitive indices. */
         if (b->shader->info.stage == MESA_SHADER_MESH &&
             sem.location == VARYING_SLOT_PRIMITIVE_INDICES &&
             !(b->shader->info.per_primitive_outputs &
               BITFIELD64_BIT(VARYING_SLOT_PRIMITIVE_INDICES)))
            continue;

         nir_src *offset = nir_get_io_offset_src(intrin);

         /* TODO: Better handling of per-view variables here */
         if (nir_src_is_const(*offset) &&
             !nir_intrinsic_io_semantics(intrin).per_view) {
            unsigned off = nir_src_as_uint(*offset);

            nir_intrinsic_set_base(intrin, nir_intrinsic_base(intrin) + off);

            sem.location += off;
            /* non-indirect indexing should reduce num_slots */
            sem.num_slots = is_dual_slot(intrin) ? 2 : 1;
            nir_intrinsic_set_io_semantics(intrin, sem);

            b->cursor = nir_before_instr(&intrin->instr);
            nir_instr_rewrite_src(&intrin->instr, offset,
                                  nir_src_for_ssa(nir_imm_int(b, 0)));
            progress = true;
         }
      }
   }

   return progress;
}

bool
nir_io_add_const_offset_to_base(nir_shader *nir, nir_variable_mode modes)
{
   bool progress = false;

   nir_foreach_function(f, nir) {
      if (f->impl) {
         bool impl_progress = false;
         nir_builder b;
         nir_builder_init(&b, f->impl);
         nir_foreach_block(block, f->impl) {
            impl_progress |= add_const_offset_to_base_block(block, &b, modes);
         }
         progress |= impl_progress;
         if (impl_progress)
            nir_metadata_preserve(f->impl, nir_metadata_block_index | nir_metadata_dominance);
         else
            nir_metadata_preserve(f->impl, nir_metadata_all);
      }
   }

   return progress;
}

static bool
nir_lower_color_inputs(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         if (intrin->intrinsic != nir_intrinsic_load_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_in))
            continue;

         b.cursor = nir_before_instr(instr);
         nir_variable *var = nir_deref_instr_get_variable(deref);
         nir_ssa_def *def;

         if (var->data.location == VARYING_SLOT_COL0) {
            def = nir_load_color0(&b);
            nir->info.fs.color0_interp = var->data.interpolation;
            nir->info.fs.color0_sample = var->data.sample;
            nir->info.fs.color0_centroid = var->data.centroid;
         } else if (var->data.location == VARYING_SLOT_COL1) {
            def = nir_load_color1(&b);
            nir->info.fs.color1_interp = var->data.interpolation;
            nir->info.fs.color1_sample = var->data.sample;
            nir->info.fs.color1_centroid = var->data.centroid;
         } else {
            continue;
         }

         nir_ssa_def_rewrite_uses(&intrin->dest.ssa, def);
         nir_instr_remove(instr);
         progress = true;
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_dominance |
                                  nir_metadata_block_index);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }
   return progress;
}

bool
nir_io_add_intrinsic_xfb_info(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;

   for (unsigned i = 0; i < NIR_MAX_XFB_BUFFERS; i++)
      nir->info.xfb_stride[i] = nir->xfb_info->buffers[i].stride / 4;

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

         if (!nir_intrinsic_has_io_xfb(intr))
            continue;

         /* No indirect indexing allowed. The index is implied to be 0. */
         ASSERTED nir_src offset = *nir_get_io_offset_src(intr);
         assert(nir_src_is_const(offset) && nir_src_as_uint(offset) == 0);

         /* Calling this pass for the second time shouldn't do anything. */
         if (nir_intrinsic_io_xfb(intr).out[0].num_components ||
             nir_intrinsic_io_xfb(intr).out[1].num_components ||
             nir_intrinsic_io_xfb2(intr).out[0].num_components ||
             nir_intrinsic_io_xfb2(intr).out[1].num_components)
            continue;

         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         unsigned writemask = nir_intrinsic_write_mask(intr) <<
                            nir_intrinsic_component(intr);

         nir_io_xfb xfb[2];
         memset(xfb, 0, sizeof(xfb));

         for (unsigned i = 0; i < nir->xfb_info->output_count; i++) {
            nir_xfb_output_info *out = &nir->xfb_info->outputs[i];
            if (out->location == sem.location) {
               unsigned xfb_mask = writemask & out->component_mask;

               /*fprintf(stdout, "output%u: buffer=%u, offset=%u, location=%u, "
                           "component_offset=%u, component_mask=0x%x, xfb_mask=0x%x, slots=%u\n",
                       i, out->buffer,
                       out->offset,
                       out->location,
                       out->component_offset,
                       out->component_mask,
                       xfb_mask, sem.num_slots);*/

               while (xfb_mask) {
                  int start, count;
                  u_bit_scan_consecutive_range(&xfb_mask, &start, &count);

                  xfb[start / 2].out[start % 2].num_components = count;
                  xfb[start / 2].out[start % 2].buffer = out->buffer;
                  /* out->offset is relative to the first stored xfb component */
                  /* start is relative to component 0 */
                  xfb[start / 2].out[start % 2].offset =
                     out->offset / 4 - out->component_offset + start;

                  progress = true;
               }
            }
         }

         nir_intrinsic_set_io_xfb(intr, xfb[0]);
         nir_intrinsic_set_io_xfb2(intr, xfb[1]);
      }
   }

   nir_metadata_preserve(impl, nir_metadata_all);
   return progress;
}

static int
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

void
nir_lower_io_passes(nir_shader *nir)
{
   if (!nir->options->lower_io_variables)
      return;

   bool has_indirect_inputs =
      (nir->options->support_indirect_inputs >> nir->info.stage) & 0x1;

   /* Transform feedback requires that indirect outputs are lowered. */
   bool has_indirect_outputs =
      (nir->options->support_indirect_outputs >> nir->info.stage) & 0x1 &&
      nir->xfb_info == NULL;

   if (!has_indirect_inputs || !has_indirect_outputs) {
      NIR_PASS_V(nir, nir_lower_io_to_temporaries,
                 nir_shader_get_entrypoint(nir), !has_indirect_outputs,
                 !has_indirect_inputs);

      /* We need to lower all the copy_deref's introduced by lower_io_to-
       * _temporaries before calling nir_lower_io.
       */
      NIR_PASS_V(nir, nir_split_var_copies);
      NIR_PASS_V(nir, nir_lower_var_copies);
      NIR_PASS_V(nir, nir_lower_global_vars_to_local);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->options->lower_fs_color_inputs)
      NIR_PASS_V(nir, nir_lower_color_inputs);

   NIR_PASS_V(nir, nir_lower_io, nir_var_shader_out | nir_var_shader_in,
              type_size_vec4, nir_lower_io_lower_64bit_to_32);

   /* nir_io_add_const_offset_to_base needs actual constants. */
   NIR_PASS_V(nir, nir_opt_constant_folding);
   NIR_PASS_V(nir, nir_io_add_const_offset_to_base, nir_var_shader_in |
                                                    nir_var_shader_out);

   /* Lower and remove dead derefs and variables to clean up the IR. */
   NIR_PASS_V(nir, nir_lower_vars_to_ssa);
   NIR_PASS_V(nir, nir_opt_dce);
   NIR_PASS_V(nir, nir_remove_dead_variables, nir_var_function_temp |
              nir_var_shader_in | nir_var_shader_out, NULL);

   if (nir->xfb_info)
      NIR_PASS_V(nir, nir_io_add_intrinsic_xfb_info);

   nir->info.io_lowered = true;
}

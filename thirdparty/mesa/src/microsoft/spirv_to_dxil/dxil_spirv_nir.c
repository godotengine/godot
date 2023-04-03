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

#include "spirv_to_dxil.h"
#include "nir_to_dxil.h"
#include "dxil_nir.h"
#include "dxil_nir_lower_int_cubemaps.h"
#include "shader_enums.h"
#include "spirv/nir_spirv.h"
#include "util/blob.h"
#include "dxil_spirv_nir.h"

#include "git_sha1.h"
#include "vulkan/vulkan.h"

static const struct spirv_to_nir_options
spirv_to_nir_options = {
   .caps = {
      .draw_parameters = true,
      .multiview = true,
      .subgroup_basic = true,
   },
   .ubo_addr_format = nir_address_format_32bit_index_offset,
   .ssbo_addr_format = nir_address_format_32bit_index_offset,
   .shared_addr_format = nir_address_format_32bit_offset,

   /* use_deref_buffer_array_length + nir_lower_explicit_io force
      * get_ssbo_size to take in the return from load_vulkan_descriptor
      * instead of vulkan_resource_index. This makes it much easier to
      * get the DXIL handle for the SSBO.
      */
   .use_deref_buffer_array_length = true
};

const struct spirv_to_nir_options*
dxil_spirv_nir_get_spirv_options(void)
{
   return &spirv_to_nir_options;
}

 /* Logic extracted from vk_spirv_to_nir() so we have the same preparation
 * steps for both the vulkan driver and the lib used by the WebGPU
 * implementation.
 * Maybe we should move those steps out of vk_spirv_to_nir() and make
 * them vk agnosting (right, the only vk specific thing is the vk_device
 * object that's used for the debug callback passed to spirv_to_nir()).
 */
void
dxil_spirv_nir_prep(nir_shader *nir)
{
   /* We have to lower away local constant initializers right before we
   * inline functions.  That way they get properly initialized at the top
   * of the function and not at the top of its caller.
   */
   NIR_PASS_V(nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS_V(nir, nir_lower_returns);
   NIR_PASS_V(nir, nir_inline_functions);
   NIR_PASS_V(nir, nir_copy_prop);
   NIR_PASS_V(nir, nir_opt_deref);

   /* Pick off the single entrypoint that we want */
   foreach_list_typed_safe(nir_function, func, node, &nir->functions) {
      if (!func->is_entrypoint)
         exec_node_remove(&func->node);
   }
   assert(exec_list_length(&nir->functions) == 1);

   /* Now that we've deleted all but the main function, we can go ahead and
   * lower the rest of the constant initializers.  We do this here so that
   * nir_remove_dead_variables and split_per_member_structs below see the
   * corresponding stores.
   */
   NIR_PASS_V(nir, nir_lower_variable_initializers, ~0);

   /* Split member structs.  We do this before lower_io_to_temporaries so that
   * it doesn't lower system values to temporaries by accident.
   */
   NIR_PASS_V(nir, nir_split_var_copies);
   NIR_PASS_V(nir, nir_split_per_member_structs);

   NIR_PASS_V(nir, nir_remove_dead_variables,
              nir_var_shader_in | nir_var_shader_out | nir_var_system_value |
              nir_var_shader_call_data | nir_var_ray_hit_attrib,
              NULL);

   NIR_PASS_V(nir, nir_propagate_invariant, false);
}

static void
shared_var_info(const struct glsl_type* type, unsigned* size, unsigned* align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length;
   *align = comp_size;
}

static nir_variable *
add_runtime_data_var(nir_shader *nir, unsigned desc_set, unsigned binding)
{
   unsigned runtime_data_size =
      nir->info.stage == MESA_SHADER_COMPUTE
         ? sizeof(struct dxil_spirv_compute_runtime_data)
         : sizeof(struct dxil_spirv_vertex_runtime_data);

   const struct glsl_type *array_type =
      glsl_array_type(glsl_uint_type(), runtime_data_size / sizeof(unsigned),
                      sizeof(unsigned));
   const struct glsl_struct_field field = {array_type, "arr"};
   nir_variable *var = nir_variable_create(
      nir, nir_var_mem_ubo,
      glsl_struct_type(&field, 1, "runtime_data", false), "runtime_data");
   var->data.descriptor_set = desc_set;
   // Check that desc_set fits on descriptor_set
   assert(var->data.descriptor_set == desc_set);
   var->data.binding = binding;
   var->data.how_declared = nir_var_hidden;
   return var;
}

static bool
lower_shader_system_values(struct nir_builder *builder, nir_instr *instr,
                           void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   /* All the intrinsics we care about are loads */
   if (!nir_intrinsic_infos[intrin->intrinsic].has_dest)
      return false;

   assert(intrin->dest.is_ssa);

   const struct dxil_spirv_runtime_conf *conf =
      (const struct dxil_spirv_runtime_conf *)cb_data;

   int offset = 0;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_num_workgroups:
      offset =
         offsetof(struct dxil_spirv_compute_runtime_data, group_count_x);
      break;
   case nir_intrinsic_load_first_vertex:
      offset = offsetof(struct dxil_spirv_vertex_runtime_data, first_vertex);
      break;
   case nir_intrinsic_load_is_indexed_draw:
      offset =
         offsetof(struct dxil_spirv_vertex_runtime_data, is_indexed_draw);
      break;
   case nir_intrinsic_load_base_instance:
      offset = offsetof(struct dxil_spirv_vertex_runtime_data, base_instance);
      break;
   case nir_intrinsic_load_draw_id:
      offset = offsetof(struct dxil_spirv_vertex_runtime_data, draw_id);
      break;
   case nir_intrinsic_load_view_index:
      if (!conf->lower_view_index)
         return false;
      offset = offsetof(struct dxil_spirv_vertex_runtime_data, view_index);
      break;
   default:
      return false;
   }

   builder->cursor = nir_after_instr(instr);
   nir_address_format ubo_format = nir_address_format_32bit_index_offset;

   nir_ssa_def *index = nir_vulkan_resource_index(
      builder, nir_address_format_num_components(ubo_format),
      nir_address_format_bit_size(ubo_format),
      nir_imm_int(builder, 0),
      .desc_set = conf->runtime_data_cbv.register_space,
      .binding = conf->runtime_data_cbv.base_shader_register,
      .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_ssa_def *load_desc = nir_load_vulkan_descriptor(
      builder, nir_address_format_num_components(ubo_format),
      nir_address_format_bit_size(ubo_format),
      index, .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_ssa_def *load_data = build_load_ubo_dxil(
      builder, nir_channel(builder, load_desc, 0),
      nir_imm_int(builder, offset),
      nir_dest_num_components(intrin->dest), nir_dest_bit_size(intrin->dest));

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, load_data);
   nir_instr_remove(instr);
   return true;
}

static bool
dxil_spirv_nir_lower_shader_system_values(nir_shader *shader,
                                          const struct dxil_spirv_runtime_conf *conf)
{
   return nir_shader_instructions_pass(shader, lower_shader_system_values,
                                       nir_metadata_block_index |
                                          nir_metadata_dominance |
                                          nir_metadata_loop_analysis,
                                       (void *)conf);
}

static nir_variable *
add_push_constant_var(nir_shader *nir, unsigned size, unsigned desc_set, unsigned binding)
{
   /* Size must be a multiple of 16 as buffer load is loading 16 bytes at a time */
   unsigned num_32bit_words = ALIGN_POT(size, 16) / 4;

   const struct glsl_type *array_type =
      glsl_array_type(glsl_uint_type(), num_32bit_words, 4);
   const struct glsl_struct_field field = {array_type, "arr"};
   nir_variable *var = nir_variable_create(
      nir, nir_var_mem_ubo,
      glsl_struct_type(&field, 1, "block", false), "push_constants");
   var->data.descriptor_set = desc_set;
   var->data.binding = binding;
   var->data.how_declared = nir_var_hidden;
   return var;
}

struct lower_load_push_constant_data {
   nir_address_format ubo_format;
   unsigned desc_set;
   unsigned binding;
   unsigned size;
};

static bool
lower_load_push_constant(struct nir_builder *builder, nir_instr *instr,
                           void *cb_data)
{
   struct lower_load_push_constant_data *data =
      (struct lower_load_push_constant_data *)cb_data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   /* All the intrinsics we care about are loads */
   if (intrin->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   uint32_t base = nir_intrinsic_base(intrin);
   uint32_t range = nir_intrinsic_range(intrin);

   data->size = MAX2(base + range, data->size);

   builder->cursor = nir_after_instr(instr);
   nir_address_format ubo_format = data->ubo_format;

   nir_ssa_def *index = nir_vulkan_resource_index(
      builder, nir_address_format_num_components(ubo_format),
      nir_address_format_bit_size(ubo_format),
      nir_imm_int(builder, 0),
      .desc_set = data->desc_set, .binding = data->binding,
      .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_ssa_def *load_desc = nir_load_vulkan_descriptor(
      builder, nir_address_format_num_components(ubo_format),
      nir_address_format_bit_size(ubo_format),
      index, .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_ssa_def *offset = nir_ssa_for_src(builder, intrin->src[0], 1);
   nir_ssa_def *load_data = build_load_ubo_dxil(
      builder, nir_channel(builder, load_desc, 0),
      nir_iadd_imm(builder, offset, base),
      nir_dest_num_components(intrin->dest), nir_dest_bit_size(intrin->dest));

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, load_data);
   nir_instr_remove(instr);
   return true;
}

static bool
dxil_spirv_nir_lower_load_push_constant(nir_shader *shader,
                                        nir_address_format ubo_format,
                                        unsigned desc_set, unsigned binding,
                                        uint32_t *size)
{
   bool ret;
   struct lower_load_push_constant_data data = {
      .ubo_format = ubo_format,
      .desc_set = desc_set,
      .binding = binding,
   };
   ret = nir_shader_instructions_pass(shader, lower_load_push_constant,
                                      nir_metadata_block_index |
                                         nir_metadata_dominance |
                                         nir_metadata_loop_analysis,
                                      &data);

   *size = data.size;

   assert(ret == (*size > 0));

   return ret;
}

struct lower_yz_flip_data {
   bool *reads_sysval_ubo;
   const struct dxil_spirv_runtime_conf *rt_conf;
};

static bool
lower_yz_flip(struct nir_builder *builder, nir_instr *instr,
              void *cb_data)
{
   struct lower_yz_flip_data *data =
      (struct lower_yz_flip_data *)cb_data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   if (intrin->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intrin, 0);
   if (var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_POS)
      return false;

   builder->cursor = nir_before_instr(instr);

   const struct dxil_spirv_runtime_conf *rt_conf = data->rt_conf;

   nir_ssa_def *pos = nir_ssa_for_src(builder, intrin->src[1], 4);
   nir_ssa_def *y_pos = nir_channel(builder, pos, 1);
   nir_ssa_def *z_pos = nir_channel(builder, pos, 2);
   nir_ssa_def *y_flip_mask = NULL, *z_flip_mask = NULL, *dyn_yz_flip_mask = NULL;

   if (rt_conf->yz_flip.mode & DXIL_SPIRV_YZ_FLIP_CONDITIONAL) {
      // conditional YZ-flip. The flip bitmask is passed through the vertex
      // runtime data UBO.
      unsigned offset =
         offsetof(struct dxil_spirv_vertex_runtime_data, yz_flip_mask);
      nir_address_format ubo_format = nir_address_format_32bit_index_offset;

      nir_ssa_def *index = nir_vulkan_resource_index(
         builder, nir_address_format_num_components(ubo_format),
         nir_address_format_bit_size(ubo_format),
         nir_imm_int(builder, 0),
         .desc_set = rt_conf->runtime_data_cbv.register_space,
         .binding = rt_conf->runtime_data_cbv.base_shader_register,
         .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

      nir_ssa_def *load_desc = nir_load_vulkan_descriptor(
         builder, nir_address_format_num_components(ubo_format),
         nir_address_format_bit_size(ubo_format),
         index, .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

      dyn_yz_flip_mask =
         build_load_ubo_dxil(builder,
                             nir_channel(builder, load_desc, 0),
                             nir_imm_int(builder, offset), 1, 32);
      *data->reads_sysval_ubo = true;
   }

   if (rt_conf->yz_flip.mode & DXIL_SPIRV_Y_FLIP_UNCONDITIONAL)
      y_flip_mask = nir_imm_int(builder, rt_conf->yz_flip.y_mask);
   else if (rt_conf->yz_flip.mode & DXIL_SPIRV_Y_FLIP_CONDITIONAL)
      y_flip_mask = nir_iand_imm(builder, dyn_yz_flip_mask, DXIL_SPIRV_Y_FLIP_MASK);

   if (rt_conf->yz_flip.mode & DXIL_SPIRV_Z_FLIP_UNCONDITIONAL)
      z_flip_mask = nir_imm_int(builder, rt_conf->yz_flip.z_mask);
   else if (rt_conf->yz_flip.mode & DXIL_SPIRV_Z_FLIP_CONDITIONAL)
      z_flip_mask = nir_ushr_imm(builder, dyn_yz_flip_mask, DXIL_SPIRV_Z_FLIP_SHIFT);

   /* TODO: Multi-viewport */

   if (y_flip_mask) {
      nir_ssa_def *flip = nir_test_mask(builder, y_flip_mask, 1);

      // Z-flip => pos.y = -pos.y
      y_pos = nir_bcsel(builder, flip, nir_fneg(builder, y_pos), y_pos);
   }

   if (z_flip_mask) {
      nir_ssa_def *flip = nir_test_mask(builder, z_flip_mask, 1);

      // Z-flip => pos.z = -pos.z + 1.0f
      z_pos = nir_bcsel(builder, flip,
                        nir_fadd_imm(builder, nir_fneg(builder, z_pos), 1.0f),
                        z_pos);
   }

   nir_ssa_def *def = nir_vec4(builder,
                               nir_channel(builder, pos, 0),
                               y_pos,
                               z_pos,
                               nir_channel(builder, pos, 3));
   nir_instr_rewrite_src(&intrin->instr, &intrin->src[1], nir_src_for_ssa(def));
   return true;
}

static bool
dxil_spirv_nir_lower_yz_flip(nir_shader *shader,
                             const struct dxil_spirv_runtime_conf *rt_conf,
                             bool *reads_sysval_ubo)
{
   struct lower_yz_flip_data data = {
      .rt_conf = rt_conf,
      .reads_sysval_ubo = reads_sysval_ubo,
   };

   return nir_shader_instructions_pass(shader, lower_yz_flip,
                                       nir_metadata_block_index |
                                       nir_metadata_dominance |
                                       nir_metadata_loop_analysis,
                                       &data);
}

static bool
discard_psiz_access(struct nir_builder *builder, nir_instr *instr,
                    void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   if (intrin->intrinsic != nir_intrinsic_store_deref &&
       intrin->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intrin, 0);
   if (!var || var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_PSIZ)
      return false;

   builder->cursor = nir_before_instr(instr);

   if (intrin->intrinsic == nir_intrinsic_load_deref)
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, nir_imm_float(builder, 1.0));

   nir_instr_remove(instr);
   return true;
}

static bool
dxil_spirv_nir_discard_point_size_var(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_VERTEX &&
       shader->info.stage != MESA_SHADER_TESS_EVAL &&
       shader->info.stage != MESA_SHADER_GEOMETRY)
      return false;

   nir_variable *psiz = NULL;
   nir_foreach_shader_out_variable(var, shader) {
      if (var->data.location == VARYING_SLOT_PSIZ) {
         psiz = var;
         break;
      }
   }

   if (!psiz)
      return false;

   if (!nir_shader_instructions_pass(shader, discard_psiz_access,
                                     nir_metadata_block_index |
                                     nir_metadata_dominance |
                                     nir_metadata_loop_analysis,
                                     NULL))
      return false;

   nir_remove_dead_derefs(shader);
   return true;
}

static bool
kill_undefined_varyings(struct nir_builder *b,
                        nir_instr *instr,
                        void *data)
{
   const nir_shader *prev_stage_nir = data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_in)
      return false;

   /* Ignore most builtins for now, some of them get default values
    * when not written from previous stages.
    */
   if (var->data.location < VARYING_SLOT_VAR0 &&
       var->data.location != VARYING_SLOT_POS)
      return false;

   uint32_t loc = var->data.patch ?
                  var->data.location - VARYING_SLOT_PATCH0 : var->data.location;
   uint64_t written = var->data.patch ?
                      prev_stage_nir->info.patch_outputs_written :
                      prev_stage_nir->info.outputs_written;
   if (BITFIELD64_RANGE(loc, glsl_varying_count(var->type)) & written)
      return false;

   b->cursor = nir_after_instr(instr);
   /* Note: zero is used instead of undef, because optimization is not run here, but is
    * run later on. If we load an undef here, and that undef ends up being used to store
    * to position later on, that can cause some or all of the components in that position
    * write to be removed, which is problematic especially in the case of all components,
    * since that would remove the store instruction, and would make it tricky to satisfy
    * the DXIL requirements of writing all position components.
    */
   unsigned int swizzle[NIR_MAX_VEC_COMPONENTS] = { 0 };
   nir_ssa_def *zero =
      nir_swizzle(b, nir_imm_intN_t(b, 0, nir_dest_bit_size(intr->dest)),
                     swizzle, nir_dest_num_components(intr->dest));
   nir_ssa_def_rewrite_uses(&intr->dest.ssa, zero);
   nir_instr_remove(instr);
   return true;
}

static bool
dxil_spirv_nir_kill_undefined_varyings(nir_shader *shader,
                                       const nir_shader *prev_stage_shader)
{
   if (!nir_shader_instructions_pass(shader,
                                     kill_undefined_varyings,
                                     nir_metadata_dominance |
                                     nir_metadata_block_index |
                                     nir_metadata_loop_analysis,
                                     (void *)prev_stage_shader))
      return false;

   nir_remove_dead_derefs(shader);
   nir_remove_dead_variables(shader, nir_var_shader_in, NULL);
   return true;
}

static bool
kill_unused_outputs(struct nir_builder *b,
                    nir_instr *instr,
                    void *data)
{
   uint64_t kill_mask = *((uint64_t *)data);

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out)
      return false;

   unsigned loc = var->data.patch ?
                  var->data.location - VARYING_SLOT_PATCH0 :
                  var->data.location;
   if (!(BITFIELD64_RANGE(loc, glsl_varying_count(var->type)) & kill_mask))
      return false;

   nir_instr_remove(instr);
   return true;
}

static bool
dxil_spirv_nir_kill_unused_outputs(nir_shader *shader,
                                   nir_shader *next_stage_shader)
{
   uint64_t kill_var_mask =
      shader->info.outputs_written & ~next_stage_shader->info.inputs_read;
   bool progress = false;

   /* Don't kill buitin vars */
   kill_var_mask &= BITFIELD64_MASK(MAX_VARYING) << VARYING_SLOT_VAR0;

   if (nir_shader_instructions_pass(shader,
                                    kill_unused_outputs,
                                    nir_metadata_dominance |
                                    nir_metadata_block_index |
                                    nir_metadata_loop_analysis,
                                    (void *)&kill_var_mask))
      progress = true;

   if (shader->info.stage == MESA_SHADER_TESS_EVAL) {
      kill_var_mask =
         (shader->info.patch_outputs_written |
          shader->info.patch_outputs_read) &
         ~next_stage_shader->info.patch_inputs_read;
      if (nir_shader_instructions_pass(shader,
                                       kill_unused_outputs,
                                       nir_metadata_dominance |
                                       nir_metadata_block_index |
                                       nir_metadata_loop_analysis,
                                       (void *)&kill_var_mask))
         progress = true;
   }

   if (progress) {
      nir_opt_dce(shader);
      nir_remove_dead_derefs(shader);
      nir_remove_dead_variables(shader, nir_var_shader_out, NULL);
   }

   return progress;
}

struct lower_pntc_data {
   const struct dxil_spirv_runtime_conf *conf;
   nir_variable *pntc;
};

static bool
write_pntc_with_pos(nir_builder *b, nir_instr *instr, void *_data)
{
   struct lower_pntc_data *data = (struct lower_pntc_data *)_data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.location != VARYING_SLOT_POS)
      return false;

   nir_ssa_def *pos = intr->src[1].ssa;

   unsigned offset =
      offsetof(struct dxil_spirv_vertex_runtime_data, viewport_width) - 4;
   static_assert(offsetof(struct dxil_spirv_vertex_runtime_data, viewport_width) % 16 == 4,
                 "Doing vector unpacking with this assumption");
   nir_address_format ubo_format = nir_address_format_32bit_index_offset;

   b->cursor = nir_before_instr(instr);
   nir_ssa_def *index = nir_vulkan_resource_index(
      b, nir_address_format_num_components(ubo_format),
      nir_address_format_bit_size(ubo_format),
      nir_imm_int(b, 0),
      .desc_set = data->conf->runtime_data_cbv.register_space,
      .binding = data->conf->runtime_data_cbv.base_shader_register,
      .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_ssa_def *load_desc = nir_load_vulkan_descriptor(
      b, nir_address_format_num_components(ubo_format),
      nir_address_format_bit_size(ubo_format),
      index, .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_ssa_def *transform = nir_channels(b,
                                         build_load_ubo_dxil(b,
                                                             nir_channel(b, load_desc, 0),
                                                             nir_imm_int(b, offset), 4, 32),
                                         0x6);
   nir_ssa_def *point_center_in_clip = nir_fmul(b, nir_channels(b, pos, 0x3), nir_frcp(b, nir_channel(b, pos, 3)));
   nir_ssa_def *point_center =
      nir_fmul(b, nir_fadd_imm(b,
                               nir_fmul(b, point_center_in_clip,
                                        nir_vec2(b, nir_imm_float(b, 0.5), nir_imm_float(b, -0.5f))),
                               0.5), transform);
   nir_store_var(b, data->pntc, nir_pad_vec4(b, point_center), 0xf);
   return true;
}

static void
dxil_spirv_write_pntc(nir_shader *nir, const struct dxil_spirv_runtime_conf *conf)
{
   struct lower_pntc_data data = { .conf = conf };
   data.pntc = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "gl_PointCoord");
   data.pntc->data.location = VARYING_SLOT_PNTC;
   nir_shader_instructions_pass(nir, write_pntc_with_pos,
                                nir_metadata_block_index |
                                nir_metadata_dominance |
                                nir_metadata_loop_analysis,
                                &data);
   nir->info.outputs_written |= VARYING_BIT_PNTC;

   /* Add the runtime data var if it's not already there */
   nir_binding binding = {
      .binding = conf->runtime_data_cbv.base_shader_register,
      .desc_set = conf->runtime_data_cbv.register_space,
      .success = true,
   };
   nir_variable *ubo_var = nir_get_binding_variable(nir, binding);
   if (!ubo_var)
      add_runtime_data_var(nir, conf->runtime_data_cbv.register_space, conf->runtime_data_cbv.base_shader_register);
}

static bool
lower_pntc_read(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.location != VARYING_SLOT_PNTC)
      return false;

   nir_ssa_def *point_center = &intr->dest.ssa;
   nir_variable *pos_var = (nir_variable *)data;

   b->cursor = nir_after_instr(instr);

   nir_ssa_def *pos;
   if (var->data.sample == pos_var->data.sample)
      pos = nir_load_var(b, pos_var);
   else if (var->data.sample)
      pos = nir_interp_deref_at_sample(b, 4, 32,
                                       &nir_build_deref_var(b, pos_var)->dest.ssa,
                                       nir_load_sample_id(b));
   else
      pos = nir_interp_deref_at_offset(b, 4, 32,
                                       &nir_build_deref_var(b, pos_var)->dest.ssa,
                                       nir_vec2(b, nir_imm_float(b, 0), nir_imm_float(b, 0)));

   nir_ssa_def *pntc = nir_fadd_imm(b,
                                    nir_fsub(b, nir_channels(b, pos, 0x3), nir_channels(b, point_center, 0x3)),
                                    0.5);
   nir_ssa_def_rewrite_uses_after(point_center, pntc, pntc->parent_instr);
   return true;
}

static void
dxil_spirv_compute_pntc(nir_shader *nir)
{
   nir_variable *pos = nir_find_variable_with_location(nir, nir_var_shader_in, VARYING_SLOT_POS);
   if (!pos) {
      pos = nir_variable_create(nir, nir_var_shader_in, glsl_vec4_type(), "gl_FragCoord");
      pos->data.location = VARYING_SLOT_POS;
      pos->data.sample = nir_find_variable_with_location(nir, nir_var_shader_in, VARYING_SLOT_PNTC)->data.sample;
   }
   nir_shader_instructions_pass(nir, lower_pntc_read,
                                nir_metadata_block_index |
                                nir_metadata_dominance |
                                nir_metadata_loop_analysis,
                                pos);
}

static bool
lower_view_index_to_rt_layer_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var ||
       var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_LAYER)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_ssa_def *layer = intr->src[1].ssa;
   nir_ssa_def *new_layer = nir_iadd(b, layer,
                                     nir_load_view_index(b));
   nir_instr_rewrite_src_ssa(instr, &intr->src[1], new_layer);
   return true;
}

static bool
add_layer_write(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr) {
      if (instr->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_emit_vertex &&
          intr->intrinsic != nir_intrinsic_emit_vertex_with_counter)
         return false;
      b->cursor = nir_before_instr(instr);
   }
   nir_variable *var = (nir_variable *)data;
   nir_store_var(b, var, nir_load_view_index(b), 0x1);
   return true;
}

static void
lower_view_index_to_rt_layer(nir_shader *nir)
{
   bool existing_write =
      nir_shader_instructions_pass(nir,
                                   lower_view_index_to_rt_layer_instr,
                                   nir_metadata_block_index |
                                   nir_metadata_dominance |
                                   nir_metadata_loop_analysis, NULL);

   if (existing_write)
      return;

   nir_variable *var = nir_variable_create(nir, nir_var_shader_out,
                                           glsl_uint_type(), "gl_Layer");
   var->data.location = VARYING_SLOT_LAYER;
   var->data.interpolation = INTERP_MODE_FLAT;
   if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      nir_shader_instructions_pass(nir,
                                   add_layer_write,
                                   nir_metadata_block_index |
                                   nir_metadata_dominance |
                                   nir_metadata_loop_analysis, var);
   } else {
      nir_function_impl *func = nir_shader_get_entrypoint(nir);
      nir_builder b;
      nir_builder_init(&b, func);
      b.cursor = nir_after_block(nir_impl_last_block(func));
      add_layer_write(&b, NULL, var);
   }
}

void
dxil_spirv_nir_link(nir_shader *nir, nir_shader *prev_stage_nir,
                    const struct dxil_spirv_runtime_conf *conf,
                    bool *requires_runtime_data)
{
   glsl_type_singleton_init_or_ref();

   *requires_runtime_data = false;
   if (prev_stage_nir) {
      if (nir->info.stage == MESA_SHADER_FRAGMENT) {
         nir->info.clip_distance_array_size = prev_stage_nir->info.clip_distance_array_size;

         if (nir->info.inputs_read & VARYING_BIT_PNTC) {
            NIR_PASS_V(prev_stage_nir, dxil_spirv_write_pntc, conf);
            NIR_PASS_V(nir, dxil_spirv_compute_pntc);
            *requires_runtime_data = true;
         }
      }

      NIR_PASS_V(nir, dxil_spirv_nir_kill_undefined_varyings, prev_stage_nir);
      NIR_PASS_V(prev_stage_nir, dxil_spirv_nir_kill_unused_outputs, nir);

      nir->info.inputs_read =
         dxil_reassign_driver_locations(nir, nir_var_shader_in,
                                        prev_stage_nir->info.outputs_written);

      prev_stage_nir->info.outputs_written =
         dxil_reassign_driver_locations(prev_stage_nir, nir_var_shader_out,
                                        nir->info.inputs_read);
   }

   glsl_type_singleton_decref();
}

void
dxil_spirv_nir_passes(nir_shader *nir,
                      const struct dxil_spirv_runtime_conf *conf,
                      bool *requires_runtime_data)
{
   glsl_type_singleton_init_or_ref();

   NIR_PASS_V(nir, dxil_nir_lower_int_cubemaps, false);
   NIR_PASS_V(nir, nir_lower_io_to_vector,
              nir_var_shader_out |
              (nir->info.stage != MESA_SHADER_VERTEX ? nir_var_shader_in : 0));
   NIR_PASS_V(nir, nir_opt_combine_stores, nir_var_shader_out);
   NIR_PASS_V(nir, nir_remove_dead_derefs);

   const struct nir_lower_sysvals_to_varyings_options sysvals_to_varyings = {
      .frag_coord = true,
      .point_coord = true,
   };
   NIR_PASS_V(nir, nir_lower_sysvals_to_varyings, &sysvals_to_varyings);

   NIR_PASS_V(nir, nir_lower_system_values);

   NIR_PASS_V(nir, dxil_nir_lower_subgroup_id);
   NIR_PASS_V(nir, dxil_nir_lower_num_subgroups);

   // Force sample-rate shading if we're asked to.
   if (conf->force_sample_rate_shading) {
      assert(nir->info.stage == MESA_SHADER_FRAGMENT);
      nir->info.fs.uses_sample_shading = true;
   }

   if (conf->zero_based_vertex_instance_id) {
      // vertex_id and instance_id should have already been transformed to
      // base zero before spirv_to_dxil was called. Therefore, we can zero out
      // base/firstVertex/Instance.
      gl_system_value system_values[] = {SYSTEM_VALUE_FIRST_VERTEX,
                                         SYSTEM_VALUE_BASE_VERTEX,
                                         SYSTEM_VALUE_BASE_INSTANCE};
      NIR_PASS_V(nir, dxil_nir_lower_system_values_to_zero, system_values,
                 ARRAY_SIZE(system_values));
   }

   if (conf->lower_view_index_to_rt_layer)
      NIR_PASS_V(nir, lower_view_index_to_rt_layer);

   *requires_runtime_data = false;
   NIR_PASS(*requires_runtime_data, nir,
            dxil_spirv_nir_lower_shader_system_values,
            conf);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS_V(nir, nir_lower_input_attachments,
                 &(nir_input_attachment_options){
                     .use_fragcoord_sysval = false,
                     .use_layer_id_sysval = !conf->lower_view_index,
                     .use_view_id_for_layer = !conf->lower_view_index,
                 });

      /* This will lower load_helper to a memoized is_helper if needed; otherwise, load_helper
       * will stay, but trivially translatable to IsHelperLane(), which will be known to be
       * constant across the invocation since no demotion would have been used.
       */
      NIR_PASS_V(nir, nir_lower_discard_or_demote, nir->info.use_legacy_math_rules);

      NIR_PASS_V(nir, dxil_nir_lower_discard_and_terminate);
      NIR_PASS_V(nir, nir_lower_returns);
      NIR_PASS_V(nir, dxil_nir_lower_sample_pos);
      NIR_PASS_V(nir, nir_lower_fragcoord_wtrans);
   }

   NIR_PASS_V(nir, nir_opt_deref);

   if (conf->read_only_images_as_srvs) {
      const nir_opt_access_options opt_access_options = {
         .is_vulkan = true,
      };
      NIR_PASS_V(nir, nir_opt_access, &opt_access_options);
   }

   NIR_PASS_V(nir, dxil_spirv_nir_discard_point_size_var);

   NIR_PASS_V(nir, nir_remove_dead_variables,
              nir_var_shader_in | nir_var_shader_out |
              nir_var_system_value | nir_var_mem_shared,
              NULL);

   uint32_t push_constant_size = 0;
   NIR_PASS_V(nir, nir_lower_explicit_io, nir_var_mem_push_const,
              nir_address_format_32bit_offset);
   NIR_PASS_V(nir, dxil_spirv_nir_lower_load_push_constant,
              nir_address_format_32bit_index_offset,
              conf->push_constant_cbv.register_space,
              conf->push_constant_cbv.base_shader_register,
              &push_constant_size);

   NIR_PASS_V(nir, nir_lower_explicit_io, nir_var_mem_ubo | nir_var_mem_ssbo,
              nir_address_format_32bit_index_offset);

   if (!nir->info.shared_memory_explicit_layout) {
      NIR_PASS_V(nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
                 shared_var_info);
   }
   NIR_PASS_V(nir, nir_lower_explicit_io, nir_var_mem_shared,
      nir_address_format_32bit_offset);

   NIR_PASS_V(nir, nir_lower_clip_cull_distance_arrays);
   NIR_PASS_V(nir, nir_lower_io_to_temporaries, nir_shader_get_entrypoint(nir), true, true);
   NIR_PASS_V(nir, nir_lower_global_vars_to_local);
   NIR_PASS_V(nir, nir_split_var_copies);
   NIR_PASS_V(nir, nir_lower_var_copies);
   NIR_PASS_V(nir, nir_lower_io_arrays_to_elements_no_indirects, false);


   if (conf->yz_flip.mode != DXIL_SPIRV_YZ_FLIP_NONE) {
      assert(nir->info.stage == MESA_SHADER_VERTEX ||
             nir->info.stage == MESA_SHADER_GEOMETRY);
      NIR_PASS_V(nir,
                 dxil_spirv_nir_lower_yz_flip,
                 conf, requires_runtime_data);
   }

   if (*requires_runtime_data) {
      add_runtime_data_var(nir, conf->runtime_data_cbv.register_space,
                           conf->runtime_data_cbv.base_shader_register);
   }

   if (push_constant_size > 0) {
      add_push_constant_var(nir, push_constant_size,
                            conf->push_constant_cbv.register_space,
                            conf->push_constant_cbv.base_shader_register);
   }

   NIR_PASS_V(nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS_V(nir, nir_opt_dce);
   NIR_PASS_V(nir, dxil_nir_lower_double_math);

   {
      bool progress;
      do
      {
         progress = false;
         NIR_PASS(progress, nir, nir_copy_prop);
         NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
         NIR_PASS(progress, nir, nir_opt_deref);
         NIR_PASS(progress, nir, nir_opt_dce);
         NIR_PASS(progress, nir, nir_opt_undef);
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_cse);
         if (nir_opt_trivial_continues(nir)) {
            progress = true;
            NIR_PASS(progress, nir, nir_copy_prop);
            NIR_PASS(progress, nir, nir_opt_dce);
         }
         NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
         NIR_PASS(progress, nir, nir_opt_algebraic);
      } while (progress);
   }

   NIR_PASS_V(nir, nir_lower_readonly_images_to_tex, true);
   nir_lower_tex_options lower_tex_options = {
      .lower_txp = UINT32_MAX,
      .lower_invalid_implicit_lod = true,
      .lower_tg4_offsets = true,
   };
   NIR_PASS_V(nir, nir_lower_tex, &lower_tex_options);

   NIR_PASS_V(nir, dxil_nir_lower_atomics_to_dxil);
   NIR_PASS_V(nir, dxil_nir_split_clip_cull_distance);
   NIR_PASS_V(nir, dxil_nir_lower_loads_stores_to_dxil);
   NIR_PASS_V(nir, dxil_nir_split_typed_samplers);
   NIR_PASS_V(nir, dxil_nir_lower_ubo_array_one_to_static);
   NIR_PASS_V(nir, nir_opt_dce);
   NIR_PASS_V(nir, nir_remove_dead_derefs);
   NIR_PASS_V(nir, nir_remove_dead_variables,
              nir_var_uniform | nir_var_shader_in | nir_var_shader_out,
              NULL);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      dxil_sort_ps_outputs(nir);
   } else {
      /* Dummy linking step so we get different driver_location
       * assigned even if there's just a single vertex shader in the
       * pipeline. The real linking happens in dxil_spirv_nir_link().
       */
      nir->info.outputs_written =
         dxil_reassign_driver_locations(nir, nir_var_shader_out, 0);
   }

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
         /* spirv_to_dxil() only emits generic vertex attributes. */
         assert(var->data.location >= VERT_ATTRIB_GENERIC0);
         var->data.driver_location = var->data.location - VERT_ATTRIB_GENERIC0;
      }

      nir->info.inputs_read =
         dxil_sort_by_driver_location(nir, nir_var_shader_in);
   } else {
      nir->info.inputs_read =
         dxil_reassign_driver_locations(nir, nir_var_shader_in, 0);
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   glsl_type_singleton_decref();
}

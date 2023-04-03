/*
 * Copyright © 2018 Valve Corporation
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
 */

#include "nir.h"

/* This pass computes for each ssa definition if it is uniform.
 * That is, the variable has the same value for all invocations
 * of the group.
 *
 * This divergence analysis pass expects the shader to be in LCSSA-form.
 *
 * This algorithm implements "The Simple Divergence Analysis" from
 * Diogo Sampaio, Rafael De Souza, Sylvain Collange, Fernando Magno Quintão Pereira.
 * Divergence Analysis.  ACM Transactions on Programming Languages and Systems (TOPLAS),
 * ACM, 2013, 35 (4), pp.13:1-13:36. <10.1145/2523815>. <hal-00909072v2>
 */

struct divergence_state {
   const gl_shader_stage stage;
   nir_shader *shader;

   /** current control flow state */
   /* True if some loop-active invocations might take a different control-flow path.
    * A divergent break does not cause subsequent control-flow to be considered
    * divergent because those invocations are no longer active in the loop.
    * For a divergent if, both sides are considered divergent flow because
    * the other side is still loop-active. */
   bool divergent_loop_cf;
   /* True if a divergent continue happened since the loop header */
   bool divergent_loop_continue;
   /* True if a divergent break happened since the loop header */
   bool divergent_loop_break;

   /* True if we visit the block for the fist time */
   bool first_visit;
};

static bool
visit_cf_list(struct exec_list *list, struct divergence_state *state);

static bool
visit_alu(nir_alu_instr *instr)
{
   if (instr->dest.dest.ssa.divergent)
      return false;

   unsigned num_src = nir_op_infos[instr->op].num_inputs;

   for (unsigned i = 0; i < num_src; i++) {
      if (instr->src[i].src.ssa->divergent) {
         instr->dest.dest.ssa.divergent = true;
         return true;
      }
   }

   return false;
}

static bool
visit_intrinsic(nir_shader *shader, nir_intrinsic_instr *instr)
{
   if (!nir_intrinsic_infos[instr->intrinsic].has_dest)
      return false;

   if (instr->dest.ssa.divergent)
      return false;

   nir_divergence_options options = shader->options->divergence_analysis_options;
   gl_shader_stage stage = shader->info.stage;
   bool is_divergent = false;
   switch (instr->intrinsic) {
   /* Intrinsics which are always uniform */
   case nir_intrinsic_shader_clock:
   case nir_intrinsic_ballot:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_load_push_constant:
   case nir_intrinsic_load_work_dim:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_workgroup_size:
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_num_subgroups:
   case nir_intrinsic_load_ray_launch_size:
   case nir_intrinsic_load_ray_launch_size_addr_amd:
   case nir_intrinsic_load_sbt_base_amd:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_last_invocation:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_draw_id:
   case nir_intrinsic_load_is_indexed_draw:
   case nir_intrinsic_load_viewport_scale:
   case nir_intrinsic_load_user_clip_plane:
   case nir_intrinsic_load_viewport_x_scale:
   case nir_intrinsic_load_viewport_y_scale:
   case nir_intrinsic_load_viewport_z_scale:
   case nir_intrinsic_load_viewport_offset:
   case nir_intrinsic_load_viewport_x_offset:
   case nir_intrinsic_load_viewport_y_offset:
   case nir_intrinsic_load_viewport_z_offset:
   case nir_intrinsic_load_viewport_xy_scale_and_offset:
   case nir_intrinsic_load_blend_const_color_a_float:
   case nir_intrinsic_load_blend_const_color_b_float:
   case nir_intrinsic_load_blend_const_color_g_float:
   case nir_intrinsic_load_blend_const_color_r_float:
   case nir_intrinsic_load_blend_const_color_rgba:
   case nir_intrinsic_load_blend_const_color_aaaa8888_unorm:
   case nir_intrinsic_load_blend_const_color_rgba8888_unorm:
   case nir_intrinsic_load_line_width:
   case nir_intrinsic_load_aa_line_width:
   case nir_intrinsic_load_xfb_address:
   case nir_intrinsic_load_num_vertices:
   case nir_intrinsic_load_fb_layers_v3d:
   case nir_intrinsic_load_tcs_num_patches_amd:
   case nir_intrinsic_load_ring_tess_factors_amd:
   case nir_intrinsic_load_ring_tess_offchip_amd:
   case nir_intrinsic_load_ring_tess_factors_offset_amd:
   case nir_intrinsic_load_ring_tess_offchip_offset_amd:
   case nir_intrinsic_load_ring_mesh_scratch_amd:
   case nir_intrinsic_load_ring_mesh_scratch_offset_amd:
   case nir_intrinsic_load_ring_esgs_amd:
   case nir_intrinsic_load_ring_es2gs_offset_amd:
   case nir_intrinsic_load_ring_task_draw_amd:
   case nir_intrinsic_load_ring_task_payload_amd:
   case nir_intrinsic_load_sample_positions_amd:
   case nir_intrinsic_load_rasterization_samples_amd:
   case nir_intrinsic_load_ring_gsvs_amd:
   case nir_intrinsic_load_ring_gs2vs_offset_amd:
   case nir_intrinsic_load_streamout_config_amd:
   case nir_intrinsic_load_streamout_write_index_amd:
   case nir_intrinsic_load_streamout_offset_amd:
   case nir_intrinsic_load_task_ring_entry_amd:
   case nir_intrinsic_load_task_ib_addr:
   case nir_intrinsic_load_task_ib_stride:
   case nir_intrinsic_load_ring_attr_amd:
   case nir_intrinsic_load_ring_attr_offset_amd:
   case nir_intrinsic_load_sample_positions_pan:
   case nir_intrinsic_load_workgroup_num_input_vertices_amd:
   case nir_intrinsic_load_workgroup_num_input_primitives_amd:
   case nir_intrinsic_load_pipeline_stat_query_enabled_amd:
   case nir_intrinsic_load_prim_gen_query_enabled_amd:
   case nir_intrinsic_load_prim_xfb_query_enabled_amd:
   case nir_intrinsic_load_merged_wave_info_amd:
   case nir_intrinsic_load_clamp_vertex_color_amd:
   case nir_intrinsic_load_cull_front_face_enabled_amd:
   case nir_intrinsic_load_cull_back_face_enabled_amd:
   case nir_intrinsic_load_cull_ccw_amd:
   case nir_intrinsic_load_cull_small_primitives_enabled_amd:
   case nir_intrinsic_load_cull_any_enabled_amd:
   case nir_intrinsic_load_cull_small_prim_precision_amd:
   case nir_intrinsic_load_user_data_amd:
   case nir_intrinsic_load_force_vrs_rates_amd:
   case nir_intrinsic_load_tess_level_inner_default:
   case nir_intrinsic_load_tess_level_outer_default:
   case nir_intrinsic_load_scalar_arg_amd:
   case nir_intrinsic_load_smem_amd:
   case nir_intrinsic_load_smem_buffer_amd:
   case nir_intrinsic_load_rt_dynamic_callable_stack_base_amd:
   case nir_intrinsic_load_global_const_block_intel:
   case nir_intrinsic_load_reloc_const_intel:
   case nir_intrinsic_load_global_block_intel:
   case nir_intrinsic_load_btd_global_arg_addr_intel:
   case nir_intrinsic_load_btd_local_arg_addr_intel:
   case nir_intrinsic_load_mesh_inline_data_intel:
   case nir_intrinsic_load_ray_num_dss_rt_stacks_intel:
   case nir_intrinsic_load_lshs_vertex_stride_amd:
   case nir_intrinsic_load_hs_out_patch_data_offset_amd:
   case nir_intrinsic_load_clip_half_line_width_amd:
   case nir_intrinsic_load_num_vertices_per_primitive_amd:
   case nir_intrinsic_load_streamout_buffer_amd:
   case nir_intrinsic_load_ordered_id_amd:
   case nir_intrinsic_load_provoking_vtx_in_prim_amd:
   case nir_intrinsic_load_lds_ngg_scratch_base_amd:
   case nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd:
   case nir_intrinsic_load_btd_shader_type_intel:
   case nir_intrinsic_load_base_workgroup_id:
      is_divergent = false;
      break;

   /* Intrinsics with divergence depending on shader stage and hardware */
   case nir_intrinsic_load_shader_record_ptr:
      is_divergent = !(options & nir_divergence_shader_record_ptr_uniform);
      break;
   case nir_intrinsic_load_frag_shading_rate:
      is_divergent = !(options & nir_divergence_single_frag_shading_rate_per_subgroup);
      break;
   case nir_intrinsic_load_input:
      is_divergent = instr->src[0].ssa->divergent;
      if (stage == MESA_SHADER_FRAGMENT)
         is_divergent |= !(options & nir_divergence_single_prim_per_subgroup);
      else if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent |= !(options & nir_divergence_single_patch_per_tes_subgroup);
      else if (stage != MESA_SHADER_MESH)
         is_divergent = true;
      break;
   case nir_intrinsic_load_per_vertex_input:
      is_divergent = instr->src[0].ssa->divergent ||
                     instr->src[1].ssa->divergent;
      if (stage == MESA_SHADER_TESS_CTRL)
         is_divergent |= !(options & nir_divergence_single_patch_per_tcs_subgroup);
      if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent |= !(options & nir_divergence_single_patch_per_tes_subgroup);
      else
         is_divergent = true;
      break;
   case nir_intrinsic_load_input_vertex:
      is_divergent = instr->src[1].ssa->divergent;
      assert(stage == MESA_SHADER_FRAGMENT);
      is_divergent |= !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_output:
      is_divergent = instr->src[0].ssa->divergent;
      switch (stage) {
      case MESA_SHADER_TESS_CTRL:
         is_divergent |= !(options & nir_divergence_single_patch_per_tcs_subgroup);
         break;
      case MESA_SHADER_FRAGMENT:
         is_divergent = true;
         break;
      case MESA_SHADER_TASK:
      case MESA_SHADER_MESH:
         /* Divergent if src[0] is, so nothing else to do. */
         break;
      default:
         unreachable("Invalid stage for load_output");
      }
      break;
   case nir_intrinsic_load_per_vertex_output:
      assert(stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_MESH);
      is_divergent = instr->src[0].ssa->divergent ||
                     instr->src[1].ssa->divergent ||
                     (stage == MESA_SHADER_TESS_CTRL &&
                      !(options & nir_divergence_single_patch_per_tcs_subgroup));
      break;
   case nir_intrinsic_load_per_primitive_output:
      assert(stage == MESA_SHADER_MESH);
      is_divergent = instr->src[0].ssa->divergent ||
                     instr->src[1].ssa->divergent;
      break;
   case nir_intrinsic_load_layer_id:
   case nir_intrinsic_load_front_face:
      assert(stage == MESA_SHADER_FRAGMENT);
      is_divergent = !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_view_index:
      assert(stage != MESA_SHADER_COMPUTE && stage != MESA_SHADER_KERNEL);
      if (options & nir_divergence_view_index_uniform)
         is_divergent = false;
      else if (stage == MESA_SHADER_FRAGMENT)
         is_divergent = !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_fs_input_interp_deltas:
      assert(stage == MESA_SHADER_FRAGMENT);
      is_divergent = instr->src[0].ssa->divergent;
      is_divergent |= !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_primitive_id:
      if (stage == MESA_SHADER_FRAGMENT)
         is_divergent = !(options & nir_divergence_single_prim_per_subgroup);
      else if (stage == MESA_SHADER_TESS_CTRL)
         is_divergent = !(options & nir_divergence_single_patch_per_tcs_subgroup);
      else if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent = !(options & nir_divergence_single_patch_per_tes_subgroup);
      else if (stage == MESA_SHADER_GEOMETRY || stage == MESA_SHADER_VERTEX)
         is_divergent = true;
      else if (stage == MESA_SHADER_ANY_HIT ||
               stage == MESA_SHADER_CLOSEST_HIT ||
               stage == MESA_SHADER_INTERSECTION)
         is_divergent = true;
      else
         unreachable("Invalid stage for load_primitive_id");
      break;
   case nir_intrinsic_load_tess_level_inner:
   case nir_intrinsic_load_tess_level_outer:
      if (stage == MESA_SHADER_TESS_CTRL)
         is_divergent = !(options & nir_divergence_single_patch_per_tcs_subgroup);
      else if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent = !(options & nir_divergence_single_patch_per_tes_subgroup);
      else
         unreachable("Invalid stage for load_primitive_tess_level_*");
      break;
   case nir_intrinsic_load_patch_vertices_in:
      if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent = !(options & nir_divergence_single_patch_per_tes_subgroup);
      else
         assert(stage == MESA_SHADER_TESS_CTRL);
      break;

   case nir_intrinsic_load_workgroup_index:
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_workgroup_id_zero_base:
      assert(gl_shader_stage_uses_workgroup(stage));
      if (stage == MESA_SHADER_COMPUTE)
         is_divergent |= (options & nir_divergence_multiple_workgroup_per_compute_subgroup);
      break;

   /* Clustered reductions are uniform if cluster_size == subgroup_size or
    * the source is uniform and the operation is invariant.
    * Inclusive scans are uniform if
    * the source is uniform and the operation is invariant
    */
   case nir_intrinsic_reduce:
      if (nir_intrinsic_cluster_size(instr) == 0)
         return false;
      FALLTHROUGH;
   case nir_intrinsic_inclusive_scan: {
      nir_op op = nir_intrinsic_reduction_op(instr);
      is_divergent = instr->src[0].ssa->divergent;
      if (op != nir_op_umin && op != nir_op_imin && op != nir_op_fmin &&
          op != nir_op_umax && op != nir_op_imax && op != nir_op_fmax &&
          op != nir_op_iand && op != nir_op_ior)
         is_divergent = true;
      break;
   }

   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
      is_divergent = (instr->src[0].ssa->divergent && (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     instr->src[1].ssa->divergent;
      break;

   case nir_intrinsic_get_ssbo_size:
   case nir_intrinsic_deref_buffer_array_length:
      is_divergent = instr->src[0].ssa->divergent && (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM);
      break;

   case nir_intrinsic_image_samples_identical:
   case nir_intrinsic_image_deref_samples_identical:
   case nir_intrinsic_bindless_image_samples_identical:
   case nir_intrinsic_image_fragment_mask_load_amd:
   case nir_intrinsic_image_deref_fragment_mask_load_amd:
   case nir_intrinsic_bindless_image_fragment_mask_load_amd:
      is_divergent = (instr->src[0].ssa->divergent && (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     instr->src[1].ssa->divergent;
      break;

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_bindless_image_sparse_load:
      is_divergent = (instr->src[0].ssa->divergent && (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     instr->src[1].ssa->divergent || instr->src[2].ssa->divergent || instr->src[3].ssa->divergent;
      break;

   case nir_intrinsic_optimization_barrier_vgpr_amd:
      is_divergent = instr->src[0].ssa->divergent;
      break;

   /* Intrinsics with divergence depending on sources */
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_byte_permute_amd:
   case nir_intrinsic_load_deref:
   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_shared2_amd:
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_2x32:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global_amd:
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_constant:
   case nir_intrinsic_load_sample_pos_from_id:
   case nir_intrinsic_load_kernel_input:
   case nir_intrinsic_load_task_payload:
   case nir_intrinsic_load_buffer_amd:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_bindless_image_samples:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_image_descriptor_amd:
   case nir_intrinsic_image_deref_descriptor_amd:
   case nir_intrinsic_bindless_image_descriptor_amd:
   case nir_intrinsic_copy_deref:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_atomic_counter_read:
   case nir_intrinsic_atomic_counter_read_deref:
   case nir_intrinsic_quad_swizzle_amd:
   case nir_intrinsic_masked_swizzle_amd:
   case nir_intrinsic_is_sparse_texels_resident:
   case nir_intrinsic_sparse_residency_code_and:
   case nir_intrinsic_bvh64_intersect_ray_amd:
   case nir_intrinsic_image_deref_load_param_intel:
   case nir_intrinsic_image_load_raw_intel:
   case nir_intrinsic_get_ubo_size:
   case nir_intrinsic_load_ssbo_address:
   case nir_intrinsic_load_desc_set_address_intel: {
      unsigned num_srcs = nir_intrinsic_infos[instr->intrinsic].num_srcs;
      for (unsigned i = 0; i < num_srcs; i++) {
         if (instr->src[i].ssa->divergent) {
            is_divergent = true;
            break;
         }
      }
      break;
   }

   case nir_intrinsic_shuffle:
      is_divergent = instr->src[0].ssa->divergent &&
                     instr->src[1].ssa->divergent;
      break;

   /* Intrinsics which are always divergent */
   case nir_intrinsic_load_color0:
   case nir_intrinsic_load_color1:
   case nir_intrinsic_load_param:
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_id_no_per_sample:
   case nir_intrinsic_load_sample_mask_in:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_point_coord_maybe_flipped:
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_model:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
   case nir_intrinsic_interp_deref_at_offset:
   case nir_intrinsic_interp_deref_at_sample:
   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_vertex:
   case nir_intrinsic_load_tess_coord:
   case nir_intrinsic_load_point_coord:
   case nir_intrinsic_load_line_coord:
   case nir_intrinsic_load_frag_coord:
   case nir_intrinsic_load_sample_pos:
   case nir_intrinsic_load_sample_pos_or_center:
   case nir_intrinsic_load_vertex_id_zero_base:
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_invocation_id:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_local_invocation_index:
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_global_invocation_id_zero_base:
   case nir_intrinsic_load_global_invocation_index:
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_load_scratch:
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
   case nir_intrinsic_deref_atomic_fcomp_swap:
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
   case nir_intrinsic_ssbo_atomic_fmax:
   case nir_intrinsic_ssbo_atomic_fmin:
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
   case nir_intrinsic_image_deref_atomic_add:
   case nir_intrinsic_image_deref_atomic_imin:
   case nir_intrinsic_image_deref_atomic_umin:
   case nir_intrinsic_image_deref_atomic_imax:
   case nir_intrinsic_image_deref_atomic_umax:
   case nir_intrinsic_image_deref_atomic_and:
   case nir_intrinsic_image_deref_atomic_or:
   case nir_intrinsic_image_deref_atomic_xor:
   case nir_intrinsic_image_deref_atomic_exchange:
   case nir_intrinsic_image_deref_atomic_comp_swap:
   case nir_intrinsic_image_deref_atomic_fadd:
   case nir_intrinsic_image_deref_atomic_fmin:
   case nir_intrinsic_image_deref_atomic_fmax:
   case nir_intrinsic_image_deref_atomic_inc_wrap:
   case nir_intrinsic_image_deref_atomic_dec_wrap:
   case nir_intrinsic_image_atomic_add:
   case nir_intrinsic_image_atomic_imin:
   case nir_intrinsic_image_atomic_umin:
   case nir_intrinsic_image_atomic_imax:
   case nir_intrinsic_image_atomic_umax:
   case nir_intrinsic_image_atomic_and:
   case nir_intrinsic_image_atomic_or:
   case nir_intrinsic_image_atomic_xor:
   case nir_intrinsic_image_atomic_exchange:
   case nir_intrinsic_image_atomic_comp_swap:
   case nir_intrinsic_image_atomic_fadd:
   case nir_intrinsic_image_atomic_fmin:
   case nir_intrinsic_image_atomic_fmax:
   case nir_intrinsic_image_atomic_inc_wrap:
   case nir_intrinsic_image_atomic_dec_wrap:
   case nir_intrinsic_bindless_image_atomic_add:
   case nir_intrinsic_bindless_image_atomic_imin:
   case nir_intrinsic_bindless_image_atomic_umin:
   case nir_intrinsic_bindless_image_atomic_imax:
   case nir_intrinsic_bindless_image_atomic_umax:
   case nir_intrinsic_bindless_image_atomic_and:
   case nir_intrinsic_bindless_image_atomic_or:
   case nir_intrinsic_bindless_image_atomic_xor:
   case nir_intrinsic_bindless_image_atomic_exchange:
   case nir_intrinsic_bindless_image_atomic_comp_swap:
   case nir_intrinsic_bindless_image_atomic_fadd:
   case nir_intrinsic_bindless_image_atomic_fmin:
   case nir_intrinsic_bindless_image_atomic_fmax:
   case nir_intrinsic_bindless_image_atomic_inc_wrap:
   case nir_intrinsic_bindless_image_atomic_dec_wrap:
   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_comp_swap:
   case nir_intrinsic_shared_atomic_fadd:
   case nir_intrinsic_shared_atomic_fmin:
   case nir_intrinsic_shared_atomic_fmax:
   case nir_intrinsic_shared_atomic_fcomp_swap:
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
   case nir_intrinsic_global_atomic_imin:
   case nir_intrinsic_global_atomic_umin:
   case nir_intrinsic_global_atomic_imax:
   case nir_intrinsic_global_atomic_umax:
   case nir_intrinsic_global_atomic_and:
   case nir_intrinsic_global_atomic_or:
   case nir_intrinsic_global_atomic_xor:
   case nir_intrinsic_global_atomic_exchange:
   case nir_intrinsic_global_atomic_comp_swap:
   case nir_intrinsic_global_atomic_fadd:
   case nir_intrinsic_global_atomic_fmin:
   case nir_intrinsic_global_atomic_fmax:
   case nir_intrinsic_global_atomic_fcomp_swap:
   case nir_intrinsic_global_atomic_add_amd:
   case nir_intrinsic_global_atomic_imin_amd:
   case nir_intrinsic_global_atomic_umin_amd:
   case nir_intrinsic_global_atomic_imax_amd:
   case nir_intrinsic_global_atomic_umax_amd:
   case nir_intrinsic_global_atomic_and_amd:
   case nir_intrinsic_global_atomic_or_amd:
   case nir_intrinsic_global_atomic_xor_amd:
   case nir_intrinsic_global_atomic_exchange_amd:
   case nir_intrinsic_global_atomic_comp_swap_amd:
   case nir_intrinsic_global_atomic_fadd_amd:
   case nir_intrinsic_global_atomic_fmin_amd:
   case nir_intrinsic_global_atomic_fmax_amd:
   case nir_intrinsic_global_atomic_fcomp_swap_amd:
   case nir_intrinsic_global_atomic_add_2x32:
   case nir_intrinsic_global_atomic_imin_2x32:
   case nir_intrinsic_global_atomic_umin_2x32:
   case nir_intrinsic_global_atomic_imax_2x32:
   case nir_intrinsic_global_atomic_umax_2x32:
   case nir_intrinsic_global_atomic_and_2x32:
   case nir_intrinsic_global_atomic_or_2x32:
   case nir_intrinsic_global_atomic_xor_2x32:
   case nir_intrinsic_global_atomic_exchange_2x32:
   case nir_intrinsic_global_atomic_comp_swap_2x32:
   case nir_intrinsic_global_atomic_fadd_2x32:
   case nir_intrinsic_global_atomic_fmin_2x32:
   case nir_intrinsic_global_atomic_fmax_2x32:
   case nir_intrinsic_global_atomic_fcomp_swap_2x32:
   case nir_intrinsic_atomic_counter_add:
   case nir_intrinsic_atomic_counter_min:
   case nir_intrinsic_atomic_counter_max:
   case nir_intrinsic_atomic_counter_and:
   case nir_intrinsic_atomic_counter_or:
   case nir_intrinsic_atomic_counter_xor:
   case nir_intrinsic_atomic_counter_inc:
   case nir_intrinsic_atomic_counter_pre_dec:
   case nir_intrinsic_atomic_counter_post_dec:
   case nir_intrinsic_atomic_counter_exchange:
   case nir_intrinsic_atomic_counter_comp_swap:
   case nir_intrinsic_atomic_counter_add_deref:
   case nir_intrinsic_atomic_counter_min_deref:
   case nir_intrinsic_atomic_counter_max_deref:
   case nir_intrinsic_atomic_counter_and_deref:
   case nir_intrinsic_atomic_counter_or_deref:
   case nir_intrinsic_atomic_counter_xor_deref:
   case nir_intrinsic_atomic_counter_inc_deref:
   case nir_intrinsic_atomic_counter_pre_dec_deref:
   case nir_intrinsic_atomic_counter_post_dec_deref:
   case nir_intrinsic_atomic_counter_exchange_deref:
   case nir_intrinsic_atomic_counter_comp_swap_deref:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_bit_count_inclusive:
   case nir_intrinsic_write_invocation_amd:
   case nir_intrinsic_mbcnt_amd:
   case nir_intrinsic_lane_permute_16_amd:
   case nir_intrinsic_elect:
   case nir_intrinsic_load_tlb_color_v3d:
   case nir_intrinsic_load_tess_rel_patch_id_amd:
   case nir_intrinsic_load_gs_vertex_offset_amd:
   case nir_intrinsic_is_subgroup_invocation_lt_amd:
   case nir_intrinsic_load_packed_passthrough_primitive_amd:
   case nir_intrinsic_load_initial_edgeflags_amd:
   case nir_intrinsic_gds_atomic_add_amd:
   case nir_intrinsic_buffer_atomic_add_amd:
   case nir_intrinsic_load_rt_arg_scratch_offset_amd:
   case nir_intrinsic_load_intersection_opaque_amd:
   case nir_intrinsic_load_vector_arg_amd:
   case nir_intrinsic_load_btd_stack_id_intel:
   case nir_intrinsic_load_topology_id_intel:
   case nir_intrinsic_load_scratch_base_ptr:
   case nir_intrinsic_ordered_xfb_counter_add_amd:
   case nir_intrinsic_load_stack:
   case nir_intrinsic_load_ray_launch_id:
   case nir_intrinsic_load_ray_instance_custom_index:
   case nir_intrinsic_load_ray_geometry_index:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_t_max:
   case nir_intrinsic_load_ray_object_to_world:
   case nir_intrinsic_load_ray_world_to_object:
   case nir_intrinsic_load_ray_hit_kind:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_cull_mask:
   case nir_intrinsic_report_ray_intersection:
   case nir_intrinsic_rq_proceed:
   case nir_intrinsic_rq_load:
      is_divergent = true;
      break;

   default:
#ifdef NDEBUG
      is_divergent = true;
      break;
#else
      nir_print_instr(&instr->instr, stderr);
      unreachable("\nNIR divergence analysis: Unhandled intrinsic.");
#endif
   }

   instr->dest.ssa.divergent = is_divergent;
   return is_divergent;
}

static bool
visit_tex(nir_tex_instr *instr)
{
   if (instr->dest.ssa.divergent)
      return false;

   bool is_divergent = false;

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_sampler_deref:
      case nir_tex_src_sampler_handle:
      case nir_tex_src_sampler_offset:
         is_divergent |= instr->src[i].src.ssa->divergent &&
                         instr->sampler_non_uniform;
         break;
      case nir_tex_src_texture_deref:
      case nir_tex_src_texture_handle:
      case nir_tex_src_texture_offset:
         is_divergent |= instr->src[i].src.ssa->divergent &&
                         instr->texture_non_uniform;
         break;
      default:
         is_divergent |= instr->src[i].src.ssa->divergent;
         break;
      }
   }

   instr->dest.ssa.divergent = is_divergent;
   return is_divergent;
}

static bool
visit_load_const(nir_load_const_instr *instr)
{
   return false;
}

static bool
visit_ssa_undef(nir_ssa_undef_instr *instr)
{
   return false;
}

static bool
nir_variable_mode_is_uniform(nir_variable_mode mode) {
   switch (mode) {
   case nir_var_uniform:
   case nir_var_mem_ubo:
   case nir_var_mem_ssbo:
   case nir_var_mem_shared:
   case nir_var_mem_task_payload:
   case nir_var_mem_global:
   case nir_var_image:
      return true;
   default:
      return false;
   }
}

static bool
nir_variable_is_uniform(nir_shader *shader, nir_variable *var)
{
   if (nir_variable_mode_is_uniform(var->data.mode))
      return true;

   nir_divergence_options options = shader->options->divergence_analysis_options;
   gl_shader_stage stage = shader->info.stage;

   if (stage == MESA_SHADER_FRAGMENT &&
       (options & nir_divergence_single_prim_per_subgroup) &&
       var->data.mode == nir_var_shader_in &&
       var->data.interpolation == INTERP_MODE_FLAT)
      return true;

   if (stage == MESA_SHADER_TESS_CTRL &&
       (options & nir_divergence_single_patch_per_tcs_subgroup) &&
       var->data.mode == nir_var_shader_out && var->data.patch)
      return true;

   if (stage == MESA_SHADER_TESS_EVAL &&
       (options & nir_divergence_single_patch_per_tes_subgroup) &&
       var->data.mode == nir_var_shader_in && var->data.patch)
      return true;

   return false;
}

static bool
visit_deref(nir_shader *shader, nir_deref_instr *deref)
{
   if (deref->dest.ssa.divergent)
      return false;

   bool is_divergent = false;
   switch (deref->deref_type) {
   case nir_deref_type_var:
      is_divergent = !nir_variable_is_uniform(shader, deref->var);
      break;
   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array:
      is_divergent = deref->arr.index.ssa->divergent;
      FALLTHROUGH;
   case nir_deref_type_struct:
   case nir_deref_type_array_wildcard:
      is_divergent |= deref->parent.ssa->divergent;
      break;
   case nir_deref_type_cast:
      is_divergent = !nir_variable_mode_is_uniform(deref->var->data.mode) ||
                     deref->parent.ssa->divergent;
      break;
   }

   deref->dest.ssa.divergent = is_divergent;
   return is_divergent;
}

static bool
visit_jump(nir_jump_instr *jump, struct divergence_state *state)
{
   switch (jump->type) {
   case nir_jump_continue:
      if (state->divergent_loop_continue)
         return false;
      if (state->divergent_loop_cf)
         state->divergent_loop_continue = true;
      return state->divergent_loop_continue;
   case nir_jump_break:
      if (state->divergent_loop_break)
         return false;
      if (state->divergent_loop_cf)
         state->divergent_loop_break = true;
      return state->divergent_loop_break;
   case nir_jump_halt:
      /* This totally kills invocations so it doesn't add divergence */
      break;
   case nir_jump_return:
      unreachable("NIR divergence analysis: Unsupported return instruction.");
      break;
   case nir_jump_goto:
   case nir_jump_goto_if:
      unreachable("NIR divergence analysis: Unsupported goto_if instruction.");
      break;
   }
   return false;
}

static bool
set_ssa_def_not_divergent(nir_ssa_def *def, UNUSED void *_state)
{
   def->divergent = false;
   return true;
}

static bool
update_instr_divergence(nir_shader *shader, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return visit_alu(nir_instr_as_alu(instr));
   case nir_instr_type_intrinsic:
      return visit_intrinsic(shader, nir_instr_as_intrinsic(instr));
   case nir_instr_type_tex:
      return visit_tex(nir_instr_as_tex(instr));
   case nir_instr_type_load_const:
      return visit_load_const(nir_instr_as_load_const(instr));
   case nir_instr_type_ssa_undef:
      return visit_ssa_undef(nir_instr_as_ssa_undef(instr));
   case nir_instr_type_deref:
      return visit_deref(shader, nir_instr_as_deref(instr));
   case nir_instr_type_jump:
   case nir_instr_type_phi:
   case nir_instr_type_call:
   case nir_instr_type_parallel_copy:
   default:
      unreachable("NIR divergence analysis: Unsupported instruction type.");
   }
}

static bool
visit_block(nir_block *block, struct divergence_state *state)
{
   bool has_changed = false;

   nir_foreach_instr(instr, block) {
      /* phis are handled when processing the branches */
      if (instr->type == nir_instr_type_phi)
         continue;

      if (state->first_visit)
         nir_foreach_ssa_def(instr, set_ssa_def_not_divergent, NULL);

      if (instr->type == nir_instr_type_jump)
         has_changed |= visit_jump(nir_instr_as_jump(instr), state);
      else
         has_changed |= update_instr_divergence(state->shader, instr);
   }

   return has_changed;
}

/* There are 3 types of phi instructions:
 * (1) gamma: represent the joining point of different paths
 *     created by an “if-then-else” branch.
 *     The resulting value is divergent if the branch condition
 *     or any of the source values is divergent. */
static bool
visit_if_merge_phi(nir_phi_instr *phi, bool if_cond_divergent)
{
   if (phi->dest.ssa.divergent)
      return false;

   unsigned defined_srcs = 0;
   nir_foreach_phi_src(src, phi) {
      /* if any source value is divergent, the resulting value is divergent */
      if (src->src.ssa->divergent) {
         phi->dest.ssa.divergent = true;
         return true;
      }
      if (src->src.ssa->parent_instr->type != nir_instr_type_ssa_undef) {
         defined_srcs++;
      }
   }

   /* if the condition is divergent and two sources defined, the definition is divergent */
   if (defined_srcs > 1 && if_cond_divergent) {
      phi->dest.ssa.divergent = true;
      return true;
   }

   return false;
}

/* There are 3 types of phi instructions:
 * (2) mu: which only exist at loop headers,
 *     merge initial and loop-carried values.
 *     The resulting value is divergent if any source value
 *     is divergent or a divergent loop continue condition
 *     is associated with a different ssa-def. */
static bool
visit_loop_header_phi(nir_phi_instr *phi, nir_block *preheader, bool divergent_continue)
{
   if (phi->dest.ssa.divergent)
      return false;

   nir_ssa_def* same = NULL;
   nir_foreach_phi_src(src, phi) {
      /* if any source value is divergent, the resulting value is divergent */
      if (src->src.ssa->divergent) {
         phi->dest.ssa.divergent = true;
         return true;
      }
      /* if this loop is uniform, we're done here */
      if (!divergent_continue)
         continue;
      /* skip the loop preheader */
      if (src->pred == preheader)
         continue;
      /* skip undef values */
      if (nir_src_is_undef(src->src))
         continue;

      /* check if all loop-carried values are from the same ssa-def */
      if (!same)
         same = src->src.ssa;
      else if (same != src->src.ssa) {
         phi->dest.ssa.divergent = true;
         return true;
      }
   }

   return false;
}

/* There are 3 types of phi instructions:
 * (3) eta: represent values that leave a loop.
 *     The resulting value is divergent if the source value is divergent
 *     or any loop exit condition is divergent for a value which is
 *     not loop-invariant.
 *     (note: there should be no phi for loop-invariant variables.) */
static bool
visit_loop_exit_phi(nir_phi_instr *phi, bool divergent_break)
{
   if (phi->dest.ssa.divergent)
      return false;

   if (divergent_break) {
      phi->dest.ssa.divergent = true;
      return true;
   }

   /* if any source value is divergent, the resulting value is divergent */
   nir_foreach_phi_src(src, phi) {
      if (src->src.ssa->divergent) {
         phi->dest.ssa.divergent = true;
         return true;
      }
   }

   return false;
}

static bool
visit_if(nir_if *if_stmt, struct divergence_state *state)
{
   bool progress = false;

   struct divergence_state then_state = *state;
   then_state.divergent_loop_cf |= if_stmt->condition.ssa->divergent;
   progress |= visit_cf_list(&if_stmt->then_list, &then_state);

   struct divergence_state else_state = *state;
   else_state.divergent_loop_cf |= if_stmt->condition.ssa->divergent;
   progress |= visit_cf_list(&if_stmt->else_list, &else_state);

   /* handle phis after the IF */
   nir_foreach_instr(instr, nir_cf_node_cf_tree_next(&if_stmt->cf_node)) {
      if (instr->type != nir_instr_type_phi)
         break;

      if (state->first_visit)
         nir_instr_as_phi(instr)->dest.ssa.divergent = false;
      progress |= visit_if_merge_phi(nir_instr_as_phi(instr),
                                     if_stmt->condition.ssa->divergent);
   }

   /* join loop divergence information from both branch legs */
   state->divergent_loop_continue |= then_state.divergent_loop_continue ||
                                     else_state.divergent_loop_continue;
   state->divergent_loop_break |= then_state.divergent_loop_break ||
                                  else_state.divergent_loop_break;

   /* A divergent continue makes succeeding loop CF divergent:
    * not all loop-active invocations participate in the remaining loop-body
    * which means that a following break might be taken by some invocations, only */
   state->divergent_loop_cf |= state->divergent_loop_continue;

   return progress;
}

static bool
visit_loop(nir_loop *loop, struct divergence_state *state)
{
   bool progress = false;
   nir_block *loop_header = nir_loop_first_block(loop);
   nir_block *loop_preheader = nir_block_cf_tree_prev(loop_header);

   /* handle loop header phis first: we have no knowledge yet about
    * the loop's control flow or any loop-carried sources. */
   nir_foreach_instr(instr, loop_header) {
      if (instr->type != nir_instr_type_phi)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);
      if (!state->first_visit && phi->dest.ssa.divergent)
         continue;

      nir_foreach_phi_src(src, phi) {
         if (src->pred == loop_preheader) {
            phi->dest.ssa.divergent = src->src.ssa->divergent;
            break;
         }
      }
      progress |= phi->dest.ssa.divergent;
   }

   /* setup loop state */
   struct divergence_state loop_state = *state;
   loop_state.divergent_loop_cf = false;
   loop_state.divergent_loop_continue = false;
   loop_state.divergent_loop_break = false;

   /* process loop body until no further changes are made */
   bool repeat;
   do {
      progress |= visit_cf_list(&loop->body, &loop_state);
      repeat = false;

      /* revisit loop header phis to see if something has changed */
      nir_foreach_instr(instr, loop_header) {
         if (instr->type != nir_instr_type_phi)
            break;

         repeat |= visit_loop_header_phi(nir_instr_as_phi(instr),
                                         loop_preheader,
                                         loop_state.divergent_loop_continue);
      }

      loop_state.divergent_loop_cf = false;
      loop_state.first_visit = false;
   } while (repeat);

   /* handle phis after the loop */
   nir_foreach_instr(instr, nir_cf_node_cf_tree_next(&loop->cf_node)) {
      if (instr->type != nir_instr_type_phi)
         break;

      if (state->first_visit)
         nir_instr_as_phi(instr)->dest.ssa.divergent = false;
      progress |= visit_loop_exit_phi(nir_instr_as_phi(instr),
                                      loop_state.divergent_loop_break);
   }

   loop->divergent = (loop_state.divergent_loop_break || loop_state.divergent_loop_continue);

   return progress;
}

static bool
visit_cf_list(struct exec_list *list, struct divergence_state *state)
{
   bool has_changed = false;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         has_changed |= visit_block(nir_cf_node_as_block(node), state);
         break;
      case nir_cf_node_if:
         has_changed |= visit_if(nir_cf_node_as_if(node), state);
         break;
      case nir_cf_node_loop:
         has_changed |= visit_loop(nir_cf_node_as_loop(node), state);
         break;
      case nir_cf_node_function:
         unreachable("NIR divergence analysis: Unsupported cf_node type.");
      }
   }

   return has_changed;
}

void
nir_divergence_analysis(nir_shader *shader)
{
   shader->info.divergence_analysis_run = true;

   struct divergence_state state = {
      .stage = shader->info.stage,
      .shader = shader,
      .divergent_loop_cf = false,
      .divergent_loop_continue = false,
      .divergent_loop_break = false,
      .first_visit = true,
   };

   visit_cf_list(&nir_shader_get_entrypoint(shader)->body, &state);
}

bool nir_update_instr_divergence(nir_shader *shader, nir_instr *instr)
{
   nir_foreach_ssa_def(instr, set_ssa_def_not_divergent, NULL);

   if (instr->type == nir_instr_type_phi) {
      nir_cf_node *prev = nir_cf_node_prev(&instr->block->cf_node);
      /* can only update gamma/if phis */
      if (!prev || prev->type != nir_cf_node_if)
         return false;

      nir_if *nif = nir_cf_node_as_if(prev);

      visit_if_merge_phi(nir_instr_as_phi(instr), nir_src_is_divergent(nif->condition));
      return true;
   }

   update_instr_divergence(shader, instr);
   return true;
}


bool
nir_has_divergent_loop(nir_shader *shader)
{
   bool divergent_loop = false;
   nir_function_impl *func = nir_shader_get_entrypoint(shader);

   foreach_list_typed(nir_cf_node, node, node, &func->body) {
      if (node->type == nir_cf_node_loop && nir_cf_node_as_loop(node)->divergent) {
         divergent_loop = true;
         break;
      }
   }

   return divergent_loop;
}

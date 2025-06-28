// Copyright 2015-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_EXTENSION_INSPECTION_HPP
#define VULKAN_EXTENSION_INSPECTION_HPP

#if defined( VULKAN_HPP_ENABLE_STD_MODULE ) && defined( VULKAN_HPP_STD_MODULE )
import VULKAN_HPP_STD_MODULE;
#else
#  include <map>
#  include <set>
#  include <string>
#  include <vector>
#  include <vulkan/vulkan.hpp>
#endif

namespace VULKAN_HPP_NAMESPACE
{
  //======================================
  //=== Extension inspection functions ===
  //======================================

  std::set<std::string> const &                                        getDeviceExtensions();
  std::set<std::string> const &                                        getInstanceExtensions();
  std::map<std::string, std::string> const &                           getDeprecatedExtensions();
  std::map<std::string, std::vector<std::vector<std::string>>> const & getExtensionDepends( std::string const & extension );
  std::pair<bool, std::vector<std::vector<std::string>> const &>       getExtensionDepends( std::string const & version, std::string const & extension );
  std::map<std::string, std::string> const &                           getObsoletedExtensions();
  std::map<std::string, std::string> const &                           getPromotedExtensions();
  VULKAN_HPP_CONSTEXPR_20 std::string getExtensionDeprecatedBy( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 std::string getExtensionObsoletedBy( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 std::string getExtensionPromotedTo( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 bool        isDeprecatedExtension( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 bool        isDeviceExtension( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 bool        isInstanceExtension( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 bool        isObsoletedExtension( std::string const & extension );
  VULKAN_HPP_CONSTEXPR_20 bool        isPromotedExtension( std::string const & extension );

  //=====================================================
  //=== Extension inspection function implementations ===
  //=====================================================

  VULKAN_HPP_INLINE std::map<std::string, std::string> const & getDeprecatedExtensions()
  {
    static const std::map<std::string, std::string> deprecatedExtensions = {
      { "VK_EXT_debug_report", "VK_EXT_debug_utils" },
      { "VK_NV_glsl_shader", "" },
      { "VK_NV_dedicated_allocation", "VK_KHR_dedicated_allocation" },
      { "VK_AMD_gpu_shader_half_float", "VK_KHR_shader_float16_int8" },
      { "VK_IMG_format_pvrtc", "" },
      { "VK_NV_external_memory_capabilities", "VK_KHR_external_memory_capabilities" },
      { "VK_NV_external_memory", "VK_KHR_external_memory" },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_NV_external_memory_win32", "VK_KHR_external_memory_win32" },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_EXT_validation_flags", "VK_EXT_layer_settings" },
      { "VK_EXT_shader_subgroup_ballot", "VK_VERSION_1_2" },
      { "VK_EXT_shader_subgroup_vote", "VK_VERSION_1_1" },
#if defined( VK_USE_PLATFORM_IOS_MVK )
      { "VK_MVK_ios_surface", "VK_EXT_metal_surface" },
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
      { "VK_MVK_macos_surface", "VK_EXT_metal_surface" },
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
      { "VK_AMD_gpu_shader_int16", "VK_KHR_shader_float16_int8" },
      { "VK_NV_ray_tracing", "VK_KHR_ray_tracing_pipeline" },
      { "VK_EXT_buffer_device_address", "VK_KHR_buffer_device_address" },
      { "VK_EXT_validation_features", "VK_EXT_layer_settings" },
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      { "VK_NV_displacement_micromap", "VK_NV_cluster_acceleration_structure" }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
    return deprecatedExtensions;
  }

  VULKAN_HPP_INLINE std::set<std::string> const & getDeviceExtensions()
  {
    static const std::set<std::string> deviceExtensions = {
      "VK_KHR_swapchain",
      "VK_KHR_display_swapchain",
      "VK_NV_glsl_shader",
      "VK_EXT_depth_range_unrestricted",
      "VK_KHR_sampler_mirror_clamp_to_edge",
      "VK_IMG_filter_cubic",
      "VK_AMD_rasterization_order",
      "VK_AMD_shader_trinary_minmax",
      "VK_AMD_shader_explicit_vertex_parameter",
      "VK_EXT_debug_marker",
      "VK_KHR_video_queue",
      "VK_KHR_video_decode_queue",
      "VK_AMD_gcn_shader",
      "VK_NV_dedicated_allocation",
      "VK_EXT_transform_feedback",
      "VK_NVX_binary_import",
      "VK_NVX_image_view_handle",
      "VK_AMD_draw_indirect_count",
      "VK_AMD_negative_viewport_height",
      "VK_AMD_gpu_shader_half_float",
      "VK_AMD_shader_ballot",
      "VK_KHR_video_encode_h264",
      "VK_KHR_video_encode_h265",
      "VK_KHR_video_decode_h264",
      "VK_AMD_texture_gather_bias_lod",
      "VK_AMD_shader_info",
      "VK_KHR_dynamic_rendering",
      "VK_AMD_shader_image_load_store_lod",
      "VK_NV_corner_sampled_image",
      "VK_KHR_multiview",
      "VK_IMG_format_pvrtc",
      "VK_NV_external_memory",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_NV_external_memory_win32",
      "VK_NV_win32_keyed_mutex",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_KHR_device_group",
      "VK_KHR_shader_draw_parameters",
      "VK_EXT_shader_subgroup_ballot",
      "VK_EXT_shader_subgroup_vote",
      "VK_EXT_texture_compression_astc_hdr",
      "VK_EXT_astc_decode_mode",
      "VK_EXT_pipeline_robustness",
      "VK_KHR_maintenance1",
      "VK_KHR_external_memory",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_KHR_external_memory_win32",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_KHR_external_memory_fd",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_KHR_win32_keyed_mutex",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_KHR_external_semaphore",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_KHR_external_semaphore_win32",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_KHR_external_semaphore_fd",
      "VK_KHR_push_descriptor",
      "VK_EXT_conditional_rendering",
      "VK_KHR_shader_float16_int8",
      "VK_KHR_16bit_storage",
      "VK_KHR_incremental_present",
      "VK_KHR_descriptor_update_template",
      "VK_NV_clip_space_w_scaling",
      "VK_EXT_display_control",
      "VK_GOOGLE_display_timing",
      "VK_NV_sample_mask_override_coverage",
      "VK_NV_geometry_shader_passthrough",
      "VK_NV_viewport_array2",
      "VK_NVX_multiview_per_view_attributes",
      "VK_NV_viewport_swizzle",
      "VK_EXT_discard_rectangles",
      "VK_EXT_conservative_rasterization",
      "VK_EXT_depth_clip_enable",
      "VK_EXT_hdr_metadata",
      "VK_KHR_imageless_framebuffer",
      "VK_KHR_create_renderpass2",
      "VK_IMG_relaxed_line_rasterization",
      "VK_KHR_shared_presentable_image",
      "VK_KHR_external_fence",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_KHR_external_fence_win32",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_KHR_external_fence_fd",
      "VK_KHR_performance_query",
      "VK_KHR_maintenance2",
      "VK_KHR_variable_pointers",
      "VK_EXT_external_memory_dma_buf",
      "VK_EXT_queue_family_foreign",
      "VK_KHR_dedicated_allocation",
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      "VK_ANDROID_external_memory_android_hardware_buffer",
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      "VK_EXT_sampler_filter_minmax",
      "VK_KHR_storage_buffer_storage_class",
      "VK_AMD_gpu_shader_int16",
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      "VK_AMDX_shader_enqueue",
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      "VK_AMD_mixed_attachment_samples",
      "VK_AMD_shader_fragment_mask",
      "VK_EXT_inline_uniform_block",
      "VK_EXT_shader_stencil_export",
      "VK_KHR_shader_bfloat16",
      "VK_EXT_sample_locations",
      "VK_KHR_relaxed_block_layout",
      "VK_KHR_get_memory_requirements2",
      "VK_KHR_image_format_list",
      "VK_EXT_blend_operation_advanced",
      "VK_NV_fragment_coverage_to_color",
      "VK_KHR_acceleration_structure",
      "VK_KHR_ray_tracing_pipeline",
      "VK_KHR_ray_query",
      "VK_NV_framebuffer_mixed_samples",
      "VK_NV_fill_rectangle",
      "VK_NV_shader_sm_builtins",
      "VK_EXT_post_depth_coverage",
      "VK_KHR_sampler_ycbcr_conversion",
      "VK_KHR_bind_memory2",
      "VK_EXT_image_drm_format_modifier",
      "VK_EXT_validation_cache",
      "VK_EXT_descriptor_indexing",
      "VK_EXT_shader_viewport_index_layer",
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      "VK_KHR_portability_subset",
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      "VK_NV_shading_rate_image",
      "VK_NV_ray_tracing",
      "VK_NV_representative_fragment_test",
      "VK_KHR_maintenance3",
      "VK_KHR_draw_indirect_count",
      "VK_EXT_filter_cubic",
      "VK_QCOM_render_pass_shader_resolve",
      "VK_EXT_global_priority",
      "VK_KHR_shader_subgroup_extended_types",
      "VK_KHR_8bit_storage",
      "VK_EXT_external_memory_host",
      "VK_AMD_buffer_marker",
      "VK_KHR_shader_atomic_int64",
      "VK_KHR_shader_clock",
      "VK_AMD_pipeline_compiler_control",
      "VK_EXT_calibrated_timestamps",
      "VK_AMD_shader_core_properties",
      "VK_KHR_video_decode_h265",
      "VK_KHR_global_priority",
      "VK_AMD_memory_overallocation_behavior",
      "VK_EXT_vertex_attribute_divisor",
#if defined( VK_USE_PLATFORM_GGP )
      "VK_GGP_frame_token",
#endif /*VK_USE_PLATFORM_GGP*/
      "VK_EXT_pipeline_creation_feedback",
      "VK_KHR_driver_properties",
      "VK_KHR_shader_float_controls",
      "VK_NV_shader_subgroup_partitioned",
      "VK_KHR_depth_stencil_resolve",
      "VK_KHR_swapchain_mutable_format",
      "VK_NV_compute_shader_derivatives",
      "VK_NV_mesh_shader",
      "VK_NV_fragment_shader_barycentric",
      "VK_NV_shader_image_footprint",
      "VK_NV_scissor_exclusive",
      "VK_NV_device_diagnostic_checkpoints",
      "VK_KHR_timeline_semaphore",
      "VK_INTEL_shader_integer_functions2",
      "VK_INTEL_performance_query",
      "VK_KHR_vulkan_memory_model",
      "VK_EXT_pci_bus_info",
      "VK_AMD_display_native_hdr",
      "VK_KHR_shader_terminate_invocation",
      "VK_EXT_fragment_density_map",
      "VK_EXT_scalar_block_layout",
      "VK_GOOGLE_hlsl_functionality1",
      "VK_GOOGLE_decorate_string",
      "VK_EXT_subgroup_size_control",
      "VK_KHR_fragment_shading_rate",
      "VK_AMD_shader_core_properties2",
      "VK_AMD_device_coherent_memory",
      "VK_KHR_dynamic_rendering_local_read",
      "VK_EXT_shader_image_atomic_int64",
      "VK_KHR_shader_quad_control",
      "VK_KHR_spirv_1_4",
      "VK_EXT_memory_budget",
      "VK_EXT_memory_priority",
      "VK_NV_dedicated_allocation_image_aliasing",
      "VK_KHR_separate_depth_stencil_layouts",
      "VK_EXT_buffer_device_address",
      "VK_EXT_tooling_info",
      "VK_EXT_separate_stencil_usage",
      "VK_KHR_present_wait",
      "VK_NV_cooperative_matrix",
      "VK_NV_coverage_reduction_mode",
      "VK_EXT_fragment_shader_interlock",
      "VK_EXT_ycbcr_image_arrays",
      "VK_KHR_uniform_buffer_standard_layout",
      "VK_EXT_provoking_vertex",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_EXT_full_screen_exclusive",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_KHR_buffer_device_address",
      "VK_EXT_line_rasterization",
      "VK_EXT_shader_atomic_float",
      "VK_EXT_host_query_reset",
      "VK_EXT_index_type_uint8",
      "VK_EXT_extended_dynamic_state",
      "VK_KHR_deferred_host_operations",
      "VK_KHR_pipeline_executable_properties",
      "VK_EXT_host_image_copy",
      "VK_KHR_map_memory2",
      "VK_EXT_map_memory_placed",
      "VK_EXT_shader_atomic_float2",
      "VK_EXT_swapchain_maintenance1",
      "VK_EXT_shader_demote_to_helper_invocation",
      "VK_NV_device_generated_commands",
      "VK_NV_inherited_viewport_scissor",
      "VK_KHR_shader_integer_dot_product",
      "VK_EXT_texel_buffer_alignment",
      "VK_QCOM_render_pass_transform",
      "VK_EXT_depth_bias_control",
      "VK_EXT_device_memory_report",
      "VK_EXT_robustness2",
      "VK_EXT_custom_border_color",
      "VK_GOOGLE_user_type",
      "VK_KHR_pipeline_library",
      "VK_NV_present_barrier",
      "VK_KHR_shader_non_semantic_info",
      "VK_KHR_present_id",
      "VK_EXT_private_data",
      "VK_EXT_pipeline_creation_cache_control",
      "VK_KHR_video_encode_queue",
      "VK_NV_device_diagnostics_config",
      "VK_QCOM_render_pass_store_ops",
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      "VK_NV_cuda_kernel_launch",
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      "VK_QCOM_tile_shading",
      "VK_NV_low_latency",
#if defined( VK_USE_PLATFORM_METAL_EXT )
      "VK_EXT_metal_objects",
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      "VK_KHR_synchronization2",
      "VK_EXT_descriptor_buffer",
      "VK_EXT_graphics_pipeline_library",
      "VK_AMD_shader_early_and_late_fragment_tests",
      "VK_KHR_fragment_shader_barycentric",
      "VK_KHR_shader_subgroup_uniform_control_flow",
      "VK_KHR_zero_initialize_workgroup_memory",
      "VK_NV_fragment_shading_rate_enums",
      "VK_NV_ray_tracing_motion_blur",
      "VK_EXT_mesh_shader",
      "VK_EXT_ycbcr_2plane_444_formats",
      "VK_EXT_fragment_density_map2",
      "VK_QCOM_rotated_copy_commands",
      "VK_EXT_image_robustness",
      "VK_KHR_workgroup_memory_explicit_layout",
      "VK_KHR_copy_commands2",
      "VK_EXT_image_compression_control",
      "VK_EXT_attachment_feedback_loop_layout",
      "VK_EXT_4444_formats",
      "VK_EXT_device_fault",
      "VK_ARM_rasterization_order_attachment_access",
      "VK_EXT_rgba10x6_formats",
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_NV_acquire_winrt_display",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_VALVE_mutable_descriptor_type",
      "VK_EXT_vertex_input_dynamic_state",
      "VK_EXT_physical_device_drm",
      "VK_EXT_device_address_binding_report",
      "VK_EXT_depth_clip_control",
      "VK_EXT_primitive_topology_list_restart",
      "VK_KHR_format_feature_flags2",
      "VK_EXT_present_mode_fifo_latest_ready",
#if defined( VK_USE_PLATFORM_FUCHSIA )
      "VK_FUCHSIA_external_memory",
      "VK_FUCHSIA_external_semaphore",
      "VK_FUCHSIA_buffer_collection",
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      "VK_HUAWEI_subpass_shading",
      "VK_HUAWEI_invocation_mask",
      "VK_NV_external_memory_rdma",
      "VK_EXT_pipeline_properties",
      "VK_EXT_frame_boundary",
      "VK_EXT_multisampled_render_to_single_sampled",
      "VK_EXT_extended_dynamic_state2",
      "VK_EXT_color_write_enable",
      "VK_EXT_primitives_generated_query",
      "VK_KHR_ray_tracing_maintenance1",
      "VK_EXT_global_priority_query",
      "VK_EXT_image_view_min_lod",
      "VK_EXT_multi_draw",
      "VK_EXT_image_2d_view_of_3d",
      "VK_EXT_shader_tile_image",
      "VK_EXT_opacity_micromap",
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      "VK_NV_displacement_micromap",
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      "VK_EXT_load_store_op_none",
      "VK_HUAWEI_cluster_culling_shader",
      "VK_EXT_border_color_swizzle",
      "VK_EXT_pageable_device_local_memory",
      "VK_KHR_maintenance4",
      "VK_ARM_shader_core_properties",
      "VK_KHR_shader_subgroup_rotate",
      "VK_ARM_scheduling_controls",
      "VK_EXT_image_sliced_view_of_3d",
      "VK_VALVE_descriptor_set_host_mapping",
      "VK_EXT_depth_clamp_zero_one",
      "VK_EXT_non_seamless_cube_map",
      "VK_ARM_render_pass_striped",
      "VK_QCOM_fragment_density_map_offset",
      "VK_NV_copy_memory_indirect",
      "VK_NV_memory_decompression",
      "VK_NV_device_generated_commands_compute",
      "VK_NV_ray_tracing_linear_swept_spheres",
      "VK_NV_linear_color_attachment",
      "VK_KHR_shader_maximal_reconvergence",
      "VK_EXT_image_compression_control_swapchain",
      "VK_QCOM_image_processing",
      "VK_EXT_nested_command_buffer",
      "VK_EXT_external_memory_acquire_unmodified",
      "VK_EXT_extended_dynamic_state3",
      "VK_EXT_subpass_merge_feedback",
      "VK_EXT_shader_module_identifier",
      "VK_EXT_rasterization_order_attachment_access",
      "VK_NV_optical_flow",
      "VK_EXT_legacy_dithering",
      "VK_EXT_pipeline_protected_access",
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      "VK_ANDROID_external_format_resolve",
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      "VK_KHR_maintenance5",
      "VK_AMD_anti_lag",
      "VK_KHR_ray_tracing_position_fetch",
      "VK_EXT_shader_object",
      "VK_KHR_pipeline_binary",
      "VK_QCOM_tile_properties",
      "VK_SEC_amigo_profiling",
      "VK_QCOM_multiview_per_view_viewports",
      "VK_NV_ray_tracing_invocation_reorder",
      "VK_NV_cooperative_vector",
      "VK_NV_extended_sparse_address_space",
      "VK_EXT_mutable_descriptor_type",
      "VK_EXT_legacy_vertex_attributes",
      "VK_ARM_shader_core_builtins",
      "VK_EXT_pipeline_library_group_handles",
      "VK_EXT_dynamic_rendering_unused_attachments",
      "VK_NV_low_latency2",
      "VK_KHR_cooperative_matrix",
      "VK_QCOM_multiview_per_view_render_areas",
      "VK_KHR_compute_shader_derivatives",
      "VK_KHR_video_decode_av1",
      "VK_KHR_video_encode_av1",
      "VK_KHR_video_maintenance1",
      "VK_NV_per_stage_descriptor_set",
      "VK_QCOM_image_processing2",
      "VK_QCOM_filter_cubic_weights",
      "VK_QCOM_ycbcr_degamma",
      "VK_QCOM_filter_cubic_clamp",
      "VK_EXT_attachment_feedback_loop_dynamic_state",
      "VK_KHR_vertex_attribute_divisor",
      "VK_KHR_load_store_op_none",
      "VK_KHR_shader_float_controls2",
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      "VK_QNX_external_memory_screen_buffer",
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      "VK_MSFT_layered_driver",
      "VK_KHR_index_type_uint8",
      "VK_KHR_line_rasterization",
      "VK_KHR_calibrated_timestamps",
      "VK_KHR_shader_expect_assume",
      "VK_KHR_maintenance6",
      "VK_NV_descriptor_pool_overallocation",
      "VK_QCOM_tile_memory_heap",
      "VK_KHR_video_encode_quantization_map",
      "VK_NV_raw_access_chains",
      "VK_NV_external_compute_queue",
      "VK_KHR_shader_relaxed_extended_instruction",
      "VK_NV_command_buffer_inheritance",
      "VK_KHR_maintenance7",
      "VK_NV_shader_atomic_float16_vector",
      "VK_EXT_shader_replicated_composites",
      "VK_NV_ray_tracing_validation",
      "VK_NV_cluster_acceleration_structure",
      "VK_NV_partitioned_acceleration_structure",
      "VK_EXT_device_generated_commands",
      "VK_KHR_maintenance8",
      "VK_MESA_image_alignment_control",
      "VK_EXT_depth_clamp_control",
      "VK_KHR_video_maintenance2",
      "VK_HUAWEI_hdr_vivid",
      "VK_NV_cooperative_matrix2",
      "VK_ARM_pipeline_opacity_micromap",
#if defined( VK_USE_PLATFORM_METAL_EXT )
      "VK_EXT_external_memory_metal",
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      "VK_KHR_depth_clamp_zero_one",
      "VK_EXT_vertex_attribute_robustness",
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      "VK_NV_present_metering",
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      "VK_EXT_fragment_density_map_offset"
    };
    return deviceExtensions;
  }

  VULKAN_HPP_INLINE std::set<std::string> const & getInstanceExtensions()
  {
    static const std::set<std::string> instanceExtensions = {
      "VK_KHR_surface",
      "VK_KHR_display",
#if defined( VK_USE_PLATFORM_XLIB_KHR )
      "VK_KHR_xlib_surface",
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
      "VK_KHR_xcb_surface",
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      "VK_KHR_wayland_surface",
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      "VK_KHR_android_surface",
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      "VK_KHR_win32_surface",
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      "VK_EXT_debug_report",
#if defined( VK_USE_PLATFORM_GGP )
      "VK_GGP_stream_descriptor_surface",
#endif /*VK_USE_PLATFORM_GGP*/
      "VK_NV_external_memory_capabilities",
      "VK_KHR_get_physical_device_properties2",
      "VK_EXT_validation_flags",
#if defined( VK_USE_PLATFORM_VI_NN )
      "VK_NN_vi_surface",
#endif /*VK_USE_PLATFORM_VI_NN*/
      "VK_KHR_device_group_creation",
      "VK_KHR_external_memory_capabilities",
      "VK_KHR_external_semaphore_capabilities",
      "VK_EXT_direct_mode_display",
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      "VK_EXT_acquire_xlib_display",
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
      "VK_EXT_display_surface_counter",
      "VK_EXT_swapchain_colorspace",
      "VK_KHR_external_fence_capabilities",
      "VK_KHR_get_surface_capabilities2",
      "VK_KHR_get_display_properties2",
#if defined( VK_USE_PLATFORM_IOS_MVK )
      "VK_MVK_ios_surface",
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
      "VK_MVK_macos_surface",
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
      "VK_EXT_debug_utils",
#if defined( VK_USE_PLATFORM_FUCHSIA )
      "VK_FUCHSIA_imagepipe_surface",
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
      "VK_EXT_metal_surface",
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      "VK_KHR_surface_protected_capabilities",
      "VK_EXT_validation_features",
      "VK_EXT_headless_surface",
      "VK_EXT_surface_maintenance1",
      "VK_EXT_acquire_drm_display",
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      "VK_EXT_directfb_surface",
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      "VK_QNX_screen_surface",
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      "VK_KHR_portability_enumeration",
      "VK_GOOGLE_surfaceless_query",
      "VK_LUNARG_direct_driver_loading",
      "VK_EXT_layer_settings",
      "VK_NV_display_stereo"
    };
    return instanceExtensions;
  }

  VULKAN_HPP_INLINE std::map<std::string, std::vector<std::vector<std::string>>> const & getExtensionDepends( std::string const & extension )
  {
    static const std::map<std::string, std::vector<std::vector<std::string>>>                        noDependencies;
    static const std::map<std::string, std::map<std::string, std::vector<std::vector<std::string>>>> dependencies = {
      { "VK_KHR_swapchain",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
      { "VK_KHR_display",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
      { "VK_KHR_display_swapchain",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_display",
              "VK_KHR_swapchain",
            } } } } },
#if defined( VK_USE_PLATFORM_XLIB_KHR )
      { "VK_KHR_xlib_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
      { "VK_KHR_xcb_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      { "VK_KHR_wayland_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      { "VK_KHR_android_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_KHR_win32_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_EXT_debug_marker",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_debug_report",
            } } } } },
      { "VK_KHR_video_queue",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_KHR_video_decode_queue",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_synchronization2",
              "VK_KHR_video_queue",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_KHR_video_queue",
            } } } } },
      { "VK_EXT_transform_feedback",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_video_encode_h264",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_encode_queue",
            } } } } },
      { "VK_KHR_video_encode_h265",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_encode_queue",
            } } } } },
      { "VK_KHR_video_decode_h264",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_decode_queue",
            } } } } },
      { "VK_AMD_texture_gather_bias_lod",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_dynamic_rendering",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_depth_stencil_resolve",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_depth_stencil_resolve",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
#if defined( VK_USE_PLATFORM_GGP )
      { "VK_GGP_stream_descriptor_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_GGP*/
      { "VK_NV_corner_sampled_image",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_multiview",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_external_memory",
        { { "VK_VERSION_1_0",
            { {
              "VK_NV_external_memory_capabilities",
            } } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_NV_external_memory_win32",
        { { "VK_VERSION_1_0",
            { {
              "VK_NV_external_memory",
            } } } } },
      { "VK_NV_win32_keyed_mutex",
        { { "VK_VERSION_1_0",
            { {
              "VK_NV_external_memory_win32",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_KHR_device_group",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_device_group_creation",
            } } } } },
#if defined( VK_USE_PLATFORM_VI_NN )
      { "VK_NN_vi_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_VI_NN*/
      { "VK_EXT_texture_compression_astc_hdr",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_astc_decode_mode",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_pipeline_robustness",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_external_memory_capabilities",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_external_memory",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory_capabilities",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_KHR_external_memory_win32",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_KHR_external_memory_fd",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_KHR_win32_keyed_mutex",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory_win32",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_KHR_external_semaphore_capabilities",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_external_semaphore",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_semaphore_capabilities",
            } } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_KHR_external_semaphore_win32",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_semaphore",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_KHR_external_semaphore_fd",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_semaphore",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_push_descriptor",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_conditional_rendering",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_float16_int8",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_16bit_storage",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_storage_buffer_storage_class",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_incremental_present",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_swapchain",
            } } } } },
      { "VK_EXT_direct_mode_display",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_display",
            } } } } },
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      { "VK_EXT_acquire_xlib_display",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_direct_mode_display",
            } } } } },
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
      { "VK_EXT_display_surface_counter",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_display",
            } } } } },
      { "VK_EXT_display_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_display_surface_counter",
              "VK_KHR_swapchain",
            } } } } },
      { "VK_GOOGLE_display_timing",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_swapchain",
            } } } } },
      { "VK_NVX_multiview_per_view_attributes",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_multiview",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_discard_rectangles",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_conservative_rasterization",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_depth_clip_enable",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_swapchain_colorspace",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
      { "VK_EXT_hdr_metadata",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_swapchain",
            } } } } },
      { "VK_KHR_imageless_framebuffer",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_image_format_list",
              "VK_KHR_maintenance2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_image_format_list",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_KHR_create_renderpass2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_maintenance2",
              "VK_KHR_multiview",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_IMG_relaxed_line_rasterization",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shared_presentable_image",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_swapchain",
            } } } } },
      { "VK_KHR_external_fence_capabilities",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_external_fence",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_fence_capabilities",
            } } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_KHR_external_fence_win32",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_fence",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_KHR_external_fence_fd",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_fence",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_performance_query",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_get_surface_capabilities2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
      { "VK_KHR_variable_pointers",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_storage_buffer_storage_class",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_get_display_properties2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_display",
            } } } } },
#if defined( VK_USE_PLATFORM_IOS_MVK )
      { "VK_MVK_ios_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
      { "VK_MVK_macos_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
      { "VK_EXT_external_memory_dma_buf",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory_fd",
            } } } } },
      { "VK_EXT_queue_family_foreign",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_dedicated_allocation",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_memory_requirements2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      { "VK_ANDROID_external_memory_android_hardware_buffer",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_queue_family_foreign",
              "VK_KHR_dedicated_allocation",
              "VK_KHR_external_memory",
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_queue_family_foreign",
            } } } } },
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      { "VK_EXT_sampler_filter_minmax",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      { "VK_AMDX_shader_enqueue",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_extended_dynamic_state",
              "VK_KHR_maintenance5",
              "VK_KHR_pipeline_library",
              "VK_KHR_spirv_1_4",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_KHR_maintenance5",
              "VK_KHR_pipeline_library",
            } } } } },
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      { "VK_EXT_inline_uniform_block",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_maintenance1",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_bfloat16",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_sample_locations",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_blend_operation_advanced",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_acceleration_structure",
        { { "VK_VERSION_1_1",
            { {
              "VK_EXT_descriptor_indexing",
              "VK_KHR_buffer_device_address",
              "VK_KHR_deferred_host_operations",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_deferred_host_operations",
            } } } } },
      { "VK_KHR_ray_tracing_pipeline",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
              "VK_KHR_spirv_1_4",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
      { "VK_KHR_ray_query",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
              "VK_KHR_spirv_1_4",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
      { "VK_NV_shader_sm_builtins", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_sampler_ycbcr_conversion",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_bind_memory2",
              "VK_KHR_get_memory_requirements2",
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_maintenance1",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_image_drm_format_modifier",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_bind_memory2",
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_image_format_list",
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_image_format_list",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_EXT_descriptor_indexing",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_maintenance3",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      { "VK_KHR_portability_subset",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      { "VK_NV_shading_rate_image",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_ray_tracing",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_memory_requirements2",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_representative_fragment_test",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_maintenance3",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_subgroup_extended_types", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_8bit_storage",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_storage_buffer_storage_class",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_external_memory_host",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_atomic_int64",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_clock",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_calibrated_timestamps",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_AMD_shader_core_properties",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_video_decode_h265",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_decode_queue",
            } } } } },
      { "VK_KHR_global_priority",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_vertex_attribute_divisor",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_GGP )
      { "VK_GGP_frame_token",
        { { "VK_VERSION_1_0",
            { {
              "VK_GGP_stream_descriptor_surface",
              "VK_KHR_swapchain",
            } } } } },
#endif /*VK_USE_PLATFORM_GGP*/
      { "VK_KHR_driver_properties",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_float_controls",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_shader_subgroup_partitioned", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_depth_stencil_resolve",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_create_renderpass2",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_KHR_swapchain_mutable_format",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_image_format_list",
              "VK_KHR_maintenance2",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_image_format_list",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_swapchain",
            } } } } },
      { "VK_NV_compute_shader_derivatives",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_mesh_shader",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_fragment_shader_barycentric",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_shader_image_footprint",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_scissor_exclusive",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_device_diagnostic_checkpoints",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_timeline_semaphore",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_INTEL_shader_integer_functions2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_vulkan_memory_model",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_pci_bus_info",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_AMD_display_native_hdr",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_swapchain",
            } } } } },
#if defined( VK_USE_PLATFORM_FUCHSIA )
      { "VK_FUCHSIA_imagepipe_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      { "VK_KHR_shader_terminate_invocation",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_METAL_EXT )
      { "VK_EXT_metal_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      { "VK_EXT_fragment_density_map",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_scalar_block_layout",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_subgroup_size_control", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_fragment_shading_rate",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_create_renderpass2",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_create_renderpass2",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_AMD_shader_core_properties2",
        { { "VK_VERSION_1_0",
            { {
              "VK_AMD_shader_core_properties",
            } } } } },
      { "VK_AMD_device_coherent_memory",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_dynamic_rendering_local_read",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_dynamic_rendering",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_shader_image_atomic_int64",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_quad_control",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_shader_maximal_reconvergence",
              "VK_KHR_vulkan_memory_model",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_shader_maximal_reconvergence",
            } } } } },
      { "VK_KHR_spirv_1_4",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_shader_float_controls",
            } } } } },
      { "VK_EXT_memory_budget",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_memory_priority",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_surface_protected_capabilities",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_get_surface_capabilities2",
            } } } } },
      { "VK_NV_dedicated_allocation_image_aliasing",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_dedicated_allocation",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_separate_depth_stencil_layouts",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_create_renderpass2",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_create_renderpass2",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_EXT_buffer_device_address",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_present_wait",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_present_id",
              "VK_KHR_swapchain",
            } } } } },
      { "VK_NV_cooperative_matrix",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_coverage_reduction_mode",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_NV_framebuffer_mixed_samples",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_NV_framebuffer_mixed_samples",
            } } } } },
      { "VK_EXT_fragment_shader_interlock",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_ycbcr_image_arrays",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_uniform_buffer_standard_layout",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_provoking_vertex",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_EXT_full_screen_exclusive",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_surface",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_surface",
              "VK_KHR_swapchain",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_EXT_headless_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
      { "VK_KHR_buffer_device_address",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_device_group",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_line_rasterization",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_shader_atomic_float",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_host_query_reset",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_index_type_uint8",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_extended_dynamic_state",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_pipeline_executable_properties",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_host_image_copy",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_copy_commands2",
              "VK_KHR_format_feature_flags2",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_copy_commands2",
              "VK_KHR_format_feature_flags2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_map_memory_placed",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_map_memory2",
            } } },
          { "VK_VERSION_1_4", { {} } } } },
      { "VK_EXT_shader_atomic_float2",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_shader_atomic_float",
            } } } } },
      { "VK_EXT_surface_maintenance1",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_surface",
            } } } } },
      { "VK_EXT_swapchain_maintenance1",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_surface_maintenance1",
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_surface_maintenance1",
              "VK_KHR_swapchain",
            } } } } },
      { "VK_EXT_shader_demote_to_helper_invocation",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_device_generated_commands",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_buffer_device_address",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_NV_inherited_viewport_scissor",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_integer_dot_product",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_texel_buffer_alignment",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_depth_bias_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_device_memory_report",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_acquire_drm_display",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_direct_mode_display",
            } } } } },
      { "VK_EXT_robustness2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_custom_border_color",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_present_barrier",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_surface",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_get_surface_capabilities2",
              "VK_KHR_surface",
              "VK_KHR_swapchain",
            } } } } },
      { "VK_KHR_present_id",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_private_data",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_pipeline_creation_cache_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_video_encode_queue",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_synchronization2",
              "VK_KHR_video_queue",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_KHR_video_queue",
            } } } } },
      { "VK_NV_device_diagnostics_config",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_QCOM_tile_shading",
        { { "VK_VERSION_1_0",
            { {
                "VK_QCOM_tile_properties",
              },
              {
                "VK_KHR_get_physical_device_properties2",
              } } } } },
      { "VK_KHR_synchronization2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_descriptor_buffer",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_descriptor_indexing",
              "VK_KHR_buffer_device_address",
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_descriptor_indexing",
              "VK_KHR_buffer_device_address",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_graphics_pipeline_library",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_pipeline_library",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_pipeline_library",
            } } } } },
      { "VK_AMD_shader_early_and_late_fragment_tests",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_fragment_shader_barycentric",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_subgroup_uniform_control_flow", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_zero_initialize_workgroup_memory",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_fragment_shading_rate_enums",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_fragment_shading_rate",
            } } } } },
      { "VK_NV_ray_tracing_motion_blur",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_ray_tracing_pipeline",
            } } } } },
      { "VK_EXT_mesh_shader",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_spirv_1_4",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_EXT_ycbcr_2plane_444_formats",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_fragment_density_map2",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_fragment_density_map",
            } } } } },
      { "VK_QCOM_rotated_copy_commands",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_copy_commands2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_image_robustness",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_workgroup_memory_explicit_layout",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_copy_commands2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_image_compression_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_attachment_feedback_loop_layout",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_4444_formats",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_device_fault",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_ARM_rasterization_order_attachment_access",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_rgba10x6_formats",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_NV_acquire_winrt_display",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_direct_mode_display",
            } } } } },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      { "VK_EXT_directfb_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
      { "VK_VALVE_mutable_descriptor_type",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_maintenance3",
            } } } } },
      { "VK_EXT_vertex_input_dynamic_state",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_physical_device_drm",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_device_address_binding_report",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_debug_utils",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_debug_utils",
            } } } } },
      { "VK_EXT_depth_clip_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_primitive_topology_list_restart",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_format_feature_flags2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_present_mode_fifo_latest_ready",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_swapchain",
            } } } } },
#if defined( VK_USE_PLATFORM_FUCHSIA )
      { "VK_FUCHSIA_external_memory",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
              "VK_KHR_external_memory_capabilities",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_FUCHSIA_external_semaphore",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_semaphore",
              "VK_KHR_external_semaphore_capabilities",
            } } } } },
      { "VK_FUCHSIA_buffer_collection",
        { { "VK_VERSION_1_0",
            { {
              "VK_FUCHSIA_external_memory",
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_FUCHSIA_external_memory",
            } } } } },
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      { "VK_HUAWEI_subpass_shading",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_create_renderpass2",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_HUAWEI_invocation_mask",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_ray_tracing_pipeline",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_KHR_ray_tracing_pipeline",
            } } } } },
      { "VK_NV_external_memory_rdma",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_pipeline_properties",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_multisampled_render_to_single_sampled",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_create_renderpass2",
              "VK_KHR_depth_stencil_resolve",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_EXT_extended_dynamic_state2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      { "VK_QNX_screen_surface",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      { "VK_EXT_color_write_enable",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_primitives_generated_query",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_transform_feedback",
            } } } } },
      { "VK_KHR_ray_tracing_maintenance1",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
      { "VK_EXT_global_priority_query",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_global_priority",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_global_priority",
            } } } } },
      { "VK_EXT_image_view_min_lod",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_multi_draw",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_image_2d_view_of_3d",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_maintenance1",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_shader_tile_image", { { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_opacity_micromap",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      { "VK_NV_displacement_micromap",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_opacity_micromap",
            } } } } },
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      { "VK_HUAWEI_cluster_culling_shader",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_border_color_swizzle",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_custom_border_color",
            } } } } },
      { "VK_EXT_pageable_device_local_memory",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_memory_priority",
            } } } } },
      { "VK_KHR_maintenance4", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_ARM_shader_core_properties", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_ARM_scheduling_controls",
        { { "VK_VERSION_1_0",
            { {
              "VK_ARM_shader_core_builtins",
            } } } } },
      { "VK_EXT_image_sliced_view_of_3d",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_maintenance1",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_VALVE_descriptor_set_host_mapping",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_depth_clamp_zero_one",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_non_seamless_cube_map",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_ARM_render_pass_striped",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_QCOM_fragment_density_map_offset",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_fragment_density_map",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_fragment_density_map",
            } } } } },
      { "VK_NV_copy_memory_indirect",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_buffer_device_address",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_buffer_device_address",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_NV_memory_decompression",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_buffer_device_address",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_buffer_device_address",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_NV_device_generated_commands_compute",
        { { "VK_VERSION_1_0",
            { {
              "VK_NV_device_generated_commands",
            } } } } },
      { "VK_NV_ray_tracing_linear_swept_spheres",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_ray_tracing_pipeline",
            } } } } },
      { "VK_NV_linear_color_attachment",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_GOOGLE_surfaceless_query",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_surface",
            } } } } },
      { "VK_KHR_shader_maximal_reconvergence", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_image_compression_control_swapchain",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_image_compression_control",
            } } } } },
      { "VK_QCOM_image_processing",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_format_feature_flags2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_nested_command_buffer",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_external_memory_acquire_unmodified",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_extended_dynamic_state3",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_subpass_merge_feedback",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_shader_module_identifier",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_pipeline_creation_cache_control",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_pipeline_creation_cache_control",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_rasterization_order_attachment_access",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_optical_flow",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_format_feature_flags2",
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_format_feature_flags2",
              "VK_KHR_synchronization2",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_EXT_legacy_dithering",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_pipeline_protected_access",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      { "VK_ANDROID_external_format_resolve",
        { { "VK_VERSION_1_0",
            { {
              "VK_ANDROID_external_memory_android_hardware_buffer",
            } } } } },
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      { "VK_KHR_maintenance5",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_dynamic_rendering",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_KHR_ray_tracing_position_fetch",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
      { "VK_EXT_shader_object",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_dynamic_rendering",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_dynamic_rendering",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_KHR_pipeline_binary",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_maintenance5",
            } } },
          { "VK_VERSION_1_4", { {} } } } },
      { "VK_QCOM_tile_properties",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_SEC_amigo_profiling",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_QCOM_multiview_per_view_viewports",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_ray_tracing_invocation_reorder",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_ray_tracing_pipeline",
            } } } } },
      { "VK_EXT_mutable_descriptor_type",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_maintenance3",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_legacy_vertex_attributes",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_vertex_input_dynamic_state",
            } } } } },
      { "VK_ARM_shader_core_builtins",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_pipeline_library_group_handles",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_pipeline_library",
              "VK_KHR_ray_tracing_pipeline",
            } } } } },
      { "VK_EXT_dynamic_rendering_unused_attachments",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_dynamic_rendering",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_KHR_dynamic_rendering",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_NV_low_latency2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_timeline_semaphore",
            } } },
          { "VK_VERSION_1_2", { {} } } } },
      { "VK_KHR_cooperative_matrix",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_compute_shader_derivatives",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_video_decode_av1",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_decode_queue",
            } } } } },
      { "VK_KHR_video_encode_av1",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_encode_queue",
            } } } } },
      { "VK_KHR_video_maintenance1",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_queue",
            } } } } },
      { "VK_NV_per_stage_descriptor_set",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_maintenance6",
            } } },
          { "VK_VERSION_1_4", { {} } } } },
      { "VK_QCOM_image_processing2",
        { { "VK_VERSION_1_0",
            { {
              "VK_QCOM_image_processing",
            } } } } },
      { "VK_QCOM_filter_cubic_weights",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_filter_cubic",
            } } } } },
      { "VK_QCOM_filter_cubic_clamp",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_filter_cubic",
              "VK_EXT_sampler_filter_minmax",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_EXT_filter_cubic",
            } } } } },
      { "VK_EXT_attachment_feedback_loop_dynamic_state",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_attachment_feedback_loop_layout",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_attachment_feedback_loop_layout",
            } } } } },
      { "VK_KHR_vertex_attribute_divisor",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_float_controls2",
        { { "VK_VERSION_1_1",
            { {
              "VK_KHR_shader_float_controls",
            } } } } },
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      { "VK_QNX_external_memory_screen_buffer",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_queue_family_foreign",
              "VK_KHR_dedicated_allocation",
              "VK_KHR_external_memory",
              "VK_KHR_sampler_ycbcr_conversion",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_queue_family_foreign",
            } } } } },
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      { "VK_MSFT_layered_driver",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_index_type_uint8",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_line_rasterization",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_calibrated_timestamps",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_shader_expect_assume",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_maintenance6", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_descriptor_pool_overallocation", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_QCOM_tile_memory_heap",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_memory_requirements2",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_display_stereo",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_display",
              "VK_KHR_get_display_properties2",
            } } } } },
      { "VK_KHR_video_encode_quantization_map",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_format_feature_flags2",
              "VK_KHR_video_encode_queue",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_KHR_video_encode_queue",
            } } } } },
      { "VK_KHR_maintenance7", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_NV_cluster_acceleration_structure",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
      { "VK_NV_partitioned_acceleration_structure",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_acceleration_structure",
            } } } } },
      { "VK_EXT_device_generated_commands",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_buffer_device_address",
              "VK_KHR_maintenance5",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_KHR_maintenance5",
            } } },
          { "VK_VERSION_1_3", { {} } } } },
      { "VK_KHR_maintenance8", { { "VK_VERSION_1_1", { {} } } } },
      { "VK_MESA_image_alignment_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_depth_clamp_control",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_KHR_video_maintenance2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_video_queue",
            } } } } },
      { "VK_HUAWEI_hdr_vivid",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_hdr_metadata",
              "VK_KHR_get_physical_device_properties2",
              "VK_KHR_swapchain",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_hdr_metadata",
              "VK_KHR_swapchain",
            } } } } },
      { "VK_NV_cooperative_matrix2",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_cooperative_matrix",
            } } } } },
      { "VK_ARM_pipeline_opacity_micromap",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_opacity_micromap",
            } } } } },
#if defined( VK_USE_PLATFORM_METAL_EXT )
      { "VK_EXT_external_memory_metal",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_external_memory",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      { "VK_KHR_depth_clamp_zero_one",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_vertex_attribute_robustness",
        { { "VK_VERSION_1_0",
            { {
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1", { {} } } } },
      { "VK_EXT_fragment_density_map_offset",
        { { "VK_VERSION_1_0",
            { {
              "VK_EXT_fragment_density_map",
              "VK_KHR_create_renderpass2",
              "VK_KHR_dynamic_rendering",
              "VK_KHR_get_physical_device_properties2",
            } } },
          { "VK_VERSION_1_1",
            { {
              "VK_EXT_fragment_density_map",
              "VK_KHR_create_renderpass2",
              "VK_KHR_dynamic_rendering",
            } } },
          { "VK_VERSION_1_2",
            { {
              "VK_EXT_fragment_density_map",
              "VK_KHR_dynamic_rendering",
            } } },
          { "VK_VERSION_1_3",
            { {
              "VK_EXT_fragment_density_map",
            } } } } }
    };
    auto depIt = dependencies.find( extension );
    return ( depIt != dependencies.end() ) ? depIt->second : noDependencies;
  }

  VULKAN_HPP_INLINE std::pair<bool, std::vector<std::vector<std::string>> const &> getExtensionDepends( std::string const & version,
                                                                                                        std::string const & extension )
  {
#if !defined( NDEBUG )
    static std::set<std::string> versions = { "VK_VERSION_1_0", "VK_VERSION_1_1", "VK_VERSION_1_2", "VK_VERSION_1_3", "VK_VERSION_1_4" };
    assert( versions.find( version ) != versions.end() );
#endif
    static std::vector<std::vector<std::string>> noDependencies;

    std::map<std::string, std::vector<std::vector<std::string>>> const & dependencies = getExtensionDepends( extension );
    if ( dependencies.empty() )
    {
      return { true, noDependencies };
    }
    auto depIt = dependencies.lower_bound( version );
    if ( ( depIt == dependencies.end() ) || ( depIt->first != version ) )
    {
      depIt = std::prev( depIt );
    }
    if ( depIt == dependencies.end() )
    {
      return { false, noDependencies };
    }
    else
    {
      return { true, depIt->second };
    }
  }

  VULKAN_HPP_INLINE std::map<std::string, std::string> const & getObsoletedExtensions()
  {
    static const std::map<std::string, std::string> obsoletedExtensions = { { "VK_AMD_negative_viewport_height", "VK_KHR_maintenance1" } };
    return obsoletedExtensions;
  }

  VULKAN_HPP_INLINE std::map<std::string, std::string> const & getPromotedExtensions()
  {
    static const std::map<std::string, std::string> promotedExtensions = {
      { "VK_KHR_sampler_mirror_clamp_to_edge", "VK_VERSION_1_2" },
      { "VK_EXT_debug_marker", "VK_EXT_debug_utils" },
      { "VK_AMD_draw_indirect_count", "VK_KHR_draw_indirect_count" },
      { "VK_KHR_dynamic_rendering", "VK_VERSION_1_3" },
      { "VK_KHR_multiview", "VK_VERSION_1_1" },
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      { "VK_NV_win32_keyed_mutex", "VK_KHR_win32_keyed_mutex" },
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      { "VK_KHR_get_physical_device_properties2", "VK_VERSION_1_1" },
      { "VK_KHR_device_group", "VK_VERSION_1_1" },
      { "VK_KHR_shader_draw_parameters", "VK_VERSION_1_1" },
      { "VK_EXT_texture_compression_astc_hdr", "VK_VERSION_1_3" },
      { "VK_EXT_pipeline_robustness", "VK_VERSION_1_4" },
      { "VK_KHR_maintenance1", "VK_VERSION_1_1" },
      { "VK_KHR_device_group_creation", "VK_VERSION_1_1" },
      { "VK_KHR_external_memory_capabilities", "VK_VERSION_1_1" },
      { "VK_KHR_external_memory", "VK_VERSION_1_1" },
      { "VK_KHR_external_semaphore_capabilities", "VK_VERSION_1_1" },
      { "VK_KHR_external_semaphore", "VK_VERSION_1_1" },
      { "VK_KHR_push_descriptor", "VK_VERSION_1_4" },
      { "VK_KHR_shader_float16_int8", "VK_VERSION_1_2" },
      { "VK_KHR_16bit_storage", "VK_VERSION_1_1" },
      { "VK_KHR_descriptor_update_template", "VK_VERSION_1_1" },
      { "VK_KHR_imageless_framebuffer", "VK_VERSION_1_2" },
      { "VK_KHR_create_renderpass2", "VK_VERSION_1_2" },
      { "VK_KHR_external_fence_capabilities", "VK_VERSION_1_1" },
      { "VK_KHR_external_fence", "VK_VERSION_1_1" },
      { "VK_KHR_maintenance2", "VK_VERSION_1_1" },
      { "VK_KHR_variable_pointers", "VK_VERSION_1_1" },
      { "VK_KHR_dedicated_allocation", "VK_VERSION_1_1" },
      { "VK_EXT_sampler_filter_minmax", "VK_VERSION_1_2" },
      { "VK_KHR_storage_buffer_storage_class", "VK_VERSION_1_1" },
      { "VK_EXT_inline_uniform_block", "VK_VERSION_1_3" },
      { "VK_KHR_relaxed_block_layout", "VK_VERSION_1_1" },
      { "VK_KHR_get_memory_requirements2", "VK_VERSION_1_1" },
      { "VK_KHR_image_format_list", "VK_VERSION_1_2" },
      { "VK_KHR_sampler_ycbcr_conversion", "VK_VERSION_1_1" },
      { "VK_KHR_bind_memory2", "VK_VERSION_1_1" },
      { "VK_EXT_descriptor_indexing", "VK_VERSION_1_2" },
      { "VK_EXT_shader_viewport_index_layer", "VK_VERSION_1_2" },
      { "VK_KHR_maintenance3", "VK_VERSION_1_1" },
      { "VK_KHR_draw_indirect_count", "VK_VERSION_1_2" },
      { "VK_EXT_global_priority", "VK_KHR_global_priority" },
      { "VK_KHR_shader_subgroup_extended_types", "VK_VERSION_1_2" },
      { "VK_KHR_8bit_storage", "VK_VERSION_1_2" },
      { "VK_KHR_shader_atomic_int64", "VK_VERSION_1_2" },
      { "VK_EXT_calibrated_timestamps", "VK_KHR_calibrated_timestamps" },
      { "VK_KHR_global_priority", "VK_VERSION_1_4" },
      { "VK_EXT_vertex_attribute_divisor", "VK_KHR_vertex_attribute_divisor" },
      { "VK_EXT_pipeline_creation_feedback", "VK_VERSION_1_3" },
      { "VK_KHR_driver_properties", "VK_VERSION_1_2" },
      { "VK_KHR_shader_float_controls", "VK_VERSION_1_2" },
      { "VK_KHR_depth_stencil_resolve", "VK_VERSION_1_2" },
      { "VK_NV_compute_shader_derivatives", "VK_KHR_compute_shader_derivatives" },
      { "VK_NV_fragment_shader_barycentric", "VK_KHR_fragment_shader_barycentric" },
      { "VK_KHR_timeline_semaphore", "VK_VERSION_1_2" },
      { "VK_KHR_vulkan_memory_model", "VK_VERSION_1_2" },
      { "VK_KHR_shader_terminate_invocation", "VK_VERSION_1_3" },
      { "VK_EXT_scalar_block_layout", "VK_VERSION_1_2" },
      { "VK_EXT_subgroup_size_control", "VK_VERSION_1_3" },
      { "VK_KHR_dynamic_rendering_local_read", "VK_VERSION_1_4" },
      { "VK_KHR_spirv_1_4", "VK_VERSION_1_2" },
      { "VK_KHR_separate_depth_stencil_layouts", "VK_VERSION_1_2" },
      { "VK_EXT_tooling_info", "VK_VERSION_1_3" },
      { "VK_EXT_separate_stencil_usage", "VK_VERSION_1_2" },
      { "VK_KHR_uniform_buffer_standard_layout", "VK_VERSION_1_2" },
      { "VK_KHR_buffer_device_address", "VK_VERSION_1_2" },
      { "VK_EXT_line_rasterization", "VK_KHR_line_rasterization" },
      { "VK_EXT_host_query_reset", "VK_VERSION_1_2" },
      { "VK_EXT_index_type_uint8", "VK_KHR_index_type_uint8" },
      { "VK_EXT_extended_dynamic_state", "VK_VERSION_1_3" },
      { "VK_EXT_host_image_copy", "VK_VERSION_1_4" },
      { "VK_KHR_map_memory2", "VK_VERSION_1_4" },
      { "VK_EXT_shader_demote_to_helper_invocation", "VK_VERSION_1_3" },
      { "VK_KHR_shader_integer_dot_product", "VK_VERSION_1_3" },
      { "VK_EXT_texel_buffer_alignment", "VK_VERSION_1_3" },
      { "VK_KHR_shader_non_semantic_info", "VK_VERSION_1_3" },
      { "VK_EXT_private_data", "VK_VERSION_1_3" },
      { "VK_EXT_pipeline_creation_cache_control", "VK_VERSION_1_3" },
      { "VK_KHR_synchronization2", "VK_VERSION_1_3" },
      { "VK_KHR_zero_initialize_workgroup_memory", "VK_VERSION_1_3" },
      { "VK_EXT_ycbcr_2plane_444_formats", "VK_VERSION_1_3" },
      { "VK_EXT_image_robustness", "VK_VERSION_1_3" },
      { "VK_KHR_copy_commands2", "VK_VERSION_1_3" },
      { "VK_EXT_4444_formats", "VK_VERSION_1_3" },
      { "VK_ARM_rasterization_order_attachment_access", "VK_EXT_rasterization_order_attachment_access" },
      { "VK_VALVE_mutable_descriptor_type", "VK_EXT_mutable_descriptor_type" },
      { "VK_KHR_format_feature_flags2", "VK_VERSION_1_3" },
      { "VK_EXT_extended_dynamic_state2", "VK_VERSION_1_3" },
      { "VK_EXT_global_priority_query", "VK_KHR_global_priority" },
      { "VK_EXT_load_store_op_none", "VK_KHR_load_store_op_none" },
      { "VK_KHR_maintenance4", "VK_VERSION_1_3" },
      { "VK_KHR_shader_subgroup_rotate", "VK_VERSION_1_4" },
      { "VK_EXT_depth_clamp_zero_one", "VK_KHR_depth_clamp_zero_one" },
      { "VK_QCOM_fragment_density_map_offset", "VK_EXT_fragment_density_map_offset" },
      { "VK_EXT_pipeline_protected_access", "VK_VERSION_1_4" },
      { "VK_KHR_maintenance5", "VK_VERSION_1_4" },
      { "VK_KHR_vertex_attribute_divisor", "VK_VERSION_1_4" },
      { "VK_KHR_load_store_op_none", "VK_VERSION_1_4" },
      { "VK_KHR_shader_float_controls2", "VK_VERSION_1_4" },
      { "VK_KHR_index_type_uint8", "VK_VERSION_1_4" },
      { "VK_KHR_line_rasterization", "VK_VERSION_1_4" },
      { "VK_KHR_shader_expect_assume", "VK_VERSION_1_4" },
      { "VK_KHR_maintenance6", "VK_VERSION_1_4" }
    };
    return promotedExtensions;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 std::string getExtensionDeprecatedBy( std::string const & extension )
  {
    if ( extension == "VK_EXT_debug_report" )
    {
      return "VK_EXT_debug_utils";
    }
    if ( extension == "VK_NV_glsl_shader" )
    {
      return "";
    }
    if ( extension == "VK_NV_dedicated_allocation" )
    {
      return "VK_KHR_dedicated_allocation";
    }
    if ( extension == "VK_AMD_gpu_shader_half_float" )
    {
      return "VK_KHR_shader_float16_int8";
    }
    if ( extension == "VK_IMG_format_pvrtc" )
    {
      return "";
    }
    if ( extension == "VK_NV_external_memory_capabilities" )
    {
      return "VK_KHR_external_memory_capabilities";
    }
    if ( extension == "VK_NV_external_memory" )
    {
      return "VK_KHR_external_memory";
    }
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    if ( extension == "VK_NV_external_memory_win32" )
    {
      return "VK_KHR_external_memory_win32";
    }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    if ( extension == "VK_EXT_validation_flags" )
    {
      return "VK_EXT_layer_settings";
    }
    if ( extension == "VK_EXT_shader_subgroup_ballot" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_shader_subgroup_vote" )
    {
      return "VK_VERSION_1_1";
    }
#if defined( VK_USE_PLATFORM_IOS_MVK )
    if ( extension == "VK_MVK_ios_surface" )
    {
      return "VK_EXT_metal_surface";
    }
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
    if ( extension == "VK_MVK_macos_surface" )
    {
      return "VK_EXT_metal_surface";
    }
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
    if ( extension == "VK_AMD_gpu_shader_int16" )
    {
      return "VK_KHR_shader_float16_int8";
    }
    if ( extension == "VK_NV_ray_tracing" )
    {
      return "VK_KHR_ray_tracing_pipeline";
    }
    if ( extension == "VK_EXT_buffer_device_address" )
    {
      return "VK_KHR_buffer_device_address";
    }
    if ( extension == "VK_EXT_validation_features" )
    {
      return "VK_EXT_layer_settings";
    }
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( extension == "VK_NV_displacement_micromap" )
    {
      return "VK_NV_cluster_acceleration_structure";
    }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

    return "";
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 std::string getExtensionObsoletedBy( std::string const & extension )
  {
    if ( extension == "VK_AMD_negative_viewport_height" )
    {
      return "VK_KHR_maintenance1";
    }
    return "";
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 std::string getExtensionPromotedTo( std::string const & extension )
  {
    if ( extension == "VK_KHR_sampler_mirror_clamp_to_edge" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_debug_marker" )
    {
      return "VK_EXT_debug_utils";
    }
    if ( extension == "VK_AMD_draw_indirect_count" )
    {
      return "VK_KHR_draw_indirect_count";
    }
    if ( extension == "VK_KHR_dynamic_rendering" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_multiview" )
    {
      return "VK_VERSION_1_1";
    }
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    if ( extension == "VK_NV_win32_keyed_mutex" )
    {
      return "VK_KHR_win32_keyed_mutex";
    }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    if ( extension == "VK_KHR_get_physical_device_properties2" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_device_group" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_shader_draw_parameters" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_EXT_texture_compression_astc_hdr" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_pipeline_robustness" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_maintenance1" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_device_group_creation" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_external_memory_capabilities" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_external_memory" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_external_semaphore_capabilities" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_external_semaphore" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_push_descriptor" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_shader_float16_int8" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_16bit_storage" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_descriptor_update_template" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_imageless_framebuffer" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_create_renderpass2" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_external_fence_capabilities" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_external_fence" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_maintenance2" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_variable_pointers" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_dedicated_allocation" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_EXT_sampler_filter_minmax" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_storage_buffer_storage_class" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_EXT_inline_uniform_block" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_relaxed_block_layout" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_get_memory_requirements2" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_image_format_list" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_sampler_ycbcr_conversion" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_bind_memory2" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_EXT_descriptor_indexing" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_shader_viewport_index_layer" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_maintenance3" )
    {
      return "VK_VERSION_1_1";
    }
    if ( extension == "VK_KHR_draw_indirect_count" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_global_priority" )
    {
      return "VK_KHR_global_priority";
    }
    if ( extension == "VK_KHR_shader_subgroup_extended_types" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_8bit_storage" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_shader_atomic_int64" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_calibrated_timestamps" )
    {
      return "VK_KHR_calibrated_timestamps";
    }
    if ( extension == "VK_KHR_global_priority" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_EXT_vertex_attribute_divisor" )
    {
      return "VK_KHR_vertex_attribute_divisor";
    }
    if ( extension == "VK_EXT_pipeline_creation_feedback" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_driver_properties" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_shader_float_controls" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_depth_stencil_resolve" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_NV_compute_shader_derivatives" )
    {
      return "VK_KHR_compute_shader_derivatives";
    }
    if ( extension == "VK_NV_fragment_shader_barycentric" )
    {
      return "VK_KHR_fragment_shader_barycentric";
    }
    if ( extension == "VK_KHR_timeline_semaphore" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_vulkan_memory_model" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_shader_terminate_invocation" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_scalar_block_layout" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_subgroup_size_control" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_dynamic_rendering_local_read" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_spirv_1_4" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_separate_depth_stencil_layouts" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_tooling_info" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_separate_stencil_usage" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_uniform_buffer_standard_layout" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_KHR_buffer_device_address" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_line_rasterization" )
    {
      return "VK_KHR_line_rasterization";
    }
    if ( extension == "VK_EXT_host_query_reset" )
    {
      return "VK_VERSION_1_2";
    }
    if ( extension == "VK_EXT_index_type_uint8" )
    {
      return "VK_KHR_index_type_uint8";
    }
    if ( extension == "VK_EXT_extended_dynamic_state" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_host_image_copy" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_map_memory2" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_EXT_shader_demote_to_helper_invocation" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_shader_integer_dot_product" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_texel_buffer_alignment" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_shader_non_semantic_info" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_private_data" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_pipeline_creation_cache_control" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_synchronization2" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_zero_initialize_workgroup_memory" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_ycbcr_2plane_444_formats" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_image_robustness" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_copy_commands2" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_4444_formats" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_ARM_rasterization_order_attachment_access" )
    {
      return "VK_EXT_rasterization_order_attachment_access";
    }
    if ( extension == "VK_VALVE_mutable_descriptor_type" )
    {
      return "VK_EXT_mutable_descriptor_type";
    }
    if ( extension == "VK_KHR_format_feature_flags2" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_extended_dynamic_state2" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_EXT_global_priority_query" )
    {
      return "VK_KHR_global_priority";
    }
    if ( extension == "VK_EXT_load_store_op_none" )
    {
      return "VK_KHR_load_store_op_none";
    }
    if ( extension == "VK_KHR_maintenance4" )
    {
      return "VK_VERSION_1_3";
    }
    if ( extension == "VK_KHR_shader_subgroup_rotate" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_EXT_depth_clamp_zero_one" )
    {
      return "VK_KHR_depth_clamp_zero_one";
    }
    if ( extension == "VK_QCOM_fragment_density_map_offset" )
    {
      return "VK_EXT_fragment_density_map_offset";
    }
    if ( extension == "VK_EXT_pipeline_protected_access" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_maintenance5" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_vertex_attribute_divisor" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_load_store_op_none" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_shader_float_controls2" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_index_type_uint8" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_line_rasterization" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_shader_expect_assume" )
    {
      return "VK_VERSION_1_4";
    }
    if ( extension == "VK_KHR_maintenance6" )
    {
      return "VK_VERSION_1_4";
    }
    return "";
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 bool isDeprecatedExtension( std::string const & extension )
  {
    return ( extension == "VK_EXT_debug_report" ) || ( extension == "VK_NV_glsl_shader" ) || ( extension == "VK_NV_dedicated_allocation" ) ||
           ( extension == "VK_AMD_gpu_shader_half_float" ) || ( extension == "VK_IMG_format_pvrtc" ) || ( extension == "VK_NV_external_memory_capabilities" ) ||
           ( extension == "VK_NV_external_memory" ) ||
#if defined( VK_USE_PLATFORM_WIN32_KHR )
           ( extension == "VK_NV_external_memory_win32" ) ||
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
           ( extension == "VK_EXT_validation_flags" ) || ( extension == "VK_EXT_shader_subgroup_ballot" ) || ( extension == "VK_EXT_shader_subgroup_vote" ) ||
#if defined( VK_USE_PLATFORM_IOS_MVK )
           ( extension == "VK_MVK_ios_surface" ) ||
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
           ( extension == "VK_MVK_macos_surface" ) ||
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
           ( extension == "VK_AMD_gpu_shader_int16" ) || ( extension == "VK_NV_ray_tracing" ) || ( extension == "VK_EXT_buffer_device_address" ) ||
           ( extension == "VK_EXT_validation_features" ) ||
#if defined( VK_ENABLE_BETA_EXTENSIONS )
           ( extension == "VK_NV_displacement_micromap" ) ||
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
           false;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 bool isDeviceExtension( std::string const & extension )
  {
    return ( extension == "VK_KHR_swapchain" ) || ( extension == "VK_KHR_display_swapchain" ) || ( extension == "VK_NV_glsl_shader" ) ||
           ( extension == "VK_EXT_depth_range_unrestricted" ) || ( extension == "VK_KHR_sampler_mirror_clamp_to_edge" ) ||
           ( extension == "VK_IMG_filter_cubic" ) || ( extension == "VK_AMD_rasterization_order" ) || ( extension == "VK_AMD_shader_trinary_minmax" ) ||
           ( extension == "VK_AMD_shader_explicit_vertex_parameter" ) || ( extension == "VK_EXT_debug_marker" ) || ( extension == "VK_KHR_video_queue" ) ||
           ( extension == "VK_KHR_video_decode_queue" ) || ( extension == "VK_AMD_gcn_shader" ) || ( extension == "VK_NV_dedicated_allocation" ) ||
           ( extension == "VK_EXT_transform_feedback" ) || ( extension == "VK_NVX_binary_import" ) || ( extension == "VK_NVX_image_view_handle" ) ||
           ( extension == "VK_AMD_draw_indirect_count" ) || ( extension == "VK_AMD_negative_viewport_height" ) ||
           ( extension == "VK_AMD_gpu_shader_half_float" ) || ( extension == "VK_AMD_shader_ballot" ) || ( extension == "VK_KHR_video_encode_h264" ) ||
           ( extension == "VK_KHR_video_encode_h265" ) || ( extension == "VK_KHR_video_decode_h264" ) || ( extension == "VK_AMD_texture_gather_bias_lod" ) ||
           ( extension == "VK_AMD_shader_info" ) || ( extension == "VK_KHR_dynamic_rendering" ) || ( extension == "VK_AMD_shader_image_load_store_lod" ) ||
           ( extension == "VK_NV_corner_sampled_image" ) || ( extension == "VK_KHR_multiview" ) || ( extension == "VK_IMG_format_pvrtc" ) ||
           ( extension == "VK_NV_external_memory" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_NV_external_memory_win32" ) || ( extension == "VK_NV_win32_keyed_mutex" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_KHR_device_group" ) || ( extension == "VK_KHR_shader_draw_parameters" ) || ( extension == "VK_EXT_shader_subgroup_ballot" ) ||
           ( extension == "VK_EXT_shader_subgroup_vote" ) || ( extension == "VK_EXT_texture_compression_astc_hdr" ) ||
           ( extension == "VK_EXT_astc_decode_mode" ) || ( extension == "VK_EXT_pipeline_robustness" ) || ( extension == "VK_KHR_maintenance1" ) ||
           ( extension == "VK_KHR_external_memory" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_KHR_external_memory_win32" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_KHR_external_memory_fd" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_KHR_win32_keyed_mutex" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_KHR_external_semaphore" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_KHR_external_semaphore_win32" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_KHR_external_semaphore_fd" ) || ( extension == "VK_KHR_push_descriptor" ) || ( extension == "VK_EXT_conditional_rendering" ) ||
           ( extension == "VK_KHR_shader_float16_int8" ) || ( extension == "VK_KHR_16bit_storage" ) || ( extension == "VK_KHR_incremental_present" ) ||
           ( extension == "VK_KHR_descriptor_update_template" ) || ( extension == "VK_NV_clip_space_w_scaling" ) || ( extension == "VK_EXT_display_control" ) ||
           ( extension == "VK_GOOGLE_display_timing" ) || ( extension == "VK_NV_sample_mask_override_coverage" ) ||
           ( extension == "VK_NV_geometry_shader_passthrough" ) || ( extension == "VK_NV_viewport_array2" ) ||
           ( extension == "VK_NVX_multiview_per_view_attributes" ) || ( extension == "VK_NV_viewport_swizzle" ) ||
           ( extension == "VK_EXT_discard_rectangles" ) || ( extension == "VK_EXT_conservative_rasterization" ) ||
           ( extension == "VK_EXT_depth_clip_enable" ) || ( extension == "VK_EXT_hdr_metadata" ) || ( extension == "VK_KHR_imageless_framebuffer" ) ||
           ( extension == "VK_KHR_create_renderpass2" ) || ( extension == "VK_IMG_relaxed_line_rasterization" ) ||
           ( extension == "VK_KHR_shared_presentable_image" ) || ( extension == "VK_KHR_external_fence" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_KHR_external_fence_win32" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_KHR_external_fence_fd" ) || ( extension == "VK_KHR_performance_query" ) || ( extension == "VK_KHR_maintenance2" ) ||
           ( extension == "VK_KHR_variable_pointers" ) || ( extension == "VK_EXT_external_memory_dma_buf" ) || ( extension == "VK_EXT_queue_family_foreign" ) ||
           ( extension == "VK_KHR_dedicated_allocation" )
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
        || ( extension == "VK_ANDROID_external_memory_android_hardware_buffer" )
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
        || ( extension == "VK_EXT_sampler_filter_minmax" ) || ( extension == "VK_KHR_storage_buffer_storage_class" ) ||
           ( extension == "VK_AMD_gpu_shader_int16" )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        || ( extension == "VK_AMDX_shader_enqueue" )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        || ( extension == "VK_AMD_mixed_attachment_samples" ) || ( extension == "VK_AMD_shader_fragment_mask" ) ||
           ( extension == "VK_EXT_inline_uniform_block" ) || ( extension == "VK_EXT_shader_stencil_export" ) || ( extension == "VK_KHR_shader_bfloat16" ) ||
           ( extension == "VK_EXT_sample_locations" ) || ( extension == "VK_KHR_relaxed_block_layout" ) || ( extension == "VK_KHR_get_memory_requirements2" ) ||
           ( extension == "VK_KHR_image_format_list" ) || ( extension == "VK_EXT_blend_operation_advanced" ) ||
           ( extension == "VK_NV_fragment_coverage_to_color" ) || ( extension == "VK_KHR_acceleration_structure" ) ||
           ( extension == "VK_KHR_ray_tracing_pipeline" ) || ( extension == "VK_KHR_ray_query" ) || ( extension == "VK_NV_framebuffer_mixed_samples" ) ||
           ( extension == "VK_NV_fill_rectangle" ) || ( extension == "VK_NV_shader_sm_builtins" ) || ( extension == "VK_EXT_post_depth_coverage" ) ||
           ( extension == "VK_KHR_sampler_ycbcr_conversion" ) || ( extension == "VK_KHR_bind_memory2" ) ||
           ( extension == "VK_EXT_image_drm_format_modifier" ) || ( extension == "VK_EXT_validation_cache" ) || ( extension == "VK_EXT_descriptor_indexing" ) ||
           ( extension == "VK_EXT_shader_viewport_index_layer" )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        || ( extension == "VK_KHR_portability_subset" )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        || ( extension == "VK_NV_shading_rate_image" ) || ( extension == "VK_NV_ray_tracing" ) || ( extension == "VK_NV_representative_fragment_test" ) ||
           ( extension == "VK_KHR_maintenance3" ) || ( extension == "VK_KHR_draw_indirect_count" ) || ( extension == "VK_EXT_filter_cubic" ) ||
           ( extension == "VK_QCOM_render_pass_shader_resolve" ) || ( extension == "VK_EXT_global_priority" ) ||
           ( extension == "VK_KHR_shader_subgroup_extended_types" ) || ( extension == "VK_KHR_8bit_storage" ) ||
           ( extension == "VK_EXT_external_memory_host" ) || ( extension == "VK_AMD_buffer_marker" ) || ( extension == "VK_KHR_shader_atomic_int64" ) ||
           ( extension == "VK_KHR_shader_clock" ) || ( extension == "VK_AMD_pipeline_compiler_control" ) || ( extension == "VK_EXT_calibrated_timestamps" ) ||
           ( extension == "VK_AMD_shader_core_properties" ) || ( extension == "VK_KHR_video_decode_h265" ) || ( extension == "VK_KHR_global_priority" ) ||
           ( extension == "VK_AMD_memory_overallocation_behavior" ) || ( extension == "VK_EXT_vertex_attribute_divisor" )
#if defined( VK_USE_PLATFORM_GGP )
        || ( extension == "VK_GGP_frame_token" )
#endif /*VK_USE_PLATFORM_GGP*/
        || ( extension == "VK_EXT_pipeline_creation_feedback" ) || ( extension == "VK_KHR_driver_properties" ) ||
           ( extension == "VK_KHR_shader_float_controls" ) || ( extension == "VK_NV_shader_subgroup_partitioned" ) ||
           ( extension == "VK_KHR_depth_stencil_resolve" ) || ( extension == "VK_KHR_swapchain_mutable_format" ) ||
           ( extension == "VK_NV_compute_shader_derivatives" ) || ( extension == "VK_NV_mesh_shader" ) ||
           ( extension == "VK_NV_fragment_shader_barycentric" ) || ( extension == "VK_NV_shader_image_footprint" ) ||
           ( extension == "VK_NV_scissor_exclusive" ) || ( extension == "VK_NV_device_diagnostic_checkpoints" ) ||
           ( extension == "VK_KHR_timeline_semaphore" ) || ( extension == "VK_INTEL_shader_integer_functions2" ) ||
           ( extension == "VK_INTEL_performance_query" ) || ( extension == "VK_KHR_vulkan_memory_model" ) || ( extension == "VK_EXT_pci_bus_info" ) ||
           ( extension == "VK_AMD_display_native_hdr" ) || ( extension == "VK_KHR_shader_terminate_invocation" ) ||
           ( extension == "VK_EXT_fragment_density_map" ) || ( extension == "VK_EXT_scalar_block_layout" ) ||
           ( extension == "VK_GOOGLE_hlsl_functionality1" ) || ( extension == "VK_GOOGLE_decorate_string" ) ||
           ( extension == "VK_EXT_subgroup_size_control" ) || ( extension == "VK_KHR_fragment_shading_rate" ) ||
           ( extension == "VK_AMD_shader_core_properties2" ) || ( extension == "VK_AMD_device_coherent_memory" ) ||
           ( extension == "VK_KHR_dynamic_rendering_local_read" ) || ( extension == "VK_EXT_shader_image_atomic_int64" ) ||
           ( extension == "VK_KHR_shader_quad_control" ) || ( extension == "VK_KHR_spirv_1_4" ) || ( extension == "VK_EXT_memory_budget" ) ||
           ( extension == "VK_EXT_memory_priority" ) || ( extension == "VK_NV_dedicated_allocation_image_aliasing" ) ||
           ( extension == "VK_KHR_separate_depth_stencil_layouts" ) || ( extension == "VK_EXT_buffer_device_address" ) ||
           ( extension == "VK_EXT_tooling_info" ) || ( extension == "VK_EXT_separate_stencil_usage" ) || ( extension == "VK_KHR_present_wait" ) ||
           ( extension == "VK_NV_cooperative_matrix" ) || ( extension == "VK_NV_coverage_reduction_mode" ) ||
           ( extension == "VK_EXT_fragment_shader_interlock" ) || ( extension == "VK_EXT_ycbcr_image_arrays" ) ||
           ( extension == "VK_KHR_uniform_buffer_standard_layout" ) || ( extension == "VK_EXT_provoking_vertex" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_EXT_full_screen_exclusive" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_KHR_buffer_device_address" ) || ( extension == "VK_EXT_line_rasterization" ) || ( extension == "VK_EXT_shader_atomic_float" ) ||
           ( extension == "VK_EXT_host_query_reset" ) || ( extension == "VK_EXT_index_type_uint8" ) || ( extension == "VK_EXT_extended_dynamic_state" ) ||
           ( extension == "VK_KHR_deferred_host_operations" ) || ( extension == "VK_KHR_pipeline_executable_properties" ) ||
           ( extension == "VK_EXT_host_image_copy" ) || ( extension == "VK_KHR_map_memory2" ) || ( extension == "VK_EXT_map_memory_placed" ) ||
           ( extension == "VK_EXT_shader_atomic_float2" ) || ( extension == "VK_EXT_swapchain_maintenance1" ) ||
           ( extension == "VK_EXT_shader_demote_to_helper_invocation" ) || ( extension == "VK_NV_device_generated_commands" ) ||
           ( extension == "VK_NV_inherited_viewport_scissor" ) || ( extension == "VK_KHR_shader_integer_dot_product" ) ||
           ( extension == "VK_EXT_texel_buffer_alignment" ) || ( extension == "VK_QCOM_render_pass_transform" ) ||
           ( extension == "VK_EXT_depth_bias_control" ) || ( extension == "VK_EXT_device_memory_report" ) || ( extension == "VK_EXT_robustness2" ) ||
           ( extension == "VK_EXT_custom_border_color" ) || ( extension == "VK_GOOGLE_user_type" ) || ( extension == "VK_KHR_pipeline_library" ) ||
           ( extension == "VK_NV_present_barrier" ) || ( extension == "VK_KHR_shader_non_semantic_info" ) || ( extension == "VK_KHR_present_id" ) ||
           ( extension == "VK_EXT_private_data" ) || ( extension == "VK_EXT_pipeline_creation_cache_control" ) ||
           ( extension == "VK_KHR_video_encode_queue" ) || ( extension == "VK_NV_device_diagnostics_config" ) ||
           ( extension == "VK_QCOM_render_pass_store_ops" )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        || ( extension == "VK_NV_cuda_kernel_launch" )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        || ( extension == "VK_QCOM_tile_shading" ) || ( extension == "VK_NV_low_latency" )
#if defined( VK_USE_PLATFORM_METAL_EXT )
        || ( extension == "VK_EXT_metal_objects" )
#endif /*VK_USE_PLATFORM_METAL_EXT*/
        || ( extension == "VK_KHR_synchronization2" ) || ( extension == "VK_EXT_descriptor_buffer" ) || ( extension == "VK_EXT_graphics_pipeline_library" ) ||
           ( extension == "VK_AMD_shader_early_and_late_fragment_tests" ) || ( extension == "VK_KHR_fragment_shader_barycentric" ) ||
           ( extension == "VK_KHR_shader_subgroup_uniform_control_flow" ) || ( extension == "VK_KHR_zero_initialize_workgroup_memory" ) ||
           ( extension == "VK_NV_fragment_shading_rate_enums" ) || ( extension == "VK_NV_ray_tracing_motion_blur" ) || ( extension == "VK_EXT_mesh_shader" ) ||
           ( extension == "VK_EXT_ycbcr_2plane_444_formats" ) || ( extension == "VK_EXT_fragment_density_map2" ) ||
           ( extension == "VK_QCOM_rotated_copy_commands" ) || ( extension == "VK_EXT_image_robustness" ) ||
           ( extension == "VK_KHR_workgroup_memory_explicit_layout" ) || ( extension == "VK_KHR_copy_commands2" ) ||
           ( extension == "VK_EXT_image_compression_control" ) || ( extension == "VK_EXT_attachment_feedback_loop_layout" ) ||
           ( extension == "VK_EXT_4444_formats" ) || ( extension == "VK_EXT_device_fault" ) ||
           ( extension == "VK_ARM_rasterization_order_attachment_access" ) || ( extension == "VK_EXT_rgba10x6_formats" )
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_NV_acquire_winrt_display" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_VALVE_mutable_descriptor_type" ) || ( extension == "VK_EXT_vertex_input_dynamic_state" ) ||
           ( extension == "VK_EXT_physical_device_drm" ) || ( extension == "VK_EXT_device_address_binding_report" ) ||
           ( extension == "VK_EXT_depth_clip_control" ) || ( extension == "VK_EXT_primitive_topology_list_restart" ) ||
           ( extension == "VK_KHR_format_feature_flags2" ) || ( extension == "VK_EXT_present_mode_fifo_latest_ready" )
#if defined( VK_USE_PLATFORM_FUCHSIA )
        || ( extension == "VK_FUCHSIA_external_memory" ) || ( extension == "VK_FUCHSIA_external_semaphore" ) || ( extension == "VK_FUCHSIA_buffer_collection" )
#endif /*VK_USE_PLATFORM_FUCHSIA*/
        || ( extension == "VK_HUAWEI_subpass_shading" ) || ( extension == "VK_HUAWEI_invocation_mask" ) || ( extension == "VK_NV_external_memory_rdma" ) ||
           ( extension == "VK_EXT_pipeline_properties" ) || ( extension == "VK_EXT_frame_boundary" ) ||
           ( extension == "VK_EXT_multisampled_render_to_single_sampled" ) || ( extension == "VK_EXT_extended_dynamic_state2" ) ||
           ( extension == "VK_EXT_color_write_enable" ) || ( extension == "VK_EXT_primitives_generated_query" ) ||
           ( extension == "VK_KHR_ray_tracing_maintenance1" ) || ( extension == "VK_EXT_global_priority_query" ) ||
           ( extension == "VK_EXT_image_view_min_lod" ) || ( extension == "VK_EXT_multi_draw" ) || ( extension == "VK_EXT_image_2d_view_of_3d" ) ||
           ( extension == "VK_EXT_shader_tile_image" ) || ( extension == "VK_EXT_opacity_micromap" )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        || ( extension == "VK_NV_displacement_micromap" )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        || ( extension == "VK_EXT_load_store_op_none" ) || ( extension == "VK_HUAWEI_cluster_culling_shader" ) ||
           ( extension == "VK_EXT_border_color_swizzle" ) || ( extension == "VK_EXT_pageable_device_local_memory" ) || ( extension == "VK_KHR_maintenance4" ) ||
           ( extension == "VK_ARM_shader_core_properties" ) || ( extension == "VK_KHR_shader_subgroup_rotate" ) ||
           ( extension == "VK_ARM_scheduling_controls" ) || ( extension == "VK_EXT_image_sliced_view_of_3d" ) ||
           ( extension == "VK_VALVE_descriptor_set_host_mapping" ) || ( extension == "VK_EXT_depth_clamp_zero_one" ) ||
           ( extension == "VK_EXT_non_seamless_cube_map" ) || ( extension == "VK_ARM_render_pass_striped" ) ||
           ( extension == "VK_QCOM_fragment_density_map_offset" ) || ( extension == "VK_NV_copy_memory_indirect" ) ||
           ( extension == "VK_NV_memory_decompression" ) || ( extension == "VK_NV_device_generated_commands_compute" ) ||
           ( extension == "VK_NV_ray_tracing_linear_swept_spheres" ) || ( extension == "VK_NV_linear_color_attachment" ) ||
           ( extension == "VK_KHR_shader_maximal_reconvergence" ) || ( extension == "VK_EXT_image_compression_control_swapchain" ) ||
           ( extension == "VK_QCOM_image_processing" ) || ( extension == "VK_EXT_nested_command_buffer" ) ||
           ( extension == "VK_EXT_external_memory_acquire_unmodified" ) || ( extension == "VK_EXT_extended_dynamic_state3" ) ||
           ( extension == "VK_EXT_subpass_merge_feedback" ) || ( extension == "VK_EXT_shader_module_identifier" ) ||
           ( extension == "VK_EXT_rasterization_order_attachment_access" ) || ( extension == "VK_NV_optical_flow" ) ||
           ( extension == "VK_EXT_legacy_dithering" ) || ( extension == "VK_EXT_pipeline_protected_access" )
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
        || ( extension == "VK_ANDROID_external_format_resolve" )
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
        || ( extension == "VK_KHR_maintenance5" ) || ( extension == "VK_AMD_anti_lag" ) || ( extension == "VK_KHR_ray_tracing_position_fetch" ) ||
           ( extension == "VK_EXT_shader_object" ) || ( extension == "VK_KHR_pipeline_binary" ) || ( extension == "VK_QCOM_tile_properties" ) ||
           ( extension == "VK_SEC_amigo_profiling" ) || ( extension == "VK_QCOM_multiview_per_view_viewports" ) ||
           ( extension == "VK_NV_ray_tracing_invocation_reorder" ) || ( extension == "VK_NV_cooperative_vector" ) ||
           ( extension == "VK_NV_extended_sparse_address_space" ) || ( extension == "VK_EXT_mutable_descriptor_type" ) ||
           ( extension == "VK_EXT_legacy_vertex_attributes" ) || ( extension == "VK_ARM_shader_core_builtins" ) ||
           ( extension == "VK_EXT_pipeline_library_group_handles" ) || ( extension == "VK_EXT_dynamic_rendering_unused_attachments" ) ||
           ( extension == "VK_NV_low_latency2" ) || ( extension == "VK_KHR_cooperative_matrix" ) ||
           ( extension == "VK_QCOM_multiview_per_view_render_areas" ) || ( extension == "VK_KHR_compute_shader_derivatives" ) ||
           ( extension == "VK_KHR_video_decode_av1" ) || ( extension == "VK_KHR_video_encode_av1" ) || ( extension == "VK_KHR_video_maintenance1" ) ||
           ( extension == "VK_NV_per_stage_descriptor_set" ) || ( extension == "VK_QCOM_image_processing2" ) ||
           ( extension == "VK_QCOM_filter_cubic_weights" ) || ( extension == "VK_QCOM_ycbcr_degamma" ) || ( extension == "VK_QCOM_filter_cubic_clamp" ) ||
           ( extension == "VK_EXT_attachment_feedback_loop_dynamic_state" ) || ( extension == "VK_KHR_vertex_attribute_divisor" ) ||
           ( extension == "VK_KHR_load_store_op_none" ) || ( extension == "VK_KHR_shader_float_controls2" )
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
        || ( extension == "VK_QNX_external_memory_screen_buffer" )
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
        || ( extension == "VK_MSFT_layered_driver" ) || ( extension == "VK_KHR_index_type_uint8" ) || ( extension == "VK_KHR_line_rasterization" ) ||
           ( extension == "VK_KHR_calibrated_timestamps" ) || ( extension == "VK_KHR_shader_expect_assume" ) || ( extension == "VK_KHR_maintenance6" ) ||
           ( extension == "VK_NV_descriptor_pool_overallocation" ) || ( extension == "VK_QCOM_tile_memory_heap" ) ||
           ( extension == "VK_KHR_video_encode_quantization_map" ) || ( extension == "VK_NV_raw_access_chains" ) ||
           ( extension == "VK_NV_external_compute_queue" ) || ( extension == "VK_KHR_shader_relaxed_extended_instruction" ) ||
           ( extension == "VK_NV_command_buffer_inheritance" ) || ( extension == "VK_KHR_maintenance7" ) ||
           ( extension == "VK_NV_shader_atomic_float16_vector" ) || ( extension == "VK_EXT_shader_replicated_composites" ) ||
           ( extension == "VK_NV_ray_tracing_validation" ) || ( extension == "VK_NV_cluster_acceleration_structure" ) ||
           ( extension == "VK_NV_partitioned_acceleration_structure" ) || ( extension == "VK_EXT_device_generated_commands" ) ||
           ( extension == "VK_KHR_maintenance8" ) || ( extension == "VK_MESA_image_alignment_control" ) || ( extension == "VK_EXT_depth_clamp_control" ) ||
           ( extension == "VK_KHR_video_maintenance2" ) || ( extension == "VK_HUAWEI_hdr_vivid" ) || ( extension == "VK_NV_cooperative_matrix2" ) ||
           ( extension == "VK_ARM_pipeline_opacity_micromap" )
#if defined( VK_USE_PLATFORM_METAL_EXT )
        || ( extension == "VK_EXT_external_memory_metal" )
#endif /*VK_USE_PLATFORM_METAL_EXT*/
        || ( extension == "VK_KHR_depth_clamp_zero_one" ) || ( extension == "VK_EXT_vertex_attribute_robustness" )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        || ( extension == "VK_NV_present_metering" )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        || ( extension == "VK_EXT_fragment_density_map_offset" );
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 bool isInstanceExtension( std::string const & extension )
  {
    return ( extension == "VK_KHR_surface" ) || ( extension == "VK_KHR_display" )
#if defined( VK_USE_PLATFORM_XLIB_KHR )
        || ( extension == "VK_KHR_xlib_surface" )
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
        || ( extension == "VK_KHR_xcb_surface" )
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
        || ( extension == "VK_KHR_wayland_surface" )
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
        || ( extension == "VK_KHR_android_surface" )
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
        || ( extension == "VK_KHR_win32_surface" )
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
        || ( extension == "VK_EXT_debug_report" )
#if defined( VK_USE_PLATFORM_GGP )
        || ( extension == "VK_GGP_stream_descriptor_surface" )
#endif /*VK_USE_PLATFORM_GGP*/
        || ( extension == "VK_NV_external_memory_capabilities" ) || ( extension == "VK_KHR_get_physical_device_properties2" ) ||
           ( extension == "VK_EXT_validation_flags" )
#if defined( VK_USE_PLATFORM_VI_NN )
        || ( extension == "VK_NN_vi_surface" )
#endif /*VK_USE_PLATFORM_VI_NN*/
        || ( extension == "VK_KHR_device_group_creation" ) || ( extension == "VK_KHR_external_memory_capabilities" ) ||
           ( extension == "VK_KHR_external_semaphore_capabilities" ) || ( extension == "VK_EXT_direct_mode_display" )
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
        || ( extension == "VK_EXT_acquire_xlib_display" )
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
        || ( extension == "VK_EXT_display_surface_counter" ) || ( extension == "VK_EXT_swapchain_colorspace" ) ||
           ( extension == "VK_KHR_external_fence_capabilities" ) || ( extension == "VK_KHR_get_surface_capabilities2" ) ||
           ( extension == "VK_KHR_get_display_properties2" )
#if defined( VK_USE_PLATFORM_IOS_MVK )
        || ( extension == "VK_MVK_ios_surface" )
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
        || ( extension == "VK_MVK_macos_surface" )
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
        || ( extension == "VK_EXT_debug_utils" )
#if defined( VK_USE_PLATFORM_FUCHSIA )
        || ( extension == "VK_FUCHSIA_imagepipe_surface" )
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
        || ( extension == "VK_EXT_metal_surface" )
#endif /*VK_USE_PLATFORM_METAL_EXT*/
        || ( extension == "VK_KHR_surface_protected_capabilities" ) || ( extension == "VK_EXT_validation_features" ) ||
           ( extension == "VK_EXT_headless_surface" ) || ( extension == "VK_EXT_surface_maintenance1" ) || ( extension == "VK_EXT_acquire_drm_display" )
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
        || ( extension == "VK_EXT_directfb_surface" )
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
        || ( extension == "VK_QNX_screen_surface" )
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
        || ( extension == "VK_KHR_portability_enumeration" ) || ( extension == "VK_GOOGLE_surfaceless_query" ) ||
           ( extension == "VK_LUNARG_direct_driver_loading" ) || ( extension == "VK_EXT_layer_settings" ) || ( extension == "VK_NV_display_stereo" );
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 bool isObsoletedExtension( std::string const & extension )
  {
    return ( extension == "VK_AMD_negative_viewport_height" );
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR_20 bool isPromotedExtension( std::string const & extension )
  {
    return ( extension == "VK_KHR_sampler_mirror_clamp_to_edge" ) || ( extension == "VK_EXT_debug_marker" ) || ( extension == "VK_AMD_draw_indirect_count" ) ||
           ( extension == "VK_KHR_dynamic_rendering" ) || ( extension == "VK_KHR_multiview" ) ||
#if defined( VK_USE_PLATFORM_WIN32_KHR )
           ( extension == "VK_NV_win32_keyed_mutex" ) ||
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
           ( extension == "VK_KHR_get_physical_device_properties2" ) || ( extension == "VK_KHR_device_group" ) ||
           ( extension == "VK_KHR_shader_draw_parameters" ) || ( extension == "VK_EXT_texture_compression_astc_hdr" ) ||
           ( extension == "VK_EXT_pipeline_robustness" ) || ( extension == "VK_KHR_maintenance1" ) || ( extension == "VK_KHR_device_group_creation" ) ||
           ( extension == "VK_KHR_external_memory_capabilities" ) || ( extension == "VK_KHR_external_memory" ) ||
           ( extension == "VK_KHR_external_semaphore_capabilities" ) || ( extension == "VK_KHR_external_semaphore" ) ||
           ( extension == "VK_KHR_push_descriptor" ) || ( extension == "VK_KHR_shader_float16_int8" ) || ( extension == "VK_KHR_16bit_storage" ) ||
           ( extension == "VK_KHR_descriptor_update_template" ) || ( extension == "VK_KHR_imageless_framebuffer" ) ||
           ( extension == "VK_KHR_create_renderpass2" ) || ( extension == "VK_KHR_external_fence_capabilities" ) || ( extension == "VK_KHR_external_fence" ) ||
           ( extension == "VK_KHR_maintenance2" ) || ( extension == "VK_KHR_variable_pointers" ) || ( extension == "VK_KHR_dedicated_allocation" ) ||
           ( extension == "VK_EXT_sampler_filter_minmax" ) || ( extension == "VK_KHR_storage_buffer_storage_class" ) ||
           ( extension == "VK_EXT_inline_uniform_block" ) || ( extension == "VK_KHR_relaxed_block_layout" ) ||
           ( extension == "VK_KHR_get_memory_requirements2" ) || ( extension == "VK_KHR_image_format_list" ) ||
           ( extension == "VK_KHR_sampler_ycbcr_conversion" ) || ( extension == "VK_KHR_bind_memory2" ) || ( extension == "VK_EXT_descriptor_indexing" ) ||
           ( extension == "VK_EXT_shader_viewport_index_layer" ) || ( extension == "VK_KHR_maintenance3" ) || ( extension == "VK_KHR_draw_indirect_count" ) ||
           ( extension == "VK_EXT_global_priority" ) || ( extension == "VK_KHR_shader_subgroup_extended_types" ) || ( extension == "VK_KHR_8bit_storage" ) ||
           ( extension == "VK_KHR_shader_atomic_int64" ) || ( extension == "VK_EXT_calibrated_timestamps" ) || ( extension == "VK_KHR_global_priority" ) ||
           ( extension == "VK_EXT_vertex_attribute_divisor" ) || ( extension == "VK_EXT_pipeline_creation_feedback" ) ||
           ( extension == "VK_KHR_driver_properties" ) || ( extension == "VK_KHR_shader_float_controls" ) || ( extension == "VK_KHR_depth_stencil_resolve" ) ||
           ( extension == "VK_NV_compute_shader_derivatives" ) || ( extension == "VK_NV_fragment_shader_barycentric" ) ||
           ( extension == "VK_KHR_timeline_semaphore" ) || ( extension == "VK_KHR_vulkan_memory_model" ) ||
           ( extension == "VK_KHR_shader_terminate_invocation" ) || ( extension == "VK_EXT_scalar_block_layout" ) ||
           ( extension == "VK_EXT_subgroup_size_control" ) || ( extension == "VK_KHR_dynamic_rendering_local_read" ) || ( extension == "VK_KHR_spirv_1_4" ) ||
           ( extension == "VK_KHR_separate_depth_stencil_layouts" ) || ( extension == "VK_EXT_tooling_info" ) ||
           ( extension == "VK_EXT_separate_stencil_usage" ) || ( extension == "VK_KHR_uniform_buffer_standard_layout" ) ||
           ( extension == "VK_KHR_buffer_device_address" ) || ( extension == "VK_EXT_line_rasterization" ) || ( extension == "VK_EXT_host_query_reset" ) ||
           ( extension == "VK_EXT_index_type_uint8" ) || ( extension == "VK_EXT_extended_dynamic_state" ) || ( extension == "VK_EXT_host_image_copy" ) ||
           ( extension == "VK_KHR_map_memory2" ) || ( extension == "VK_EXT_shader_demote_to_helper_invocation" ) ||
           ( extension == "VK_KHR_shader_integer_dot_product" ) || ( extension == "VK_EXT_texel_buffer_alignment" ) ||
           ( extension == "VK_KHR_shader_non_semantic_info" ) || ( extension == "VK_EXT_private_data" ) ||
           ( extension == "VK_EXT_pipeline_creation_cache_control" ) || ( extension == "VK_KHR_synchronization2" ) ||
           ( extension == "VK_KHR_zero_initialize_workgroup_memory" ) || ( extension == "VK_EXT_ycbcr_2plane_444_formats" ) ||
           ( extension == "VK_EXT_image_robustness" ) || ( extension == "VK_KHR_copy_commands2" ) || ( extension == "VK_EXT_4444_formats" ) ||
           ( extension == "VK_ARM_rasterization_order_attachment_access" ) || ( extension == "VK_VALVE_mutable_descriptor_type" ) ||
           ( extension == "VK_KHR_format_feature_flags2" ) || ( extension == "VK_EXT_extended_dynamic_state2" ) ||
           ( extension == "VK_EXT_global_priority_query" ) || ( extension == "VK_EXT_load_store_op_none" ) || ( extension == "VK_KHR_maintenance4" ) ||
           ( extension == "VK_KHR_shader_subgroup_rotate" ) || ( extension == "VK_EXT_depth_clamp_zero_one" ) ||
           ( extension == "VK_QCOM_fragment_density_map_offset" ) || ( extension == "VK_EXT_pipeline_protected_access" ) ||
           ( extension == "VK_KHR_maintenance5" ) || ( extension == "VK_KHR_vertex_attribute_divisor" ) || ( extension == "VK_KHR_load_store_op_none" ) ||
           ( extension == "VK_KHR_shader_float_controls2" ) || ( extension == "VK_KHR_index_type_uint8" ) || ( extension == "VK_KHR_line_rasterization" ) ||
           ( extension == "VK_KHR_shader_expect_assume" ) || ( extension == "VK_KHR_maintenance6" );
  }
}  // namespace VULKAN_HPP_NAMESPACE

#endif

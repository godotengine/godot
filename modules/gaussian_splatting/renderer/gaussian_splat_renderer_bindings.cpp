// Phase 13: Property bindings extracted from gaussian_splat_renderer.cpp
// This file contains all ClassDB bindings, ADD_PROPERTY, and BIND_ENUM_CONSTANT macros.
// Keeping bindings separate from logic improves code organization.

#include "gaussian_splat_renderer.h"

void GaussianSplatRenderer::_bind_methods() {
    // Initialization
    ClassDB::bind_method(D_METHOD("initialize"), &GaussianSplatRenderer::initialize);

    // Data management
    ClassDB::bind_method(D_METHOD("set_gaussian_data", "data"), &GaussianSplatRenderer::set_gaussian_data);
    ClassDB::bind_method(D_METHOD("get_gaussian_data"), &GaussianSplatRenderer::get_gaussian_data);
    ClassDB::bind_method(D_METHOD("set_painterly_material", "material"), &GaussianSplatRenderer::set_painterly_material);
    ClassDB::bind_method(D_METHOD("get_painterly_material"), &GaussianSplatRenderer::get_painterly_material);
    ClassDB::bind_method(D_METHOD("force_sort_for_view", "camera_transform"), &GaussianSplatRenderer::force_sort_for_view);

    // Rendering configuration
    ClassDB::bind_method(D_METHOD("set_render_mode", "mode"), &GaussianSplatRenderer::set_render_mode);
    ClassDB::bind_method(D_METHOD("get_render_mode"), &GaussianSplatRenderer::get_render_mode);
    ClassDB::bind_method(D_METHOD("set_opacity_multiplier", "opacity"), &GaussianSplatRenderer::set_opacity_multiplier);
    ClassDB::bind_method(D_METHOD("get_opacity_multiplier"), &GaussianSplatRenderer::get_opacity_multiplier);
    ClassDB::bind_method(D_METHOD("set_static_sort_cache_enabled", "enabled"), &GaussianSplatRenderer::set_static_sort_cache_enabled);
    ClassDB::bind_method(D_METHOD("is_static_sort_cache_enabled"), &GaussianSplatRenderer::is_static_sort_cache_enabled);
    ClassDB::bind_method(D_METHOD("set_cached_render_reuse_enabled", "enabled"), &GaussianSplatRenderer::set_cached_render_reuse_enabled);
    ClassDB::bind_method(D_METHOD("is_cached_render_reuse_enabled"), &GaussianSplatRenderer::is_cached_render_reuse_enabled);

    // Performance controls
    ClassDB::bind_method(D_METHOD("set_lod_enabled", "enabled"), &GaussianSplatRenderer::set_lod_enabled);
    ClassDB::bind_method(D_METHOD("get_lod_enabled"), &GaussianSplatRenderer::get_lod_enabled);
    ClassDB::bind_method(D_METHOD("set_lod_bias", "bias"), &GaussianSplatRenderer::set_lod_bias);
    ClassDB::bind_method(D_METHOD("get_lod_bias"), &GaussianSplatRenderer::get_lod_bias);
    ClassDB::bind_method(D_METHOD("set_lod_min_screen_size", "pixels"), &GaussianSplatRenderer::set_lod_min_screen_size);
    ClassDB::bind_method(D_METHOD("get_lod_min_screen_size"), &GaussianSplatRenderer::get_lod_min_screen_size);
    ClassDB::bind_method(D_METHOD("set_lod_max_distance", "distance"), &GaussianSplatRenderer::set_lod_max_distance);
    ClassDB::bind_method(D_METHOD("get_lod_max_distance"), &GaussianSplatRenderer::get_lod_max_distance);
    ClassDB::bind_method(D_METHOD("set_importance_cull_threshold", "threshold"), &GaussianSplatRenderer::set_importance_cull_threshold);
    ClassDB::bind_method(D_METHOD("get_importance_cull_threshold"), &GaussianSplatRenderer::get_importance_cull_threshold);
    ClassDB::bind_method(D_METHOD("set_cull_radius_multiplier", "multiplier"), &GaussianSplatRenderer::set_cull_radius_multiplier);
    ClassDB::bind_method(D_METHOD("get_cull_radius_multiplier"), &GaussianSplatRenderer::get_cull_radius_multiplier);
    ClassDB::bind_method(D_METHOD("set_cull_frustum_plane_slack", "slack"), &GaussianSplatRenderer::set_cull_frustum_plane_slack);
    ClassDB::bind_method(D_METHOD("get_cull_frustum_plane_slack"), &GaussianSplatRenderer::get_cull_frustum_plane_slack);
    ClassDB::bind_method(D_METHOD("set_cull_near_tolerance", "tolerance"), &GaussianSplatRenderer::set_cull_near_tolerance);
    ClassDB::bind_method(D_METHOD("get_cull_near_tolerance"), &GaussianSplatRenderer::get_cull_near_tolerance);
    ClassDB::bind_method(D_METHOD("set_cull_far_tolerance", "tolerance"), &GaussianSplatRenderer::set_cull_far_tolerance);
    ClassDB::bind_method(D_METHOD("get_cull_far_tolerance"), &GaussianSplatRenderer::get_cull_far_tolerance);
    ClassDB::bind_method(D_METHOD("set_tiny_splat_screen_radius", "pixels"), &GaussianSplatRenderer::set_tiny_splat_screen_radius);
    ClassDB::bind_method(D_METHOD("get_tiny_splat_screen_radius"), &GaussianSplatRenderer::get_tiny_splat_screen_radius);
    ClassDB::bind_method(D_METHOD("set_opacity_aware_culling", "enabled"), &GaussianSplatRenderer::set_opacity_aware_culling);
    ClassDB::bind_method(D_METHOD("is_opacity_aware_culling"), &GaussianSplatRenderer::is_opacity_aware_culling);
    ClassDB::bind_method(D_METHOD("set_visibility_threshold", "threshold"), &GaussianSplatRenderer::set_visibility_threshold);
    ClassDB::bind_method(D_METHOD("get_visibility_threshold"), &GaussianSplatRenderer::get_visibility_threshold);
    ClassDB::bind_method(D_METHOD("set_distance_cull_enabled", "enabled"), &GaussianSplatRenderer::set_distance_cull_enabled);
    ClassDB::bind_method(D_METHOD("is_distance_cull_enabled"), &GaussianSplatRenderer::is_distance_cull_enabled);
    ClassDB::bind_method(D_METHOD("set_distance_cull_start", "distance"), &GaussianSplatRenderer::set_distance_cull_start);
    ClassDB::bind_method(D_METHOD("get_distance_cull_start"), &GaussianSplatRenderer::get_distance_cull_start);
    ClassDB::bind_method(D_METHOD("set_distance_cull_max_rate", "rate"), &GaussianSplatRenderer::set_distance_cull_max_rate);
    ClassDB::bind_method(D_METHOD("get_distance_cull_max_rate"), &GaussianSplatRenderer::get_distance_cull_max_rate);
    ClassDB::bind_method(D_METHOD("set_overflow_autotune_enabled", "enabled"), &GaussianSplatRenderer::set_overflow_autotune_enabled);
    ClassDB::bind_method(D_METHOD("is_overflow_autotune_enabled"), &GaussianSplatRenderer::is_overflow_autotune_enabled);
    ClassDB::bind_method(D_METHOD("set_max_splats", "count"), &GaussianSplatRenderer::set_max_splats);
    ClassDB::bind_method(D_METHOD("get_max_splats"), &GaussianSplatRenderer::get_max_splats);
    ClassDB::bind_method(D_METHOD("set_frustum_culling", "enabled"), &GaussianSplatRenderer::set_frustum_culling);
    ClassDB::bind_method(D_METHOD("get_frustum_culling"), &GaussianSplatRenderer::get_frustum_culling);

    // Painterly configuration
    ClassDB::bind_method(D_METHOD("set_painterly_enabled", "enabled"), &GaussianSplatRenderer::set_painterly_enabled);
    ClassDB::bind_method(D_METHOD("get_painterly_enabled"), &GaussianSplatRenderer::get_painterly_enabled);
    ClassDB::bind_method(D_METHOD("set_painterly_low_end_mode", "enabled"), &GaussianSplatRenderer::set_painterly_low_end_mode);
    ClassDB::bind_method(D_METHOD("get_painterly_low_end_mode"), &GaussianSplatRenderer::get_painterly_low_end_mode);
    ClassDB::bind_method(D_METHOD("set_painterly_enable_strokes", "enabled"), &GaussianSplatRenderer::set_painterly_enable_strokes);
    ClassDB::bind_method(D_METHOD("get_painterly_enable_strokes"), &GaussianSplatRenderer::get_painterly_enable_strokes);
    ClassDB::bind_method(D_METHOD("set_painterly_internal_scale", "scale"), &GaussianSplatRenderer::set_painterly_internal_scale);
    ClassDB::bind_method(D_METHOD("get_painterly_internal_scale"), &GaussianSplatRenderer::get_painterly_internal_scale);
    ClassDB::bind_method(D_METHOD("set_painterly_edge_threshold", "threshold"), &GaussianSplatRenderer::set_painterly_edge_threshold);
    ClassDB::bind_method(D_METHOD("get_painterly_edge_threshold"), &GaussianSplatRenderer::get_painterly_edge_threshold);
    ClassDB::bind_method(D_METHOD("set_painterly_edge_intensity", "intensity"), &GaussianSplatRenderer::set_painterly_edge_intensity);
    ClassDB::bind_method(D_METHOD("get_painterly_edge_intensity"), &GaussianSplatRenderer::get_painterly_edge_intensity);
    ClassDB::bind_method(D_METHOD("set_painterly_stroke_length", "pixels"), &GaussianSplatRenderer::set_painterly_stroke_length);
    ClassDB::bind_method(D_METHOD("get_painterly_stroke_length"), &GaussianSplatRenderer::get_painterly_stroke_length);
    ClassDB::bind_method(D_METHOD("set_painterly_stroke_opacity", "opacity"), &GaussianSplatRenderer::set_painterly_stroke_opacity);
    ClassDB::bind_method(D_METHOD("get_painterly_stroke_opacity"), &GaussianSplatRenderer::get_painterly_stroke_opacity);
    ClassDB::bind_method(D_METHOD("set_painterly_gamma", "gamma"), &GaussianSplatRenderer::set_painterly_gamma);
    ClassDB::bind_method(D_METHOD("get_painterly_gamma"), &GaussianSplatRenderer::get_painterly_gamma);

    // Quality presets
    ClassDB::bind_method(D_METHOD("set_quality_preset", "preset"), &GaussianSplatRenderer::set_quality_preset);
    ClassDB::bind_method(D_METHOD("get_quality_preset"), &GaussianSplatRenderer::get_quality_preset);

    // Interactive State System
    ClassDB::bind_method(D_METHOD("set_interactive_state", "state"), &GaussianSplatRenderer::set_interactive_state);
    ClassDB::bind_method(D_METHOD("get_interactive_state"), &GaussianSplatRenderer::get_interactive_state);
    ClassDB::bind_method(D_METHOD("enable_highlight_effect", "color"), &GaussianSplatRenderer::enable_highlight_effect);
    ClassDB::bind_method(D_METHOD("enable_outline_effect", "color", "width"), &GaussianSplatRenderer::enable_outline_effect);
    ClassDB::bind_method(D_METHOD("remove_visual_effects"), &GaussianSplatRenderer::remove_visual_effects);

    // Performance monitoring
    ClassDB::bind_method(D_METHOD("get_render_stats"), &GaussianSplatRenderer::get_render_stats);
    ClassDB::bind_method(D_METHOD("get_binning_debug_counters"), &GaussianSplatRenderer::get_binning_debug_counters);
    ClassDB::bind_method(D_METHOD("benchmark_sorting_performance"), &GaussianSplatRenderer::benchmark_sorting_performance);
    ClassDB::bind_method(D_METHOD("run_sort_benchmark", "sizes"), &GaussianSplatRenderer::run_sort_benchmark);
    ClassDB::bind_method(D_METHOD("get_last_sort_metrics"), &GaussianSplatRenderer::get_last_sort_metrics);
	ClassDB::bind_method(D_METHOD("get_sort_metrics_history"), &GaussianSplatRenderer::get_sort_metrics_history);
	ClassDB::bind_method(D_METHOD("get_sort_time_ms"), &GaussianSplatRenderer::get_sort_time_ms);
	ClassDB::bind_method(D_METHOD("get_render_time_ms"), &GaussianSplatRenderer::get_render_time_ms);
	ClassDB::bind_method(D_METHOD("get_overflow_tile_count"), &GaussianSplatRenderer::get_overflow_tile_count);
	ClassDB::bind_method(D_METHOD("get_clamped_records"), &GaussianSplatRenderer::get_clamped_records);
	ClassDB::bind_method(D_METHOD("get_aggregated_count"), &GaussianSplatRenderer::get_aggregated_count);
	ClassDB::bind_method(D_METHOD("get_overflow_stats"), &GaussianSplatRenderer::get_overflow_stats);
	ClassDB::bind_method(D_METHOD("get_visible_splat_count"), &GaussianSplatRenderer::get_visible_splat_count);
	ClassDB::bind_method(D_METHOD("was_last_viewport_copy_successful"), &GaussianSplatRenderer::was_last_viewport_copy_successful);
	ClassDB::bind_method(D_METHOD("get_last_viewport_copy_source_size"), &GaussianSplatRenderer::get_last_viewport_copy_source_size);
	ClassDB::bind_method(D_METHOD("get_last_viewport_copy_dest_size"), &GaussianSplatRenderer::get_last_viewport_copy_dest_size);
	ClassDB::bind_method(D_METHOD("get_pipeline_trace_snapshot"), &GaussianSplatRenderer::get_pipeline_trace_snapshot);
	ClassDB::bind_method(D_METHOD("get_pipeline_trace_json"), &GaussianSplatRenderer::get_pipeline_trace_json);
	ClassDB::bind_method(D_METHOD("dump_pipeline_trace_to_file", "path"), &GaussianSplatRenderer::dump_pipeline_trace_to_file);

    ClassDB::bind_method(D_METHOD("set_debug_show_tile_grid", "enabled"), &GaussianSplatRenderer::set_debug_show_tile_grid);
    ClassDB::bind_method(D_METHOD("is_debug_show_tile_grid"), &GaussianSplatRenderer::is_debug_show_tile_grid);
    ClassDB::bind_method(D_METHOD("set_debug_show_density_heatmap", "enabled"), &GaussianSplatRenderer::set_debug_show_density_heatmap);
    ClassDB::bind_method(D_METHOD("is_debug_show_density_heatmap"), &GaussianSplatRenderer::is_debug_show_density_heatmap);
    ClassDB::bind_method(D_METHOD("set_debug_show_performance_hud", "enabled"), &GaussianSplatRenderer::set_debug_show_performance_hud);
    ClassDB::bind_method(D_METHOD("is_debug_show_performance_hud"), &GaussianSplatRenderer::is_debug_show_performance_hud);
    ClassDB::bind_method(D_METHOD("set_debug_show_residency_hud", "enabled"), &GaussianSplatRenderer::set_debug_show_residency_hud);
    ClassDB::bind_method(D_METHOD("is_debug_show_residency_hud"), &GaussianSplatRenderer::is_debug_show_residency_hud);
    ClassDB::bind_method(D_METHOD("set_debug_show_tile_bounds", "enabled"), &GaussianSplatRenderer::set_debug_show_tile_bounds);
    ClassDB::bind_method(D_METHOD("get_debug_show_tile_bounds"), &GaussianSplatRenderer::get_debug_show_tile_bounds);
    ClassDB::bind_method(D_METHOD("set_debug_show_splat_coverage", "enabled"), &GaussianSplatRenderer::set_debug_show_splat_coverage);
    ClassDB::bind_method(D_METHOD("get_debug_show_splat_coverage"), &GaussianSplatRenderer::get_debug_show_splat_coverage);
    ClassDB::bind_method(D_METHOD("set_debug_show_overflow_tiles", "enabled"), &GaussianSplatRenderer::set_debug_show_overflow_tiles);
    ClassDB::bind_method(D_METHOD("get_debug_show_overflow_tiles"), &GaussianSplatRenderer::get_debug_show_overflow_tiles);
    ClassDB::bind_method(D_METHOD("set_debug_show_projection_issues", "enabled"), &GaussianSplatRenderer::set_debug_show_projection_issues);
    ClassDB::bind_method(D_METHOD("get_debug_show_projection_issues"), &GaussianSplatRenderer::get_debug_show_projection_issues);
    ClassDB::bind_method(D_METHOD("set_debug_show_white_albedo", "enabled"), &GaussianSplatRenderer::set_debug_show_white_albedo);
    ClassDB::bind_method(D_METHOD("get_debug_show_white_albedo"), &GaussianSplatRenderer::get_debug_show_white_albedo);
    ClassDB::bind_method(D_METHOD("set_debug_show_shadow_opacity", "enabled"), &GaussianSplatRenderer::set_debug_show_shadow_opacity);
    ClassDB::bind_method(D_METHOD("get_debug_show_shadow_opacity"), &GaussianSplatRenderer::get_debug_show_shadow_opacity);
    ClassDB::bind_method(D_METHOD("set_debug_show_device_boundaries", "enabled"), &GaussianSplatRenderer::set_debug_show_device_boundaries);
    ClassDB::bind_method(D_METHOD("is_debug_show_device_boundaries"), &GaussianSplatRenderer::is_debug_show_device_boundaries);
	ClassDB::bind_method(D_METHOD("set_debug_show_texture_states", "enabled"), &GaussianSplatRenderer::set_debug_show_texture_states);
	ClassDB::bind_method(D_METHOD("is_debug_show_texture_states"), &GaussianSplatRenderer::is_debug_show_texture_states);
	ClassDB::bind_method(D_METHOD("set_debug_compute_raster_policy", "policy"), &GaussianSplatRenderer::set_debug_compute_raster_policy);
	ClassDB::bind_method(D_METHOD("get_debug_compute_raster_policy"), &GaussianSplatRenderer::get_debug_compute_raster_policy);
	ClassDB::bind_method(D_METHOD("set_debug_dump_gpu_counters", "enabled"), &GaussianSplatRenderer::set_debug_dump_gpu_counters);
	ClassDB::bind_method(D_METHOD("get_debug_dump_gpu_counters"), &GaussianSplatRenderer::get_debug_dump_gpu_counters);
	ClassDB::bind_method(D_METHOD("set_debug_binning_counters_enabled", "enabled"), &GaussianSplatRenderer::set_debug_binning_counters_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_binning_counters_enabled"), &GaussianSplatRenderer::get_debug_binning_counters_enabled);
	ClassDB::bind_method(D_METHOD("set_debug_pipeline_trace_enabled", "enabled"), &GaussianSplatRenderer::set_debug_pipeline_trace_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_pipeline_trace_enabled"), &GaussianSplatRenderer::get_debug_pipeline_trace_enabled);
	ClassDB::bind_method(D_METHOD("set_debug_state_guardrails_enabled", "enabled"), &GaussianSplatRenderer::set_debug_state_guardrails_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_state_guardrails_enabled"), &GaussianSplatRenderer::get_debug_state_guardrails_enabled);
	ClassDB::bind_method(D_METHOD("set_debug_cull_guardrails_enabled", "enabled"), &GaussianSplatRenderer::set_debug_cull_guardrails_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_cull_guardrails_enabled"), &GaussianSplatRenderer::get_debug_cull_guardrails_enabled);
	ClassDB::bind_method(D_METHOD("set_debug_splat_audit_enabled", "enabled"), &GaussianSplatRenderer::set_debug_splat_audit_enabled);
	ClassDB::bind_method(D_METHOD("get_debug_splat_audit_enabled"), &GaussianSplatRenderer::get_debug_splat_audit_enabled);
	ClassDB::bind_method(D_METHOD("set_debug_splat_audit_sample_count", "count"), &GaussianSplatRenderer::set_debug_splat_audit_sample_count);
	ClassDB::bind_method(D_METHOD("get_debug_splat_audit_sample_count"), &GaussianSplatRenderer::get_debug_splat_audit_sample_count);
	ClassDB::bind_method(D_METHOD("set_debug_overlay_opacity", "opacity"), &GaussianSplatRenderer::set_debug_overlay_opacity);
	ClassDB::bind_method(D_METHOD("get_debug_overlay_opacity"), &GaussianSplatRenderer::get_debug_overlay_opacity);
    ClassDB::bind_method(D_METHOD("set_solid_coverage_enabled", "enabled"), &GaussianSplatRenderer::set_solid_coverage_enabled);
    ClassDB::bind_method(D_METHOD("is_solid_coverage_enabled"), &GaussianSplatRenderer::is_solid_coverage_enabled);
    ClassDB::bind_method(D_METHOD("set_solid_coverage_alpha_floor", "alpha_floor"), &GaussianSplatRenderer::set_solid_coverage_alpha_floor);
    ClassDB::bind_method(D_METHOD("get_solid_coverage_alpha_floor"), &GaussianSplatRenderer::get_solid_coverage_alpha_floor);
	ClassDB::bind_method(D_METHOD("set_debug_show_resolve_input", "enabled"), &GaussianSplatRenderer::set_debug_show_resolve_input);
	ClassDB::bind_method(D_METHOD("get_debug_show_resolve_input"), &GaussianSplatRenderer::get_debug_show_resolve_input);
	ClassDB::bind_method(D_METHOD("set_debug_show_resolve_output", "enabled"), &GaussianSplatRenderer::set_debug_show_resolve_output);
	ClassDB::bind_method(D_METHOD("get_debug_show_resolve_output"), &GaussianSplatRenderer::get_debug_show_resolve_output);
	ClassDB::bind_method(D_METHOD("reload_pipeline_feature_set"), &GaussianSplatRenderer::reload_pipeline_feature_set);

    // Jacobian diagnostic toggles for radial stretching investigation
    ClassDB::bind_method(D_METHOD("set_jacobian_bypass_radius_depth_floor", "enabled"), &GaussianSplatRenderer::set_jacobian_bypass_radius_depth_floor);
    ClassDB::bind_method(D_METHOD("get_jacobian_bypass_radius_depth_floor"), &GaussianSplatRenderer::get_jacobian_bypass_radius_depth_floor);
    ClassDB::bind_method(D_METHOD("set_jacobian_bypass_j_col2_clamp", "enabled"), &GaussianSplatRenderer::set_jacobian_bypass_j_col2_clamp);
    ClassDB::bind_method(D_METHOD("get_jacobian_bypass_j_col2_clamp"), &GaussianSplatRenderer::get_jacobian_bypass_j_col2_clamp);
    ClassDB::bind_method(D_METHOD("set_jacobian_invert_j_col2_sign", "enabled"), &GaussianSplatRenderer::set_jacobian_invert_j_col2_sign);
    ClassDB::bind_method(D_METHOD("get_jacobian_invert_j_col2_sign"), &GaussianSplatRenderer::get_jacobian_invert_j_col2_sign);
    ClassDB::bind_method(D_METHOD("set_max_conic_aspect", "aspect"), &GaussianSplatRenderer::set_max_conic_aspect);
    ClassDB::bind_method(D_METHOD("get_max_conic_aspect"), &GaussianSplatRenderer::get_max_conic_aspect);

    ClassDB::bind_method(D_METHOD("set_debug_preview_mode", "mode"), &GaussianSplatRenderer::set_debug_preview_mode);
    ClassDB::bind_method(D_METHOD("get_debug_preview_mode"), &GaussianSplatRenderer::get_debug_preview_mode);
    ClassDB::bind_method(D_METHOD("get_runtime_diagnostic_snapshot"), &GaussianSplatRenderer::get_runtime_diagnostic_snapshot);
    ClassDB::bind_method(D_METHOD("get_tile_renderer"), &GaussianSplatRenderer::get_tile_renderer);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gaussian_data", PROPERTY_HINT_RESOURCE_TYPE, "GaussianData"),
                 "set_gaussian_data", "get_gaussian_data");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "painterly_material", PROPERTY_HINT_RESOURCE_TYPE, "PainterlyMaterial"),
                 "set_painterly_material", "get_painterly_material");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "render_mode", PROPERTY_HINT_ENUM, "3D,2D,Hybrid"),
            "set_render_mode", "get_render_mode");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "render/opacity_multiplier", PROPERTY_HINT_RANGE, "0,1,0.01"),
            "set_opacity_multiplier", "get_opacity_multiplier");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sort/static_cache_enabled"),
                 "set_static_sort_cache_enabled", "is_static_sort_cache_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "render/cached_render_reuse_enabled"),
            "set_cached_render_reuse_enabled", "is_cached_render_reuse_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "interactive_state", PROPERTY_HINT_ENUM, "Normal,Hovered,Selected,Disabled"),
                 "set_interactive_state", "get_interactive_state");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lod_enabled"), "set_lod_enabled", "get_lod_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_bias", PROPERTY_HINT_RANGE, "0.01,8.0,0.01"), "set_lod_bias", "get_lod_bias");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_min_screen_size", PROPERTY_HINT_RANGE, "0.0,64.0,0.1"), "set_lod_min_screen_size", "get_lod_min_screen_size");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_max_distance", PROPERTY_HINT_RANGE, "0.0,1000.0,1.0"), "set_lod_max_distance", "get_lod_max_distance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/importance_threshold", PROPERTY_HINT_RANGE, "0.0,1.0,0.001"), "set_importance_cull_threshold", "get_importance_cull_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/radius_multiplier", PROPERTY_HINT_RANGE, "0.5,16.0,0.1"), "set_cull_radius_multiplier", "get_cull_radius_multiplier");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/frustum_plane_slack", PROPERTY_HINT_RANGE, "1.0,8.0,0.1"), "set_cull_frustum_plane_slack", "get_cull_frustum_plane_slack");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/near_tolerance", PROPERTY_HINT_RANGE, "0.0,1.0,0.001"), "set_cull_near_tolerance", "get_cull_near_tolerance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/far_tolerance", PROPERTY_HINT_RANGE, "0.0,1.0,0.001"), "set_cull_far_tolerance", "get_cull_far_tolerance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/tiny_splat_screen_radius", PROPERTY_HINT_RANGE, "0.0,10.0,0.05"), "set_tiny_splat_screen_radius", "get_tiny_splat_screen_radius");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cull/opacity_aware_culling"), "set_opacity_aware_culling", "is_opacity_aware_culling");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/visibility_threshold", PROPERTY_HINT_RANGE, "0.001,0.1,0.001"), "set_visibility_threshold", "get_visibility_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cull/distance_cull_enabled"), "set_distance_cull_enabled", "is_distance_cull_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/distance_cull_start", PROPERTY_HINT_RANGE, "0.0,1000.0,1.0"), "set_distance_cull_start", "get_distance_cull_start");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cull/distance_cull_max_rate", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_distance_cull_max_rate", "get_distance_cull_max_rate");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cull/overflow_autotune_enabled"), "set_overflow_autotune_enabled", "is_overflow_autotune_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_splats", PROPERTY_HINT_RANGE, "1000,10000000,1000"),
                 "set_max_splats", "get_max_splats");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "frustum_culling"), "set_frustum_culling", "get_frustum_culling");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "painterly/enabled"), "set_painterly_enabled", "get_painterly_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "painterly/low_end_mode"), "set_painterly_low_end_mode", "get_painterly_low_end_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "painterly/enable_strokes"), "set_painterly_enable_strokes", "get_painterly_enable_strokes");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/internal_scale", PROPERTY_HINT_RANGE, "0.25,1.0,0.05"),
                 "set_painterly_internal_scale", "get_painterly_internal_scale");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/edge_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"),
                 "set_painterly_edge_threshold", "get_painterly_edge_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/edge_intensity", PROPERTY_HINT_RANGE, "0,8,0.1"),
                 "set_painterly_edge_intensity", "get_painterly_edge_intensity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/stroke_length", PROPERTY_HINT_RANGE, "1,128,1"),
                 "set_painterly_stroke_length", "get_painterly_stroke_length");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/stroke_opacity", PROPERTY_HINT_RANGE, "0,1,0.01"),
                 "set_painterly_stroke_opacity", "get_painterly_stroke_opacity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/gamma", PROPERTY_HINT_RANGE, "0.5,4.0,0.05"),
                 "set_painterly_gamma", "get_painterly_gamma");

    // Debug overlays from PR #145
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_tile_grid"), "set_debug_show_tile_grid", "is_debug_show_tile_grid");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_density_heatmap"), "set_debug_show_density_heatmap", "is_debug_show_density_heatmap");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_performance_hud"), "set_debug_show_performance_hud", "is_debug_show_performance_hud");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_residency_hud"), "set_debug_show_residency_hud", "is_debug_show_residency_hud");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_tile_bounds"), "set_debug_show_tile_bounds", "get_debug_show_tile_bounds");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_splat_coverage"), "set_debug_show_splat_coverage", "get_debug_show_splat_coverage");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_overflow_tiles"), "set_debug_show_overflow_tiles", "get_debug_show_overflow_tiles");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_projection_issues"), "set_debug_show_projection_issues", "get_debug_show_projection_issues");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_white_albedo"), "set_debug_show_white_albedo", "get_debug_show_white_albedo");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_shadow_opacity"), "set_debug_show_shadow_opacity", "get_debug_show_shadow_opacity");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_resolve_input"), "set_debug_show_resolve_input", "get_debug_show_resolve_input");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_resolve_output"), "set_debug_show_resolve_output", "get_debug_show_resolve_output");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_device_boundaries"), "set_debug_show_device_boundaries", "is_debug_show_device_boundaries");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_texture_states"), "set_debug_show_texture_states", "is_debug_show_texture_states");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug/compute_raster_policy", PROPERTY_HINT_ENUM, "Default,ForceOn,ForceOff"),
			"set_debug_compute_raster_policy", "get_debug_compute_raster_policy");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/dump_gpu_counters"), "set_debug_dump_gpu_counters", "get_debug_dump_gpu_counters");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/enable_pipeline_trace"), "set_debug_pipeline_trace_enabled", "get_debug_pipeline_trace_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/enable_state_guardrails"), "set_debug_state_guardrails_enabled", "get_debug_state_guardrails_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/enable_splat_audit"), "set_debug_splat_audit_enabled", "get_debug_splat_audit_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug/splat_audit_sample_count", PROPERTY_HINT_RANGE, "1,64,1"),
			"set_debug_splat_audit_sample_count", "get_debug_splat_audit_sample_count");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "debug/overlay_opacity", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_debug_overlay_opacity", "get_debug_overlay_opacity");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "render/solid_coverage_enabled"), "set_solid_coverage_enabled", "is_solid_coverage_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "render/solid_coverage_alpha_floor", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_solid_coverage_alpha_floor", "get_solid_coverage_alpha_floor");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "debug/preview_mode", PROPERTY_HINT_ENUM, "Off,Wireframe,Points,Depth,Heatmap,Runtime Modifications"), "set_debug_preview_mode", "get_debug_preview_mode");

    // Enums
    BIND_ENUM_CONSTANT(MODE_3D);
    BIND_ENUM_CONSTANT(MODE_2D);
    BIND_ENUM_CONSTANT(MODE_HYBRID);

    BIND_ENUM_CONSTANT(STATE_NORMAL);
    BIND_ENUM_CONSTANT(STATE_HOVERED);
    BIND_ENUM_CONSTANT(STATE_SELECTED);
    BIND_ENUM_CONSTANT(STATE_DISABLED);

    BIND_ENUM_CONSTANT(DEBUG_PREVIEW_OFF);
    BIND_ENUM_CONSTANT(DEBUG_PREVIEW_WIREFRAME);
    BIND_ENUM_CONSTANT(DEBUG_PREVIEW_POINTS);
    BIND_ENUM_CONSTANT(DEBUG_PREVIEW_DEPTH);
    BIND_ENUM_CONSTANT(DEBUG_PREVIEW_HEATMAP);
    BIND_ENUM_CONSTANT(DEBUG_PREVIEW_RUNTIME_MODIFICATIONS);
}

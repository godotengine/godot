#include "tile_rasterizer.h"
#include "../renderer/tile_renderer.h"
#include "../renderer/gpu_sorting_config.h"
#include "../logger/gs_logger.h"
#include "core/config/project_settings.h"

static bool _is_raster_ready_log_enabled() {
	static int cached = -1;
	if (cached >= 0) {
		return cached == 1;
	}
	bool enabled = false;
	if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
		const bool enable_all = ps->get_setting("rendering/gaussian_splatting/debug/enable_all_debug", false);
		const bool enable_tile_pipeline = ps->get_setting("rendering/gaussian_splatting/debug/enable_tile_pipeline_logs", false);
		const bool enable_tile_logs = ps->get_setting("rendering/gaussian_splatting/debug/enable_tile_logs", false);
		enabled = enable_all || enable_tile_pipeline || enable_tile_logs;
	}
	cached = enabled ? 1 : 0;
	return enabled;
}

enum class OutputOwnershipContractResult {
	VALID,
	REMAPPED_TO_MAIN,
	VIOLATION,
};

static RenderingDevice *_get_contract_main_device(const Ref<RenderDeviceManager> &p_device_manager) {
	if (p_device_manager.is_valid() && p_device_manager->get_main_device()) {
		return p_device_manager->get_main_device();
	}
	return RenderingDevice::get_singleton();
}

static RenderingDevice *_get_manager_main_device(const Ref<RenderDeviceManager> &p_device_manager) {
	if (!p_device_manager.is_valid()) {
		return nullptr;
	}
	return p_device_manager->get_main_device();
}

static OutputOwnershipContractResult _enforce_texture_owner_contract(const char *p_label, const RID &p_texture,
		RenderingDevice *&r_owner, RenderDeviceManager *p_device_manager, RenderingDevice *p_main_device) {
	if (!p_texture.is_valid()) {
		r_owner = nullptr;
		return OutputOwnershipContractResult::VALID;
	}

	if (!r_owner && p_device_manager) {
		r_owner = p_device_manager->get_resource_owner(p_texture, nullptr);
	}

	if (!r_owner && p_main_device && p_main_device->texture_is_valid(p_texture)) {
		r_owner = p_main_device;
		GS_LOG_WARN_DEFAULT(vformat("[TileRasterizer] %s owner missing; using main RenderingDevice for RID=%s",
				String(p_label ? p_label : "texture"), String::num_uint64(p_texture.get_id())));
		return OutputOwnershipContractResult::REMAPPED_TO_MAIN;
	}

	if (!r_owner) {
		GS_LOG_ERROR_DEFAULT(vformat("[TileRasterizer] %s ownership contract violation: missing owner for RID=%s",
				String(p_label ? p_label : "texture"), String::num_uint64(p_texture.get_id())));
		return OutputOwnershipContractResult::VIOLATION;
	}

	if (!r_owner->texture_is_valid(p_texture)) {
		if (p_main_device && p_main_device != r_owner && p_main_device->texture_is_valid(p_texture)) {
			if (p_device_manager) {
				p_device_manager->push_cross_device_operation(
						String(p_label ? p_label : "texture") + String("_owner_remap"), r_owner, p_main_device);
			}
			r_owner = p_main_device;
			GS_LOG_WARN_DEFAULT(vformat("[TileRasterizer] %s owner contract remap: using main RenderingDevice for RID=%s",
					String(p_label ? p_label : "texture"), String::num_uint64(p_texture.get_id())));
			return OutputOwnershipContractResult::REMAPPED_TO_MAIN;
		}
		GS_LOG_ERROR_DEFAULT(vformat("[TileRasterizer] %s ownership contract violation: owner does not validate RID=%s",
				String(p_label ? p_label : "texture"), String::num_uint64(p_texture.get_id())));
		return OutputOwnershipContractResult::VIOLATION;
	}

	if (p_main_device && r_owner != p_main_device) {
		const bool visible_on_main = p_main_device->texture_is_valid(p_texture);
		if (visible_on_main) {
			if (p_device_manager) {
				p_device_manager->push_cross_device_operation(
						String(p_label ? p_label : "texture") + String("_owner_remap"), r_owner, p_main_device);
			}
			r_owner = p_main_device;
			GS_LOG_WARN_DEFAULT(vformat("[TileRasterizer] %s owner contract remap: exposed on main RenderingDevice for RID=%s",
					String(p_label ? p_label : "texture"), String::num_uint64(p_texture.get_id())));
			return OutputOwnershipContractResult::REMAPPED_TO_MAIN;
		}

		if (p_device_manager) {
			p_device_manager->push_cross_device_operation(
					String(p_label ? p_label : "texture") + String("_contract_violation"), r_owner, p_main_device);
		}
		GS_LOG_ERROR_DEFAULT(vformat("[TileRasterizer] %s ownership contract violation: RID=%s is not visible on the main RenderingDevice",
				String(p_label ? p_label : "texture"), String::num_uint64(p_texture.get_id())));
		return OutputOwnershipContractResult::VIOLATION;
	}

	return OutputOwnershipContractResult::VALID;
}

void TileRasterizer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_tile_renderer"), &TileRasterizer::get_tile_renderer);
}

TileRasterizer::TileRasterizer() {
    // Don't instantiate tile_renderer here - it may be set externally
}

TileRasterizer::~TileRasterizer() {
    shutdown();
}

void TileRasterizer::set_tile_renderer(Ref<TileRenderer> p_renderer) {
    if (tile_renderer == p_renderer) {
        return;
    }
	_unbind_output_invalidation_callback();
	tile_renderer = p_renderer;
	using_external_renderer = p_renderer.is_valid();
	if (tile_renderer.is_valid()) {
		tile_renderer->set_contract_main_device(_get_manager_main_device(device_manager));
	}
	_bind_output_invalidation_callback();
}

void TileRasterizer::set_device_manager(Ref<RenderDeviceManager> p_device_manager) {
    device_manager = p_device_manager;
    if (tile_renderer.is_valid()) {
		tile_renderer->set_contract_main_device(_get_manager_main_device(device_manager));
    }
}

void TileRasterizer::_bind_output_invalidation_callback() {
    if (!tile_renderer.is_valid()) {
        return;
    }
    tile_renderer->set_output_invalidation_callback([this]() {
        clear_output_resource_tracking();
    });
}

void TileRasterizer::_unbind_output_invalidation_callback() {
    if (!tile_renderer.is_valid()) {
        return;
    }
    tile_renderer->clear_output_invalidation_callback();
}

void TileRasterizer::track_output_resources(const RID &p_color_output, RenderingDevice *p_color_device,
		const RID &p_depth_output, RenderingDevice *p_depth_device) {
	if (!device_manager.is_valid()) {
		return;
	}

	RenderingDevice *main_device = _get_contract_main_device(device_manager);
	RenderDeviceManager *manager_ptr = device_manager.ptr();

	if (p_color_output.is_valid()) {
		if (tracked_color_output.is_valid() && tracked_color_output != p_color_output) {
			device_manager->forget_resource(tracked_color_output);
		}
		RenderingDevice *owner = p_color_device ? p_color_device : main_device;
		OutputOwnershipContractResult contract = _enforce_texture_owner_contract(
				"tile_color_output", p_color_output, owner, manager_ptr, main_device);
		if (contract != OutputOwnershipContractResult::VIOLATION) {
			device_manager->track_resource(p_color_output, owner, false, "tile_renderer_color_output");
			tracked_color_output = p_color_output;
		} else if (tracked_color_output.is_valid()) {
			device_manager->forget_resource(tracked_color_output);
			tracked_color_output = RID();
		}
	} else if (tracked_color_output.is_valid()) {
		device_manager->forget_resource(tracked_color_output);
		tracked_color_output = RID();
    }

	if (p_depth_output.is_valid()) {
		if (tracked_depth_output.is_valid() && tracked_depth_output != p_depth_output) {
			device_manager->forget_resource(tracked_depth_output);
		}
		RenderingDevice *owner = p_depth_device ? p_depth_device : main_device;
		OutputOwnershipContractResult contract = _enforce_texture_owner_contract(
				"tile_depth_output", p_depth_output, owner, manager_ptr, main_device);
		if (contract != OutputOwnershipContractResult::VIOLATION) {
			device_manager->track_resource(p_depth_output, owner, false, "tile_renderer_depth_output");
			tracked_depth_output = p_depth_output;
		} else if (tracked_depth_output.is_valid()) {
			device_manager->forget_resource(tracked_depth_output);
			tracked_depth_output = RID();
		}
	} else if (tracked_depth_output.is_valid()) {
		device_manager->forget_resource(tracked_depth_output);
        tracked_depth_output = RID();
    }
}

void TileRasterizer::clear_output_resource_tracking() {
    if (device_manager.is_valid()) {
        if (tracked_color_output.is_valid()) {
            device_manager->forget_resource(tracked_color_output);
        }
        if (tracked_depth_output.is_valid()) {
            device_manager->forget_resource(tracked_depth_output);
        }
    }
    tracked_color_output = RID();
    tracked_depth_output = RID();
}

Error TileRasterizer::initialize(RenderingDevice *p_device, const Vector2i &p_initial_viewport,
        int p_tile_size, RD::DataFormat p_format) {
    // Only create internal tile_renderer if not using external one
    if (tile_renderer.is_null() && !using_external_renderer) {
        tile_renderer.instantiate();
    }
    _bind_output_invalidation_callback();

    if (tile_renderer.is_valid()) {
        tile_renderer->set_contract_main_device(_get_manager_main_device(device_manager));
        tile_renderer->set_gpu_timestamp_capture_enabled(g_gpu_sorting_config.enable_stage_timestamps);
    }

    rd = p_device;
    return tile_renderer->initialize(p_device, p_initial_viewport, p_tile_size, p_format);
}

void TileRasterizer::shutdown() {
    _unbind_output_invalidation_callback();
    clear_output_resource_tracking();
    // Only cleanup if we own the tile_renderer (not using external)
    if (tile_renderer.is_valid() && !using_external_renderer) {
        tile_renderer->cleanup();
    }
    rd = nullptr;
}

bool TileRasterizer::is_ready() const {
    static int ready_check = 0;
    if (++ready_check <= 5 && _is_raster_ready_log_enabled()) {
        bool valid = tile_renderer.is_valid();
        bool init = valid && tile_renderer->is_initialized();
        GS_LOG_RENDERER_DEBUG(vformat("[RASTER-READY] valid=%s init=%s using_external=%s",
                valid ? "yes" : "no", init ? "yes" : "no",
                using_external_renderer ? "yes" : "no"));
    }
    return tile_renderer.is_valid() && tile_renderer->is_initialized();
}

RasterResult TileRasterizer::render(const RasterParams &p_params) {
    RasterResult result;
    result.success = false;

    if (!is_ready()) {
        return result;
    }

    // Build TileRenderer::RenderParams from RasterParams
    TileRenderer::RenderParams render_params;
    render_params.gaussian_buffer = p_params.gaussian_buffer;
    render_params.sorted_indices = p_params.sorted_indices;
    render_params.splat_count = p_params.splat_count;
    render_params.total_gaussians = p_params.total_gaussians;
    render_params.overlap_record_count = p_params.overlap_record_count;
    render_params.world_to_camera_transform = p_params.world_to_camera_transform;
    render_params.projection = p_params.projection;
    render_params.render_projection = p_params.render_projection;
    render_params.viewport_size = p_params.viewport_size;
    render_params.interactive_state_uniform = p_params.interactive_state_uniform;
    render_params.tile_size = p_params.tile_size;
    render_params.compute_raster_policy = p_params.compute_raster_policy;
    render_params.output_is_premultiplied = p_params.output_is_premultiplied;
    render_params.opacity_multiplier = p_params.opacity_multiplier;
    render_params.alpha_floor = p_params.alpha_floor;
    render_params.force_solid_coverage = p_params.force_solid_coverage;
    render_params.cull_far_tolerance = p_params.cull_far_tolerance;
    render_params.tiny_splat_screen_radius = p_params.tiny_splat_screen_radius;
    render_params.max_conic_aspect = p_params.max_conic_aspect;
    render_params.low_pass_filter = p_params.low_pass_filter;
    render_params.opacity_aware_culling = p_params.opacity_aware_culling;
    render_params.visibility_threshold = p_params.visibility_threshold;
    render_params.distance_cull_enabled = p_params.distance_cull_enabled;
    render_params.distance_cull_start = p_params.distance_cull_start;
    render_params.distance_cull_max_rate = p_params.distance_cull_max_rate;
    render_params.lod_blend_enabled = p_params.lod_blend_enabled;
    render_params.lod_blend_factor = p_params.lod_blend_factor;
    render_params.lod_blend_distance = p_params.lod_blend_distance;
    render_params.frame_serial = p_params.frame_serial;
    // Instance rotation inverse for SH view direction correction
    render_params.instance_rotation_inverse = p_params.instance_rotation_inverse;
    render_params.instance_rotation_valid = p_params.instance_rotation_valid;

    // Apply debug options
    render_params.debug_show_tile_bounds = current_debug_options.show_tile_bounds;
    render_params.debug_show_splat_coverage = current_debug_options.show_splat_coverage;
    render_params.debug_show_overflow_tiles = current_debug_options.show_overflow_tiles;
    render_params.debug_show_projection_issues = current_debug_options.show_projection_issues;
    render_params.debug_show_white_albedo = current_debug_options.show_white_albedo;
    render_params.debug_dump_gpu_counters = current_debug_options.dump_gpu_counters;
    render_params.debug_show_tile_grid = current_debug_options.show_tile_grid;
    render_params.debug_show_density_heatmap = current_debug_options.show_density_heatmap;
    render_params.debug_show_performance_hud = current_debug_options.show_performance_hud;
    render_params.debug_overlay_opacity = current_debug_options.overlay_opacity;

    // Set frame serial for timing
    tile_renderer->set_frame_serial(p_params.frame_serial);
    tile_renderer->set_gpu_timestamp_capture_enabled(g_gpu_sorting_config.enable_stage_timestamps);

    // Use device from params if provided, otherwise use the one from initialization
    RenderingDevice *render_device = p_params.device ? p_params.device : rd;

    // Perform render
    RID output = tile_renderer->render(render_device, render_params);

    if (output.is_valid()) {
        result.output_texture = output;
        result.depth_texture = tile_renderer->get_depth_texture();
        result.output_owner = tile_renderer->get_output_texture_owner();
        result.depth_owner = tile_renderer->get_depth_texture_owner();
        if (!result.output_owner) {
            result.output_owner = render_device;
        }
        if (!result.depth_owner) {
            result.depth_owner = render_device;
        }
        result.has_depth = tile_renderer->has_depth_output() && result.depth_texture.is_valid();
        result.depth_copy_compatible = result.has_depth && tile_renderer->is_depth_copy_compatible();

        RenderingDevice *main_device = _get_contract_main_device(device_manager);
        RenderDeviceManager *manager_ptr = device_manager.is_valid() ? device_manager.ptr() : nullptr;

        OutputOwnershipContractResult color_contract = _enforce_texture_owner_contract(
                "tile_color_output", result.output_texture, result.output_owner, manager_ptr, main_device);
        if (color_contract == OutputOwnershipContractResult::VIOLATION) {
            GS_LOG_ERROR_DEFAULT("[TileRasterizer] Color output contract failed; output is disabled for this frame");
            clear_output_resource_tracking();
            result.output_texture = RID();
            result.output_owner = nullptr;
            result.depth_texture = RID();
            result.depth_owner = nullptr;
            result.has_depth = false;
            result.depth_copy_compatible = false;
            result.success = false;
            return result;
        }

        if (result.has_depth) {
            OutputOwnershipContractResult depth_contract = _enforce_texture_owner_contract(
                    "tile_depth_output", result.depth_texture, result.depth_owner, manager_ptr, main_device);
            if (depth_contract == OutputOwnershipContractResult::VIOLATION) {
                GS_LOG_WARN_DEFAULT("[TileRasterizer] Depth output contract failed; depth output is disabled for this frame");
                result.depth_texture = RID();
                result.depth_owner = nullptr;
                result.has_depth = false;
                result.depth_copy_compatible = false;
            }
        }

        result.success = true;
        track_output_resources(result.output_texture, result.output_owner,
                result.depth_texture, result.has_depth ? result.depth_owner : nullptr);
    }

    return result;
}

RasterResult TileRasterizer::render_direct(RenderingDevice *p_device, const TileRenderer::RenderParams &p_params) {
    RasterResult result;
    result.success = false;

    if (!is_ready()) {
        return result;
    }

    // Set frame serial for timing
    tile_renderer->set_frame_serial(p_params.frame_serial);
    tile_renderer->set_gpu_timestamp_capture_enabled(g_gpu_sorting_config.enable_stage_timestamps);

    // Perform render directly with TileRenderer::RenderParams
    RID output = tile_renderer->render(p_device, p_params);

    if (output.is_valid()) {
        result.output_texture = output;
        result.depth_texture = tile_renderer->get_depth_texture();
        result.output_owner = tile_renderer->get_output_texture_owner();
        result.depth_owner = tile_renderer->get_depth_texture_owner();
        if (!result.output_owner) {
            result.output_owner = p_device ? p_device : rd;
        }
        if (!result.depth_owner) {
            result.depth_owner = p_device ? p_device : rd;
        }
        result.has_depth = tile_renderer->has_depth_output() && result.depth_texture.is_valid();
        result.depth_copy_compatible = result.has_depth && tile_renderer->is_depth_copy_compatible();

        RenderingDevice *main_device = _get_contract_main_device(device_manager);
        RenderDeviceManager *manager_ptr = device_manager.is_valid() ? device_manager.ptr() : nullptr;

        OutputOwnershipContractResult color_contract = _enforce_texture_owner_contract(
                "tile_color_output", result.output_texture, result.output_owner, manager_ptr, main_device);
        if (color_contract == OutputOwnershipContractResult::VIOLATION) {
            GS_LOG_ERROR_DEFAULT("[TileRasterizer] Color output contract failed; output is disabled for this frame");
            clear_output_resource_tracking();
            result.output_texture = RID();
            result.output_owner = nullptr;
            result.depth_texture = RID();
            result.depth_owner = nullptr;
            result.has_depth = false;
            result.depth_copy_compatible = false;
            result.success = false;
            return result;
        }

        if (result.has_depth) {
            OutputOwnershipContractResult depth_contract = _enforce_texture_owner_contract(
                    "tile_depth_output", result.depth_texture, result.depth_owner, manager_ptr, main_device);
            if (depth_contract == OutputOwnershipContractResult::VIOLATION) {
                GS_LOG_WARN_DEFAULT("[TileRasterizer] Depth output contract failed; depth output is disabled for this frame");
                result.depth_texture = RID();
                result.depth_owner = nullptr;
                result.has_depth = false;
                result.depth_copy_compatible = false;
            }
        }

        result.success = true;
        track_output_resources(result.output_texture, result.output_owner,
                result.depth_texture, result.has_depth ? result.depth_owner : nullptr);
    }

    return result;
}

Error TileRasterizer::resize(const Vector2i &p_size, RD::DataFormat p_format) {
    if (!tile_renderer.is_valid()) {
        return ERR_UNCONFIGURED;
    }
    return tile_renderer->resize(p_size, p_format);
}

void TileRasterizer::set_output_format(RD::DataFormat p_format) {
    if (tile_renderer.is_valid()) {
        tile_renderer->set_output_format(p_format);
    }
}

RD::DataFormat TileRasterizer::get_output_format() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_output_format();
    }
    return RD::DATA_FORMAT_MAX;
}

RID TileRasterizer::get_output_texture() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_output_texture();
    }
    return RID();
}

RID TileRasterizer::get_depth_texture() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_depth_texture();
    }
    return RID();
}

RenderingDevice *TileRasterizer::get_output_texture_owner() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_output_texture_owner();
    }
    return nullptr;
}

RenderingDevice *TileRasterizer::get_depth_texture_owner() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_depth_texture_owner();
    }
    return nullptr;
}

bool TileRasterizer::has_depth_output() const {
	if (!tile_renderer.is_valid()) {
		return false;
	}
	RID depth = tile_renderer->get_depth_texture();
	if (!depth.is_valid()) {
		return false;
	}
	RenderingDevice *owner = tile_renderer->get_depth_texture_owner();
	if (!owner || !owner->texture_is_valid(depth)) {
		return false;
	}
	RenderingDevice *main_device = _get_contract_main_device(device_manager);
	if (!main_device) {
		return true;
	}
	if (owner == main_device) {
		return true;
	}
	return main_device->texture_is_valid(depth);
}

void TileRasterizer::set_debug_options(const RasterDebugOptions &p_options) {
    current_debug_options = p_options;
}

RasterDebugOptions TileRasterizer::get_debug_options() const {
    return current_debug_options;
}

RasterDebugCounters TileRasterizer::get_debug_counters() const {
    RasterDebugCounters counters;
    if (!tile_renderer.is_valid()) {
        return counters;
    }

    TileRenderer::DebugCounterSnapshot snapshot = tile_renderer->get_debug_counters();
    counters.total_processed = snapshot.total_processed;
    counters.near_far_reject = snapshot.near_far_reject;
    counters.view_distance_reject = snapshot.view_distance_reject;
    counters.quaternion_reject = snapshot.quaternion_reject;
    counters.scale_reject = snapshot.scale_reject;
    counters.clip_w_reject = snapshot.clip_w_reject;
    counters.clip_bounds_reject = snapshot.clip_bounds_reject;
    counters.screen_nan_reject = snapshot.screen_nan_reject;
    counters.focal_length_reject = snapshot.focal_length_reject;
    counters.z_inverse_reject = snapshot.z_inverse_reject;
    counters.covariance_nan_reject = snapshot.covariance_nan_reject;
    counters.determinant_reject = snapshot.determinant_reject;
    counters.radius_reject = snapshot.radius_reject;
    counters.viewport_bounds_reject = snapshot.viewport_bounds_reject;
    counters.bbox_integrity_reject = snapshot.bbox_integrity_reject;
    counters.tile_extent_reject = snapshot.tile_extent_reject;
    counters.success_count = snapshot.success_count;
    counters.extreme_conic_count = snapshot.extreme_conic_count;
    counters.index_mismatch_count = snapshot.index_mismatch_count;

    return counters;
}

RasterOverflowStats TileRasterizer::get_overflow_stats() const {
    RasterOverflowStats stats;
    if (!tile_renderer.is_valid()) {
        return stats;
    }

    TileRenderer::OverflowStatsSnapshot snapshot = tile_renderer->get_overflow_stats();
    stats.overflow_tile_count = snapshot.overflow_tile_count;
    stats.overflow_splats_clamped = snapshot.overflow_splats_clamped;
    stats.overflow_splats_aggregated = snapshot.overflow_splats_aggregated;
    stats.raster_sample_count = snapshot.raster_sample_count;
    stats.raster_splats_iterated = snapshot.raster_splats_iterated;
    stats.raster_splats_contributed = snapshot.raster_splats_contributed;
    // Stamp with the frame serial from the async GPU readback so consumers
    // (e.g. the overflow auto-tuner) can detect stale stats.
    stats.frame_number = tile_renderer->get_overflow_stats_frame_serial();

    return stats;
}

RasterStats TileRasterizer::get_render_stats() const {
    RasterStats stats;
    if (!tile_renderer.is_valid()) {
        return stats;
    }

    TileRenderer::RenderStats renderer_stats = tile_renderer->get_last_render_stats();
    stats.total_tiles = renderer_stats.total_tiles;
    stats.tiles_with_overflow = renderer_stats.tiles_with_overflow;
    stats.empty_tiles = renderer_stats.empty_tiles;
    stats.max_splats_in_tile = renderer_stats.max_splats_in_tile;
    stats.average_splats_per_tile = renderer_stats.average_splats_per_tile;
    stats.has_rendering_errors = renderer_stats.has_rendering_errors;
    stats.overlap_records = renderer_stats.overlap_records;
    stats.overlap_record_budget = renderer_stats.overlap_record_budget;
    stats.overlap_record_budget_effective = renderer_stats.overlap_record_budget_effective;
    stats.overlap_record_budget_configured = renderer_stats.overlap_record_budget_configured;
    stats.overlap_thinning_keep_ratio = renderer_stats.overlap_thinning_keep_ratio;
    stats.occupancy_ratio = renderer_stats.density_metrics.occupancy_ratio;
    stats.dense_ratio = renderer_stats.density_metrics.dense_ratio;
    stats.overflow_ratio = renderer_stats.density_metrics.overflow_ratio;
    stats.compute_raster_frames = renderer_stats.compute_raster_frames;
    stats.fragment_raster_frames = renderer_stats.fragment_raster_frames;
    stats.last_raster_used_compute = renderer_stats.last_raster_used_compute;
    stats.sorted_indices_blend_fallback_active = renderer_stats.sorted_indices_blend_fallback_active;
    stats.sorted_indices_blend_fallback_reason = renderer_stats.sorted_indices_blend_fallback_reason;

    return stats;
}

RasterPerformance TileRasterizer::get_performance() const {
    RasterPerformance perf;
    if (!tile_renderer.is_valid()) {
        return perf;
    }

    perf.tile_assignment_ms = tile_renderer->get_tile_assignment_time();
    perf.rasterization_ms = tile_renderer->get_rasterization_time();
    perf.submission_cpu_ms = tile_renderer->get_last_submission_cpu_ms();
    perf.binning_gpu_ms = tile_renderer->get_last_gpu_binning_time_ms();
    perf.raster_gpu_ms = tile_renderer->get_last_gpu_raster_time_ms();
    perf.prefix_gpu_ms = tile_renderer->get_last_gpu_prefix_time_ms();
    perf.resolve_gpu_ms = tile_renderer->get_last_gpu_resolve_time_ms();
    perf.frame_gpu_ms = tile_renderer->get_last_gpu_frame_time_ms();
    perf.sort_sync_fallback_count = tile_renderer->get_sort_sync_fallback_count();
    perf.timing_frame_serial = tile_renderer->get_gpu_timing_frame_serial();
    perf.timing_frames_behind = tile_renderer->get_gpu_timing_frames_behind();

    return perf;
}

int TileRasterizer::get_tile_size() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_tile_size();
    }
    return TileRenderer::DEFAULT_TILE_SIZE;
}

Vector2i TileRasterizer::get_tile_grid_size() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_tile_grid_size();
    }
    return Vector2i();
}

int TileRasterizer::get_tile_splat_capacity() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_tile_splat_capacity();
    }
    return TileRenderer::MAX_SPLATS_PER_TILE;
}

int TileRasterizer::get_tile_count() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_tile_count();
    }
    return 0;
}

bool TileRasterizer::is_depth_copy_compatible() const {
	if (!tile_renderer.is_valid()) {
		return false;
	}
	if (!tile_renderer->is_depth_copy_compatible()) {
		return false;
	}
	return has_depth_output();
}

void TileRasterizer::set_frame_serial(uint64_t p_serial) {
    if (tile_renderer.is_valid()) {
        tile_renderer->set_frame_serial(p_serial);
    }
}

void TileRasterizer::resolve_gpu_timestamps_async() {
    if (tile_renderer.is_valid()) {
        tile_renderer->resolve_gpu_timestamps_async();
    }
}

void TileRasterizer::set_resolve_debug_mode(int p_mode) {
    if (tile_renderer.is_valid()) {
        tile_renderer->set_resolve_debug_mode(static_cast<TileRenderer::ResolveDebugMode>(p_mode));
    }
}

RID TileRasterizer::get_debug_counter_buffer() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_debug_counter_buffer();
    }
    return RID();
}

Vector<uint32_t> TileRasterizer::get_tile_density_snapshot() const {
    if (tile_renderer.is_valid()) {
        return tile_renderer->get_tile_density_snapshot();
    }
    return Vector<uint32_t>();
}

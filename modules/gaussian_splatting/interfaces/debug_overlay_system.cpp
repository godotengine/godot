#include "debug_overlay_system.h"
#include "debug_overlay_macros.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"
#include "servers/rendering_server.h"
#include "../core/gaussian_splat_manager.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/render_route_labels.h"
#include "gpu_sorting_pipeline.h"

DebugOverlayOptions DebugOverlayQueryView::get_options() const {
    return system ? system->get_options() : DebugOverlayOptions();
}

DebugCounterSnapshot DebugOverlayQueryView::get_debug_counters() const {
    return system ? system->get_debug_counters() : DebugCounterSnapshot();
}

Dictionary DebugOverlayQueryView::get_binning_debug_counters() const {
    return system ? system->get_binning_debug_counters() : Dictionary();
}

bool DebugOverlayQueryView::is_dirty() const {
    return system ? system->is_dirty() : false;
}

uint64_t DebugOverlayQueryView::get_version() const {
    return system ? system->get_version() : 0;
}

bool DebugOverlayQueryView::has_active_overlays() const {
    return system ? system->has_active_overlays() : false;
}

const Vector<String> &DebugOverlayQueryView::get_hud_lines() const {
    static const Vector<String> empty;
    if (debug_state) {
        return debug_state->hud_lines;
    }
    return system ? system->get_hud_lines() : empty;
}

uint32_t DebugOverlayQueryView::get_tile_density_peak() const {
    if (debug_state) {
        return debug_state->tile_density_peak;
    }
    return system ? system->get_tile_density_peak() : 0;
}

float DebugOverlayQueryView::get_tile_density_average() const {
    if (debug_state) {
        return debug_state->tile_density_average;
    }
    return system ? system->get_tile_density_average() : 0.0f;
}

const Vector<uint32_t> &DebugOverlayQueryView::get_tile_density_cache() const {
    static const Vector<uint32_t> empty;
    if (debug_state) {
        return debug_state->tile_density_cache;
    }
    return system ? system->get_tile_density_cache() : empty;
}

int DebugOverlayQueryView::get_tile_density_width() const {
    if (debug_state) {
        return debug_state->tile_density_width;
    }
    return system ? system->get_tile_density_width() : 0;
}

int DebugOverlayQueryView::get_tile_density_height() const {
    if (debug_state) {
        return debug_state->tile_density_height;
    }
    return system ? system->get_tile_density_height() : 0;
}

void DebugOverlayCommandSink::set_show_tile_grid(bool p_enabled) const {
    if (system) {
        system->set_renderer_show_tile_grid(*this, p_enabled);
    }
}

void DebugOverlayCommandSink::set_show_density_heatmap(bool p_enabled) const {
    if (system) {
        system->set_renderer_show_density_heatmap(*this, p_enabled);
    }
}

void DebugOverlayCommandSink::set_show_performance_hud(bool p_enabled) const {
    if (system) {
        system->set_renderer_show_performance_hud(*this, p_enabled);
    }
}

void DebugOverlayCommandSink::set_show_residency_hud(bool p_enabled) const {
    if (system) {
        system->set_renderer_show_residency_hud(*this, p_enabled);
    }
}

void DebugOverlayCommandSink::set_show_device_boundaries(bool p_enabled) const {
    if (system) {
        system->set_renderer_show_device_boundaries(*this, p_enabled);
    }
}

void DebugOverlayCommandSink::set_show_texture_states(bool p_enabled) const {
    if (system) {
        system->set_renderer_show_texture_states(*this, p_enabled);
    }
}

void DebugOverlayCommandSink::set_show_shadow_opacity(bool p_enabled) const {
    if (system) {
        system->set_show_shadow_opacity(p_enabled);
    }
}

void DebugOverlayCommandSink::set_overlay_opacity(float p_opacity) const {
    if (system) {
        system->set_renderer_overlay_opacity(*this, p_opacity);
    }
}

void DebugOverlayCommandSink::set_dump_gpu_counters(bool p_enabled) const {
    if (system) {
        system->set_dump_gpu_counters(p_enabled);
    }
}

void DebugOverlayCommandSink::invalidate_overlay(bool p_increment_version) const {
    if (system) {
        system->invalidate_renderer_overlay(*this, p_increment_version);
    }
}

void DebugOverlayCommandSink::invalidate_hud(bool p_increment_version) const {
    if (system) {
        system->invalidate_renderer_hud(*this, p_increment_version);
    }
}

DebugOverlayQueryView DebugOverlaySystem::build_query_view(const GaussianSplatRenderer *p_renderer) const {
    DebugOverlayQueryView query_view(this);
    if (!p_renderer) {
        return query_view;
    }

    GaussianSplatRenderer::FrameStateProvider frame_provider(const_cast<GaussianSplatRenderer *>(p_renderer));
    const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;

    query_view.debug_state = &p_renderer->get_debug_state();
    query_view.frame_state = &state_view.get_frame_state_view();
    query_view.sorting_state = &state_view.get_sorting_state_view();
    query_view.performance_state = &state_view.get_performance_state_view();
    query_view.device_state = &p_renderer->get_device_state();
    query_view.subsystem_state = &state_view.get_subsystem_state_view();
    query_view.submission_device = const_cast<GaussianSplatRenderer *>(p_renderer)->get_submission_device();
    query_view.main_rendering_device = p_renderer->get_main_rendering_device();
    return query_view;
}

DebugOverlayCommandSink DebugOverlaySystem::build_command_sink(GaussianSplatRenderer *p_renderer) {
    DebugOverlayCommandSink command_sink(this);
    if (!p_renderer) {
        return command_sink;
    }

    command_sink.debug_state = &p_renderer->get_debug_state();
    command_sink.debug_config = &p_renderer->get_debug_config();
    return command_sink;
}

void DebugOverlaySystem::_bind_methods() {
    // Bind methods for script access if needed
}

DebugOverlaySystem::DebugOverlaySystem() {
}

DebugOverlaySystem::~DebugOverlaySystem() {
    shutdown();
}

void DebugOverlaySystem::initialize() {
    options = DebugOverlayOptions();
    counters = DebugCounterSnapshot();
    binning_counters.clear();
    hud_lines.clear();
    tile_density_cache.clear();
    tile_density_width = 0;
    tile_density_height = 0;
    tile_density_peak = 0;
    tile_density_average = 0.0f;
    dirty = false;
    version = 0;
}

void DebugOverlaySystem::shutdown() {
    binning_counters.clear();
    hud_lines.clear();
    tile_density_cache.clear();
}

void DebugOverlaySystem::set_options(const DebugOverlayOptions &p_options) {
    options = p_options;
    _mark_dirty();
}

DebugOverlayOptions DebugOverlaySystem::get_options() const {
    return options;
}

// Standard boolean setters - use macros to reduce boilerplate
GS_DEBUG_OVERLAY_SETTER_IMPL(show_tile_bounds)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_splat_coverage)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_tile_grid)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_overflow_tiles)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_projection_issues)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_white_albedo)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_density_heatmap)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_shadow_opacity)

// Mutually exclusive setters - resolve_input and resolve_output cannot both be enabled
GS_DEBUG_OVERLAY_SETTER_EXCLUSIVE_IMPL(show_resolve_input, show_resolve_output)
GS_DEBUG_OVERLAY_SETTER_EXCLUSIVE_IMPL(show_resolve_output, show_resolve_input)

GS_DEBUG_OVERLAY_SETTER_IMPL(show_performance_hud)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_residency_hud)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_device_boundaries)
GS_DEBUG_OVERLAY_SETTER_IMPL(show_texture_states)

void DebugOverlaySystem::set_overlay_opacity(float p_opacity) {
    p_opacity = CLAMP(p_opacity, 0.0f, 1.0f);
    if (options.overlay_opacity != p_opacity) {
        options.overlay_opacity = p_opacity;
        _mark_dirty();
    }
}

GS_DEBUG_OVERLAY_SETTER_IMPL(dump_gpu_counters)

DebugCounterSnapshot DebugOverlaySystem::get_debug_counters() const {
    return counters;
}

Dictionary DebugOverlaySystem::get_binning_debug_counters() const {
    return binning_counters;
}

void DebugOverlaySystem::reset_counters() {
    counters = DebugCounterSnapshot();
    binning_counters.clear();
}

bool DebugOverlaySystem::has_active_overlays() const {
    return options.show_tile_bounds ||
           options.show_splat_coverage ||
           options.show_tile_grid ||
           options.show_overflow_tiles ||
           options.show_projection_issues ||
           options.show_white_albedo ||
           options.show_density_heatmap ||
           options.show_shadow_opacity ||
           options.show_resolve_input ||
           options.show_resolve_output ||
           options.show_performance_hud ||
           options.show_residency_hud ||
           options.show_device_boundaries ||
           options.show_texture_states;
}

void DebugOverlaySystem::update_counters(const DebugCounterSnapshot &p_counters) {
    counters = p_counters;
}

void DebugOverlaySystem::update_binning_counters(const Dictionary &p_counters) {
    binning_counters = p_counters;
}

void DebugOverlaySystem::_mark_dirty() {
    dirty = true;
    version++;
}

// HUD building and overlay statistics - extracted from GaussianSplatRenderer god class

void DebugOverlaySystem::rebuild_overlay_statistics_from_tile_density() {
    if (!options.show_tile_grid && !options.show_density_heatmap) {
        tile_density_peak = 0;
        tile_density_average = 0.0f;
        return;
    }

    if (tile_density_cache.is_empty()) {
        tile_density_peak = 0;
        tile_density_average = 0.0f;
        return;
    }

    const uint32_t *density_ptr = tile_density_cache.ptr();
    uint64_t total = 0;
    uint32_t peak = 0;
    uint32_t non_zero_tiles = 0;

    const int density_count = tile_density_cache.size();
    for (int i = 0; i < density_count; i++) {
        uint32_t value = density_ptr[i];
        peak = MAX(peak, value);
        if (value > 0) {
            total += value;
            non_zero_tiles++;
        }
    }

    tile_density_peak = peak;
    tile_density_average = non_zero_tiles > 0 ? float(total) / float(non_zero_tiles) : 0.0f;
}

void DebugOverlaySystem::update_tile_density_cache(const Vector<uint32_t> &p_tile_counts,
        const Vector2i &p_tile_grid, uint32_t p_peak, float p_average) {
    tile_density_cache = p_tile_counts;
    tile_density_width = p_tile_grid.x;
    tile_density_height = p_tile_grid.y;
    tile_density_peak = p_peak;
    tile_density_average = p_average;
}

void DebugOverlaySystem::clear_tile_density_cache() {
    tile_density_cache.clear();
    tile_density_width = 0;
    tile_density_height = 0;
    tile_density_peak = 0;
    tile_density_average = 0.0f;
}

// Renderer-syncing setters - overlay invalidation variants
GS_DEBUG_OVERLAY_RENDERER_SETTER_OVERLAY_IMPL(show_tile_grid)
GS_DEBUG_OVERLAY_RENDERER_SETTER_OVERLAY_IMPL(show_density_heatmap)
GS_DEBUG_OVERLAY_RENDERER_SETTER_OVERLAY_IMPL(show_device_boundaries)
GS_DEBUG_OVERLAY_RENDERER_SETTER_OVERLAY_IMPL(show_texture_states)

// Renderer-syncing setters - HUD invalidation variants
GS_DEBUG_OVERLAY_RENDERER_SETTER_HUD_IMPL(show_performance_hud)
GS_DEBUG_OVERLAY_RENDERER_SETTER_HUD_IMPL(show_residency_hud)

void DebugOverlaySystem::set_renderer_show_tile_grid(GaussianSplatRenderer *p_renderer, bool p_enabled) {
    set_renderer_show_tile_grid(build_command_sink(p_renderer), p_enabled);
}

void DebugOverlaySystem::set_renderer_show_density_heatmap(GaussianSplatRenderer *p_renderer, bool p_enabled) {
    set_renderer_show_density_heatmap(build_command_sink(p_renderer), p_enabled);
}

void DebugOverlaySystem::set_renderer_show_performance_hud(GaussianSplatRenderer *p_renderer, bool p_enabled) {
    set_renderer_show_performance_hud(build_command_sink(p_renderer), p_enabled);
}

void DebugOverlaySystem::set_renderer_show_residency_hud(GaussianSplatRenderer *p_renderer, bool p_enabled) {
    set_renderer_show_residency_hud(build_command_sink(p_renderer), p_enabled);
}

void DebugOverlaySystem::set_renderer_show_device_boundaries(GaussianSplatRenderer *p_renderer, bool p_enabled) {
    set_renderer_show_device_boundaries(build_command_sink(p_renderer), p_enabled);
}

void DebugOverlaySystem::set_renderer_show_texture_states(GaussianSplatRenderer *p_renderer, bool p_enabled) {
    set_renderer_show_texture_states(build_command_sink(p_renderer), p_enabled);
}

void DebugOverlaySystem::set_renderer_overlay_opacity(const DebugOverlayCommandSink &p_sink, float p_opacity) {
    if (!p_sink.debug_config) {
        return;
    }

    float clamped = CLAMP(p_opacity, 0.0f, 1.0f);
    auto &debug_config = *p_sink.debug_config;
    if (Math::is_equal_approx(debug_config.overlay_opacity, clamped)) {
        return;
    }

    debug_config.overlay_opacity = clamped;
    set_overlay_opacity(clamped);
    invalidate_renderer_overlay(p_sink, true);
}

void DebugOverlaySystem::set_renderer_overlay_opacity(GaussianSplatRenderer *p_renderer, float p_opacity) {
    set_renderer_overlay_opacity(build_command_sink(p_renderer), p_opacity);
}

void DebugOverlaySystem::invalidate_renderer_overlay(const DebugOverlayCommandSink &p_sink, bool p_increment_version) {
    if (!p_sink.debug_state) {
        return;
    }

    auto &debug_state = *p_sink.debug_state;
    if (p_increment_version) {
        debug_state.overlay_version++;
    }

    debug_state.overlay_dirty = true;

    if (!debug_state.show_tile_grid && !debug_state.show_density_heatmap) {
        debug_state.tile_density_cache.clear();
        debug_state.tile_density_width = 0;
        debug_state.tile_density_height = 0;
        debug_state.tile_density_peak = 0;
        debug_state.tile_density_average = 0.0f;
    }
}

void DebugOverlaySystem::invalidate_renderer_overlay(GaussianSplatRenderer *p_renderer, bool p_increment_version) {
    invalidate_renderer_overlay(build_command_sink(p_renderer), p_increment_version);
}

void DebugOverlaySystem::invalidate_renderer_hud(const DebugOverlayCommandSink &p_sink, bool p_increment_version) {
    if (!p_sink.debug_state) {
        return;
    }

    auto &debug_state = *p_sink.debug_state;
    if (p_increment_version) {
        debug_state.hud_version++;
    }

    if (!debug_state.show_performance_hud && !debug_state.show_residency_hud) {
        debug_state.hud_lines.clear();
    }

    debug_state.hud_dirty = true;
}

void DebugOverlaySystem::invalidate_renderer_hud(GaussianSplatRenderer *p_renderer, bool p_increment_version) {
    invalidate_renderer_hud(build_command_sink(p_renderer), p_increment_version);
}

void DebugOverlaySystem::rebuild_renderer_overlay_statistics_from_cache(const DebugOverlayQueryView &p_query_view,
        const DebugOverlayCommandSink &p_sink) {
    if (!p_query_view.debug_state || !p_sink.debug_state) {
        return;
    }

    const auto &debug_state_view = *p_query_view.debug_state;
    auto &debug_state = *p_sink.debug_state;
    if (!debug_state_view.show_tile_grid && !debug_state_view.show_density_heatmap) {
        debug_state.overlay_dirty = false;
        debug_state.tile_density_peak = 0;
        debug_state.tile_density_average = 0.0f;
        return;
    }

    if (debug_state_view.tile_density_cache.is_empty()) {
        debug_state.overlay_dirty = false;
        debug_state.tile_density_peak = 0;
        debug_state.tile_density_average = 0.0f;
        return;
    }

    const uint32_t *density_ptr = debug_state_view.tile_density_cache.ptr();
    uint64_t total = 0;
    uint32_t peak = 0;
    uint32_t non_zero_tiles = 0;

    const int density_count = debug_state_view.tile_density_cache.size();
    for (int i = 0; i < density_count; i++) {
        uint32_t value = density_ptr[i];
        peak = MAX(peak, value);
        if (value > 0) {
            total += value;
            non_zero_tiles++;
        }
    }

    debug_state.tile_density_peak = peak;
    debug_state.tile_density_average = non_zero_tiles > 0 ? float(total) / float(non_zero_tiles) : 0.0f;
    debug_state.overlay_dirty = false;
}

void DebugOverlaySystem::rebuild_renderer_overlay_statistics_from_cache(GaussianSplatRenderer *p_renderer) {
    DebugOverlayQueryView query_view = build_query_view(p_renderer);
    rebuild_renderer_overlay_statistics_from_cache(query_view, build_command_sink(p_renderer));
}

void DebugOverlaySystem::rebuild_renderer_performance_hud_lines(const DebugOverlayQueryView &p_query_view,
        const DebugOverlayCommandSink &p_sink) {
    if (!p_query_view.debug_state || !p_query_view.frame_state || !p_query_view.sorting_state ||
            !p_query_view.performance_state || !p_query_view.device_state ||
            !p_query_view.subsystem_state || !p_sink.debug_state) {
        return;
    }

    const auto &debug_state_view = *p_query_view.debug_state;
    const auto &frame_state = *p_query_view.frame_state;
    const auto &sorting_state = *p_query_view.sorting_state;
    const auto &performance_state = *p_query_view.performance_state;
    const auto &device_state = *p_query_view.device_state;
    const auto &subsystem_state = *p_query_view.subsystem_state;
    auto &debug_state = *p_sink.debug_state;

    debug_state.hud_lines.clear();

    if (debug_state_view.show_performance_hud) {
        debug_state.hud_lines.push_back(vformat("Route: %s",
                GaussianRenderRouteLabels::format_route_uid(debug_state_view.route_uid)));
        debug_state.hud_lines.push_back(vformat("Sort Route: %s",
                GaussianRenderRouteLabels::format_sort_route_uid(debug_state_view.sort_route_uid)));
        debug_state.hud_lines.push_back(vformat("Cull Route: %s",
                GaussianRenderRouteLabels::format_cull_route_uid(performance_state.metrics.cull_route_uid)));
        debug_state.hud_lines.push_back(vformat("Requested Policy: %s (%s)",
                debug_state_view.requested_route_policy,
                debug_state_view.requested_route_policy_source));
        debug_state.hud_lines.push_back(vformat("Instance Backend: %s",
                debug_state_view.instance_backend_policy));
        if (!debug_state_view.backend_selection_reason.is_empty() &&
                debug_state_view.backend_selection_reason != "not_evaluated") {
            debug_state.hud_lines.push_back(vformat("Backend Reason: %s",
                    GaussianRenderRouteLabels::format_backend_selection_reason(
                            debug_state_view.backend_selection_reason)));
        }
        debug_state.hud_lines.push_back(vformat("Instance Contract: %s (%s)",
                debug_state_view.instance_contract_shape,
                debug_state_view.instance_contract_ready ? "ready" : "not ready"));
        if (!performance_state.metrics.cull_route_reason.is_empty()) {
            debug_state.hud_lines.push_back(vformat("Cull Reason: %s",
                    GaussianRenderRouteLabels::format_cull_route_reason(
                            performance_state.metrics.cull_route_reason)));
        }
		debug_state.hud_lines.push_back(String("Visible Splats: ") +
				String::num_uint64(frame_state.visible_splat_count.load(std::memory_order_acquire)));
		debug_state.hud_lines.push_back(String("Sorted Splats: ") + String::num_uint64(sorting_state.sorted_splat_count));
		if (!performance_state.metrics.data_source.is_empty()) {
			debug_state.hud_lines.push_back(vformat("Data Source: %s", performance_state.metrics.data_source));
			if (!performance_state.metrics.data_source_error.is_empty()) {
				debug_state.hud_lines.push_back(vformat("Data Error: %s", performance_state.metrics.data_source_error));
			}
		}
		float sort_time_ms = debug_state_view.last_sort_time_ms;
		float render_time_ms = debug_state_view.last_render_time_ms;
        if (debug_state_view.last_stage_metrics_valid) {
            sort_time_ms = debug_state_view.last_stage_metrics.sort.sort_time_ms;
            render_time_ms = debug_state_view.last_stage_metrics.raster.render_time_ms;
        }
        debug_state.hud_lines.push_back(vformat("Sort Time: %.2f ms", sort_time_ms));
        debug_state.hud_lines.push_back(vformat("Render Time: %.2f ms", render_time_ms));
        debug_state.hud_lines.push_back(vformat("Tile Assign: %.2f ms", debug_state_view.last_tile_assignment_ms));
        debug_state.hud_lines.push_back(vformat("Tile Raster: %.2f ms", debug_state_view.last_tile_rasterization_ms));
        if (subsystem_state.rasterizer.is_valid()) {
            RasterStats raster_stats = subsystem_state.rasterizer->get_render_stats();
            debug_state.hud_lines.push_back(vformat("Raster Path: %s (compute=%s, fragment=%s)",
                    raster_stats.last_raster_used_compute ? "compute" : "fragment",
                    String::num_uint64(raster_stats.compute_raster_frames),
                    String::num_uint64(raster_stats.fragment_raster_frames)));
            debug_state.hud_lines.push_back(vformat("Tile Size: %d (compute max 32)", subsystem_state.rasterizer->get_tile_size()));
            debug_state.hud_lines.push_back(vformat("Overlap Records: %u / %u (effective %u)",
                    raster_stats.overlap_records,
                    raster_stats.overlap_record_budget_configured,
                    raster_stats.overlap_record_budget_effective));
            if (raster_stats.sorted_indices_blend_fallback_active) {
                debug_state.hud_lines.push_back(vformat("Blend Sort Fallback: %s",
                        raster_stats.sorted_indices_blend_fallback_reason));
            }
            if (raster_stats.overlap_thinning_keep_ratio < 0.999f) {
                debug_state.hud_lines.push_back(vformat("Overlap Thinning: keep %.1f%%",
                        raster_stats.overlap_thinning_keep_ratio * 100.0f));
            }
        }
        if (debug_state_view.last_stage_metrics_valid) {
            const auto &stage_metrics = debug_state_view.last_stage_metrics;
            debug_state.hud_lines.push_back(vformat("Cull: %.2f ms (cand %u -> vis %u)",
                    stage_metrics.cull.cull_time_ms, stage_metrics.cull.candidate_count, stage_metrics.cull.visible_count));
            if (stage_metrics.sort.did_sort) {
                debug_state.hud_lines.push_back(vformat("Sort: %.2f ms (in %u -> %u)",
                        stage_metrics.sort.sort_time_ms, stage_metrics.sort.input_count, stage_metrics.sort.sorted_count));
            } else {
                debug_state.hud_lines.push_back(vformat("Sort: skipped (in %u)", stage_metrics.sort.input_count));
            }
            const char *raster_label = stage_metrics.raster.reused_cached_render
                    ? "cached"
                    : (stage_metrics.raster.painterly_active ? "painterly" : "baseline");
            debug_state.hud_lines.push_back(vformat("Raster: %.2f ms (%s)", stage_metrics.raster.render_time_ms, raster_label));
            if (stage_metrics.composite_executed) {
                debug_state.hud_lines.push_back(vformat("Composite: %.2f ms", stage_metrics.composite_time_ms));
            } else {
                debug_state.hud_lines.push_back(String("Composite: skipped"));
            }
            auto stage_io_label = [](const GaussianSplatRenderer::StageIO &p_io) -> const char * {
                if (!p_io.validated) {
                    return "n/a";
                }
                return p_io.validation_failed ? "fail" : "ok";
            };
            debug_state.hud_lines.push_back(vformat("Stage IO: cull %s | sort %s | raster %s | comp %s",
                    stage_io_label(stage_metrics.cull_io),
                    stage_io_label(stage_metrics.sort_io),
                    stage_io_label(stage_metrics.raster_io),
                    stage_io_label(stage_metrics.composite_io)));
            const GaussianSplatRenderer::StageIO *failed_io = nullptr;
            const char *failed_label = nullptr;
            if (stage_metrics.cull_io.validation_failed) {
                failed_io = &stage_metrics.cull_io;
                failed_label = "cull";
            } else if (stage_metrics.sort_io.validation_failed) {
                failed_io = &stage_metrics.sort_io;
                failed_label = "sort";
            } else if (stage_metrics.raster_io.validation_failed) {
                failed_io = &stage_metrics.raster_io;
                failed_label = "raster";
            } else if (stage_metrics.composite_io.validation_failed) {
                failed_io = &stage_metrics.composite_io;
                failed_label = "comp";
            }
            if (failed_io && !failed_io->validation_error.is_empty()) {
                debug_state.hud_lines.push_back(vformat("IO Error (%s): %s", failed_label, failed_io->validation_error));
            }
        }
        float fps = (performance_state.metrics.avg_frame_to_frame_ms > 0.001f)
                ? (1000.0f / performance_state.metrics.avg_frame_to_frame_ms)
                : 0.0f;
        debug_state.hud_lines.push_back(
                vformat("Frame Time: %.2f ms (%.1f FPS)", performance_state.metrics.avg_frame_to_frame_ms, fps));
        debug_state.hud_lines.push_back(vformat("GPU Utilization: %.1f%%", performance_state.metrics.gpu_utilization));
        debug_state.hud_lines.push_back(vformat("GPU Frame Time: %.2f ms", performance_state.metrics.gpu_frame_time_ms));
        debug_state.hud_lines.push_back(vformat("GPU Binning: %.2f ms", performance_state.metrics.gpu_tile_binning_time_ms));
        debug_state.hud_lines.push_back(vformat("GPU Prefix: %.2f ms", performance_state.metrics.gpu_tile_prefix_time_ms));
        debug_state.hud_lines.push_back(vformat("GPU Resolve: %.2f ms", performance_state.metrics.gpu_tile_resolve_time_ms));
        debug_state.hud_lines.push_back(vformat("GPU Raster: %.2f ms", performance_state.metrics.gpu_tile_raster_time_ms));
        debug_state.hud_lines.push_back(vformat("GPU Memory: %.2f MB", performance_state.metrics.gpu_memory_usage_mb));

        const Dictionary binning = get_binning_debug_counters();
        if (!binning.is_empty()) {
            debug_state.hud_lines.push_back(vformat("Binning counters: hits=%lld updates=%lld forced=%lld hitrate=%.2f",
                    int64_t(binning.get("sh_cache_hits", int64_t(0))),
                    int64_t(binning.get("sh_cache_updates", int64_t(0))),
                    int64_t(binning.get("sh_cache_forced_updates", int64_t(0))),
                    double(binning.get("sh_cache_hit_rate", 0.0))));
        }


        if (debug_state_view.show_device_boundaries) {
            debug_state.hud_lines.push_back(String());
            debug_state.hud_lines.push_back(String("Device Boundaries"));
            auto append_device = [&](const char *p_label, RenderingDevice *p_device) {
                if (!p_device) {
                    return;
                }
                debug_state.hud_lines.push_back(vformat("%s: %s", p_label, p_device->get_device_name()));
            };
            RenderingDevice *tile_device = nullptr;
            if (subsystem_state.rasterizer.is_valid()) {
                tile_device = subsystem_state.rasterizer->get_output_texture_owner();
                if (!tile_device && subsystem_state.rasterizer->has_depth_output()) {
                    tile_device = subsystem_state.rasterizer->get_depth_texture_owner();
                }
            }

            RenderingDevice *sort_device = nullptr;
            if (subsystem_state.sorting_pipeline.is_valid()) {
                sort_device = subsystem_state.sorting_pipeline->get_sort_resource_device();
            }

            append_device("Primary", device_state.rd);
            append_device("Local", p_query_view.submission_device);
            append_device("Viewport", p_query_view.main_rendering_device);
            append_device("Tile", tile_device);
            append_device("Sorter", sort_device);
        }

        if (debug_state_view.show_texture_states) {
            debug_state.hud_lines.push_back(String());
            debug_state.hud_lines.push_back(String("Texture States"));
            if (subsystem_state.device_manager.is_valid()) {
                uint32_t tracked_count = subsystem_state.device_manager->get_tracked_resource_count();
                debug_state.hud_lines.push_back(vformat("Total tracked resources: %d", tracked_count));
            } else {
                debug_state.hud_lines.push_back(String("RenderDeviceManager not initialized"));
            }
        }
    }

    if (debug_state_view.show_residency_hud) {
        if (!debug_state.hud_lines.is_empty()) {
            debug_state.hud_lines.push_back(String());
        }
        debug_state.hud_lines.push_back(String("Residency"));

        if (GaussianSplatManager *mgr = GaussianSplatManager::get_singleton()) {
            Dictionary stats = mgr->get_global_stats();
            int buffer_count = stats.get(StringName("buffer_count"), 0);
            double total_gaussians = stats.get(StringName("total_gaussians"), 0.0);
            double reported_gaussians = stats.get(StringName("reported_gaussians"), 0.0);
            double total_memory = stats.get(StringName("total_memory_mb"), 0.0);
            double reported_memory = stats.get(StringName("reported_memory_mb"), 0.0);
            debug_state.hud_lines.push_back(vformat("GPU Buffers: %d", buffer_count));
            debug_state.hud_lines.push_back(vformat("Resident Splats: %.0f / %.0f", reported_gaussians, total_gaussians));
            debug_state.hud_lines.push_back(vformat("Resident Memory: %.2f / %.2f MB", reported_memory, total_memory));
        } else {
            debug_state.hud_lines.push_back(String("GPU Buffers: n/a"));
        }
    }

    if (debug_state.hud_lines.is_empty()) {
        debug_state.hud_dirty = false;
        return;
    }

    debug_state.hud_dirty = false;
}

void DebugOverlaySystem::rebuild_renderer_performance_hud_lines(GaussianSplatRenderer *p_renderer) {
    DebugOverlayQueryView query_view = build_query_view(p_renderer);
    rebuild_renderer_performance_hud_lines(query_view, build_command_sink(p_renderer));
}

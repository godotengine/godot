#ifndef GS_DEBUG_OVERLAY_INTERFACES_H
#define GS_DEBUG_OVERLAY_INTERFACES_H

#include "core/variant/dictionary.h"
#include "core/string/ustring.h"

// Debug overlay options - all the visual debug modes
struct DebugOverlayOptions {
    // Tile and coverage visualization
    bool show_tile_bounds = false;
    bool show_splat_coverage = false;
    bool show_tile_grid = false;
    bool show_overflow_tiles = false;

    // Projection and rendering debug
    bool show_projection_issues = false;
    bool show_white_albedo = false;
    bool show_density_heatmap = false;
    bool show_shadow_opacity = false;

    // Resolve pass debug
    bool show_resolve_input = false;
    bool show_resolve_output = false;

    // Performance overlays
    bool show_performance_hud = false;
    bool show_residency_hud = false;

    // Device and texture debug
    bool show_device_boundaries = false;
    bool show_texture_states = false;

    // Overlay appearance
    float overlay_opacity = 0.3f;

    // GPU counter dump
    bool dump_gpu_counters = false;
};

// Debug counter categories
struct DebugCounterSnapshot {
    // Projection rejection counters
    uint32_t total_processed = 0;
    uint32_t near_far_reject = 0;
    uint32_t view_distance_reject = 0;
    uint32_t quaternion_reject = 0;
    uint32_t scale_reject = 0;
    uint32_t clip_w_reject = 0;
    uint32_t clip_bounds_reject = 0;
    uint32_t screen_nan_reject = 0;
    uint32_t focal_length_reject = 0;
    uint32_t z_inverse_reject = 0;
    uint32_t covariance_nan_reject = 0;
    uint32_t determinant_reject = 0;
    uint32_t radius_reject = 0;
    uint32_t viewport_bounds_reject = 0;
    uint32_t bbox_integrity_reject = 0;
    uint32_t tile_extent_reject = 0;
    uint32_t success_count = 0;

    // Conic and index validation
    uint32_t extreme_conic_count = 0;
    uint32_t index_mismatch_count = 0;

    // Tile overflow stats
    uint32_t overflow_tile_count = 0;
    uint32_t overflow_splats_clamped = 0;
    uint32_t overflow_splats_aggregated = 0;

    // Rasterization stats
    uint32_t raster_sample_count = 0;
    uint32_t raster_splats_iterated = 0;
    uint32_t raster_splats_contributed = 0;
};

// Pure abstract interface for debug overlay system
class IDebugOverlaySystem {
public:
    virtual ~IDebugOverlaySystem() = default;

    // Lifecycle
    virtual void initialize() = 0;
    virtual void shutdown() = 0;

    // Bulk options management
    virtual void set_options(const DebugOverlayOptions &p_options) = 0;
    virtual DebugOverlayOptions get_options() const = 0;

    // Individual toggle setters
    virtual void set_show_tile_bounds(bool p_enabled) = 0;
    virtual void set_show_splat_coverage(bool p_enabled) = 0;
    virtual void set_show_tile_grid(bool p_enabled) = 0;
    virtual void set_show_overflow_tiles(bool p_enabled) = 0;
    virtual void set_show_projection_issues(bool p_enabled) = 0;
    virtual void set_show_density_heatmap(bool p_enabled) = 0;
    virtual void set_show_shadow_opacity(bool p_enabled) = 0;
    virtual void set_show_resolve_input(bool p_enabled) = 0;
    virtual void set_show_resolve_output(bool p_enabled) = 0;
    virtual void set_show_performance_hud(bool p_enabled) = 0;
    virtual void set_show_residency_hud(bool p_enabled) = 0;
    virtual void set_show_device_boundaries(bool p_enabled) = 0;
    virtual void set_show_texture_states(bool p_enabled) = 0;
    virtual void set_overlay_opacity(float p_opacity) = 0;
    virtual void set_dump_gpu_counters(bool p_enabled) = 0;

    // Individual toggle getters
    virtual bool get_show_tile_bounds() const = 0;
    virtual bool get_show_splat_coverage() const = 0;
    virtual bool get_show_tile_grid() const = 0;
    virtual bool get_show_overflow_tiles() const = 0;
    virtual bool get_show_projection_issues() const = 0;
    virtual bool get_show_density_heatmap() const = 0;
    virtual bool get_show_shadow_opacity() const = 0;
    virtual bool get_show_resolve_input() const = 0;
    virtual bool get_show_resolve_output() const = 0;
    virtual bool get_show_performance_hud() const = 0;
    virtual bool get_show_residency_hud() const = 0;
    virtual bool get_show_device_boundaries() const = 0;
    virtual bool get_show_texture_states() const = 0;
    virtual float get_overlay_opacity() const = 0;
    virtual bool get_dump_gpu_counters() const = 0;

    // Counter access
    virtual DebugCounterSnapshot get_debug_counters() const = 0;
    virtual Dictionary get_binning_debug_counters() const = 0;
    virtual void reset_counters() = 0;

    // State tracking
    virtual bool is_dirty() const = 0;
    virtual void clear_dirty_flag() = 0;
    virtual uint64_t get_version() const = 0;

    // Check if any debug overlay is active
    virtual bool has_active_overlays() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

#endif // GS_DEBUG_OVERLAY_INTERFACES_H

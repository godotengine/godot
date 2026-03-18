#ifndef GS_RASTERIZER_INTERFACES_H
#define GS_RASTERIZER_INTERFACES_H

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2i.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"

#include "../renderer/tile_render_types.h"

// Rasterization parameters
struct RasterParams {
    // Rendering device (can override the one from initialization)
    RenderingDevice *device = nullptr;

    // Input buffers
    RID gaussian_buffer;
    RID sorted_indices;
    // Post-cull/visible splat count (must match the size of sorted_indices).
    uint32_t splat_count = 0;
    // Required when sorted_indices stores global Gaussian indices (global composite sort).
    // Only safe to fall back to splat_count when indices are known to be local.
    uint32_t total_gaussians = 0;
    // Required in global sort mode: set from the prefix scan IndirectDispatch.element_count.
    uint32_t overlap_record_count = 0;

    // Transform data
    Transform3D world_to_camera_transform;
    Transform3D camera_to_world_transform; // PERF (#659): Pre-computed inverse to avoid affine_inverse() in render()
    Projection projection;
    Projection render_projection; // GPU projection with depth/jitter correction applied.
    Vector2i viewport_size;

    // Optional interactive state
    RID interactive_state_uniform;

    // Rendering options
    int tile_size = 8;
    bool output_is_premultiplied = false;
    float opacity_multiplier = 1.0f;
    float alpha_floor = 0.0f;
    bool force_solid_coverage = false;
    float cull_far_tolerance = 0.05f;
    float tiny_splat_screen_radius = 0.3f;  // Drop subpixel splats to prevent tile overflow (#797)
    float max_conic_aspect = 10.0f;
    float low_pass_filter = 0.35f; // Minimum covariance variance added in projection (lower = sharper)

    // Opacity-aware bounding (FlashGS optimization)
    // When enabled, reduces tile-Gaussian pairs by ~94% using opacity-based radius calculation
    bool opacity_aware_culling = true;
    float visibility_threshold = 0.01f;
    bool distance_cull_enabled = true;
    float distance_cull_start = 30.0f;
    float distance_cull_max_rate = 0.5f;

    // LOD blending (LODGE technique) - eliminates popping during LOD transitions
    bool lod_blend_enabled = true;
    float lod_blend_factor = 1.0f;
    float lod_blend_distance = 5.0f;

    // Compute raster selection (defaults to global config).
    GaussianSplatting::ComputeRasterPolicy compute_raster_policy = GaussianSplatting::ComputeRasterPolicy::Default;

    // Frame tracking
    uint64_t frame_serial = 0;

    // Instance rotation inverse for SH view direction correction.
    // When a GaussianSplatNode3D has a rotation transform, SH coefficients (stored in
    // original capture coordinates) expect view directions in the same frame.
    Basis instance_rotation_inverse = Basis();
    bool instance_rotation_valid = false;
};

// Debug overlay options
struct RasterDebugOptions {
    bool show_tile_bounds = false;
    bool show_splat_coverage = false;
    bool show_overflow_tiles = false;
    bool show_projection_issues = false;
    bool show_white_albedo = false;
    bool dump_gpu_counters = false;
    bool show_tile_grid = false;
    bool show_density_heatmap = false;
    bool show_performance_hud = false;
    float overlay_opacity = 0.3f;
};

// Debug counter statistics
struct RasterDebugCounters {
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
    uint32_t extreme_conic_count = 0;
    uint32_t index_mismatch_count = 0;
};

// Overflow/rasterization statistics
struct RasterOverflowStats {
    uint32_t overflow_tile_count = 0;
    uint32_t overflow_splats_clamped = 0;
    uint32_t overflow_splats_aggregated = 0;
    uint32_t raster_sample_count = 0;
    uint32_t raster_splats_iterated = 0;
    uint32_t raster_splats_contributed = 0;
    // Frame number when these stats were captured. Used by the auto-tuner to detect
    // stale stats from async GPU readback. 0 means the frame number is unknown.
    uint64_t frame_number = 0;
};

// Render statistics
struct RasterStats {
    uint32_t total_tiles = 0;
    uint32_t tiles_with_overflow = 0;
    uint32_t empty_tiles = 0;
    uint32_t max_splats_in_tile = 0;
    float average_splats_per_tile = 0.0f;
    bool has_rendering_errors = false;
    uint32_t overlap_records = 0;
    uint32_t overlap_record_budget = 0;
    uint32_t overlap_record_budget_effective = 0;
    uint32_t overlap_record_budget_configured = 0;
    float overlap_thinning_keep_ratio = 1.0f;
    float occupancy_ratio = 0.0f;
    float dense_ratio = 0.0f;
    float overflow_ratio = 0.0f;
    uint64_t compute_raster_frames = 0;
    uint64_t fragment_raster_frames = 0;
    bool last_raster_used_compute = false;
    bool sorted_indices_blend_fallback_active = false;
    String sorted_indices_blend_fallback_reason;
};

// Performance timing
struct RasterPerformance {
    float tile_assignment_ms = 0.0f;
    float rasterization_ms = 0.0f;
    float submission_cpu_ms = 0.0f;
    float binning_gpu_ms = 0.0f;
    float raster_gpu_ms = 0.0f;
    float prefix_gpu_ms = 0.0f;
    float resolve_gpu_ms = 0.0f;
    float frame_gpu_ms = 0.0f;
    uint64_t sort_sync_fallback_count = 0;
    uint64_t timing_frame_serial = 0;
    uint32_t timing_frames_behind = 0;
};

// Rasterization result
struct RasterResult {
    RID output_texture;
    RID depth_texture;
    RenderingDevice *output_owner = nullptr;
    RenderingDevice *depth_owner = nullptr;
    bool success = false;
    bool has_depth = false;
    bool depth_copy_compatible = false;
};

// Pure abstract interface for rasterization implementations
class IRasterizer {
public:
    virtual ~IRasterizer() = default;

    // Lifecycle
    virtual Error initialize(RenderingDevice *p_device, const Vector2i &p_initial_viewport = Vector2i(),
            int p_tile_size = -1, RD::DataFormat p_format = RD::DATA_FORMAT_MAX) = 0;
    virtual void shutdown() = 0;
    virtual bool is_ready() const = 0;

    // Rendering
    virtual RasterResult render(const RasterParams &p_params) = 0;
    virtual Error resize(const Vector2i &p_size, RD::DataFormat p_format = RD::DATA_FORMAT_MAX) = 0;

    // Output configuration
    virtual void set_output_format(RD::DataFormat p_format) = 0;
    virtual RD::DataFormat get_output_format() const = 0;

    // Output access
    virtual RID get_output_texture() const = 0;
    virtual RID get_depth_texture() const = 0;
    virtual RenderingDevice *get_output_texture_owner() const = 0;
    virtual RenderingDevice *get_depth_texture_owner() const = 0;
    virtual bool has_depth_output() const = 0;

    // Debug options
    virtual void set_debug_options(const RasterDebugOptions &p_options) = 0;
    virtual RasterDebugOptions get_debug_options() const = 0;

    // Statistics
    virtual RasterDebugCounters get_debug_counters() const = 0;
    virtual RasterOverflowStats get_overflow_stats() const = 0;
    virtual RasterStats get_render_stats() const = 0;
    virtual RasterPerformance get_performance() const = 0;

    // Configuration
    virtual int get_tile_size() const = 0;
    virtual Vector2i get_tile_grid_size() const = 0;
    virtual int get_tile_splat_capacity() const = 0;
    virtual int get_tile_count() const = 0;
    virtual bool is_depth_copy_compatible() const = 0;

    // Frame management
    virtual void set_frame_serial(uint64_t p_serial) = 0;

    // GPU timing
    virtual void resolve_gpu_timestamps_async() = 0;

    // Debug mode
    virtual void set_resolve_debug_mode(int p_mode) = 0;

    // Advanced debug access (for diagnostic tools)
    virtual RID get_debug_counter_buffer() const = 0;
    virtual Vector<uint32_t> get_tile_density_snapshot() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

#endif // GS_RASTERIZER_INTERFACES_H

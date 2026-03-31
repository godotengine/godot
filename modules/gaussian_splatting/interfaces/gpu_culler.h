#ifndef GS_GPU_CULLER_H
#define GS_GPU_CULLER_H

#include "culler_interfaces.h"
#include "core/math/aabb.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"
#include "../lod/hierarchical_splat_structure.h"
#include "cluster_culler.h"
#include "../renderer/batched_async_readback.h"
#include <cstdint>
#include <memory>

// Forward declarations
class GaussianSplatManager;
class ClusterCuller;
class GaussianData;
class IOverflowAutoTuner;
struct RasterOverflowStats;

struct StaticChunk {
    AABB bounds;
    Vector<uint32_t> indices;
    Vector3 center;
    float radius = 0.0f;
};

// GPU-based frustum culler using compute shaders
class GPUCuller : public RefCounted, public ICuller {
    GDCLASS(GPUCuller, RefCounted);

public:
    // Shader variant groups for feature detection
    enum ShaderGroup {
        SHADER_GROUP_STANDARD = 0,   // Fallback path using atomicAdd
        SHADER_GROUP_SUBGROUPS = 1,  // Optimized path using subgroup operations
    };
    struct CullingConfig {
        bool lod_enabled = true;
        float lod_bias = 1.0f;
        bool frustum_culling = true;
        bool gpu_culling_enabled = true;
        bool gpu_culling_readback_enabled = true; // enable GPU readback so GPU cull can drive visibility
        bool temporal_coherence = true;

        // Cluster-level coarse culling (LiteGS-style)
        // Groups splats into clusters of 32-256 and tests cluster AABBs first
        bool cluster_culling_enabled = true;         // Enable two-level cluster culling
        uint32_t cluster_target_size = 128;          // Target splats per cluster (32-256)
        float cluster_frustum_slack = 2.0f;          // Slack factor for cluster AABB tests
        bool cluster_use_morton_order = true;        // Use Morton (Z-order) curve for spatial locality
        bool cluster_use_indirect_dispatch = true;   // Use indirect dispatch for fine pass
        float lod_min_screen_size = 1.5f;
        float lod_max_distance = 150.0f;
        bool lod_bias_override = false;
        bool lod_min_screen_size_override = false;
        bool lod_max_distance_override = false;
        float lod_cached_min_screen_threshold = 0.0f;
        float lod_cached_max_distance = 0.0f;
        float lod_cached_max_distance_sq = 0.0f;
        bool lod_cache_dirty = true;
        float lod_project_bias = 1.0f;
        float importance_cull_threshold = 0.0f;
        float importance_cull_baseline = 0.0f;
        float cull_radius_multiplier = 3.0f;
        float cull_frustum_plane_slack = 2.0f;
        float cull_near_tolerance = 0.05f;
        float cull_far_tolerance = 0.05f; // extra band on far plane (5% default)
        bool cull_params_dirty = false;
        bool importance_cull_override = false;
        Size2i last_cull_viewport_size = Size2i(1280, 720);

        // Opacity-aware bounding (FlashGS optimization)
        // When enabled, reduces tile-Gaussian pairs by ~94% using opacity-based radius
        bool opacity_aware_culling = true;
        float visibility_threshold = 0.01f; // tau: minimum visible contribution

        // Distance-based probabilistic culling (tile binning)
        bool distance_cull_enabled = true;
        float distance_cull_start = 30.0f;
        float distance_cull_max_rate = 0.5f;
    };

    struct CullingState {
        LocalVector<uint32_t> culled_indices;
        LocalVector<float> culled_distances_sq;
        LocalVector<float> culled_importance_weights;
        RID gpu_visible_indices_buffer;
        RenderingDevice *gpu_visible_indices_device = nullptr;
        uint32_t gpu_visible_indices_count = 0;
        uint32_t total_splats_pre_cull = 0;
        uint64_t preculled_generation = 0;
        uint32_t culled_by_frustum = 0;
        uint32_t culled_by_distance = 0;
        uint32_t culled_by_screen = 0;
        uint32_t culled_by_importance = 0;
        uint32_t culled_by_limit = 0;
        float cull_time_ms = 0.0f;
        int culling_octree_max_depth = 8;
        uint32_t culling_min_gaussians = 32;
        float tiny_splat_screen_radius_px = 0.3f; // drop subpixel splats (minor axis < 0.3px) to prevent tile overflow (#797)
        float tiny_splat_screen_radius_baseline = 0.3f;
        // Overflow auto-tuner: DISABLED by default (January 2025)
        // The auto-tuner's feedback loop caused exponential splat decay bug:
        // - importance_threshold kept increasing even without true overflow
        // - This caused more splats to be culled each frame
        // - Result: 84K -> 36K -> 12K -> 538 -> 0 splats over ~60 frames
        // Re-enable at your own risk via cull/overflow_autotune_enabled property.
        // Root cause (stale overflow stats) fixed in ISSUE-033: RasterOverflowStats now
        // carries a frame_number, and OverflowAutoTuner discards stats > 2 frames old.
        bool overflow_autotune_enabled = false;
        float overflow_autotune_trigger_ratio = 0.0025f; // 0.25% overflow triggers tightening
        float overflow_autotune_importance_step = 0.001f;
        float overflow_autotune_importance_decay = 0.0005f;
        float overflow_autotune_importance_max = 0.03f; // max delta above baseline
        float overflow_autotune_tiny_step = 0.05f; // pixels
        float overflow_autotune_tiny_decay = 0.02f; // pixels per frame when overflow relieved
        float overflow_autotune_tiny_max = 1.0f; // max delta above baseline
        bool hierarchical_structure_dirty = true;
        std::unique_ptr<GaussianSplatting::HierarchicalSplatStructure> hierarchical_structure;
        Vector<StaticChunk> static_chunks;
        uint64_t static_chunks_revision = 1;
        LocalVector<int> visible_static_chunk_indices;
        float sort_cache_angle_cos_threshold = 0.0f;
        float sort_cache_position_threshold_sq = 0.0f;
        bool static_sort_cache_enabled = false;
    };

    struct CullingInputs {
        Ref<GaussianData> gaussian_data;
        const LocalVector<Vector3> *test_positions = nullptr;
        const LocalVector<Vector3> *test_scales = nullptr;
        const LocalVector<uint32_t> *preculled_indices = nullptr;
        uint64_t preculled_generation = 0;
        uint32_t preculled_total_splats = 0;
        String cull_route_uid;
        String cull_route_reason;
        uint32_t max_splats = 0;
        bool readback_indices = true;
        bool readback_distances = true;
        bool readback_importance = true;
    };

    struct InstancePipelineInputs {
        RID instance_buffer;
        RID asset_meta_buffer;
        RID asset_chunk_index_buffer;
        RID chunk_meta_buffer;
        RID visible_chunk_buffer;
        RID counter_buffer;
        RenderingDevice *device = nullptr;
        uint32_t instance_count = 0;
        uint32_t dispatch_chunk_count = 0;
        uint32_t max_visible_chunks = 0;
    };

    struct CullingSummary {
        uint32_t visible_after_culling = 0;
        uint32_t culling_candidate_count = 0;
        uint32_t culled_frustum_count = 0;
        uint32_t culled_distance_count = 0;
        uint32_t culled_screen_count = 0;
        uint32_t culled_importance_count = 0;
        String cull_route_uid;
        String cull_route_reason;
        bool used_instance_pipeline = false;
        bool used_hierarchical_culling = false;
        float culling_time_ms = 0.0f;

        // Cluster-level culling stats
        bool used_cluster_culling = false;
        uint32_t total_clusters = 0;
        uint32_t visible_clusters = 0;
        uint32_t culled_clusters = 0;
        float cluster_cull_time_ms = 0.0f;
        float cluster_cull_ratio = 0.0f;  // Percentage of splats skipped via cluster cull
    };

    GPUCuller();
    ~GPUCuller();

    // ICuller interface
    Error initialize(RenderingDevice *p_device) override;
    void shutdown() override;
    bool is_ready() const override;
    CullResult cull(const CullParams &p_params, const CullInputBuffers &p_input) override;
    String get_name() const override { return "GPUCuller"; }
    bool is_gpu_based() const override { return true; }

    // GPU-specific configuration
    void set_readback_enabled(bool p_enabled) { readback_enabled = p_enabled; }
    bool is_readback_enabled() const { return readback_enabled; }
    CullingConfig &get_config() { return culling_config; }
    const CullingConfig &get_config() const { return culling_config; }
    CullingState &get_state() { return culling_state; }
    const CullingState &get_state() const { return culling_state; }
    void set_instance_pipeline_inputs(const InstancePipelineInputs &p_inputs);
    void clear_instance_pipeline_inputs();
    uint32_t get_last_instance_visible_chunk_count() const { return last_instance_visible_chunk_count; }
    void invalidate_lod_cache();
    void update_lod_cache();
    void update_culling_settings();
    void ensure_hierarchical_structure(const Ref<GaussianData> &p_data);
    CullingSummary cull_for_view(const Transform3D &p_cam_transform, const Projection &p_projection, const Size2i &p_viewport_size,
            const CullingInputs &p_inputs);
    // Apply overflow feedback from rasterizer stats
    // p_avg_splats_per_tile: average splats per tile (proxy for close-up detection)
    void apply_overflow_feedback(const RasterOverflowStats &p_stats, uint32_t p_splat_count, uint32_t p_tile_count,
            IOverflowAutoTuner *p_auto_tuner, float p_avg_splats_per_tile = 0.0f);

protected:
    static void _bind_methods();

private:
    enum AsyncReadbackType {
        ASYNC_READBACK_COUNTERS = 0,
        ASYNC_READBACK_INDICES = 1,
        ASYNC_READBACK_DISTANCES = 2,
        ASYNC_READBACK_IMPORTANCE = 3,
    };

    struct AsyncReadbackRequest {
        uint64_t request_id = 0;
        uint64_t generation = 0;
        uint32_t max_visible = 0;
        uint32_t total_splats = 0;
        float dispatch_time_ms = 0.0f;
        bool readback_indices = true;
        bool readback_distances = true;
        bool readback_importance = true;
        bool counters_ready = false;
        bool indices_ready = false;
        bool distances_ready = false;
        bool importance_ready = false;
        CullCounters counters;
        Vector<uint8_t> indices_bytes;
        Vector<uint8_t> distance_bytes;
        Vector<uint8_t> importance_bytes;
    };

    struct AsyncReadbackState {
        Vector<AsyncReadbackRequest> pending;
        CullResult last_result;
        uint64_t last_result_id = 0;
        bool last_result_valid = false;
        uint64_t next_request_id = 1;
        uint64_t generation = 1;
        uint64_t last_submitted_id = 0;
    };

    struct InstanceReadbackState {
        bool pending = false;
        uint64_t generation = 1;
        uint64_t pending_request_id = 0;
        uint64_t next_request_id = 1;
        uint64_t last_applied_request_id = 0;
        uint32_t last_frame_chunk_limit = UINT32_MAX;
    };

    struct InstanceUniformSetCache {
        RID uniform_set;
        RID instance_buffer;
        RID asset_meta_buffer;
        RID asset_chunk_index_buffer;
        RID chunk_meta_buffer;
        RID visible_chunk_buffer;
        RID counter_buffer;
        RID param_buffer;
        RID shader_rid;
        RenderingDevice *device = nullptr;
    };

    // GPU resources
    RenderingDevice *rd = nullptr;
    RID shader;
    RID frustum_shader_version;
    RID pipeline;

    // Subgroup support tracking
    bool subgroups_available = false;
    int active_shader_group = SHADER_GROUP_STANDARD;
    RID param_buffer;
    RID counter_buffer;
    RID visible_buffer;
    RID distance_buffer;
    RID importance_buffer;
    RID consolidated_buffer; // Binding 6 - packed readback buffer (shader requires it even if unused)
    RID instance_param_buffer;
    RenderingDevice *instance_resource_device = nullptr;

    // Buffer capacity tracking
    uint32_t buffer_capacity = 0;
    RenderingDevice *resource_device = nullptr;

    // Staging data
    Vector<uint8_t> param_bytes;
    Vector<uint8_t> counter_bytes;
    Vector<uint8_t> instance_param_bytes;

    // Configuration
    bool initialized = false;
    bool readback_enabled = true;
    CullingConfig culling_config;
    CullingState culling_state;
    AsyncReadbackState async_readback_state;
    InstanceReadbackState instance_readback_state;
    InstanceUniformSetCache instance_uniform_set_cache;
    InstancePipelineInputs instance_inputs;
    bool instance_inputs_valid = false;
    uint32_t last_instance_visible_chunk_count = 0;

    // Cluster-level coarse culling
    Ref<ClusterCuller> cluster_culler;
    bool clusters_need_rebuild = true;

    // PERF (#634): Batched async readback to reduce CPU/GPU sync points
    Ref<BatchedAsyncReadback> batched_readback;
    void _on_batched_cull_readback(const Vector<uint8_t> &p_data, int64_t p_user_data);

    // Internal helpers
    void _ensure_shader(RenderingDevice *p_device);
    void _ensure_buffers(RenderingDevice *p_device, uint32_t p_required_capacity);
    void _ensure_instance_param_buffer(RenderingDevice *p_device);
    RID _get_instance_cull_uniform_set(RenderingDevice *p_device, const InstancePipelineInputs &p_inputs);
    void _invalidate_instance_uniform_set_cache();
    void _release_resources();
    RenderingDevice *_acquire_submission_device(RenderingDevice *p_device);
    void _on_cull_readback(const Vector<uint8_t> &p_data, int p_type, int64_t p_request_id, int64_t p_generation);
    void _on_instance_counter_readback(const Vector<uint8_t> &p_data, int64_t p_generation,
            int64_t p_max_visible_chunks, int64_t p_request_id);
    void _reset_async_readback_state();
    bool _gpu_frustum_cull_instance(const CullParams &p_params, const InstancePipelineInputs &p_inputs,
            uint64_t p_start_time_usec, CullingSummary &r_summary);
};

#endif // GS_GPU_CULLER_H

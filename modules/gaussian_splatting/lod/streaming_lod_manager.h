#ifndef STREAMING_LOD_MANAGER_H
#define STREAMING_LOD_MANAGER_H

#include "scene/3d/camera_3d.h"
#include "core/math/aabb.h"
#include "core/templates/vector.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_set.h"
#include "servers/rendering_server.h"
#include "core/os/thread.h"
#include "../core/gaussian_data.h"
#include "../core/painterly_manager.h"
#include "../renderer/gpu_memory_stream.h"
#include "hierarchical_splat_structure.h"
#include "adaptive_lod_system.h"
#include "splat_clusterer.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace GaussianSplatting {

class StreamingLODManager {
public:
    struct LODLevel {
        uint32_t level_index;
        RID gpu_buffer;
        uint32_t splat_count;
        uint32_t buffer_capacity;
        float min_distance;
        float max_distance;
        bool is_loaded;
        bool is_loading;
        bool needs_update;
        uint64_t last_access_time;
        uint32_t access_count;

        // Memory statistics
        uint64_t gpu_memory_bytes;
        uint64_t cpu_memory_bytes;

        LODLevel() : level_index(0), splat_count(0), buffer_capacity(0),
                    min_distance(0.0f), max_distance(FLT_MAX),
                    is_loaded(false), is_loading(false), needs_update(false),
                    last_access_time(0), access_count(0),
                    gpu_memory_bytes(0), cpu_memory_bytes(0) {}
    };

    struct StreamingConfig {
        // Memory limits
        uint64_t max_gpu_memory = 1024ull * 1024ull * 1024ull;  // 1GB
        uint64_t max_cpu_memory = 2048ull * 1024ull * 1024ull;  // 2GB
        uint64_t target_gpu_memory = 768ull * 1024ull * 1024ull; // 768MB

        // Streaming parameters
        float load_ahead_distance = 50.0f;   // Pre-load distance
        float unload_distance = 200.0f;      // Unload distance
        uint32_t max_concurrent_loads = 2;   // Max parallel loads
        bool enable_predictive_loading = true;
        float prediction_time = 0.5f;        // Seconds ahead to predict

        // LOD parameters
        uint32_t num_lod_levels = 4;
        float lod_distance_multiplier = 2.0f;
        bool enable_adaptive_quality = true;
        bool enable_painterly_mode = false;
        uint32_t painterly_seed = 1337;
        float painterly_transition_rate = 4.0f;
        float painterly_hold_strength = 0.2f;

        // Performance
        uint32_t stream_budget_ms = 2;       // Max ms per frame for streaming
        bool enable_async_loading = true;
        bool enable_compression = true;
    };

    struct StreamingStats {
        uint64_t total_gpu_memory;
        uint64_t total_cpu_memory;
        uint32_t loaded_lod_levels;
        uint32_t loading_lod_levels;
        uint32_t total_visible_splats;
        uint32_t load_requests_last_frame;
        uint32_t frustum_rejected_lods_last_frame;
        bool content_visible_last_frame;
        bool content_distance_valid_last_frame;
        uint32_t async_prepare_jobs_completed;
        uint32_t async_apply_jobs_completed;
        bool async_prepare_observed_off_main_thread;
        bool async_apply_observed_on_main_thread;
        uint32_t async_prepare_main_thread_violations;
        uint32_t async_apply_off_main_thread_violations;

        struct PerformanceMetrics {
            float avg_load_time_ms;
            float avg_unload_time_ms;
            float avg_async_prepare_time_ms;
            float avg_async_apply_time_ms;
            uint32_t cache_hits;
            uint32_t cache_misses;
            float prediction_accuracy;
        } performance;

        void reset() {
            total_gpu_memory = 0;
            total_cpu_memory = 0;
            loaded_lod_levels = 0;
            loading_lod_levels = 0;
            total_visible_splats = 0;
            load_requests_last_frame = 0;
            frustum_rejected_lods_last_frame = 0;
            content_visible_last_frame = false;
            content_distance_valid_last_frame = false;
            async_prepare_jobs_completed = 0;
            async_apply_jobs_completed = 0;
            async_prepare_observed_off_main_thread = false;
            async_apply_observed_on_main_thread = false;
            async_prepare_main_thread_violations = 0;
            async_apply_off_main_thread_violations = 0;
            performance = {};
        }
    };

    struct PredictedMovement {
        Vector3 predicted_position;
        Vector3 predicted_direction;
        float confidence;
    };

public:
    StreamingLODManager();
    ~StreamingLODManager();

    // Initialize with splat data and configuration
    void initialize(
        const Vector<GaussianData>& base_splats,
        const StreamingConfig& p_config
    );

    // Update streaming based on camera position
    void update(
        const Camera3D* camera,
        float delta_time
    );

    // Get visible splats for current frame
    struct VisibleSplats {
        LocalVector<uint32_t> indices;
        LocalVector<float> lod_weights;
        LocalVector<uint8_t> lod_levels;
        LocalVector<PainterlyMetadata> painterly_metadata;
        LocalVector<uint32_t> painterly_seeds;
        LocalVector<uint32_t> painterly_prev_seeds;
        LocalVector<float> painterly_blend_weights;
        uint32_t total_count;
    };

    VisibleSplats get_visible_splats(
        const Camera3D* camera,
        uint32_t max_splats = 500000
    );

    // Manual LOD management
    void load_lod_level(uint32_t level);
    void unload_lod_level(uint32_t level);
    void prefetch_lod_range(float min_distance, float max_distance);

    // Memory management
    void compact_memory();
    void clear_cache();
    uint64_t get_memory_usage() const;

    // Statistics
    StreamingStats get_stats() const {
        std::lock_guard<std::mutex> lock(lod_mutex);
        return stats;
    }
    StreamingStats get_stats_snapshot() const { return get_stats(); }

    // Configuration
    void set_config(const StreamingConfig& p_config);
    const StreamingConfig& get_config() const { return config; }

private:
    void _initialize_gpu_streaming(const Vector<GaussianData>& base_splats);
    void start_async_loading();
    void stop_async_loading();
    void clear_async_load_queues();

    // LOD generation
    void generate_lod_levels(const Vector<GaussianData>& base_splats);
    void generate_single_lod(
        uint32_t level,
        const Vector<GaussianData>& source_splats
    );

    // Streaming operations
    void stream_lod_level(uint32_t level);
    void unload_distant_lods(const Vector3& camera_pos, const Frustum& frustum);
    void reprioritize_async_loads(const LocalVector<uint32_t>& desired_levels);

    // Predictive loading
    PredictedMovement predict_camera_movement(
        const Vector3& current_pos,
        float delta_time
    );
    void collect_predicted_lods(
        const PredictedMovement& prediction,
        const Frustum& frustum,
        LocalVector<uint32_t>& r_desired_levels,
        LocalVector<uint32_t>& r_new_levels);

    // Memory management
    void enforce_memory_limits();
    uint32_t select_lod_to_unload();

    // GPU operations
    void prepare_gpu_upload_data(const Vector<GaussianData>& data, LocalVector<::Gaussian>& gpu_gaussians) const;
    void upload_to_gpu(LODLevel& lod, const LocalVector<::Gaussian>& gpu_gaussians, uint32_t splat_count);
    void release_gpu_buffer(LODLevel& lod);

    // Async loading
    void async_load_worker();
    void enqueue_load_request(uint32_t level, const LocalVector<uint32_t>& desired_levels);
    void apply_completed_async_loads(uint64_t frame_start_usec);

    // Visibility determination
    bool is_lod_visible(
        const LODLevel& lod,
        const Vector3& camera_pos,
        const Frustum* frustum = nullptr
    ) const;

    float compute_lod_priority(
        const LODLevel& lod,
        const Vector3& camera_pos,
        const Frustum* frustum = nullptr
    ) const;

    float compute_distance_to_content(const Vector3& camera_pos) const;
    bool is_content_in_frustum(const Frustum& frustum) const;

private:
    StreamingConfig config;
    StreamingStats stats;
    mutable std::mutex lod_mutex;

    // LOD data
    Vector<LODLevel> lod_levels;
    Vector<Vector<GaussianData>> lod_data;  // CPU cache
    AABB content_bounds;
    bool has_content_bounds;

    // Spatial structures
    std::unique_ptr<HierarchicalSplatStructure> spatial_structure;
    std::unique_ptr<AdaptiveLODSystem> adaptive_lod;
    std::unique_ptr<SplatClusterer> clusterer;
    std::unique_ptr<PainterlyManager> painterly_manager;

    // GPU memory streaming
    std::unique_ptr<GaussianMemoryStream> gpu_stream;
    RenderingDevice* rd;

    float last_frame_delta;

    // Camera tracking
    struct CameraHistory {
        static constexpr uint32_t HISTORY_SIZE = 10;
        LocalVector<Vector3> positions;
        LocalVector<Vector3> velocities;
        Vector3 last_position;
        float avg_speed;

        void update(const Vector3& pos, float delta_time);
        Vector3 predict_position(float time_ahead) const;
    } camera_history;

    // Async loading
    std::thread loading_thread;
    std::atomic<bool> should_stop_loading;
    struct AsyncLoadRequest {
        uint32_t level = UINT32_MAX;
        Vector<GaussianData> source_data;
        uint64_t enqueue_time_usec = 0;
    };
    struct AsyncLoadJob {
        uint32_t level = UINT32_MAX;
        LocalVector<::Gaussian> gpu_gaussians;
        uint32_t splat_count = 0;
        float prepare_time_ms = 0.0f;
        uint64_t enqueue_time_usec = 0;
    };
    std::queue<AsyncLoadRequest> load_queue;
    std::queue<AsyncLoadJob> ready_queue;
    std::mutex load_queue_mutex;
    std::condition_variable load_cv;
    HashSet<uint32_t> pending_async_loads;

    // Performance monitoring
    uint64_t frame_start_time;
    uint64_t streaming_time_budget;
};

} // namespace GaussianSplatting

#endif // STREAMING_LOD_MANAGER_H

#ifndef GAUSSIAN_STREAMING_H
#define GAUSSIAN_STREAMING_H

#include "gaussian_data.h"
#include "gaussian_splat_manager.h"
#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/transform_3d.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/safe_refcount.h"
#include "servers/rendering/rendering_device.h"
#include <atomic>
#include "../renderer/gaussian_gpu_layout.h"
#include "../lod/lod_config.h"

class GaussianMemoryStream;
class OS;
class ResidencyBudgetController;
class UploadSchedulerFacade;

// Extracted subsystem headers (ISSUE-006 split)
#include "streaming_quantization.h"
#include "streaming_vram_regulator.h"
#include "streaming_atlas.h"

// GPU memory streaming system with ring buffer for large datasets
class GaussianStreamingSystem : public RefCounted {
    GDCLASS(GaussianStreamingSystem, RefCounted);

public:
    static constexpr uint32_t CHUNK_SIZE = 65536;  // 64K splats per chunk
    static constexpr uint32_t MAX_CHUNKS_IN_VRAM = 32;  // ~2GB @ 64K splats/chunk
    static constexpr uint32_t RING_BUFFER_FRAMES = 3;  // Triple buffering

    struct ConfigOverrides {
        bool override_chunk_culling = false;
        bool chunk_frustum_culling_enabled = true;
        float chunk_frustum_padding = 1.5f;

        bool override_prefetch = false;
        bool predictive_prefetch_enabled = true;
        float prefetch_lookahead_distance = 10.0f;

        bool override_vram_budget = false;
        VRAMBudgetConfig vram_budget_config;

        bool override_lod_config = false;
        LODConfig lod_config;

        bool override_lod_blend = false;
        LODBlendConfig lod_blend_config;

        bool override_streaming_tuning = false;
        uint32_t max_chunk_loads_per_frame = 0;

        bool override_io_source = false;
        String io_source_path;

        bool has_any_override() const {
            return override_chunk_culling || override_prefetch || override_vram_budget ||
                    override_lod_config || override_lod_blend || override_streaming_tuning ||
                    override_io_source;
        }
    };

    struct ChunkLayoutHint {
        uint32_t start_idx = 0;
        uint32_t count = 0;
        uint32_t source_index_offset = 0;
        bool source_indices_remapped = false;
        AABB bounds;
        Vector3 center;
        float radius = 0.0f;
    };

private:
    RenderingDevice *primary_device_override = nullptr;
    static constexpr uint32_t PRIMARY_ASSET_ID = 0;
    static constexpr uint32_t INVALID_ASSET_ID = UINT32_MAX;
    static constexpr uint32_t MAX_REQUESTED_LOD = GS_MAX_ASSET_LODS - 1;

    struct StreamingChunk {
        uint32_t start_idx = 0;
        uint32_t count = 0;
        bool source_index_remapped = false;
        RID gpu_buffer;
        Vector3 center;  // Precomputed center position
        AABB bounds;     // Precomputed axis-aligned bounding box for frustum culling
        float max_radius = 0.0f;
        float distance = 0.0f;  // Distance from camera for LOD
        bool is_loaded = false;
        bool is_visible = true;  // Set by frustum culling (true = potentially visible)
        bool upload_pending = false;  // For async upload tracking
        RenderingDevice *upload_device = nullptr;  // Device handling upload
        uint64_t last_used_frame = 0;
        uint64_t last_loaded_frame = 0;
        uint32_t buffer_slot = UINT32_MAX;  // Slot in persistent buffer

        // LOD blending state (LODGE technique + Octree-GS)
        float lod_blend_factor = 1.0f;      // Current blend factor (0.0 = fading out, 1.0 = fully visible)
        float previous_distance = 0.0f;     // Previous frame distance for hysteresis
        uint32_t current_lod_level = 0;     // Current LOD level (0 = highest quality)
        uint32_t target_lod_level = 0;      // Target LOD level (for smooth transitions)

        // Octree-GS LOD reduction parameters
        int sh_band_level = 3;              // Current SH band level (0-3, 3 = full quality)
        int splat_skip_factor = 1;          // Current splat skip factor (1 = render all)
        float opacity_multiplier = 1.0f;    // Distance-based opacity multiplier
        uint32_t effective_count = 0;       // Count after LOD reduction (= count / splat_skip_factor)

        // Per-chunk quantization info (Unity technique for 4x compression)
        ChunkQuantizationInfo quantization;
        bool quantization_computed = false;  // True if quantization bounds have been computed
    };

    struct RequestedChunkState {
        uint64_t stamp = 0;
        uint32_t lod_mask = 0;
    };

    struct AtlasAssetState {
        uint32_t asset_id = PRIMARY_ASSET_ID;
        uint32_t dense_id = PRIMARY_ASSET_ID;
        Ref<GaussianData> data;
        bool uses_primary_chunks = false;
        LocalVector<StreamingChunk> asset_chunks;
        LocalVector<uint32_t> requested_chunks;
        HashMap<uint32_t, RequestedChunkState> requested_chunk_state;
        uint32_t lod_count = 1;
        uint32_t sh_degree = 0;
        AABB bounds;
        uint32_t chunk_meta_base = 0;
        uint32_t chunk_meta_count = 0;
        uint32_t chunk_index_base = 0;
        uint32_t chunk_index_count = 0;
        uint32_t quant_base = 0;
        uint32_t quant_count = 0;
        bool metadata_dirty = true;
        uint32_t generation = 0; // Incremented on re-registration to invalidate queued pack jobs
    };

    uint32_t max_chunk_count_per_asset = 0;
    uint32_t max_chunk_splats = 0;

    struct FrameData {
        LocalVector<uint32_t> visible_chunks;
        uint64_t frame_number = 0;
    };

    struct PackJob {
        uint32_t asset_id = PRIMARY_ASSET_ID;
        uint32_t chunk_idx = UINT32_MAX;
        uint32_t buffer_slot = UINT32_MAX;
        uint32_t asset_generation = 0; // Captured at enqueue time for staleness detection
        uint64_t enqueue_usec = 0;
        uint32_t chunk_start = 0;
        uint32_t chunk_count = 0;
        bool uses_explicit_source_indices = false;
        LocalVector<uint32_t> source_indices;
        // Snapshot capture is deferred to worker threads to keep queueing lightweight.
        Ref<GaussianData> data_ref;
    };

    struct PendingChunkUpload {
        uint32_t asset_id = PRIMARY_ASSET_ID;
        uint32_t chunk_idx = UINT32_MAX;
        uint32_t buffer_slot = UINT32_MAX;
        uint32_t asset_generation = 0; // For staleness detection
        uint64_t enqueue_usec = 0;
        Vector<PackedGaussian> packed_data;
        SHCompressionMetrics metrics;
        uint32_t bytes_uploaded = 0;
    };

    struct PackThreadContext {
        GaussianStreamingSystem *system = nullptr;
        uint32_t thread_index = 0;
    };

    enum class EvictionResult {
        NoEviction,
        EvictedNonVisible,
        EvictedVisible,
        SkippedAllVisible,
    };

    // Debug counters for chunk culling
    struct ChunkCullingStats {
        uint32_t total_chunks = 0;
        uint32_t visible_chunks = 0;
        uint32_t frustum_culled_chunks = 0;
        uint32_t loaded_chunks = 0;
        void reset() {
            total_chunks = 0;
            visible_chunks = 0;
            frustum_culled_chunks = 0;
            loaded_chunks = 0;
        }
    };

    // Camera velocity tracking for predictive prefetch
    struct CameraVelocityTracker {
        Vector3 last_position;
        Vector3 velocity;
        bool has_previous_position = false;

        void update(const Vector3 &current_pos, float delta_time);
        Vector3 predict_position(const Vector3 &current_pos, float lookahead_distance) const;
    };

    struct VisibilityState {
        bool chunk_frustum_culling_enabled = true;
        float chunk_frustum_padding = 1.5f;  // Conservative padding multiplier to prevent popping
        float chunk_radius_multiplier = 1.0f;
        ChunkCullingStats culling_stats;
        LocalVector<uint32_t> visible_chunk_indices;
        CameraVelocityTracker camera_tracker;
        bool predictive_prefetch_enabled = true;
        float prefetch_lookahead_distance = 10.0f;
        LODBlendConfig lod_blend_config;
        float current_lod_blend_factor = 1.0f;  // Global blend factor for the frame
        int global_sh_band_level = 3;  // 0-3 for DC only to full SH
        uint32_t lod_transitions_this_frame = 0; // Chunks whose LOD target/current level changed this update.
        uint32_t prev_visible_count = 0;  // For change ratio calculation

        void update_chunk_visibility(GaussianStreamingSystem &system, const Transform3D &camera_transform, const Projection &projection);
        bool is_chunk_in_frustum(const AABB &p_bounds, const Vector<Plane> &p_frustum_planes) const;
        void update_culling_config_from_project_settings(GaussianStreamingSystem &system);
        void load_prefetch_config_from_project_settings(GaussianStreamingSystem &system);
        void load_lod_blend_config_from_project_settings(GaussianStreamingSystem &system);
        float calculate_lod_blend_factor(float distance, float lod_distance) const;
        void update_chunk_lod_blend_factors(GaussianStreamingSystem &system, const Vector3 &camera_pos);
        void update_chunk_lod_parameters(GaussianStreamingSystem &system, const Vector3 &camera_pos);
        uint32_t get_prefetch_limit(GaussianStreamingSystem &system, uint32_t available_slots, uint32_t load_budget) const;
        void collect_prefetch_candidates(GaussianStreamingSystem &system, const Vector3 &predicted_pos,
                uint32_t max_prefetch, uint32_t max_scan_budget, LocalVector<uint32_t> &out_candidates) const;
        uint32_t schedule_prefetch_loads(GaussianStreamingSystem &system, const Vector3 &predicted_pos,
                const LocalVector<uint32_t> &candidates, uint32_t available_slots, uint32_t load_budget);
        uint32_t prefetch_chunks_at_predicted_position(GaussianStreamingSystem &system, const Vector3 &predicted_pos,
                uint32_t available_slots, uint32_t load_budget, uint32_t max_scan_budget);
    };

    enum class ZeroVisibleRecoveryMode : uint8_t {
        STARTUP_ONLY = 0,
        PERSISTENT = 1,
    };

    struct ZeroVisibleRecoveryState {
        ZeroVisibleRecoveryMode mode = ZeroVisibleRecoveryMode::PERSISTENT;
        uint32_t persistent_trigger_frames = 16;
        uint32_t persistent_cooldown_frames = 30;
        uint32_t stall_log_interval_frames = 120;
        uint32_t zero_visible_consecutive_frames = 0;
        uint64_t last_recovery_frame = UINT64_MAX;
        uint64_t last_stall_log_frame = UINT64_MAX;
        uint64_t recoveries_triggered = 0;
        uint64_t stall_detections = 0;
    };

    struct EvictionState {
        struct NonPrimaryEvictionCandidate {
            uint32_t asset_id = UINT32_MAX;
            uint32_t chunk_id = UINT32_MAX;
            uint64_t last_used_frame = UINT64_MAX;
            float distance = 0.0f;
        };
        uint64_t chunk_load_counter = 0;  // Strictly increasing counter for chunk loads
        uint32_t eviction_hysteresis_frames = 5;
        uint32_t max_evictions_per_frame = 4;
        uint32_t chunks_evicted_this_frame = 0;
        uint32_t visible_chunks_evicted_this_frame = 0;  // Only visible chunks that were evicted
        uint64_t last_stabilize_log_frame = 0;  // For throttling stabilize log messages
        uint64_t cached_eviction_frame = UINT64_MAX;
        uint64_t cached_non_primary_lru_frame = UINT64_MAX;
        uint32_t cached_non_primary_lru_cursor = 0;
        LocalVector<NonPrimaryEvictionCandidate> cached_non_primary_lru_candidates;
        LocalVector<uint32_t> cached_visible_chunks;
        LocalVector<uint32_t> cached_nonvisible_chunks;

        EvictionResult evict_least_recently_used(GaussianStreamingSystem &system, bool p_allow_visible_eviction);
        bool evict_non_primary_lru(GaussianStreamingSystem &system);
        bool ensure_atlas_slot_available(GaussianStreamingSystem &system, uint32_t requesting_asset_id);
    };

    // Cache-line-aligned pack/upload telemetry counters.
    // Groups 15 atomic counters that were previously scattered across
    // UploadQueueState, eliminating false-sharing between pack worker
    // threads and the main thread that samples analytics.
    // Compiles to an empty struct in release builds (non-DEV_ENABLED).
    struct alignas(64) PackTelemetry {
#ifdef DEV_ENABLED
        SafeFlag enabled;

        // Pack timing
        SafeNumeric<uint64_t> pack_time_usec_total;
        SafeNumeric<uint64_t> pack_time_usec_max;
        SafeNumeric<uint32_t> pack_jobs_completed;

        // Upload volume
        SafeNumeric<uint64_t> upload_bytes_total;
        SafeNumeric<uint32_t> upload_chunks_completed;

        // Pack-queue latency
        SafeNumeric<uint64_t> pack_queue_latency_usec_total;
        SafeNumeric<uint64_t> pack_queue_latency_usec_max;
        SafeNumeric<uint32_t> pack_queue_latency_samples;

        // Upload-queue latency
        SafeNumeric<uint64_t> upload_queue_latency_usec_total;
        SafeNumeric<uint64_t> upload_queue_latency_usec_max;
        SafeNumeric<uint32_t> upload_queue_latency_samples;

        // Mutex contention
        SafeNumeric<uint64_t> pack_mutex_wait_usec_total;
        SafeNumeric<uint64_t> pack_mutex_wait_usec_max;
        SafeNumeric<uint32_t> pack_mutex_wait_samples;

        // --- Convenience accessors (keep call-site API identical) ---

        _ALWAYS_INLINE_ bool is_enabled() const { return enabled.is_set(); }
        _ALWAYS_INLINE_ void set_enabled(bool p_on) { enabled.set_to(p_on); }

        _ALWAYS_INLINE_ void add_pack_time(uint64_t p_usec) {
            pack_time_usec_total.add(p_usec);
            pack_jobs_completed.increment();
            pack_time_usec_max.exchange_if_greater(p_usec);
        }

        _ALWAYS_INLINE_ void add_upload_bytes(uint64_t p_bytes) {
            upload_bytes_total.add(p_bytes);
        }

        _ALWAYS_INLINE_ void add_upload_chunk() {
            upload_chunks_completed.increment();
        }

        _ALWAYS_INLINE_ void add_pack_queue_latency(uint64_t p_usec) {
            pack_queue_latency_usec_total.add(p_usec);
            pack_queue_latency_samples.increment();
            pack_queue_latency_usec_max.exchange_if_greater(p_usec);
        }

        _ALWAYS_INLINE_ void add_upload_queue_latency(uint64_t p_usec) {
            upload_queue_latency_usec_total.add(p_usec);
            upload_queue_latency_samples.increment();
            upload_queue_latency_usec_max.exchange_if_greater(p_usec);
        }

        _ALWAYS_INLINE_ void add_mutex_wait(uint64_t p_usec) {
            pack_mutex_wait_usec_total.add(p_usec);
            pack_mutex_wait_samples.increment();
            if (p_usec > 0) {
                pack_mutex_wait_usec_max.exchange_if_greater(p_usec);
            }
        }

        // Snapshot-and-reset used by the analytics sampler each frame.
        struct Snapshot {
            uint64_t pack_time_total = 0;
            uint64_t pack_time_max = 0;
            uint32_t pack_jobs = 0;
            uint64_t upload_bytes = 0;
            uint32_t upload_chunks = 0;
            uint64_t pack_queue_lat_total = 0;
            uint64_t pack_queue_lat_max = 0;
            uint32_t pack_queue_lat_samples = 0;
            uint64_t upload_queue_lat_total = 0;
            uint64_t upload_queue_lat_max = 0;
            uint32_t upload_queue_lat_samples = 0;
            uint64_t mutex_wait_total = 0;
            uint64_t mutex_wait_max = 0;
            uint32_t mutex_wait_samples = 0;
        };

        // Atomically exchange every counter to zero and return the snapshot.
        Snapshot exchange_and_reset();

        // Non-destructive read (for debug logging that shouldn't drain counters).
        Snapshot read_current() const;

#else // !DEV_ENABLED -- every method compiles to nothing.
        _ALWAYS_INLINE_ bool is_enabled() const { return false; }
        _ALWAYS_INLINE_ void set_enabled(bool) {}
        _ALWAYS_INLINE_ void add_pack_time(uint64_t) {}
        _ALWAYS_INLINE_ void add_upload_bytes(uint64_t) {}
        _ALWAYS_INLINE_ void add_upload_chunk() {}
        _ALWAYS_INLINE_ void add_pack_queue_latency(uint64_t) {}
        _ALWAYS_INLINE_ void add_upload_queue_latency(uint64_t) {}
        _ALWAYS_INLINE_ void add_mutex_wait(uint64_t) {}

        struct Snapshot {
            uint64_t pack_time_total = 0;
            uint64_t pack_time_max = 0;
            uint32_t pack_jobs = 0;
            uint64_t upload_bytes = 0;
            uint32_t upload_chunks = 0;
            uint64_t pack_queue_lat_total = 0;
            uint64_t pack_queue_lat_max = 0;
            uint32_t pack_queue_lat_samples = 0;
            uint64_t upload_queue_lat_total = 0;
            uint64_t upload_queue_lat_max = 0;
            uint32_t upload_queue_lat_samples = 0;
            uint64_t mutex_wait_total = 0;
            uint64_t mutex_wait_max = 0;
            uint32_t mutex_wait_samples = 0;
        };

        _ALWAYS_INLINE_ Snapshot exchange_and_reset() { return {}; }
        _ALWAYS_INLINE_ Snapshot read_current() const { return {}; }
#endif // DEV_ENABLED
    };

    struct UploadQueueState {
        struct UploadBudgetState {
            uint64_t upload_budget = 0;
            uint64_t slice_limit = 0;
            uint32_t completed_chunks = 0;
            uint32_t chunk_limit = 0;
        };
        static constexpr uint32_t QUEUE_COMPACT_MIN_PREFIX = 128;

        bool async_pack_enabled = true;
        uint32_t pack_worker_threads = 2;
        uint32_t max_pack_jobs_in_flight = 4;
        uint32_t max_chunk_loads_per_frame = 16;
        uint64_t max_upload_bytes_per_frame = 128 * 1024 * 1024;
        uint64_t max_upload_bytes_per_slice = 16 * 1024 * 1024;
        // Optional bandwidth cap. 0 disables token-bucket throttling.
        uint64_t max_upload_bytes_per_second = 0;
        // Token-bucket state for bandwidth-based upload throttling.
        uint64_t upload_budget_tokens = 0;
        uint64_t upload_budget_last_update_usec = 0;
        uint32_t queued_chunk_loads_this_frame = 0;
        std::atomic<uint32_t> pack_jobs_in_flight{0};
        PackTelemetry telemetry;

        // Last sampled pack/upload metrics (for performance monitors)
        double last_pack_avg_ms = 0.0;
        double last_pack_max_ms = 0.0;
        uint32_t last_pack_jobs = 0;
        double last_upload_mb = 0.0;
        uint32_t last_upload_chunks = 0;
        String cap_tier_preset = "custom";
        bool cap_tier_active = false;
        String cap_source_upload_mb_per_frame = "project_default";
        String cap_source_upload_mb_per_slice = "project_default";
        String cap_source_upload_mb_per_second = "project_default";
        uint32_t effective_upload_cap_mb_per_frame = 128;
        uint32_t effective_upload_cap_mb_per_slice = 16;
        uint32_t effective_upload_cap_mb_per_second = 0;
        bool upload_frame_cap_hit_this_frame = false;
        bool upload_slice_cap_hit_this_frame = false;
        bool upload_bandwidth_cap_hit_this_frame = false;
        bool chunk_load_cap_hit_this_frame = false;
        // Queue-pressure latch contract:
        // - inactive => source=none, reason=none
        // - active => source/reason are known contract tokens
        // Enforced via StreamingQueuePressureController in queue/enqueue/process paths.
        bool queue_pressure_active = false;
        String queue_pressure_source = "none";
        String queue_pressure_reason = "none";
        double last_pack_queue_latency_avg_ms = 0.0;
        double last_pack_queue_latency_max_ms = 0.0;
        double last_upload_queue_latency_avg_ms = 0.0;
        double last_upload_queue_latency_max_ms = 0.0;
        double last_pack_mutex_wait_avg_ms = 0.0;
        double last_pack_mutex_wait_max_ms = 0.0;

        // Async pack thread state
        Mutex pack_thread_lifecycle_mutex;
        Mutex pack_mutex;
        Semaphore pack_semaphore;
        LocalVector<Thread *> pack_threads;
        LocalVector<PackThreadContext> pack_thread_contexts;
        std::atomic<bool> pack_thread_running{false};
        std::atomic<bool> pack_thread_exit{false};
        // Queue head cursors avoid O(n) front erases under sustained streaming pressure.
        LocalVector<PackJob> pack_queue;
        uint32_t pack_queue_read_idx = 0;
        LocalVector<PendingChunkUpload *> upload_queue;
        uint32_t upload_queue_read_idx = 0;
        std::atomic<uint32_t> pack_queue_depth_cached{0};
        std::atomic<uint32_t> upload_queue_depth_cached{0};

        void start_pack_threads(GaussianStreamingSystem &system);
        void stop_pack_threads(GaussianStreamingSystem &system);
        void pack_thread_func(GaussianStreamingSystem &system, uint32_t p_thread_index);
        bool queue_chunk_load(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx);
        void process_upload_queue(GaussianStreamingSystem &system);
        void clear_pending_uploads(GaussianStreamingSystem &system);
        void cancel_asset_jobs(GaussianStreamingSystem &system, uint32_t asset_id);
        void cancel_chunk_jobs(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx, uint32_t buffer_slot);
        void load_streaming_tuning_config_from_project_settings(GaussianStreamingSystem &system);
        bool has_pending_uploads();
        bool pop_upload_job(PendingChunkUpload *&job);
        UploadBudgetState prepare_upload_budget_state();
        bool resolve_upload_chunk(GaussianStreamingSystem &system, PendingChunkUpload *job, StreamingChunk *&chunk);
        bool upload_job_slices(GaussianStreamingSystem &system, RenderingDevice *submission_rd,
                StreamingChunk &chunk, PendingChunkUpload *job, uint64_t total_bytes,
                uint64_t &upload_budget, uint64_t slice_limit, bool &submitted);
        void finalize_upload_job(GaussianStreamingSystem &system, PendingChunkUpload *job,
                StreamingChunk &chunk, UploadBudgetState &budget);
        void requeue_upload_job(PendingChunkUpload *job);
        uint32_t get_pack_queue_depth_cached() const;
        uint32_t get_upload_queue_depth_cached() const;
        void get_pending_queue_depths_cached(uint32_t &r_pack_queue_depth, uint32_t &r_upload_queue_depth) const;
        uint32_t get_pack_queue_depth_unsafe() const;
        uint32_t get_upload_queue_depth_unsafe() const;
        void compact_queues_locked();
        void sync_cached_queue_depths_locked();
        void record_pack_mutex_wait(uint64_t wait_start_usec);
        void record_upload_queue_latency(uint64_t enqueue_usec);

    private:
        bool start_pack_threads_locked(GaussianStreamingSystem &system);
        void stop_pack_threads_locked(GaussianStreamingSystem &system);
    };

    struct BudgetState {
        Ref<VRAMBudgetRegulator> vram_regulator;
        uint32_t loaded_chunks_count = 0;
        uint64_t vram_usage = 0;
        uint64_t evicted_bytes_total = 0;
        uint32_t chunks_loaded_this_frame = 0;
        bool vram_chunk_cap_hit_this_frame = false;

        Dictionary get_vram_debug_stats() const;
        bool is_vram_budget_warning_active() const;
        uint32_t get_effective_max_chunks() const;
    };

    // Ring buffer for frame synchronization
    FrameData frame_data[RING_BUFFER_FRAMES];
    uint32_t current_frame_idx = 0;
    uint64_t total_frame_count = 0;
    bool debug_logging_enabled = false;
    int debug_frame_log_frequency = 0;

    VisibilityState visibility;
    ZeroVisibleRecoveryState zero_visible_recovery;
    EvictionState eviction;
    UploadQueueState uploads;
    BudgetState budget;

    // Chunk management
    LocalVector<StreamingChunk> chunks;
    uint32_t total_splat_count = 0;

    // GPU resources
    RID persistent_buffer;  // Main GPU buffer (persistent mapped)
    uint32_t persistent_buffer_size = 0;
    GaussianAtlasAllocator atlas_allocator;
    HashMap<uint32_t, AtlasAssetState> atlas_assets;
    LocalVector<uint32_t> atlas_asset_order;
    HashMap<uint32_t, uint32_t> asset_id_to_dense;
    LocalVector<uint32_t> dense_to_asset_id;
    LocalVector<uint32_t> dense_id_generation;
    LocalVector<uint32_t> free_dense_ids;
    HashMap<uint32_t, uint32_t> asset_generation_tracker;
    uint64_t request_generation = 1;
    bool request_collection_active = false;
    bool request_pending = false;

    // Source data reference
    Ref<::GaussianData> source_data;

    // Compression tracking
    SHCompressionMetrics total_sh_metrics;

    // Configuration reload (triggered on ProjectSettings changes)
    static constexpr uint64_t BYTES_PER_MB = 1024u * 1024u;
    static constexpr float STREAMING_LOAD_DISTANCE_BASE = 100000.0f;
    static constexpr float ESTIMATED_FRAME_DELTA_60FPS = 1.0f / 60.0f;
    static constexpr double USEC_PER_MS = 1000.0;
    uint64_t last_config_reload_frame = 0;
    bool config_dirty = true;  // Force initial load
    bool project_settings_connected = false;
    bool streaming_initialized = false;

    struct SchedulerState {
        static constexpr uint32_t DEFAULT_PREFETCH_LOADS_PER_FRAME = 6;
        static constexpr uint32_t MAX_PREFETCH_LOADS_PER_FRAME = 64;
        static constexpr uint32_t DEFAULT_SYNC_FALLBACK_LOADS_PER_FRAME = 1;
        static constexpr uint32_t MAX_SYNC_FALLBACK_LOADS_PER_FRAME = 8;
        static constexpr uint32_t DEFAULT_SYNC_FALLBACK_QUEUE_SIZE = 2048;
        static constexpr uint32_t SYNC_FALLBACK_QUEUE_COMPACT_MIN_PREFIX = 128;
        static constexpr bool DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_ENABLED = false;
        static constexpr uint32_t DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_MIN_QUEUE_DEPTH = 1;
        static constexpr uint32_t DEFAULT_QUEUE_PRESSURE_VISIBLE_SCAN_BUDGET = 1024;
        static constexpr uint32_t DEFAULT_QUEUE_PRESSURE_PREFETCH_SCAN_BUDGET = 1024;
        uint32_t max_visible_chunk_scan_per_frame = 4096;
        uint32_t max_prefetch_chunk_scan_per_frame = 4096;
        uint32_t max_prefetch_loads_per_frame = DEFAULT_PREFETCH_LOADS_PER_FRAME;
        uint32_t max_sync_fallback_loads_per_frame = DEFAULT_SYNC_FALLBACK_LOADS_PER_FRAME;
        uint32_t max_sync_fallback_queue_size = DEFAULT_SYNC_FALLBACK_QUEUE_SIZE;
        bool queue_pressure_candidate_scan_throttle_enabled =
                DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_ENABLED;
        uint32_t queue_pressure_candidate_scan_throttle_min_queue_depth =
                DEFAULT_QUEUE_PRESSURE_CANDIDATE_SCAN_THROTTLE_MIN_QUEUE_DEPTH;
        uint32_t queue_pressure_candidate_scan_throttle_visible_scan_cap =
                DEFAULT_QUEUE_PRESSURE_VISIBLE_SCAN_BUDGET;
        uint32_t queue_pressure_candidate_scan_throttle_prefetch_scan_cap =
                DEFAULT_QUEUE_PRESSURE_PREFETCH_SCAN_BUDGET;
        uint32_t visible_scan_cursor = 0;
        uint32_t prefetch_scan_cursor = 0;
        uint32_t last_visible_scan_count = 0;
        uint32_t last_visible_scan_budget_effective = 0;
        uint32_t last_load_candidate_count = 0;
        uint32_t last_non_primary_scan_count = 0;
        uint32_t last_prefetch_scan_count = 0;
        uint32_t last_prefetch_scan_budget_effective = 0;
        uint32_t last_prefetch_candidate_count = 0;
        uint32_t last_prefetch_upload_pending_skip_count = 0;
        uint32_t last_prefetch_enqueued_count = 0;
        uint32_t last_prefetch_enqueue_headroom_stall_count = 0;
        uint32_t last_sync_fallback_queue_depth = 0;
        uint32_t last_sync_fallback_enqueued_count = 0;
        uint32_t last_sync_fallback_drained_count = 0;
        uint32_t last_sync_fallback_dropped_count = 0;
        uint32_t last_sync_fallback_stalled_count = 0;
        uint32_t prefetch_loads_remaining_this_frame = DEFAULT_PREFETCH_LOADS_PER_FRAME;
        uint32_t prefetch_scan_budget_remaining_this_frame = 0;
        bool queue_pressure_candidate_scan_throttle_active = false;
        uint32_t queue_pressure_candidate_scan_throttle_queue_depth = 0;
        LocalVector<uint64_t> sync_fallback_chunk_load_queue;
        HashSet<uint64_t> sync_fallback_chunk_load_set;
        uint32_t sync_fallback_chunk_load_queue_read_idx = 0;
        double last_update_cpu_ms = 0.0;
        double last_visibility_cpu_ms = 0.0;
        double last_load_cpu_ms = 0.0;
        double last_build_visible_cpu_ms = 0.0;
        double last_prefetch_cpu_ms = 0.0;
        double last_sync_fallback_cpu_ms = 0.0;
        double last_cpu_total_attributed_ms = 0.0;
        double last_cpu_unattributed_ms = 0.0;
    } scheduler;

    struct DiagnosticsState {
        static constexpr uint32_t STALL_THRESHOLD_FRAMES = 30;
        static constexpr uint32_t LOG_INTERVAL_FRAMES = 120;

        uint32_t init_invalid_frames = 0;
        uint32_t culling_empty_frames = 0;
        uint32_t scheduler_stall_frames = 0;
        uint32_t upload_stall_frames = 0;
        uint32_t sync_fallback_stall_frames = 0;
        uint32_t queue_pressure_frames = 0;
        uint32_t vram_cap_hit_frames = 0;
        uint64_t visible_evict_fallback_attempts = 0;
        uint64_t visible_evict_fallback_successes = 0;

        uint32_t last_total_chunks = 0;
        uint32_t last_visible_chunks = 0;
        uint32_t last_loaded_chunks = 0;

        uint64_t invariant_slot_ownership_violations = 0;
        uint64_t invariant_upload_lifecycle_violations = 0;
        uint64_t invariant_generation_violations = 0;
        String last_invariant_context;
        String last_invariant_message;

        String active_category = "ok";
        String active_reason = "healthy";
        String active_fingerprint = "ok";
        String last_logged_fingerprint;
        uint64_t last_fingerprint_log_frame = 0;
    } diagnostics;

    struct PrimaryChunkLayoutMetrics {
        bool spatial_partition_enabled = false;
        uint32_t source_index_count = 0;
        float avg_chunk_radius_ratio = 0.0f;
        float max_chunk_radius_ratio = 0.0f;
        float bounds_volume_ratio = 0.0f;

        void reset() {
            spatial_partition_enabled = false;
            source_index_count = 0;
            avg_chunk_radius_ratio = 0.0f;
            max_chunk_radius_ratio = 0.0f;
            bounds_volume_ratio = 0.0f;
        }
    } primary_chunk_layout_metrics;

    uint64_t last_streaming_update_usec = 0;
    float last_streaming_frame_delta_seconds = ESTIMATED_FRAME_DELTA_60FPS;
    bool effective_max_guard_warning_emitted = false;
    uint32_t effective_max_guard_warning_regulated = 0;
    uint32_t effective_max_guard_warning_capacity = 0;
    bool runtime_capacity_guard_logged = false;
    uint32_t runtime_capacity_guard_effective_max = UINT32_MAX;
    uint32_t runtime_capacity_guard_runtime_capacity = UINT32_MAX;
    bool runtime_capacity_guard_buffer_valid = true;
    bool runtime_capacity_guard_initialized = true;
    uint64_t invalid_camera_input_events = 0;
    uint64_t last_invalid_camera_log_frame = UINT64_MAX;
    uint32_t invalid_camera_log_interval_frames = 120;
    ConfigOverrides config_overrides;
    bool config_overrides_active = false;

    // Streaming orchestration hooks
    Ref<GaussianMemoryStream> memory_stream_proxy;
    Dictionary analytics_snapshot;
    RenderingDevice *last_upload_device = nullptr;

    // Per-chunk quantization (Unity technique for 4x compression)
    bool per_chunk_quantization_enabled = false;
    RID quantization_buffer;  // GPU buffer for ChunkQuantizationGPU data
    uint32_t quantization_buffer_size = 0;
    LocalVector<ChunkQuantizationGPU> quantization_gpu_data;  // CPU-side copy for upload
    bool quantization_release_deferred_logged = false;
    uint32_t quantization_position_bits = 16;
    uint32_t quantization_scale_bits = 12;
    bool quantization_scales_enabled = false;
    bool quantization_cpu_cache_valid = false;

    // Global atlas metadata (instance pipeline)
    GlobalAtlasState global_atlas_state;
    RID asset_meta_buffer;
    RID chunk_meta_buffer;
    RID asset_chunk_index_buffer;
    uint32_t asset_meta_buffer_size = 0;
    uint32_t chunk_meta_buffer_size = 0;
    uint32_t asset_chunk_index_buffer_size = 0;
    LocalVector<AssetMetaGPU> asset_meta_cpu;
    LocalVector<ChunkMetaGPU> chunk_meta_cpu;
    LocalVector<AssetChunkIndexGPU> asset_chunk_index_cpu;
    LocalVector<uint32_t> chunk_meta_dirty_indices;
    LocalVector<uint8_t> chunk_meta_dirty_flags;
    bool asset_meta_dirty = false;
    bool asset_chunk_index_dirty = false;
    bool chunk_meta_dirty_all = false;
    bool atlas_asset_registry_dirty = false;
    bool quantization_dirty = false;
    Vector<ChunkLayoutHint> io_chunk_layout_hints;
    uint32_t io_chunk_layout_asset_id = INVALID_ASSET_ID;
    Vector<ChunkLayoutHint> primary_chunk_layout_hints;
    LocalVector<uint32_t> primary_chunk_layout_source_indices;
    LocalVector<uint32_t> primary_chunk_source_indices;

    void _connect_project_settings();
    void _on_project_settings_changed();

protected:
    static void _bind_methods();

public:
    GaussianStreamingSystem();
    ~GaussianStreamingSystem();

    // Initialize streaming system with source data
    void initialize(Ref<::GaussianData> p_data);
    void initialize_with_device(Ref<::GaussianData> p_data, RenderingDevice *p_device);
    void initialize_empty(RenderingDevice *p_device);
    void update_primary_asset_data(Ref<::GaussianData> p_data);
    void attach_memory_stream(const Ref<GaussianMemoryStream> &p_stream);
    void set_config_overrides(const ConfigOverrides &p_overrides);
    void clear_config_overrides();
    void set_io_chunk_layout_hints(const Vector<ChunkLayoutHint> &p_hints, uint32_t p_asset_id = INVALID_ASSET_ID);
    void set_primary_chunk_layout(const Vector<ChunkLayoutHint> &p_hints, const Vector<uint32_t> &p_source_indices);
    bool has_config_overrides() const { return config_overrides_active; }
    const ConfigOverrides &get_config_overrides() const { return config_overrides; }

    // Update visibility and load/unload chunks based on camera
    // Note: camera_transform is in the same space as the Gaussian data (camera-to-world or camera-to-local).
    void update_streaming(const Transform3D &camera_transform, const Projection &projection, float frame_delta_seconds = -1.0f);

    // Global atlas residency requests (instance pipeline).
    void begin_residency_requests();
    void request_chunk_residency(uint32_t asset_id, uint32_t chunk_id, uint32_t lod_level);
    void request_asset_residency(uint32_t asset_id, uint32_t lod_level);
    void finalize_residency_requests();

    // Get GPU buffer for current frame's visible splats
    RID get_frame_buffer() const;
    uint32_t get_visible_count() const;
    LocalVector<Gaussian> get_visible_gaussians() const;
    LocalVector<uint32_t> get_visible_indices() const;

    // Frame management
    void begin_frame();
    void end_frame();

    // Memory management
    uint64_t get_vram_usage() const;
    uint32_t get_loaded_chunks() const { return budget.loaded_chunks_count; }
    uint32_t get_pending_pack_jobs();
    uint32_t get_pending_upload_jobs();
    SHCompressionMetrics get_total_sh_metrics() const { return total_sh_metrics; }
    Dictionary get_task_debug_state() const;
    Dictionary get_streaming_analytics() const;
    bool is_runtime_ready(String *r_reason = nullptr) const;
    bool is_runtime_capacity_zero() const;
    bool is_persistent_buffer_invalid() const;
    uint32_t get_registered_asset_count_with_data() const;

    // Chunk frustum culling configuration
    void set_chunk_frustum_culling_enabled(bool p_enabled) { visibility.chunk_frustum_culling_enabled = p_enabled; }
    bool is_chunk_frustum_culling_enabled() const { return visibility.chunk_frustum_culling_enabled; }
    void set_chunk_frustum_padding(float p_padding) { visibility.chunk_frustum_padding = MAX(1.0f, p_padding); }
    float get_chunk_frustum_padding() const { return visibility.chunk_frustum_padding; }
    void set_chunk_radius_multiplier(float p_multiplier);
    float get_chunk_radius_multiplier() const { return visibility.chunk_radius_multiplier; }

    // Debug statistics for chunk culling
    Dictionary get_chunk_culling_stats() const;

    // VRAM budget regulation
    Ref<VRAMBudgetRegulator> get_vram_regulator() const { return budget.vram_regulator; }
    Dictionary get_vram_debug_stats() const;
    bool is_vram_budget_warning_active() const;
    uint32_t get_effective_max_chunks() const;
    uint32_t get_max_chunk_count_per_asset() const { return max_chunk_count_per_asset; }
    uint32_t get_max_chunk_splats() const { return max_chunk_splats; }

    float get_global_lod_blend_factor() const { return visibility.current_lod_blend_factor; }
    void set_lod_blend_enabled(bool p_enabled) { visibility.lod_blend_config.blend_enabled = p_enabled; }
    bool is_lod_blend_enabled() const { return visibility.lod_blend_config.blend_enabled; }
    void set_lod_blend_distance(float p_distance) { visibility.lod_blend_config.blend_distance = MAX(0.1f, p_distance); }
    float get_lod_blend_distance() const { return visibility.lod_blend_config.blend_distance; }
    float get_lod_hysteresis_zone() const { return visibility.lod_blend_config.hysteresis_zone; }
    const LODConfig &get_lod_config() const { return _get_lod_config(); }

    // SH band level and visibility change tracking
    int get_global_sh_band_level() const { return visibility.global_sh_band_level; }
    void set_global_sh_band_level(int p_level) { visibility.global_sh_band_level = CLAMP(p_level, 0, 3); }
    float get_visible_count_change_ratio() const;

    // Streaming change tracking
    uint32_t get_chunks_loaded_this_frame() const { return budget.chunks_loaded_this_frame; }
    uint32_t get_chunks_evicted_this_frame() const { return eviction.chunks_evicted_this_frame; }
    uint32_t get_visible_chunks_evicted_this_frame() const { return eviction.visible_chunks_evicted_this_frame; }
    float get_visible_chunk_change_ratio() const;
    float get_effective_count_change_ratio() const;
    uint32_t get_buffer_capacity_splats() const;
    void build_visible_indices(LocalVector<uint32_t> &out_buffer_indices, LocalVector<uint32_t> &out_source_indices) const;

    // Buffer index mapping for streaming
    bool map_buffer_index_to_source(uint32_t buffer_index, uint32_t &out_source_index) const;
    bool is_asset_registered(uint32_t asset_id) const { return atlas_assets.has(asset_id); }

    // Per-chunk quantization (Unity technique for 4x compression)
    bool is_per_chunk_quantization_enabled() const { return per_chunk_quantization_enabled; }
    RID get_quantization_buffer() const { return quantization_buffer; }
    uint32_t get_quantization_position_bits() const { return quantization_position_bits; }
    uint32_t get_quantization_scale_bits() const { return quantization_scale_bits; }
    bool is_quantization_scales_enabled() const { return quantization_scales_enabled; }
    Dictionary get_quantization_stats() const;

    // Multi-asset registration (scaffolding).
    void register_asset(uint32_t asset_id, const Ref<GaussianData> &p_data);
    void unregister_asset(uint32_t asset_id);
    bool has_asset(uint32_t asset_id) const { return atlas_assets.has(asset_id); }
    uint32_t get_dense_asset_id(uint32_t asset_id) const;
    bool remap_instance_asset_ids(LocalVector<InstanceDataGPU> &p_instances, bool p_warn_on_missing = true) const;

    // Global atlas (instance pipeline) accessors
    const GlobalAtlasState &get_global_atlas_state() const { return global_atlas_state; }
    RID get_atlas_gaussian_buffer() const { return global_atlas_state.atlas_gaussian_buffer; }
    uint32_t get_atlas_gaussian_count() const { return global_atlas_state.atlas_gaussian_count; }
    RID get_asset_meta_buffer() const { return global_atlas_state.asset_meta_buffer; }
    RID get_chunk_meta_buffer() const { return global_atlas_state.chunk_meta_buffer; }
    RID get_asset_chunk_index_buffer() const { return global_atlas_state.asset_chunk_index_buffer; }
    RID get_atlas_quantization_buffer() const { return global_atlas_state.quantization_buffer; }
    uint64_t get_atlas_generation() const { return global_atlas_state.atlas_generation; }

    // Distance-based LOD (Octree-GS) - chunk-level LOD selection and reduction
    Dictionary get_lod_debug_stats() const;
    uint32_t get_effective_splat_count() const;  // Total splats after LOD reduction

    // Internal upload scheduler access (not user-facing API).
    UploadQueueState &_internal_get_upload_state() { return uploads; }

private:
    bool _create_chunks();
    void _build_chunks_for_data(const Ref<GaussianData> &p_data, LocalVector<StreamingChunk> &out_chunks);
    bool _build_primary_chunks_from_layout_hints(const Ref<GaussianData> &p_data, const Vector<ChunkLayoutHint> &p_hints,
            const LocalVector<uint32_t> &p_source_indices, LocalVector<StreamingChunk> &out_chunks);
    bool _build_chunks_from_layout_hints(const Ref<GaussianData> &p_data, const Vector<ChunkLayoutHint> &p_hints, LocalVector<StreamingChunk> &out_chunks);
    bool _resolve_primary_chunk_source_index(const StreamingChunk &chunk, uint32_t p_offset_in_chunk, uint32_t &r_source_index) const;
    void _refresh_primary_chunk_layout_metrics();
    void _register_primary_asset();
    uint32_t _advance_asset_generation(uint32_t asset_id);
    uint32_t _alloc_dense_id(uint32_t asset_id);
    void _release_dense_id(uint32_t dense_id);
    uint32_t _get_dense_generation(uint32_t dense_id) const;
    AtlasAssetState *_get_asset_state(uint32_t asset_id);
    const AtlasAssetState *_get_asset_state(uint32_t asset_id) const;
    LocalVector<StreamingChunk> &_get_asset_chunks(AtlasAssetState &asset);
    const LocalVector<StreamingChunk> &_get_asset_chunks(const AtlasAssetState &asset) const;
    uint64_t _make_chunk_key(uint32_t asset_id, uint32_t chunk_id) const;
    void _update_chunk_visibility(const Transform3D &camera_transform, const Projection &projection);
    void _log_streaming_telemetry();
    void _reload_config_if_dirty();
    float _resolve_frame_delta_seconds(float p_frame_delta_seconds);
    uint32_t _compute_runtime_chunk_capacity_limit() const;
    uint64_t _get_auxiliary_vram_overhead_bytes() const;
    uint64_t _get_total_vram_usage_bytes() const;
    void _load_zero_visible_recovery_config_from_project_settings();
    void _update_camera_tracking(const Vector3 &camera_pos, float p_frame_delta_seconds);
    void _handle_zero_visible_chunk_recovery();
    void _run_streaming_frame_pipeline(const Transform3D &camera_transform, const Projection &projection,
            const Vector3 &camera_pos, float resolved_frame_delta_seconds, uint32_t regulated_max,
            uint32_t effective_max, uint32_t runtime_capacity_max, OS *os, uint64_t scheduler_start_usec);
    void _reset_per_frame_counters();
    void _evict_for_vram_budget(uint32_t &evictions_left, bool &eviction_blocked);
    void _load_visible_chunks(uint32_t effective_max, uint32_t &evictions_left, bool &eviction_blocked);
    void _build_visible_chunk_list();
    void _handle_predictive_prefetch(const Vector3 &camera_pos, uint32_t effective_max);
    void _update_vram_regulator();
    void _log_streaming_frame_stats(uint32_t effective_max);
    void _load_chunk(uint32_t chunk_idx);
    void _load_chunk(uint32_t asset_id, uint32_t chunk_idx);
    RenderingDevice *_resolve_submission_device(GaussianSplatManager *manager,
            GaussianSplatManager::ScopedSubmissionLock &submission_lock) const;
    bool _pack_chunk_data(uint32_t asset_id, uint32_t chunk_idx, const AtlasAssetState &asset, StreamingChunk &chunk,
            Vector<PackedGaussian> &chunk_data, SHCompressionMetrics &metrics);
    void _log_chunk_load_metrics(uint32_t chunk_idx, const SHCompressionMetrics &metrics);
    bool _upload_chunk_to_gpu(RenderingDevice *submission_rd, uint32_t buffer_offset,
            const Vector<PackedGaussian> &chunk_data, uint32_t asset_id, uint32_t chunk_idx,
            uint32_t buffer_slot, uint32_t chunk_count) const;
    bool _begin_chunk_upload(uint32_t asset_id, uint32_t chunk_idx, StreamingChunk &chunk, uint32_t buffer_slot);
    void _rollback_pending_chunk(uint32_t asset_id, uint32_t chunk_idx, StreamingChunk &chunk, bool release_slot);
    void _assert_chunk_state_invariant(uint32_t asset_id, uint32_t chunk_idx,
            const StreamingChunk &chunk, const char *context,
            bool allow_deferred_allocator_release = false);
    void _finalize_chunk_load(uint32_t asset_id, uint32_t chunk_idx, StreamingChunk &chunk, uint32_t buffer_slot, uint32_t asset_chunk_count);
    void _unload_chunk(uint32_t chunk_idx);
    void _unload_chunk(uint32_t asset_id, uint32_t chunk_idx);
    EvictionResult _evict_least_recently_used(bool p_allow_visible_eviction);
    bool _is_chunk_in_frustum(const AABB &p_bounds, const Vector<Plane> &p_frustum_planes) const;
    void _reload_debug_logging_config();
    void _load_streaming_tuning_config_from_project_settings();
    void _start_pack_threads();
    void _stop_pack_threads();
    void _pack_thread_func(uint32_t p_thread_index);
    static void _pack_thread_entry(void *p_userdata);
    bool _queue_chunk_load(uint32_t chunk_idx);
    bool _queue_chunk_load(uint32_t asset_id, uint32_t chunk_idx);
    bool _enqueue_chunk_load_request(uint32_t asset_id, uint32_t chunk_idx,
            bool can_async_pack, bool prioritize_sync_fallback = false);
    bool _enqueue_sync_fallback_chunk_load(uint32_t asset_id, uint32_t chunk_idx, bool prioritize = false);
    uint32_t _drain_sync_fallback_chunk_loads(uint32_t effective_max, uint32_t &evictions_left, bool &eviction_blocked);
    uint32_t _get_sync_fallback_queue_depth() const;
    void _compact_sync_fallback_queue();
    void _process_upload_queue();
    void _clear_pending_uploads();
    void _release_persistent_buffer(RenderingDevice *p_rd, const char *p_context);
    void _apply_config_overrides();
    const LODConfig &_get_lod_config() const;
    void _update_culling_config_from_project_settings();
    uint32_t _prefetch_chunks_at_predicted_position(const Vector3 &predicted_pos,
            uint32_t available_slots, uint32_t load_budget, uint32_t max_scan_budget);
    void _load_prefetch_config_from_project_settings();
    void _load_lod_blend_config_from_project_settings();
    void _update_chunk_lod_blend_factors(const Vector3 &camera_pos);
    float _calculate_lod_blend_factor(float distance, float lod_distance) const;

    // Per-chunk quantization helpers
    void _load_quantization_config_from_project_settings();
    void _compute_chunk_quantization(uint32_t chunk_idx);
    void _compute_chunk_quantization(uint32_t asset_id, uint32_t chunk_idx);
    bool _release_quantization_buffer(RenderingDevice *p_rd, const char *p_context, bool p_allow_deferred_release);
    bool _upload_quantization_buffer(RenderingDevice *p_rd);
    ChunkQuantizationGPU _create_gpu_quantization_data(const ChunkQuantizationInfo &info, uint32_t start_idx, uint32_t count) const;

    // Global atlas metadata helpers (instance pipeline)
    void _build_global_atlas_cpu_state();
    void _update_chunk_meta_entry(uint32_t asset_id, uint32_t chunk_idx);
    void _mark_chunk_meta_dirty(uint32_t chunk_idx);
    void _mark_chunk_meta_dirty(uint32_t asset_id, uint32_t chunk_idx);
    void _apply_requested_residency();
    void _evict_unrequested_chunks(uint32_t asset_id, AtlasAssetState &asset,
            LocalVector<StreamingChunk> &asset_chunks);
    bool _load_requested_chunks(uint32_t asset_id, AtlasAssetState &asset,
            LocalVector<StreamingChunk> &asset_chunks, bool trace_enabled, bool can_async_pack);
    void _sync_global_atlas_state(RenderingDevice *p_rd);
    bool _ensure_atlas_slot_available(uint32_t requesting_asset_id);
    bool _evict_non_primary_lru();

    // Distance-based LOD (Octree-GS) helpers
    void _update_chunk_lod_parameters(const Vector3 &camera_pos);
    void _load_lod_config_from_project_settings();
    void _collect_lod_debug_stats(const FrameData &frame,
            uint32_t (&lod_level_counts)[8],
            uint32_t (&sh_band_counts)[4],
            uint32_t &total_original_splats,
            uint32_t &total_effective_splats,
            float &min_distance,
            float &max_distance,
            float &total_distance,
            uint32_t &visible_count,
            uint32_t &total_lod_level,
            int &max_skip_factor,
            float &min_opacity,
            uint32_t &chunks_in_transition) const;
    Dictionary _build_streaming_diagnostics_snapshot(uint32_t pack_queue_depth, uint32_t upload_queue_depth,
            uint32_t sync_fallback_queue_depth);
};

#endif // GAUSSIAN_STREAMING_H

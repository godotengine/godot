#ifndef STREAMING_UPLOAD_PIPELINE_H
#define STREAMING_UPLOAD_PIPELINE_H

#include "gaussian_data.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/string/ustring.h"
#include "core/templates/safe_refcount.h"
#include "servers/rendering/rendering_device.h"
#include "../renderer/gaussian_gpu_layout.h"
#include <atomic>

class GaussianStreamingSystem;

class StreamingUploadPipeline {
public:
    struct PackJob {
        uint32_t asset_id = 0;
        uint32_t chunk_idx = UINT32_MAX;
        uint32_t buffer_slot = UINT32_MAX;
        uint32_t asset_generation = 0;
        uint64_t enqueue_usec = 0;
        uint32_t chunk_start = 0;
        uint32_t chunk_count = 0;
        bool uses_explicit_source_indices = false;
        LocalVector<uint32_t> source_indices;
        Ref<GaussianData> data_ref;
    };

    struct PendingChunkUpload {
        uint32_t asset_id = 0;
        uint32_t chunk_idx = UINT32_MAX;
        uint32_t buffer_slot = UINT32_MAX;
        uint32_t asset_generation = 0;
        uint64_t enqueue_usec = 0;
        Vector<PackedGaussian> packed_data;
        uint32_t payload_checksum = 0;
        SHCompressionMetrics metrics;
        uint32_t bytes_uploaded = 0;
    };

    struct PackThreadContext {
        GaussianStreamingSystem *system = nullptr;
        uint32_t thread_index = 0;
    };

    struct alignas(64) PackTelemetry {
#ifdef DEV_ENABLED
        SafeFlag enabled;
        SafeNumeric<uint64_t> pack_time_usec_total;
        SafeNumeric<uint64_t> pack_time_usec_max;
        SafeNumeric<uint32_t> pack_jobs_completed;
        SafeNumeric<uint64_t> upload_bytes_total;
        SafeNumeric<uint32_t> upload_chunks_completed;
        SafeNumeric<uint64_t> pack_queue_latency_usec_total;
        SafeNumeric<uint64_t> pack_queue_latency_usec_max;
        SafeNumeric<uint32_t> pack_queue_latency_samples;
        SafeNumeric<uint64_t> upload_queue_latency_usec_total;
        SafeNumeric<uint64_t> upload_queue_latency_usec_max;
        SafeNumeric<uint32_t> upload_queue_latency_samples;
        SafeNumeric<uint64_t> pack_mutex_wait_usec_total;
        SafeNumeric<uint64_t> pack_mutex_wait_usec_max;
        SafeNumeric<uint32_t> pack_mutex_wait_samples;

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

        Snapshot exchange_and_reset();
        Snapshot read_current() const;
#else
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
#endif
    };

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
    uint64_t max_upload_bytes_per_second = 0;
    uint64_t upload_budget_tokens = 0;
    uint64_t upload_budget_last_update_usec = 0;
    uint32_t queued_chunk_loads_this_frame = 0;
    std::atomic<uint32_t> pack_jobs_in_flight{0};
    PackTelemetry telemetry;

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
    bool queue_pressure_active = false;
    String queue_pressure_source = "none";
    String queue_pressure_reason = "none";
    double last_pack_queue_latency_avg_ms = 0.0;
    double last_pack_queue_latency_max_ms = 0.0;
    double last_upload_queue_latency_avg_ms = 0.0;
    double last_upload_queue_latency_max_ms = 0.0;
    double last_pack_mutex_wait_avg_ms = 0.0;
    double last_pack_mutex_wait_max_ms = 0.0;

    Mutex pack_mutex;
    Semaphore pack_semaphore;
    LocalVector<Thread *> pack_threads;
    LocalVector<PackThreadContext> pack_thread_contexts;
    std::atomic<bool> pack_thread_running{false};
    std::atomic<bool> pack_thread_exit{false};
    LocalVector<PackJob> pack_queue;
    uint32_t pack_queue_read_idx = 0;
    LocalVector<PendingChunkUpload *> upload_queue;
    uint32_t upload_queue_read_idx = 0;
    std::atomic<uint32_t> pack_queue_depth_cached{0};
    std::atomic<uint32_t> upload_queue_depth_cached{0};

    void load_streaming_tuning_config_from_project_settings(GaussianStreamingSystem &system);
    void start_pack_threads(GaussianStreamingSystem &system);
    void stop_pack_threads(GaussianStreamingSystem &system);
    static void pack_thread_entry(void *p_userdata);
    void pack_thread_func(GaussianStreamingSystem &system, uint32_t p_thread_index);
    bool queue_chunk_load(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx);
    void process_upload_queue(GaussianStreamingSystem &system);
    void clear_pending_uploads(GaussianStreamingSystem &system);
    void cancel_asset_jobs(GaussianStreamingSystem &system, uint32_t asset_id);
    void cancel_chunk_jobs(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx, uint32_t buffer_slot);
    bool has_pending_uploads();
    uint32_t get_pack_queue_depth_cached() const;
    uint32_t get_upload_queue_depth_cached() const;
    void get_pending_queue_depths_cached(uint32_t &r_pack_queue_depth, uint32_t &r_upload_queue_depth) const;

private:
    bool pop_upload_job(PendingChunkUpload *&job);
    UploadBudgetState prepare_upload_budget_state();
    uint32_t get_pack_queue_depth_unsafe() const;
    uint32_t get_upload_queue_depth_unsafe() const;
    void compact_queues_locked();
    void sync_cached_queue_depths_locked();
    void record_pack_mutex_wait(uint64_t wait_start_usec);
    void record_upload_queue_latency(uint64_t enqueue_usec);
    void requeue_upload_job(PendingChunkUpload *job);
};

#endif // STREAMING_UPLOAD_PIPELINE_H

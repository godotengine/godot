#ifndef GPU_MEMORY_STREAM_H
#define GPU_MEMORY_STREAM_H

#include "core/object/ref_counted.h"
#include "core/object/object_id.h"
#include "core/os/mutex.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"
#include "../core/gaussian_data.h"
#include "gaussian_gpu_layout.h"
#include "core/variant/dictionary.h"
#include <atomic>

// Forward declarations
class RenderingDevice;
class RenderDeviceManager;

// Memory streaming statistics for performance monitoring
struct StreamingStats {
    uint64_t total_bytes_uploaded = 0;
    uint64_t total_bytes_downloaded = 0;
    uint32_t buffer_switches = 0;
    uint32_t stalls = 0;
    uint32_t total_frames = 0;  // For stall percentage
    uint32_t pool_hits = 0;      // Memory pool effectiveness
    uint32_t pool_misses = 0;
    float avg_upload_time_ms = 0.0f;
    float peak_memory_mb = 0.0f;
    uint32_t defrag_count = 0;
    uint64_t sh_raw_bytes_uploaded = 0;
    uint64_t sh_compressed_bytes_uploaded = 0;
    uint64_t sh_coefficients_streamed = 0;
    uint32_t reused_ready_buffers = 0;
};

// GPU memory stream with triple buffering for async uploads
class GaussianMemoryStream : public RefCounted {
    GDCLASS(GaussianMemoryStream, RefCounted);

public:
    enum BufferState {
        BUFFER_FREE = 0,      // Buffer available for writing
        BUFFER_UPLOADING = 1, // Being uploaded to GPU
        BUFFER_READY = 2,     // Ready for rendering
        BUFFER_RENDERING = 3  // Currently being used for rendering
    };

    // Stream buffer with atomic state management.
    struct StreamBuffer {
        RID gpu_buffer;
        RenderingDevice *gpu_allocation_device = nullptr;
        ObjectID gpu_allocation_device_id;
        uint64_t upload_fence = 0; // Timeline/fence value for pending uploads
        uint64_t upload_submit_frame = 0;
        uint32_t upload_frame_delay = 0;
        uint32_t capacity = 0;
        uint32_t used = 0;
        std::atomic<BufferState> state{BUFFER_FREE};
        uint64_t frame_last_used = 0;

        // Memory pool info
        bool from_pool = false;
        uint32_t pool_offset = UINT32_MAX;

        // Memory layout info
        uint32_t gaussian_offset = 0;
        uint32_t gaussian_stride = sizeof(PackedGaussian);
        uint32_t sort_key_offset = 0;
        uint32_t sort_key_stride = sizeof(uint32_t);

        // Transfer optimization (Issue #108)
        uint64_t transfer_start_time = 0;
        float last_bandwidth_mbps = 0.0f;
    };

    // Memory pool for efficient suballocation
    struct MemoryPool {
        struct Block {
            uint32_t offset = 0;
            uint32_t size = 0;
            bool free = true;
            uint32_t last_frame_used = 0;
        };

        LocalVector<Block> blocks;
        uint32_t total_size = 0;
        uint32_t used_size = 0;
        uint32_t fragmentation_threshold = 30; // Percentage

        uint32_t allocate(uint32_t size, uint32_t alignment = 16);
        void deallocate(uint32_t offset);
        void defragment();
        float get_fragmentation_ratio() const;
    };

private:
    // Triple buffering system
    static constexpr int BUFFER_COUNT = 3;
    StreamBuffer buffers[BUFFER_COUNT];
    std::atomic<int> write_index{0};
    std::atomic<int> read_index{0};
    std::atomic<int> upload_index{0};
    std::atomic<int> active_rendering_index{-1};

    // Upload fence tracking
    std::atomic<uint64_t> upload_timeline{0};
    std::atomic<uint64_t> completed_timeline{0};
    bool fine_grained_upload_supported = false;

    // Memory management
    MemoryPool gpu_memory_pool;

    // Configuration
    uint32_t max_gaussians = 1000000;
    uint32_t buffer_size_mb = 256;
    bool use_persistent_mapping = false;
    bool enable_async_upload = true;
    uint32_t sh_coefficient_limit = PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS;

    // Performance monitoring
    StreamingStats stats;
    uint64_t current_frame = 0;

    // Thread safety
    mutable Mutex buffer_mutex;
    mutable Mutex pool_mutex;

    // Rendering device reference
    RenderingDevice *rd = nullptr;
    Ref<RenderDeviceManager> device_manager;

    // Helper methods
    void _create_buffer(StreamBuffer &buffer, uint32_t size);
    void _destroy_buffer(StreamBuffer &buffer);
    int _get_next_write_buffer();
    int _get_current_read_buffer() const;
    Error _stream_internal(const LocalVector<Gaussian> &gaussians,
            uint32_t start,
            uint32_t count,
            const Vector3 *higher_order_coeffs,
            uint32_t first_order_count,
            uint32_t higher_order_count,
            const uint8_t *coefficient_limits,
            bool async_mode);
    void _upload_buffer_async(int buffer_index, const uint8_t *data, uint32_t size);
    void _wait_for_upload(int buffer_index);
    void _wait_for_buffer_complete(int buffer_index, bool p_block);
    bool _wait_for_upload_fence_value(uint64_t fence_value);
    RID _allocate_from_pool(RenderingDevice *p_device, uint32_t size, uint32_t pool_offset);
    void _poll_uploads();

    // Quaternion validation and data copy (Issue #108, #744)
    void _upload_buffer_coalesced(int buffer_index, const PackedGaussian *data, uint32_t count);
    void _validate_and_copy_gaussians(const PackedGaussian *src, PackedGaussian *dst, uint32_t count);
    float _measure_transfer_bandwidth(uint32_t bytes, uint64_t start_time, uint64_t end_time);
    void _update_bandwidth_stats(int buffer_index, uint32_t bytes, float bandwidth_mbps);

    // Reusable staging buffers
    Vector<PackedGaussian> coalesced_upload_scratch;

protected:
    static void _bind_methods();

public:
    GaussianMemoryStream();
    ~GaussianMemoryStream();

    // Initialization
    Error initialize(RenderingDevice *p_rd, uint32_t p_max_gaussians, uint32_t p_buffer_size_mb = 256);
    void shutdown();
    void set_device_manager(const Ref<RenderDeviceManager> &p_device_manager);

    // Streaming operations
    Error stream_gaussians_async(const LocalVector<Gaussian> &gaussians,
            uint32_t start = 0,
            uint32_t count = UINT32_MAX,
            const Vector3 *higher_order_coeffs = nullptr,
            uint32_t first_order_count = 3,
            uint32_t higher_order_count = 0,
            const uint8_t *coefficient_limits = nullptr);
    Error stream_gaussians_immediate(const LocalVector<Gaussian> &gaussians,
            uint32_t start = 0,
            uint32_t count = UINT32_MAX,
            const Vector3 *higher_order_coeffs = nullptr,
            uint32_t first_order_count = 3,
            uint32_t higher_order_count = 0,
            const uint8_t *coefficient_limits = nullptr);

    // Buffer management
    RID get_current_gpu_buffer();
    RID get_sort_keys_buffer() const;
    void swap_buffers();
    bool is_upload_complete() const;
    void wait_for_all_uploads();

    // Memory management
    void update_visible_range(uint32_t start, uint32_t count);
    void compact_memory();
    void defragment_if_needed();

    // LOD and culling support
    void set_lod_ranges(const LocalVector<uint32_t> &lod_starts, const LocalVector<uint32_t> &lod_counts);
    void update_culling_mask(const LocalVector<uint8_t> &visibility_mask);

    // Statistics and debugging
    StreamingStats get_stats() const { return stats; }
    void reset_stats() { stats = StreamingStats(); }
    Dictionary get_task_debug_state() const;
    uint32_t get_allocated_memory_mb() const;
    uint32_t get_used_memory_mb() const;
    float get_memory_efficiency() const;

    // Configuration
    void set_max_gaussians(uint32_t count) { max_gaussians = count; }
    uint32_t get_max_gaussians() const { return max_gaussians; }
    void set_async_upload(bool enabled) { enable_async_upload = enabled; }
    bool get_async_upload() const { return enable_async_upload; }
    void set_sh_coefficient_limit(uint32_t limit) {
        sh_coefficient_limit = MIN<uint32_t>(limit, PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);
    }
    uint32_t get_sh_coefficient_limit() const { return sh_coefficient_limit; }

    // Frame synchronization
    void begin_frame(uint64_t frame_number);
    void end_frame();
};

// Streaming pipeline for managing the entire streaming process
class StreamingPipeline : public RefCounted {
    GDCLASS(StreamingPipeline, RefCounted);

private:
    Ref<GaussianMemoryStream> memory_stream;
    Ref<::GaussianData> gaussian_data;

    // Streaming state
    struct StreamState {
        uint32_t current_lod = 0;
        uint32_t visible_start = 0;
        uint32_t visible_count = 0;
        bool needs_update = false;
        bool is_streaming = false;
    } state;
    mutable Mutex state_mutex;

    // Prefetching and prediction
    struct PrefetchData {
        Vector3 predicted_camera_pos;
        Vector3 predicted_camera_dir;
        float predicted_fov = 60.0f;
        uint32_t prefetch_distance = 100; // In splat units
    } prefetch;

    // Thread for async streaming
    Thread streaming_thread;
    std::atomic<bool> thread_running{false};
    std::atomic<bool> thread_exit{false};
    Semaphore stream_semaphore;

    void _streaming_thread_func();
    static void _streaming_thread_entry(void *p_userdata);

protected:
    static void _bind_methods();

public:
    StreamingPipeline();
    ~StreamingPipeline();

    // Initialization
    Error initialize(Ref<GaussianMemoryStream> p_stream, Ref<::GaussianData> p_data);
    void shutdown();

    // Streaming control
    void start_streaming();
    void stop_streaming();
    void update_view(const Transform3D &camera_transform, const Projection &projection);

    // LOD management
    void set_lod_level(uint32_t lod);
    uint32_t get_current_lod() const;

    // Visibility updates
    void update_visible_range(uint32_t start, uint32_t count);
    void update_from_frustum(const Vector<Plane> &frustum_planes);

    // Prefetching
    void enable_prefetching(bool enabled);
    void set_prefetch_distance(uint32_t distance) { prefetch.prefetch_distance = distance; }

    // Memory management
    void compact_memory() { if (memory_stream.is_valid()) memory_stream->compact_memory(); }
    void force_defragment() { if (memory_stream.is_valid()) memory_stream->defragment_if_needed(); }

    // Current buffer access
    RID get_current_buffer() { return memory_stream.is_valid() ? memory_stream->get_current_gpu_buffer() : RID(); }

    // Statistics
    Dictionary get_streaming_stats() const;
};

#endif // GPU_MEMORY_STREAM_H

#include "gpu_memory_stream.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "servers/rendering/rendering_device.h"
#include "../logger/gs_logger.h"
#include "servers/rendering/rendering_device_commons.h"

#include "../core/gaussian_splat_manager.h"
#include "../interfaces/render_device_manager.h"
#include "../interfaces/sync_policy.h"
#include <cstring>

namespace {

static constexpr uint32_t kUploadAlignmentBytes = 256;
static constexpr uint32_t kBandwidthSampleMinBytes = 4 * 1024 * 1024;
static constexpr float kBandwidthWarnThresholdMBps = 8000.0f;
static constexpr float kBandwidthTargetMBps = 10000.0f;
static constexpr uint64_t kUploadStallWarnUsec = 5000;

uint32_t _get_frame_fence_delay(RenderingDevice *p_device) {
    if (!p_device) {
        return 1;
    }
    uint32_t delay = p_device->get_frame_delay();
    return delay > 0 ? delay : 1;
}

void _advance_completed_timeline(std::atomic<uint64_t> &p_timeline, uint64_t p_value) {
    uint64_t observed = p_timeline.load(std::memory_order_acquire);
    while (observed < p_value &&
            !p_timeline.compare_exchange_weak(observed, p_value,
                    std::memory_order_release, std::memory_order_relaxed)) {
    }
}

RenderingDevice *_resolve_device(RenderingDevice *p_fallback, const ObjectID &p_id) {
	if (p_id.is_valid()) {
		if (Object *obj = ObjectDB::get_instance(p_id)) {
			if (RenderingDevice *resolved = Object::cast_to<RenderingDevice>(obj)) {
				return resolved;
			}
		}
		// Owner was tracked but no longer resolves: avoid using fallback device for explicit free.
		return nullptr;
	}
	return p_fallback;
}

} // namespace

// ==============================================================================
// GaussianMemoryStream Implementation
// ==============================================================================

GaussianMemoryStream::GaussianMemoryStream() {
    // Initialize with default values
}

GaussianMemoryStream::~GaussianMemoryStream() {
    shutdown();
}

void GaussianMemoryStream::set_device_manager(const Ref<RenderDeviceManager> &p_device_manager) {
    device_manager = p_device_manager;
    if (!device_manager.is_valid()) {
        return;
    }

    for (int i = 0; i < BUFFER_COUNT; i++) {
        StreamBuffer &buffer = buffers[i];
        if (!buffer.gpu_buffer.is_valid()) {
            continue;
        }
        RenderingDevice *owner = _resolve_device(buffer.gpu_allocation_device, buffer.gpu_allocation_device_id);
        if (!owner) {
            owner = rd;
        }
        if (!owner) {
            continue;
        }
        device_manager->track_resource(buffer.gpu_buffer, owner, true, "gaussian_memory_stream_buffer");
    }
}

void GaussianMemoryStream::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "rendering_device", "max_gaussians", "buffer_size_mb"), &GaussianMemoryStream::initialize, DEFVAL(256));
    ClassDB::bind_method(D_METHOD("shutdown"), &GaussianMemoryStream::shutdown);

    ClassDB::bind_method(D_METHOD("get_current_gpu_buffer"), &GaussianMemoryStream::get_current_gpu_buffer);
    ClassDB::bind_method(D_METHOD("swap_buffers"), &GaussianMemoryStream::swap_buffers);
    ClassDB::bind_method(D_METHOD("is_upload_complete"), &GaussianMemoryStream::is_upload_complete);
    ClassDB::bind_method(D_METHOD("wait_for_all_uploads"), &GaussianMemoryStream::wait_for_all_uploads);

    ClassDB::bind_method(D_METHOD("compact_memory"), &GaussianMemoryStream::compact_memory);
    ClassDB::bind_method(D_METHOD("get_allocated_memory_mb"), &GaussianMemoryStream::get_allocated_memory_mb);
    ClassDB::bind_method(D_METHOD("get_used_memory_mb"), &GaussianMemoryStream::get_used_memory_mb);
    ClassDB::bind_method(D_METHOD("get_memory_efficiency"), &GaussianMemoryStream::get_memory_efficiency);
    ClassDB::bind_method(D_METHOD("get_task_debug_state"), &GaussianMemoryStream::get_task_debug_state);
}

Error GaussianMemoryStream::initialize(RenderingDevice *p_rd, uint32_t p_max_gaussians, uint32_t p_buffer_size_mb) {
    ERR_FAIL_NULL_V_MSG(p_rd, ERR_INVALID_PARAMETER, "RenderingDevice is null");
    ERR_FAIL_COND_V_MSG(p_max_gaussians == 0, ERR_INVALID_PARAMETER, "Max gaussians must be > 0");

    rd = p_rd;
    max_gaussians = p_max_gaussians;
    buffer_size_mb = p_buffer_size_mb;

    // Calculate buffer size
    uint32_t gaussian_buffer_size = max_gaussians * sizeof(PackedGaussian);
    uint32_t sort_key_buffer_size = max_gaussians * sizeof(uint32_t);
    uint32_t total_buffer_size = gaussian_buffer_size + sort_key_buffer_size;
    uint32_t aligned_total_buffer_size =
            (total_buffer_size + kUploadAlignmentBytes - 1) & ~(kUploadAlignmentBytes - 1);

    // Make pool sizing responsive to configured memory budget, while guaranteeing enough
    // space for the active triple-buffer footprint.
    const uint64_t minimum_pool_size = uint64_t(aligned_total_buffer_size) * BUFFER_COUNT;
    uint64_t requested_pool_size = uint64_t(MAX(buffer_size_mb, 1u)) * 1024u * 1024u;
    uint64_t pool_size_u64 = MAX(minimum_pool_size, requested_pool_size);
    pool_size_u64 = (pool_size_u64 + kUploadAlignmentBytes - 1) & ~(uint64_t(kUploadAlignmentBytes) - 1);
    if (pool_size_u64 > UINT32_MAX) {
        GS_LOG_GPU_MEMORY_WARN(vformat("[MEMORY POOL] Requested size exceeds addressable range; clamping to %d bytes.",
                UINT32_MAX));
        pool_size_u64 = (uint64_t(UINT32_MAX) / kUploadAlignmentBytes) * kUploadAlignmentBytes;
    }
    uint32_t pool_size = static_cast<uint32_t>(pool_size_u64);
    gpu_memory_pool.total_size = pool_size;
    gpu_memory_pool.blocks.clear();
    gpu_memory_pool.blocks.push_back({0, pool_size, true, 0});
    gpu_memory_pool.used_size = 0;

    GS_LOG_GPU_MEMORY_INFO(vformat("[MEMORY POOL] Initialized pool with %d MB (%d bytes), requested=%d MB, minimum_required=%d MB",
                      pool_size / (1024 * 1024), pool_size,
                      MAX(buffer_size_mb, 1u),
                      uint32_t((minimum_pool_size + (1024u * 1024u - 1)) / (1024u * 1024u))));

    // Reset statistics
    stats = StreamingStats();

    // Create triple buffers to minimize GPU stalls
    // With proper pool allocation and async upload, target is <5% stall rate
    for (int i = 0; i < BUFFER_COUNT; i++) {
        _create_buffer(buffers[i], aligned_total_buffer_size);
        buffers[i].gaussian_offset = 0;
        buffers[i].sort_key_offset = gaussian_buffer_size;
    }

    // Explicitly log the triple buffering configuration for performance diagnostics
    GS_LOG_GPU_MEMORY_INFO("[GPU Memory Stream] Triple buffering enabled for asynchronous uploads");

    GS_LOG_GPU_MEMORY_INFO(vformat("[GPU Memory Stream] Initialized with %d buffers, %d max gaussians, %d MB per buffer",
                      BUFFER_COUNT, max_gaussians, aligned_total_buffer_size / (1024 * 1024)));
    GS_LOG_GPU_MEMORY_INFO(vformat("[GPU Memory Stream] Total pool size: %d MB, fragmentation threshold: %d%%",
                      pool_size / (1024 * 1024), gpu_memory_pool.fragmentation_threshold));

    return OK;
}

void GaussianMemoryStream::shutdown() {
    if (!rd) return;

    wait_for_all_uploads();

    // Destroy all buffers
    for (int i = 0; i < BUFFER_COUNT; i++) {
        _destroy_buffer(buffers[i]);
    }

    // Clear memory pool
    gpu_memory_pool.blocks.clear();
    gpu_memory_pool.total_size = 0;
    gpu_memory_pool.used_size = 0;

    rd = nullptr;
    GS_LOG_GPU_MEMORY_INFO("[GPU Memory Stream] Shutdown complete");
}

void GaussianMemoryStream::_create_buffer(StreamBuffer &buffer, uint32_t size) {
    ERR_FAIL_NULL_MSG(rd, "RenderingDevice is null");

    RenderingDevice *submission_device = rd;
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        submission_device = manager->acquire_submission_device(rd, submission_lock);
    }
    ERR_FAIL_NULL_MSG(submission_device, "RenderingDevice unavailable for buffer allocation");

    // Try to allocate from memory pool first
    MutexLock lock(pool_mutex);
    uint32_t pool_offset = gpu_memory_pool.allocate(size, kUploadAlignmentBytes);

    if (pool_offset != UINT32_MAX) {
        // Successfully allocated from pool
        buffer.gpu_buffer = _allocate_from_pool(submission_device, size, pool_offset);
        buffer.pool_offset = pool_offset;
        buffer.from_pool = true;
        stats.pool_hits++;
        GS_LOG_GPU_MEMORY_DEBUG(vformat("[MEMORY POOL] Allocated %d bytes from pool at offset %d", size, pool_offset));
    } else {
        // Pool allocation failed, allocate directly
        buffer.gpu_buffer = submission_device->storage_buffer_create(size, Vector<uint8_t>());
        if (buffer.gpu_buffer.is_valid()) {
            submission_device->set_resource_name(buffer.gpu_buffer, "GS_GaussianMemoryStream_DirectBuffer");
        }
        buffer.from_pool = false;
        stats.pool_misses++;
        GS_LOG_GPU_MEMORY_WARN(vformat("[MEMORY POOL] Direct allocation of %d bytes (pool full)", size));
    }

    ERR_FAIL_COND_MSG(!buffer.gpu_buffer.is_valid(), "Failed to create GPU buffer");
    buffer.gpu_allocation_device = submission_device;
    buffer.gpu_allocation_device_id = submission_device->get_instance_id();
    if (device_manager.is_valid()) {
        device_manager->track_resource(buffer.gpu_buffer, submission_device, true, "gaussian_memory_stream_buffer");
    }

    // Per-buffer staging buffers removed; rely on RenderingDevice internal staging.

    // Initialize transfer optimization metrics
    buffer.transfer_start_time = 0;
    buffer.last_bandwidth_mbps = 0.0f;

    buffer.capacity = size;
    buffer.used = 0;
    buffer.upload_fence = 0;
    buffer.upload_submit_frame = 0;
    buffer.upload_frame_delay = 0;
    buffer.state = BUFFER_FREE;
    buffer.frame_last_used = 0;

    // Print pool effectiveness metrics
    if (stats.pool_hits + stats.pool_misses > 0) {
        float hit_rate = (stats.pool_hits * 100.0f) / (stats.pool_hits + stats.pool_misses);
        GS_LOG_GPU_MEMORY_INFO(vformat("[MEMORY POOL] Hits: %d, Misses: %d, Hit Rate: %.1f%%",
                          stats.pool_hits, stats.pool_misses, hit_rate));
    }
}

void GaussianMemoryStream::_destroy_buffer(StreamBuffer &buffer) {
    if (buffer.gpu_buffer.is_valid()) {
        if (buffer.from_pool) {
            // Return memory to pool
            MutexLock lock(pool_mutex);
            gpu_memory_pool.deallocate(buffer.pool_offset);
            GS_LOG_GPU_MEMORY_DEBUG(vformat("[MEMORY POOL] Returned %d bytes to pool at offset %d",
                              buffer.capacity, buffer.pool_offset));
        }

        RenderingDevice *allocation_device = _resolve_device(
                buffer.gpu_allocation_device, buffer.gpu_allocation_device_id);
        if (device_manager.is_valid()) {
            if (allocation_device) {
                RenderingDevice *tracked_owner = device_manager->get_resource_owner(buffer.gpu_buffer, nullptr);
                if (!tracked_owner) {
                    device_manager->track_resource(buffer.gpu_buffer, allocation_device, true, "gaussian_memory_stream_buffer");
                }
            }
            device_manager->free_owned_resource(allocation_device, buffer.gpu_buffer);
        } else if (allocation_device) {
            allocation_device->free(buffer.gpu_buffer);
            buffer.gpu_buffer = RID();
        } else {
            buffer.gpu_buffer = RID();
        }
    }

    buffer.capacity = 0;
    buffer.used = 0;
    buffer.upload_fence = 0;
    buffer.upload_submit_frame = 0;
    buffer.upload_frame_delay = 0;
    buffer.state = BUFFER_FREE;
    buffer.from_pool = false;
    buffer.pool_offset = UINT32_MAX;
    buffer.gpu_allocation_device = nullptr;
    buffer.gpu_allocation_device_id = ObjectID();
}

int GaussianMemoryStream::_get_next_write_buffer() {
    // Find next free buffer using atomic operations
    for (int attempts = 0; attempts < BUFFER_COUNT * 2; attempts++) {
        int idx = write_index.load() % BUFFER_COUNT;
        BufferState expected = BUFFER_FREE;

        if (buffers[idx].state.compare_exchange_weak(expected, BUFFER_UPLOADING)) {
            write_index = (idx + 1) % BUFFER_COUNT;
            return idx;
        }

        // Try next buffer
        write_index = (write_index + 1) % BUFFER_COUNT;
    }

    // No free buffer available - try to complete uploads and reuse a ready buffer.
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (buffers[i].state.load() == BUFFER_UPLOADING) {
            _wait_for_buffer_complete(i, false);
        }
    }

    for (int attempts = 0; attempts < BUFFER_COUNT * 2; attempts++) {
        int idx = write_index.load() % BUFFER_COUNT;
        BufferState expected = BUFFER_FREE;
        if (buffers[idx].state.compare_exchange_weak(expected, BUFFER_UPLOADING)) {
            write_index = (idx + 1) % BUFFER_COUNT;
            return idx;
        }
        write_index = (write_index + 1) % BUFFER_COUNT;
    }

    int ready_idx = -1;
    uint64_t oldest_frame = UINT64_MAX;
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (buffers[i].state.load() == BUFFER_READY && buffers[i].frame_last_used < oldest_frame) {
            oldest_frame = buffers[i].frame_last_used;
            ready_idx = i;
        }
    }
    if (ready_idx >= 0) {
        BufferState expected = BUFFER_READY;
        if (buffers[ready_idx].state.compare_exchange_weak(expected, BUFFER_UPLOADING)) {
            stats.reused_ready_buffers++;
            if (stats.reused_ready_buffers == 1 || stats.reused_ready_buffers % 100 == 0) {
                GS_LOG_GPU_MEMORY_WARN(vformat(
                        "[STREAM] Reusing READY buffer %d due to upload backpressure (count=%d)",
                        ready_idx, stats.reused_ready_buffers));
            }
            return ready_idx;
        }
    }

    // Still no buffer available - caller may choose to block.
    stats.stalls++;
    stats.total_frames++;

    return -1;
}

int GaussianMemoryStream::_get_current_read_buffer() const {
    // Find buffer ready for rendering
    for (int i = 0; i < BUFFER_COUNT; i++) {
        int idx = (read_index + i) % BUFFER_COUNT;
        BufferState state = buffers[idx].state.load();

        if (state == BUFFER_READY || state == BUFFER_RENDERING) {
            // Don't change state in const method, just return the index
            return idx;
        }
    }

    return -1; // No buffer ready
}

Error GaussianMemoryStream::stream_gaussians_async(const LocalVector<Gaussian> &gaussians,
        uint32_t start,
        uint32_t count,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        const uint8_t *coefficient_limits) {
    return _stream_internal(gaussians, start, count, higher_order_coeffs, first_order_count,
            higher_order_count, coefficient_limits, true);
}

Error GaussianMemoryStream::stream_gaussians_immediate(const LocalVector<Gaussian> &gaussians,
        uint32_t start,
        uint32_t count,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        const uint8_t *coefficient_limits) {
    return _stream_internal(gaussians, start, count, higher_order_coeffs, first_order_count,
            higher_order_count, coefficient_limits, false);
}

Error GaussianMemoryStream::_stream_internal(const LocalVector<Gaussian> &gaussians,
        uint32_t start,
        uint32_t count,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        const uint8_t *coefficient_limits,
        bool async_mode) {
    MutexLock lock(buffer_mutex);
    ERR_FAIL_NULL_V_MSG(rd, ERR_UNCONFIGURED, "RenderingDevice not initialized");

    if (count == UINT32_MAX) {
        count = gaussians.size() - start;
    }

    ERR_FAIL_COND_V_MSG(start + count > gaussians.size(), ERR_INVALID_PARAMETER, "Invalid range for streaming gaussians");

    int buffer_idx = _get_next_write_buffer();
    if (buffer_idx < 0) {
        if (async_mode) {
            wait_for_all_uploads();
            buffer_idx = _get_next_write_buffer();
        }
        ERR_FAIL_COND_V_MSG(buffer_idx < 0, ERR_BUSY, "No buffers available for streaming");
    }

    StreamBuffer &buffer = buffers[buffer_idx];

    Vector<PackedGaussian> packed_gaussians;
    SHCompressionMetrics compression_metrics;
    if (coefficient_limits) {
        pack_gaussians_range_limited(gaussians,
                start,
                count,
                packed_gaussians,
                compression_metrics,
                higher_order_coeffs,
                first_order_count,
                higher_order_count,
                coefficient_limits,
                sh_coefficient_limit);
    } else {
        pack_gaussians_range(gaussians,
                start,
                count,
                packed_gaussians,
                compression_metrics,
                higher_order_coeffs,
                first_order_count,
                higher_order_count,
                sh_coefficient_limit);
    }

    uint32_t data_size = packed_gaussians.size() * sizeof(PackedGaussian);
    ERR_FAIL_COND_V_MSG(data_size > buffer.capacity, ERR_OUT_OF_MEMORY,
            vformat("Data size (%d bytes) exceeds buffer capacity (%d bytes)", data_size, buffer.capacity));

    if (async_mode && enable_async_upload) {
        _upload_buffer_coalesced(buffer_idx, packed_gaussians.ptr(), packed_gaussians.size());
    } else {
        uint32_t dst_offset = 0;
        if (!buffer.from_pool && buffer.pool_offset != UINT32_MAX) {
            dst_offset = buffer.pool_offset;
        }
        rd->buffer_update(buffer.gpu_buffer, dst_offset, data_size, packed_gaussians.ptr());
        // submit() and sync() only work on local (non-main) RenderingDevices
        if (!rd->is_main_rendering_device()) {
            rd->submit();
            rd->sync();
        }
        buffer.used = data_size;
        buffer.upload_fence = 0;
        buffer.upload_submit_frame = 0;
        buffer.upload_frame_delay = 0;
        buffer.state = BUFFER_READY;
    }

    stats.total_bytes_uploaded += data_size;
    stats.buffer_switches++;
    stats.sh_raw_bytes_uploaded += compression_metrics.raw_bytes;
    stats.sh_compressed_bytes_uploaded += compression_metrics.compressed_bytes;
    stats.sh_coefficients_streamed += compression_metrics.coefficient_count;

    return OK;
}

void GaussianMemoryStream::_upload_buffer_async(int buffer_index, const uint8_t *data, uint32_t size) {
    ERR_FAIL_INDEX_MSG(buffer_index, BUFFER_COUNT, "Invalid buffer index");
    ERR_FAIL_NULL_MSG(data, "Data pointer is null");

    StreamBuffer &buffer = buffers[buffer_index];
    RenderingDevice *transfer_rd = buffer.gpu_allocation_device;
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        transfer_rd = manager->acquire_submission_device(buffer.gpu_allocation_device, submission_lock);
    }
    ERR_FAIL_NULL_MSG(transfer_rd, "RenderingDevice unavailable for GPU transfer");

    if (size == 0) {
        buffer.used = 0;
        buffer.upload_fence = 0;
        buffer.upload_submit_frame = 0;
        buffer.upload_frame_delay = 0;
        buffer.state = BUFFER_READY;
        return;
    }
    transfer_rd->buffer_update(buffer.gpu_buffer, 0, size, data);
    static constexpr const char *async_upload_context = "GaussianMemoryStream::_upload_buffer_async";
    (void)async_upload_context;
    gs_device_utils::safe_submit(transfer_rd);

    uint64_t fence_value = upload_timeline.fetch_add(1, std::memory_order_relaxed) + 1;
    buffer.upload_fence = fence_value;
    buffer.upload_submit_frame = current_frame;
    buffer.upload_frame_delay = _get_frame_fence_delay(transfer_rd);

    buffer.used = size;
    buffer.state = BUFFER_UPLOADING;
}

// ======================================================================
// MEMORY ACCESS PATTERN OPTIMIZATION (Issue #108)
// ======================================================================

void GaussianMemoryStream::_upload_buffer_coalesced(int buffer_index, const PackedGaussian *data, uint32_t count) {
    ERR_FAIL_INDEX_MSG(buffer_index, BUFFER_COUNT, "Invalid buffer index");
    ERR_FAIL_NULL_MSG(data, "Data pointer is null");

    StreamBuffer &buffer = buffers[buffer_index];
    RenderingDevice *transfer_rd = buffer.gpu_allocation_device;
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        transfer_rd = manager->acquire_submission_device(buffer.gpu_allocation_device, submission_lock);
    }
    ERR_FAIL_NULL_MSG(transfer_rd, "RenderingDevice unavailable for GPU transfer");

    if (count == 0) {
        buffer.used = 0;
        buffer.upload_fence = 0;
        buffer.upload_submit_frame = 0;
        buffer.upload_frame_delay = 0;
        buffer.state = BUFFER_READY;
        return;
    }

    uint32_t size = count * sizeof(PackedGaussian);

    const uint32_t alignment = kUploadAlignmentBytes;

    uint32_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    ERR_FAIL_COND_MSG(aligned_size > buffer.capacity,
            vformat("Upload size (%d) exceeds buffer capacity (%d)", aligned_size, buffer.capacity));

    uint32_t aligned_count = aligned_size / sizeof(PackedGaussian);
    if (aligned_count == 0) {
        aligned_count = 1;
        aligned_size = sizeof(PackedGaussian);
    }

    uint32_t scratch_size = static_cast<uint32_t>(coalesced_upload_scratch.size());
    if (scratch_size < aligned_count) {
        uint32_t grown = scratch_size > 0 ? (scratch_size + scratch_size / 2) : aligned_count;
        uint32_t new_size = MAX(aligned_count, grown);
        coalesced_upload_scratch.resize(new_size);
    }

    PackedGaussian *scratch_ptr = coalesced_upload_scratch.ptrw();
    _validate_and_copy_gaussians(data, scratch_ptr, count);

    if (aligned_count > count) {
        memset(scratch_ptr + count, 0, (aligned_count - count) * sizeof(PackedGaussian));
    }

    const uint8_t *upload_ptr = reinterpret_cast<const uint8_t *>(scratch_ptr);
    // Measure CPU submission throughput; GPU-side timings aren't exposed for buffer_update.
    uint64_t submit_start = OS::get_singleton()->get_ticks_usec();
    buffer.transfer_start_time = submit_start;
    transfer_rd->buffer_update(buffer.gpu_buffer, 0, aligned_size, upload_ptr);

    gs_device_utils::safe_submit(transfer_rd);

    uint64_t fence_value = upload_timeline.fetch_add(1, std::memory_order_relaxed) + 1;

    buffer.upload_fence = fence_value;
    buffer.upload_submit_frame = current_frame;
    buffer.upload_frame_delay = _get_frame_fence_delay(transfer_rd);
    buffer.used = count * sizeof(PackedGaussian);
    buffer.state = BUFFER_UPLOADING;

    // Measure and update bandwidth statistics (CPU submit path only).
    uint64_t submit_end = OS::get_singleton()->get_ticks_usec();
    if (aligned_size >= kBandwidthSampleMinBytes) {
        float bandwidth_mbps = _measure_transfer_bandwidth(aligned_size, submit_start, submit_end);
        _update_bandwidth_stats(buffer_index, aligned_size, bandwidth_mbps);
    }
}

void GaussianMemoryStream::_validate_and_copy_gaussians(const PackedGaussian *src, PackedGaussian *dst, uint32_t count) {
    if (count == 0) {
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        const PackedGaussian &gaussian = src[i];
        PackedGaussian &opt_gaussian = dst[i];

        opt_gaussian = gaussian;

        Quaternion rotation(
                gaussian.rotation[0],
                gaussian.rotation[1],
                gaussian.rotation[2],
                gaussian.rotation[3]);
        float len_sq = rotation.length_squared();
        if (len_sq > 0.0f && Math::abs(len_sq - 1.0f) > 1e-3f) {
            rotation.normalize();
        } else if (len_sq == 0.0f) {
            rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
        }

        opt_gaussian.rotation[0] = rotation.x;
        opt_gaussian.rotation[1] = rotation.y;
        opt_gaussian.rotation[2] = rotation.z;
        opt_gaussian.rotation[3] = rotation.w;
    }
}

float GaussianMemoryStream::_measure_transfer_bandwidth(uint32_t bytes, uint64_t start_time, uint64_t end_time) {
    if (end_time <= start_time) {
        return 0.0f;
    }
    uint64_t transfer_time_usec = end_time - start_time;
    if (transfer_time_usec == 0) return 0.0f;

    // Calculate bandwidth in MB/s
    float transfer_time_sec = transfer_time_usec / 1000000.0f;
    float bytes_mb = bytes / (1024.0f * 1024.0f);
    float bandwidth_mbps = bytes_mb / transfer_time_sec;

    return bandwidth_mbps;
}

void GaussianMemoryStream::_update_bandwidth_stats(int buffer_index, uint32_t bytes, float bandwidth_mbps) {
    ERR_FAIL_INDEX(buffer_index, BUFFER_COUNT);

    StreamBuffer &buffer = buffers[buffer_index];
    buffer.last_bandwidth_mbps = bandwidth_mbps;

    // Log CPU submission throughput (buffer_update + submit), not GPU transfer bandwidth.
    GS_LOG_GPU_MEMORY_DEBUG(vformat("[GPU Memory] Buffer %d: %.2f MB submitted at %.1f MB/s (CPU submit, target: >%.0f MB/s)",
                      buffer_index, bytes / (1024.0f * 1024.0f), bandwidth_mbps, kBandwidthTargetMBps));

    // Performance warning if CPU submission throughput is too low.
    if (bandwidth_mbps > 0.0f && bandwidth_mbps < kBandwidthWarnThresholdMBps) {
        GS_LOG_GPU_MEMORY_WARN(vformat("[PERF WARNING] Low upload submission throughput: %.1f MB/s (CPU submit)",
                bandwidth_mbps));
    }
}

void GaussianMemoryStream::wait_for_all_uploads() {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        _wait_for_buffer_complete(i, true);
    }
}

bool GaussianMemoryStream::is_upload_complete() const {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (buffers[i].state.load() == BUFFER_UPLOADING) {
            return false;
        }
    }
    return true;
}

void GaussianMemoryStream::_poll_uploads() {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        _wait_for_buffer_complete(i, false);
    }
}

void GaussianMemoryStream::_wait_for_buffer_complete(int buffer_index, bool p_block) {
    if (buffer_index < 0 || buffer_index >= BUFFER_COUNT) {
        return;
    }

    StreamBuffer &buffer = buffers[buffer_index];
    BufferState state = buffer.state.load();
    if (state != BUFFER_UPLOADING) {
        return;
    }

    uint64_t fence_value = buffer.upload_fence;
    bool completed = (fence_value == 0);

    if (!completed) {
        uint64_t known_completed = completed_timeline.load(std::memory_order_acquire);
        if (known_completed >= fence_value) {
            completed = true;
        } else if (fine_grained_upload_supported) {
            completed = _wait_for_upload_fence_value(fence_value);
        }
    }

    if (!completed) {
        uint32_t delay = buffer.upload_frame_delay;
        if (delay == 0) {
            RenderingDevice *device_hint = buffer.gpu_allocation_device ? buffer.gpu_allocation_device : rd;
            delay = _get_frame_fence_delay(device_hint);
        }
        uint64_t submit_frame = buffer.upload_submit_frame;
        if (current_frame >= submit_frame + delay) {
            completed = true;
        }
    }

    if (!completed && p_block) {
        uint64_t latest_known = upload_timeline.load(std::memory_order_acquire);
        RenderingDevice *transfer_rd = buffer.gpu_allocation_device;
        GaussianSplatManager::ScopedSubmissionLock submission_lock;
        if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
            transfer_rd = manager->acquire_submission_device(buffer.gpu_allocation_device, submission_lock);
        }
        if (transfer_rd) {
            uint64_t block_start = OS::get_singleton()->get_ticks_usec();
            gs_device_utils::safe_submit_and_sync(transfer_rd);
            uint64_t block_end = OS::get_singleton()->get_ticks_usec();
            if (block_end > block_start && (block_end - block_start) > kUploadStallWarnUsec) {
                GS_LOG_GPU_MEMORY_WARN(vformat("[STALL] Upload sync blocked for %.2f ms",
                        (block_end - block_start) / 1000.0f));
            }
        }
        _advance_completed_timeline(completed_timeline, latest_known);
        completed = latest_known >= fence_value;
    }

    if (completed) {
        buffer.state = BUFFER_READY;
        buffer.upload_fence = 0;
        buffer.upload_submit_frame = 0;
        buffer.upload_frame_delay = 0;
        if (fence_value > 0) {
            _advance_completed_timeline(completed_timeline, fence_value);
        }
    }
}

bool GaussianMemoryStream::_wait_for_upload_fence_value(uint64_t fence_value) {
    if (fence_value == 0) {
        return true;
    }

    uint64_t known_completed = completed_timeline.load(std::memory_order_acquire);
    if (known_completed >= fence_value) {
        return true;
    }

    // No direct GPU fence wait is exposed yet; indicate incomplete so caller can fall back.
    return false;
}

RID GaussianMemoryStream::get_current_gpu_buffer() {
    int idx = _get_current_read_buffer();
    if (idx >= 0) {
        _wait_for_buffer_complete(idx, false);
        BufferState expected = BUFFER_READY;
        if (!buffers[idx].state.compare_exchange_strong(expected, BUFFER_RENDERING)) {
            if (expected == BUFFER_RENDERING) {
                // Already being rendered - still safe to return
                active_rendering_index.store(idx);
                return buffers[idx].gpu_buffer;
            }
            // Buffer is in unexpected state (FREE/UPLOADING) - fall through to linear scan
        } else {
            active_rendering_index.store(idx);
            return buffers[idx].gpu_buffer;
        }
    }

    for (int i = 0; i < BUFFER_COUNT; i++) {
        BufferState state = buffers[i].state.load();
        if (state == BUFFER_UPLOADING) {
            _wait_for_buffer_complete(i, false);
            state = buffers[i].state.load();
        }

        if (state == BUFFER_READY || state == BUFFER_RENDERING) {
            if (state == BUFFER_READY) {
                BufferState expected2 = BUFFER_READY;
                if (!buffers[i].state.compare_exchange_strong(expected2, BUFFER_RENDERING)) {
                    if (expected2 == BUFFER_RENDERING) {
                        active_rendering_index.store(i);
                        return buffers[i].gpu_buffer;
                    }
                    continue;  // State changed unexpectedly, try next buffer
                }
            }
            active_rendering_index.store(i);
            return buffers[i].gpu_buffer;
        }
    }

    return RID();
}

void GaussianMemoryStream::swap_buffers() {
    int current_idx = active_rendering_index.load();
    if (current_idx >= 0 && current_idx < BUFFER_COUNT) {
        BufferState expected = BUFFER_RENDERING;
        if (buffers[current_idx].state.compare_exchange_strong(expected, BUFFER_FREE)) {
            buffers[current_idx].frame_last_used = current_frame;
        }
    }
    active_rendering_index.store(-1);

    read_index = (read_index + 1) % BUFFER_COUNT;
}

void GaussianMemoryStream::compact_memory() {
    MutexLock lock(pool_mutex);

    bool merged;
    do {
        merged = false;
        for (uint32_t i = 0; i < gpu_memory_pool.blocks.size() - 1; i++) {
            auto &block1 = gpu_memory_pool.blocks[i];
            auto &block2 = gpu_memory_pool.blocks[i + 1];

            if (block1.free && block2.free && block1.offset + block1.size == block2.offset) {
                block1.size += block2.size;
                gpu_memory_pool.blocks.remove_at(i + 1);
                merged = true;
                break;
            }
        }
    } while (merged);
}

void GaussianMemoryStream::defragment_if_needed() {
    float fragmentation = gpu_memory_pool.get_fragmentation_ratio();
    if (fragmentation > gpu_memory_pool.fragmentation_threshold / 100.0f) {
        // Hard guard: pool defragmentation rewrites offsets, but active owners still reference
        // their previous pool offsets and RIDs. Without coordinated owner remap + GPU data
        // relocation this corrupts allocation bookkeeping.
        static bool unsafe_defrag_warned = false;
        if (!unsafe_defrag_warned) {
            GS_LOG_GPU_MEMORY_WARN(vformat("[GPU Memory Stream] Skipping unsafe defragmentation at %.1f%% fragmentation "
                                           "(offset remap/data relocation path not implemented).",
                    fragmentation * 100.0f));
            unsafe_defrag_warned = true;
        }
        compact_memory();
    }
}

uint32_t GaussianMemoryStream::get_allocated_memory_mb() const {
    uint64_t total_bytes = 0;
    for (int i = 0; i < BUFFER_COUNT; i++) {
        total_bytes += buffers[i].capacity;
    }
    return static_cast<uint32_t>(total_bytes / (1024 * 1024));
}

uint32_t GaussianMemoryStream::get_used_memory_mb() const {
    uint32_t total_used = 0;
    for (int i = 0; i < BUFFER_COUNT; i++) {
        total_used += buffers[i].used;
    }
    return total_used / (1024 * 1024);
}

float GaussianMemoryStream::get_memory_efficiency() const {
    uint32_t allocated = get_allocated_memory_mb();
    uint32_t used = get_used_memory_mb();

    if (allocated == 0) return 0.0f;
    return (float)used / (float)allocated;
}

Dictionary GaussianMemoryStream::get_task_debug_state() const {
    Dictionary result;
    Dictionary upload_state;
    upload_state["name"] = "upload";
    Dictionary residency_state;
    residency_state["name"] = "residency";
    Dictionary eviction_state;
    eviction_state["name"] = "eviction";
    result["upload"] = upload_state;
    result["residency"] = residency_state;
    result["eviction"] = eviction_state;
    return result;
}

void GaussianMemoryStream::begin_frame(uint64_t frame_number) {
    current_frame = frame_number;
    stats.total_frames++;
    _poll_uploads();
}

void GaussianMemoryStream::end_frame() {
    float current_mb = get_used_memory_mb();
    if (current_mb > stats.peak_memory_mb) {
        stats.peak_memory_mb = current_mb;
    }

    if (current_frame > 0 && current_frame % 100 == 0) {
        float stall_rate = (stats.stalls * 100.0f) / MAX(1u, stats.total_frames);
        float pool_hit_rate = (stats.pool_hits * 100.0f) /
                MAX(1u, stats.pool_hits + stats.pool_misses);

        float total_bandwidth = 0.0f;
        uint32_t buffer_count = 0;
        for (int i = 0; i < BUFFER_COUNT; i++) {
            if (buffers[i].last_bandwidth_mbps > 0.0f) {
                total_bandwidth += buffers[i].last_bandwidth_mbps;
                buffer_count++;
            }
        }
        float avg_bandwidth_mbps = (buffer_count > 0) ? (total_bandwidth / buffer_count) : 0.0f;

        GS_LOG_GPU_MEMORY_DEBUG("======== GPU PERFORMANCE METRICS (Issues #107 & #108) ========");
        GS_LOG_GPU_MEMORY_DEBUG(vformat("Frame %d - Stalls: %d (%.1f%%), Ready Reuse: %d, Pool Hit Rate: %.1f%%",
                current_frame, stats.stalls, stall_rate,
                stats.reused_ready_buffers, pool_hit_rate));
        GS_LOG_GPU_MEMORY_DEBUG(vformat("Memory: Used %.1f MB / Peak %.1f MB, Efficiency: %.1f%%",
                current_mb, stats.peak_memory_mb, get_memory_efficiency() * 100.0f));
        GS_LOG_GPU_MEMORY_DEBUG(vformat("Upload submit: %.1f MB/s (CPU target: >10000 MB/s), Buffer switches: %d",
                avg_bandwidth_mbps, stats.buffer_switches));
        GS_LOG_GPU_MEMORY_DEBUG(vformat("Defragmentation: %d operations, GPU utilization target: >85%%",
                stats.defrag_count));
        GS_LOG_GPU_MEMORY_DEBUG("==============================================================");

        bool performance_good = true;
        if (stall_rate > 5.0f) {
            GS_LOG_GPU_MEMORY_WARN(vformat("[ISSUE #107 WARNING] Stall rate %.1f%% exceeds 5%% target!", stall_rate));
            performance_good = false;
        }
        if (avg_bandwidth_mbps < 8000.0f && avg_bandwidth_mbps > 0.0f) {
            GS_LOG_GPU_MEMORY_WARN(vformat("[ISSUE #108 WARNING] Upload submit %.1f MB/s below 8000 MB/s target!", avg_bandwidth_mbps));
            performance_good = false;
        }
        float memory_efficiency = get_memory_efficiency() * 100.0f;
        if (memory_efficiency < 80.0f) {
            performance_good = false;
        }
        if (performance_good) {
            GS_LOG_GPU_MEMORY_DEBUG("[SUCCESS] All performance targets met!");
        }
    }

    defragment_if_needed();
}

RID GaussianMemoryStream::_allocate_from_pool(RenderingDevice *p_device, uint32_t size, uint32_t pool_offset) {
    ERR_FAIL_NULL_V_MSG(p_device, RID(), "RenderingDevice is null");

    // Create GPU buffer from pool allocation
    // Note: Godot's RenderingDevice doesn't support suballocations directly
    // We create a buffer but track it as part of our pool for management
    // In a real implementation with direct memory access, we would:
    // 1. Use a single large buffer for the pool
    // 2. Return views/offsets into that buffer
    // Since Godot doesn't expose this, we create individual buffers
    // but track them in our pool for statistics and management
    RID buffer = p_device->storage_buffer_create(size, Vector<uint8_t>());

    if (buffer.is_valid()) {
        p_device->set_resource_name(buffer, "GS_GaussianMemoryStream_PoolBuffer");
        GS_LOG_GPU_MEMORY_DEBUG(vformat("[MEMORY POOL] Created GPU buffer from pool: size=%d, offset=%d", size, pool_offset));
    }

    return buffer;
}

// ==============================================================================
// MemoryPool Implementation
// ==============================================================================

uint32_t GaussianMemoryStream::MemoryPool::allocate(uint32_t size, uint32_t alignment) {
    // Align size
    size = (size + alignment - 1) & ~(alignment - 1);

    // Find first fit
    for (uint32_t i = 0; i < blocks.size(); i++) {
        Block &block = blocks[i];
        if (block.free && block.size >= size) {
            uint32_t offset = block.offset;

            if (block.size == size) {
                // Exact fit
                block.free = false;
            } else {
                // Split block
                Block remainder_block;
                remainder_block.offset = block.offset + size;
                remainder_block.size = block.size - size;
                remainder_block.free = true;
                remainder_block.last_frame_used = block.last_frame_used;

                block.size = size;
                block.free = false;

                // Insert new block after current
                blocks.insert(i + 1, remainder_block);
            }

            used_size += size;
            return offset;
        }
    }

    return UINT32_MAX; // Allocation failed
}

void GaussianMemoryStream::MemoryPool::deallocate(uint32_t offset) {
    for (auto &block : blocks) {
        if (block.offset == offset && !block.free) {
            block.free = true;
            used_size -= block.size;
            return;
        }
    }
}

void GaussianMemoryStream::MemoryPool::defragment() {
    // Disabled: this method used to rewrite allocation offsets without remapping all
    // owners that hold pool_offset/RID pairs. That is unsafe until a coordinated owner
    // remap + GPU relocation pipeline exists.
    GS_LOG_GPU_MEMORY_WARN("[MEMORY POOL] Defragmentation disabled: unsafe without owner remap/data relocation.");
}

float GaussianMemoryStream::MemoryPool::get_fragmentation_ratio() const {
    if (total_size == 0 || used_size == 0) return 0.0f;

    uint32_t free_blocks = 0;
    uint32_t largest_free_block = 0;

    for (const auto &block : blocks) {
        if (block.free) {
            free_blocks++;
            if (block.size > largest_free_block) {
                largest_free_block = block.size;
            }
        }
    }

    uint32_t total_free = total_size - used_size;
    if (total_free == 0) return 0.0f;

    // Fragmentation ratio: 1.0 - (largest free block / total free space)
    return 1.0f - ((float)largest_free_block / (float)total_free);
}

// ==============================================================================
// StreamingPipeline Implementation
// ==============================================================================

StreamingPipeline::StreamingPipeline() {
    // Initialize with defaults
}

StreamingPipeline::~StreamingPipeline() {
    shutdown();
}

void StreamingPipeline::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "memory_stream", "gaussian_data"), &StreamingPipeline::initialize);
    ClassDB::bind_method(D_METHOD("shutdown"), &StreamingPipeline::shutdown);

    ClassDB::bind_method(D_METHOD("start_streaming"), &StreamingPipeline::start_streaming);
    ClassDB::bind_method(D_METHOD("stop_streaming"), &StreamingPipeline::stop_streaming);

    ClassDB::bind_method(D_METHOD("set_lod_level", "lod"), &StreamingPipeline::set_lod_level);
    ClassDB::bind_method(D_METHOD("get_current_lod"), &StreamingPipeline::get_current_lod);

    ClassDB::bind_method(D_METHOD("compact_memory"), &StreamingPipeline::compact_memory);
    ClassDB::bind_method(D_METHOD("get_current_buffer"), &StreamingPipeline::get_current_buffer);
    ClassDB::bind_method(D_METHOD("get_streaming_stats"), &StreamingPipeline::get_streaming_stats);
}

Error StreamingPipeline::initialize(Ref<GaussianMemoryStream> p_stream, Ref<::GaussianData> p_data) {
    ERR_FAIL_COND_V(!p_stream.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!p_data.is_valid(), ERR_INVALID_PARAMETER);

    memory_stream = p_stream;
    gaussian_data = p_data;

    GS_LOG_STREAMING_INFO("[Streaming Pipeline] Initialized with " + itos(gaussian_data->get_count()) + " gaussians");

    return OK;
}

void StreamingPipeline::shutdown() {
    stop_streaming();

    if (memory_stream.is_valid()) {
        memory_stream->shutdown();
    }

    memory_stream.unref();
    gaussian_data.unref();
}

void StreamingPipeline::start_streaming() {
    if (thread_running.load(std::memory_order_acquire)) {
        return;
    }

    thread_running.store(true, std::memory_order_release);
    thread_exit.store(false, std::memory_order_release);

    streaming_thread.start(_streaming_thread_entry, this);

    bool should_post = false;
    {
        MutexLock lock(state_mutex);
        if (state.needs_update && !state.is_streaming) {
            state.is_streaming = true;
            should_post = true;
        }
    }

    if (should_post) {
        stream_semaphore.post();
    }
    GS_LOG_STREAMING_INFO("[Streaming Pipeline] Streaming thread started");
}

void StreamingPipeline::stop_streaming() {
    if (!thread_running.load(std::memory_order_acquire)) {
        return;
    }

    thread_exit.store(true, std::memory_order_release);
    stream_semaphore.post(); // Wake up thread

    streaming_thread.wait_to_finish();
    thread_running.store(false, std::memory_order_release);

    MutexLock lock(state_mutex);
    state.is_streaming = false;

    GS_LOG_STREAMING_INFO("[Streaming Pipeline] Streaming thread stopped");
}

void StreamingPipeline::set_lod_level(uint32_t p_lod) {
    bool should_post = false;
    {
        MutexLock lock(state_mutex);
        if (state.current_lod == p_lod) {
            return;
        }

        state.current_lod = p_lod;
        state.needs_update = true;

        if (thread_running.load(std::memory_order_acquire) && !state.is_streaming) {
            state.is_streaming = true;
            should_post = true;
        }
    }

    if (should_post) {
        stream_semaphore.post();
    }
}

void StreamingPipeline::_streaming_thread_func() {
    while (!thread_exit.load(std::memory_order_acquire)) {
        // Wait for work
        stream_semaphore.wait();

        if (thread_exit.load(std::memory_order_acquire)) {
            break;
        }

        while (!thread_exit.load(std::memory_order_acquire)) {
            uint32_t visible_start = 0;
            uint32_t visible_count = 0;
            {
                MutexLock lock(state_mutex);
                if (!state.needs_update) {
                    state.is_streaming = false;
                    break;
                }

                visible_start = state.visible_start;
                visible_count = state.visible_count;
                state.needs_update = false;
            }

            if (memory_stream.is_valid() && gaussian_data.is_valid()) {
                const LocalVector<Gaussian> &gaussians = gaussian_data->get_gaussian_storage();
                const uint64_t end_index = uint64_t(visible_start) + uint64_t(visible_count);
                if (visible_count > 0 && end_index <= uint64_t(gaussians.size())) {
                    memory_stream->stream_gaussians_async(gaussians,
                            visible_start,
                            visible_count,
                            gaussian_data->get_sh_high_order_coefficients_ptr(),
                            gaussian_data->get_sh_first_order_count(),
                            gaussian_data->get_sh_high_order_count());
                }
            }
        }
    }

    MutexLock lock(state_mutex);
    state.is_streaming = false;
}

void StreamingPipeline::_streaming_thread_entry(void *p_userdata) {
    StreamingPipeline *pipeline = static_cast<StreamingPipeline *>(p_userdata);
    if (pipeline != nullptr) {
        pipeline->_streaming_thread_func();
    }
}

void StreamingPipeline::update_visible_range(uint32_t start, uint32_t count) {
    bool should_post = false;
    {
        MutexLock lock(state_mutex);
        if (start != state.visible_start || count != state.visible_count) {
            state.visible_start = start;
            state.visible_count = count;
            state.needs_update = true;

            if (thread_running.load(std::memory_order_acquire) && !state.is_streaming) {
                state.is_streaming = true;
                should_post = true;
            }
        }
    }

    if (should_post) {
        stream_semaphore.post(); // Trigger streaming
    }
}

uint32_t StreamingPipeline::get_current_lod() const {
    MutexLock lock(state_mutex);
    return state.current_lod;
}

Dictionary StreamingPipeline::get_streaming_stats() const {
    Dictionary stats;

    if (memory_stream.is_valid()) {
        StreamingStats mem_stats = memory_stream->get_stats();
        stats["bytes_uploaded"] = mem_stats.total_bytes_uploaded;
        stats["buffer_switches"] = mem_stats.buffer_switches;
        stats["stalls"] = mem_stats.stalls;
        stats["peak_memory_mb"] = mem_stats.peak_memory_mb;
        stats["memory_efficiency"] = memory_stream->get_memory_efficiency();
        stats["sh_raw_bytes"] = mem_stats.sh_raw_bytes_uploaded;
        stats["sh_compressed_bytes"] = mem_stats.sh_compressed_bytes_uploaded;
        stats["sh_coefficients"] = mem_stats.sh_coefficients_streamed;
        float sh_ratio = mem_stats.sh_raw_bytes_uploaded > 0
                ? float(mem_stats.sh_compressed_bytes_uploaded) / float(mem_stats.sh_raw_bytes_uploaded)
                : 0.0f;
        stats["sh_compression_ratio"] = sh_ratio;
    }

    uint32_t visible_start = 0;
    uint32_t visible_count = 0;
    uint32_t current_lod = 0;
    bool is_streaming = false;
    {
        MutexLock lock(state_mutex);
        visible_start = state.visible_start;
        visible_count = state.visible_count;
        current_lod = state.current_lod;
        is_streaming = state.is_streaming;
    }

    stats["visible_start"] = visible_start;
    stats["visible_count"] = visible_count;
    stats["current_lod"] = current_lod;
    stats["is_streaming"] = is_streaming;

    return stats;
}

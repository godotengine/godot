#include "gpu_buffer_manager.h"

#include "core/os/os.h"
#include "servers/rendering/rendering_device.h"

#include "../core/gaussian_splat_manager.h"
#include "../interfaces/sync_policy.h"
#include "../logger/gs_logger.h"
#include "../logger/gs_debug_trace.h"

namespace {

RenderingDevice *_acquire_submission_device(RenderingDevice *p_device, GaussianSplatManager::ScopedSubmissionLock &r_lock) {
    RenderingDevice *device = p_device;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        device = manager->acquire_submission_device(device, r_lock);
    }
    return device;
}

uint32_t _get_frame_fence_delay(RenderingDevice *p_device) {
    if (!p_device) {
        return 1;
    }
    uint32_t delay = p_device->get_frame_delay();
    return delay > 0 ? delay : 1;
}

} // namespace

GPUBufferManager::BufferHandle GPUBufferManager::_make_handle(const BufferSet &p_set, const RID &p_rid, RenderingDevice *p_fallback) {
    BufferHandle handle;
    handle.buffer = p_rid;
    handle.device = p_set.device ? p_set.device : p_fallback;
    return handle;
}

GPUBufferManager::GPUBufferManager() {
    _reset_state(true);
}

GPUBufferManager::~GPUBufferManager() {
    cleanup_buffers();
}

void GPUBufferManager::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "rendering_device", "max_gaussians"), &GPUBufferManager::initialize, DEFVAL(2000000));
    ClassDB::bind_method(D_METHOD("upload_gaussian_data", "data"), &GPUBufferManager::upload_gaussian_data);
    ClassDB::bind_method(D_METHOD("begin_frame"), &GPUBufferManager::begin_frame);
    ClassDB::bind_method(D_METHOD("end_frame"), &GPUBufferManager::end_frame);
    ClassDB::bind_method(D_METHOD("swap_buffers"), &GPUBufferManager::swap_buffers);
    ClassDB::bind_method(D_METHOD("is_ready_for_update"), &GPUBufferManager::is_ready_for_update);
    ClassDB::bind_method(D_METHOD("get_current_read_buffer"), &GPUBufferManager::get_current_read_buffer);
    ClassDB::bind_method(D_METHOD("get_current_write_buffer"), &GPUBufferManager::get_current_write_buffer);
    ClassDB::bind_method(D_METHOD("get_gaussian_buffer"), &GPUBufferManager::get_gaussian_buffer);
    ClassDB::bind_method(D_METHOD("get_sort_key_buffer"), &GPUBufferManager::get_sort_key_buffer);
    ClassDB::bind_method(D_METHOD("get_sorted_indices_buffer"), &GPUBufferManager::get_sorted_indices_buffer);
    ClassDB::bind_method(D_METHOD("get_gaussian_count"), &GPUBufferManager::get_gaussian_count);
    ClassDB::bind_method(D_METHOD("get_memory_usage_mb"), &GPUBufferManager::get_memory_usage_mb);
}

GPUBufferManager::BufferSet &GPUBufferManager::_get_buffer_set(uint32_t p_index) {
    ERR_FAIL_UNSIGNED_INDEX_V(p_index, BUFFER_COUNT, buffer_sets[0]);
    return buffer_sets[p_index];
}

const GPUBufferManager::BufferSet &GPUBufferManager::_get_buffer_set(uint32_t p_index) const {
    ERR_FAIL_UNSIGNED_INDEX_V(p_index, BUFFER_COUNT, buffer_sets[0]);
    return buffer_sets[p_index];
}

void GPUBufferManager::_reset_state(bool p_reset_handles) {
    for (uint32_t i = 0; i < BUFFER_COUNT; i++) {
        BufferSet &set = buffer_sets[i];
        if (p_reset_handles) {
            set.reset();
        } else {
            set.gaussian_count = 0;
            set.visible_count = 0;
            set.in_flight = false;
            set.has_data = false;
            set.in_flight_frame = 0;
            set.in_flight_delay = 0;
#ifdef DEBUG_ENABLED
            set.debug_cpu_writing = false;
            set.debug_gpu_reading = false;
#endif
        }
    }
    read_index = 0;
    write_index = BUFFER_COUNT > 1 ? 1 : 0;
    frame_index = 0;
    frame_active = false;
    swap_performed_this_frame = false;
    current_count = 0;
    current_visible_count = 0;
    pending_submission_device = nullptr;
    pending_submission_needs_submit = false;
    pending_submission_requires_sync = false;
    if (p_reset_handles) {
        uniform_buffer_device = nullptr;
        sequential_index_cache.reset();
    }
}

Error GPUBufferManager::initialize(RenderingDevice *p_rd, uint32_t p_max_gaussians) {
    ERR_FAIL_NULL_V(p_rd, ERR_INVALID_PARAMETER);

    if (buffers_created) {
        cleanup_buffers();
    }

    rd = p_rd;
    max_gaussians = p_max_gaussians;

    return create_buffers();
}

Error GPUBufferManager::_create_buffer_set(BufferSet &r_set, uint32_t p_gaussian_size, uint32_t p_sort_key_size, uint32_t p_index_size) {
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    // Buffers are updated and consumed on the submission RenderingDevice that
    // records compute uploads. Allocate them on that same context so command
    // lists never reference foreign resources.
    RenderingDevice *device = _acquire_submission_device(rd, submission_lock);
    ERR_FAIL_NULL_V(device, ERR_CANT_CREATE);

    Vector<uint8_t> empty_data;

    empty_data.resize(p_gaussian_size);
    r_set.gaussian_buffer = device->storage_buffer_create(p_gaussian_size, empty_data);
    ERR_FAIL_COND_V(!r_set.gaussian_buffer.is_valid(), ERR_CANT_CREATE);
    device->set_resource_name(r_set.gaussian_buffer, "GS_GPUBufferManager_GaussianBuffer");

    empty_data.resize(p_sort_key_size);
    r_set.sort_key_buffer = device->storage_buffer_create(p_sort_key_size, empty_data);
    ERR_FAIL_COND_V(!r_set.sort_key_buffer.is_valid(), ERR_CANT_CREATE);
    device->set_resource_name(r_set.sort_key_buffer, "GS_GPUBufferManager_SortKeyBuffer");

    empty_data.resize(p_index_size);
    r_set.sorted_indices_buffer = device->storage_buffer_create(p_index_size, empty_data);
    ERR_FAIL_COND_V(!r_set.sorted_indices_buffer.is_valid(), ERR_CANT_CREATE);
    device->set_resource_name(r_set.sorted_indices_buffer, "GS_GPUBufferManager_SortedIndicesBuffer");

    r_set.fence = RID();
    r_set.gaussian_count = 0;
    r_set.visible_count = 0;
    r_set.in_flight = false;
    r_set.has_data = false;
    r_set.in_flight_frame = 0;
    r_set.in_flight_delay = 0;
#ifdef DEBUG_ENABLED
    r_set.debug_cpu_writing = false;
    r_set.debug_gpu_reading = false;
#endif

    r_set.device = device;

    return OK;
}

void GPUBufferManager::_destroy_buffer_set(BufferSet &p_set) {
    RenderingDevice *device_hint = p_set.device ? p_set.device : rd;
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *device = _acquire_submission_device(device_hint, submission_lock);
    if (!device) {
        p_set.reset();
        return;
    }

    if (p_set.gaussian_buffer.is_valid()) {
        device->free(p_set.gaussian_buffer);
    }
    if (p_set.sort_key_buffer.is_valid()) {
        device->free(p_set.sort_key_buffer);
    }
    if (p_set.sorted_indices_buffer.is_valid()) {
        device->free(p_set.sorted_indices_buffer);
    }
    if (p_set.fence.is_valid()) {
        device->free(p_set.fence);
    }

    p_set.reset();
}

Error GPUBufferManager::create_buffers() {
    cleanup_buffers();

    if (!rd) {
        return ERR_UNCONFIGURED;
    }

    // Calculate buffer sizes
    uint32_t gaussian_buffer_size = sizeof(PackedGaussian) * max_gaussians;
    uint32_t sort_key_buffer_size = sizeof(SortKey) * max_gaussians;
    uint32_t indices_buffer_size = sizeof(uint32_t) * max_gaussians;
    uint32_t uniform_buffer_size = 256; // For view matrix and other uniforms

    for (uint32_t i = 0; i < BUFFER_COUNT; i++) {
        Error err = _create_buffer_set(buffer_sets[i], gaussian_buffer_size, sort_key_buffer_size, indices_buffer_size);
        if (err != OK) {
            cleanup_buffers();
            return err;
        }
    }

    sequential_index_cache.resize_uninitialized(max_gaussians);
    for (uint32_t i = 0; i < max_gaussians; i++) {
        sequential_index_cache[i] = i;
    }

    Vector<uint8_t> uniform_data;
    uniform_data.resize(uniform_buffer_size);
    {
        GaussianSplatManager::ScopedSubmissionLock submission_lock;
        RenderingDevice *device = _acquire_submission_device(rd, submission_lock);
        ERR_FAIL_NULL_V(device, ERR_CANT_CREATE);
        uniform_buffer = device->uniform_buffer_create(uniform_buffer_size, uniform_data);
        ERR_FAIL_COND_V(!uniform_buffer.is_valid(), ERR_CANT_CREATE);
        device->set_resource_name(uniform_buffer, "GS_GPUBufferManager_UniformBuffer");
        uniform_buffer_device = device;
    }

    buffers_created = true;
    _reset_state(false);

    GS_LOG_GPU_MEMORY_INFO(vformat("GPU buffers created for %d gaussians (%.2f MB)", max_gaussians, get_memory_usage_mb()));

    return OK;
}

void GPUBufferManager::_mark_buffer_ready(uint32_t p_index) {
    BufferSet &set = _get_buffer_set(p_index);

    set.in_flight = false;
    set.in_flight_frame = 0;
    set.in_flight_delay = 0;
    if (set.fence.is_valid()) {
        RenderingDevice *device_hint = set.device ? set.device : rd;
        GaussianSplatManager::ScopedSubmissionLock submission_lock;
        RenderingDevice *device = _acquire_submission_device(device_hint, submission_lock);
        if (device) {
            device->free(set.fence);
        }
    }
    set.fence = RID();
#ifdef DEBUG_ENABLED
    set.debug_gpu_reading = false;
#endif
}

void GPUBufferManager::_queue_submission(RenderingDevice *p_device, bool p_requires_sync) {
    if (!p_device) {
        return;
    }

    if (pending_submission_device && pending_submission_device != p_device) {
        _flush_pending_submission(true);
    }

    pending_submission_device = p_device;
    pending_submission_needs_submit = true;
    pending_submission_requires_sync = pending_submission_requires_sync || p_requires_sync;
}

void GPUBufferManager::_flush_pending_submission(bool p_block) {
    if (!pending_submission_device) {
        if (p_block && rd) {
            gs_device_utils::safe_submit_and_sync(rd);
        }
        return;
    }

    RenderingDevice *device = pending_submission_device;
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        device = manager->acquire_submission_device(device, submission_lock);
    }
    if (!device) {
        return;
    }

    if (pending_submission_needs_submit || p_block || pending_submission_requires_sync) {
        if (p_block) {
            gs_device_utils::safe_submit_and_sync(device);
        } else {
            gs_device_utils::safe_submit(device);
        }
        pending_submission_needs_submit = false;
        pending_submission_requires_sync = false;
    }

    if (!pending_submission_needs_submit && !pending_submission_requires_sync) {
        pending_submission_device = nullptr;
    }
}

void GPUBufferManager::_wait_for_buffer(uint32_t p_index, bool p_block) {
    BufferSet &set = _get_buffer_set(p_index);

    if (!set.in_flight) {
        return;
    }

    if (!rd) {
        _mark_buffer_ready(p_index);
        return;
    }

    (void)p_block;

    _flush_pending_submission(false);

    RenderingDevice *device_hint = set.device ? set.device : rd;
    uint32_t delay = set.in_flight_delay > 0 ? set.in_flight_delay : _get_frame_fence_delay(device_hint);
    uint64_t frame_age = frame_index >= set.in_flight_frame ? (frame_index - set.in_flight_frame) : 0;
    if (frame_age >= delay) {
        _mark_buffer_ready(p_index);
    }
}

void GPUBufferManager::_force_wait_for_buffer(uint32_t p_index) {
    BufferSet &set = _get_buffer_set(p_index);

    if (!set.in_flight) {
        return;
    }

    _flush_pending_submission(true);
    _mark_buffer_ready(p_index);
}

bool GPUBufferManager::_acquire_write_buffer(bool p_block) {
    if (!buffers_created) {
        return false;
    }

    BufferSet &write_set = buffer_sets[write_index];
    if (!write_set.in_flight) {
        return true;
    }

    _wait_for_buffer(write_index, false);
    if (!buffer_sets[write_index].in_flight) {
        return true;
    }

    if (!p_block) {
        return false;
    }

    _force_wait_for_buffer(write_index);
    return !buffer_sets[write_index].in_flight;
}

bool GPUBufferManager::_begin_frame_internal(bool p_block) {
    if (frame_active) {
        return true;
    }

    if (!_acquire_write_buffer(p_block)) {
        return false;
    }

    frame_active = true;
    swap_performed_this_frame = false;
    frame_index++;

#ifdef DEBUG_ENABLED
    buffer_sets[write_index].debug_cpu_writing = true;
    buffer_sets[read_index].debug_gpu_reading = true;
#endif

    return true;
}

void GPUBufferManager::cleanup_buffers() {
    if (!rd || !buffers_created) {
        _reset_state(true);
        return;
    }

    _flush_pending_submission(true);
    gs_device_utils::safe_submit_and_sync(rd);

    for (uint32_t i = 0; i < BUFFER_COUNT; i++) {
        _destroy_buffer_set(buffer_sets[i]);
    }

    if (uniform_buffer.is_valid()) {
        GaussianSplatManager::ScopedSubmissionLock submission_lock;
        RenderingDevice *device = _acquire_submission_device(uniform_buffer_device ? uniform_buffer_device : rd, submission_lock);
        if (device) {
            device->free(uniform_buffer);
        }
        uniform_buffer = RID();
        uniform_buffer_device = nullptr;
    }

    buffers_created = false;
    _reset_state(true);
}

Error GPUBufferManager::create_buffers_from_data(const Ref<::GaussianData> &p_data) {
    ERR_FAIL_COND_V(!p_data.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!rd, ERR_UNCONFIGURED);

    if (!buffers_created) {
        Error err = create_buffers();
        if (err != OK) {
            return err;
        }
    }

    return upload_gaussian_data(p_data);
}

void GPUBufferManager::begin_frame() {
    ERR_FAIL_COND_MSG(!buffers_created, "GPUBufferManager not initialized");
    ERR_FAIL_COND_MSG(frame_active, "begin_frame called while a frame is already active");

    if (!is_ready_for_update()) {
#ifdef DEBUG_ENABLED
        GS_LOG_INFO_DEFAULT("[GPUBufferManager] Waiting for GPU to finish before beginning frame update");
#endif
    }

    bool ok = _begin_frame_internal(false);
    if (!ok) {
        GS_LOG_INFO_DEFAULT("[GPUBufferManager] Write buffer still in flight; skipping frame update");
        return;
    }
}

void GPUBufferManager::end_frame() {
    ERR_FAIL_COND_MSG(!buffers_created, "GPUBufferManager not initialized");
    ERR_FAIL_COND_MSG(!frame_active, "end_frame called without a matching begin_frame");

    BufferSet &read_set = buffer_sets[read_index];
    BufferSet &write_set = buffer_sets[write_index];

#ifdef DEBUG_ENABLED
    write_set.debug_cpu_writing = false;
#endif

    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *submission_rd = nullptr;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        submission_rd = manager->acquire_submission_device(nullptr, submission_lock);
    }
    if (submission_rd) {
        if (read_set.has_data && current_count > 0) {
            read_set.in_flight = true;
            read_set.in_flight_frame = frame_index;
            read_set.in_flight_delay = _get_frame_fence_delay(submission_rd);
        } else {
            read_set.in_flight = false;
            read_set.in_flight_frame = 0;
            read_set.in_flight_delay = 0;
        }
        _queue_submission(submission_rd, read_set.in_flight);
        _flush_pending_submission(false);
    } else {
        WARN_PRINT_ONCE("[GPUBufferManager] Unable to acquire shared submission RenderingDevice; skipping submission");
        if (read_set.has_data && current_count > 0) {
            RenderingDevice *delay_device = read_set.device ? read_set.device : rd;
            read_set.in_flight = true;
            read_set.in_flight_frame = frame_index;
            read_set.in_flight_delay = _get_frame_fence_delay(delay_device);
        } else {
            read_set.in_flight = false;
            read_set.in_flight_frame = 0;
            read_set.in_flight_delay = 0;
        }
    }

    frame_active = false;
    swap_performed_this_frame = false;
}

void GPUBufferManager::swap_buffers() {
    ERR_FAIL_COND_MSG(!buffers_created, "GPUBufferManager not initialized");
    ERR_FAIL_COND_MSG(!frame_active, "swap_buffers must be called between begin_frame and end_frame");
    ERR_FAIL_COND_MSG(swap_performed_this_frame, "swap_buffers has already been called this frame");

    BufferSet &current_read = buffer_sets[read_index];

#ifdef DEBUG_ENABLED
    ERR_FAIL_COND_MSG(current_read.debug_cpu_writing, "GPU read buffer is currently marked for CPU writes");
#endif

    std::swap(read_index, write_index);
    swap_performed_this_frame = true;

    BufferSet &new_read = buffer_sets[read_index];
    BufferSet &new_write = buffer_sets[write_index];

    current_count = new_read.has_data ? new_read.gaussian_count : 0;
    current_visible_count = new_read.has_data ? new_read.visible_count : 0;

#ifdef DEBUG_ENABLED
    new_read.debug_cpu_writing = false;
    new_read.debug_gpu_reading = true;
    new_write.debug_cpu_writing = false;
    new_write.debug_gpu_reading = new_write.in_flight;
#endif
}

bool GPUBufferManager::is_ready_for_update() const {
    if (!buffers_created) {
        return false;
    }
    const BufferSet &set = buffer_sets[write_index];
    if (!set.in_flight) {
        return true;
    }
    RenderingDevice *device_hint = set.device ? set.device : rd;
    uint32_t delay = set.in_flight_delay > 0 ? set.in_flight_delay : _get_frame_fence_delay(device_hint);
    uint64_t frame_age = frame_index >= set.in_flight_frame ? (frame_index - set.in_flight_frame) : 0;
    return frame_age >= delay;
}

RID GPUBufferManager::get_current_read_buffer() const {
    if (!buffers_created) {
        return RID();
    }
    return buffer_sets[read_index].gaussian_buffer;
}

RID GPUBufferManager::get_current_write_buffer() const {
    if (!buffers_created) {
        return RID();
    }
    return buffer_sets[write_index].gaussian_buffer;
}

Error GPUBufferManager::upload_gaussian_data(const Ref<::GaussianData> &p_data) {
    ERR_FAIL_COND_V(!p_data.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!buffers_created, ERR_UNCONFIGURED);

    bool frame_was_active = frame_active;
    GS_LOG_GPU_MEMORY_DEBUG(vformat("[BUFFER-DBG] upload_gaussian_data: frame_active=%d read_idx=%d write_idx=%d count=%d",
            frame_active ? 1 : 0, read_index, write_index, p_data->get_count()));
    if (!frame_was_active) {
        bool ok = _begin_frame_internal(false);
        if (!ok) {
            _force_wait_for_buffer(write_index);
            ok = _begin_frame_internal(false);
        }
        ERR_FAIL_COND_V_MSG(!ok, ERR_BUSY, "Unable to acquire GPU write buffer for upload");
    }

    uint32_t target_count = p_data->get_count();
    BufferSet &write_set = buffer_sets[write_index];

    if (target_count == 0) {
        write_set.gaussian_count = 0;
        write_set.visible_count = 0;
        write_set.has_data = false;
    } else {
        ERR_FAIL_COND_V(target_count > max_gaussians, ERR_PARAMETER_RANGE_ERROR);

        Vector<PackedGaussian> packed_data;
        packed_data.resize(target_count);

        SHCompressionMetrics compression_metrics;
        const LocalVector<Gaussian> &gaussians = p_data->get_gaussian_storage();
        ERR_FAIL_COND_V((uint32_t)gaussians.size() < target_count, ERR_BUG);

        // Debug: Check source sh_dc values
        static int buffer_mgr_debug = 0;
        if (++buffer_mgr_debug <= 3 && gaussians.size() > 0) {
            const Gaussian &g0 = gaussians[0];
            GaussianSplatting::debug_trace_record_buffer_mgr(gaussians.size(), g0.sh_dc, g0.opacity);
        }

        pack_gaussians_range(gaussians,
                0,
                target_count,
                packed_data,
                compression_metrics,
                p_data->get_sh_high_order_coefficients_ptr(),
                p_data->get_sh_first_order_count(),
                p_data->get_sh_high_order_count());

        GaussianSplatManager::ScopedSubmissionLock upload_lock;
        RenderingDevice *upload_device = _acquire_submission_device(write_set.device ? write_set.device : rd, upload_lock);
        ERR_FAIL_NULL_V(upload_device, ERR_CANT_CREATE);
        ERR_FAIL_COND_V_MSG(sequential_index_cache.size() < target_count, ERR_BUG, "Sequential index cache is smaller than the requested upload count");

        uint64_t start_time = OS::get_singleton()->get_ticks_usec();
        upload_device->buffer_update(write_set.gaussian_buffer, 0, sizeof(PackedGaussian) * target_count, packed_data.ptr());
        upload_device->buffer_update(write_set.sorted_indices_buffer, 0, sizeof(uint32_t) * target_count, sequential_index_cache.ptr());
        gs_device_utils::safe_submit_and_sync(upload_device);

        uint64_t upload_time = OS::get_singleton()->get_ticks_usec() - start_time;
        GS_LOG_GPU_MEMORY_INFO(vformat("Uploaded %d gaussians to GPU in %.2f ms (SH coeffs: %d, compression %.2f%%)",
                target_count,
                upload_time / 1000.0,
                compression_metrics.coefficient_count,
                compression_metrics.raw_bytes > 0
                        ? (compression_metrics.compressed_bytes * 100.0) / (double)compression_metrics.raw_bytes
                        : 0.0));

        write_set.gaussian_count = target_count;
        write_set.visible_count = target_count;
        write_set.has_data = true;
    }

    if (!frame_was_active) {
        if (!swap_performed_this_frame) {
            swap_buffers();
        }
        end_frame();
    }

    return OK;
}

void GPUBufferManager::clear_gaussian_data() {
    if (!buffers_created) {
        return;
    }

    for (uint32_t i = 0; i < BUFFER_COUNT; i++) {
        buffer_sets[i].gaussian_count = 0;
        buffer_sets[i].visible_count = 0;
        buffer_sets[i].has_data = false;
    }

    current_count = 0;
    current_visible_count = 0;
}

RID GPUBufferManager::get_gaussian_buffer() const {
    return get_current_read_buffer();
}

RID GPUBufferManager::get_sort_key_buffer() const {
    if (!buffers_created) {
        return RID();
    }
    return buffer_sets[read_index].sort_key_buffer;
}

RID GPUBufferManager::get_sorted_indices_buffer() const {
    if (!buffers_created) {
        return RID();
    }
    static int dbg_count = 0;
    if (++dbg_count <= 3) {
        GS_LOG_GPU_MEMORY_DEBUG(vformat("[BUFFER-DBG] get_sorted_indices: read_idx=%d has_data=%d count=%d",
                read_index, buffer_sets[read_index].has_data ? 1 : 0, buffer_sets[read_index].gaussian_count));
    }
    return buffer_sets[read_index].sorted_indices_buffer;
}

RID GPUBufferManager::get_current_write_sort_key_buffer() const {
    if (!buffers_created) {
        return RID();
    }
    return buffer_sets[write_index].sort_key_buffer;
}

RID GPUBufferManager::get_current_write_sorted_indices_buffer() const {
    if (!buffers_created) {
        return RID();
    }
    return buffer_sets[write_index].sorted_indices_buffer;
}

GPUBufferManager::BufferHandle GPUBufferManager::get_gaussian_handle() const {
    if (!buffers_created) {
        return BufferHandle();
    }
    return _make_handle(buffer_sets[read_index], buffer_sets[read_index].gaussian_buffer, rd);
}

GPUBufferManager::BufferHandle GPUBufferManager::get_sort_key_handle() const {
    if (!buffers_created) {
        return BufferHandle();
    }
    return _make_handle(buffer_sets[read_index], buffer_sets[read_index].sort_key_buffer, rd);
}

GPUBufferManager::BufferHandle GPUBufferManager::get_sorted_indices_handle() const {
    if (!buffers_created) {
        return BufferHandle();
    }
    return _make_handle(buffer_sets[read_index], buffer_sets[read_index].sorted_indices_buffer, rd);
}

GPUBufferManager::BufferHandle GPUBufferManager::get_current_read_handle() const {
    if (!buffers_created) {
        return BufferHandle();
    }
    return _make_handle(buffer_sets[read_index], buffer_sets[read_index].gaussian_buffer, rd);
}

GPUBufferManager::BufferHandle GPUBufferManager::get_current_write_handle() const {
    if (!buffers_created) {
        return BufferHandle();
    }
    return _make_handle(buffer_sets[write_index], buffer_sets[write_index].gaussian_buffer, rd);
}

void GPUBufferManager::set_visible_count(uint32_t p_visible) {
    if (!buffers_created) {
        return;
    }

    BufferSet &read_set = buffer_sets[read_index];
    uint32_t maximum = read_set.gaussian_count;
    uint32_t clamped = maximum > 0 ? MIN(p_visible, maximum) : p_visible;
    read_set.visible_count = clamped;
    current_visible_count = clamped;

    BufferSet &write_set = buffer_sets[write_index];
    if (!write_set.has_data) {
        write_set.visible_count = clamped;
    }
}

float GPUBufferManager::get_memory_usage_mb() const {
    if (!buffers_created) {
        return 0.0f;
    }

    uint64_t total_bytes = 0;
    total_bytes += sizeof(PackedGaussian) * max_gaussians * BUFFER_COUNT;
    total_bytes += sizeof(SortKey) * max_gaussians * BUFFER_COUNT;
    total_bytes += sizeof(uint32_t) * max_gaussians * BUFFER_COUNT;
    total_bytes += 256; // Uniform buffer

    return total_bytes / (1024.0f * 1024.0f);
}

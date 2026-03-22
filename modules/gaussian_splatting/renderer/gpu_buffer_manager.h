#ifndef GPU_BUFFER_MANAGER_H
#define GPU_BUFFER_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"
#include "../core/gaussian_data.h"
#include "gaussian_gpu_layout.h"

class GPUBufferManager : public RefCounted {
    GDCLASS(GPUBufferManager, RefCounted);

public:
    struct BufferHandle {
        RID buffer;
        RenderingDevice *device = nullptr;

        bool is_valid() const {
            return buffer.is_valid() && device != nullptr;
        }
    };

    // Sort key for depth sorting
    struct SortKey {
        float depth;
        uint32_t index;
    };

    class DeferredDeletionQueue {
        struct PendingDelete {
            RID rid;
            RenderingDevice *device = nullptr;
            uint32_t frame_delay = 0;
            bool auto_free = false;
        };

        LocalVector<PendingDelete> pending;
        uint32_t current_frame = 0;

        static bool _auto_free_is_valid(RenderingDevice *p_device, const RID &p_rid) {
            return p_device->uniform_set_is_valid(p_rid) ||
                    p_device->compute_pipeline_is_valid(p_rid) ||
                    p_device->render_pipeline_is_valid(p_rid) ||
                    p_device->framebuffer_is_valid(p_rid);
        }

        static void _free_pending_delete(const PendingDelete &p_pending_delete) {
            if (!p_pending_delete.rid.is_valid() || !p_pending_delete.device) {
                return;
            }

            if (p_pending_delete.auto_free) {
                // Auto-free types (PR 103113): only free if still valid.
                if (_auto_free_is_valid(p_pending_delete.device, p_pending_delete.rid)) {
                    p_pending_delete.device->free(p_pending_delete.rid);
                }
            } else {
                p_pending_delete.device->free(p_pending_delete.rid);
            }
        }

    public:
        void queue_free(RenderingDevice *p_device, const RID &p_rid, uint32_t p_delay = 2) {
            if (!p_device || !p_rid.is_valid()) {
                return;
            }
            bool auto_free = _auto_free_is_valid(p_device, p_rid);
            pending.push_back({p_rid, p_device, current_frame + p_delay, auto_free});
        }

        void process_frame() {
            current_frame++;
            uint32_t write_index = 0;
            const uint32_t pending_count = pending.size();
            for (uint32_t read_index = 0; read_index < pending_count; read_index++) {
                if (pending[read_index].frame_delay <= current_frame) {
                    _free_pending_delete(pending[read_index]);
                    continue;
                }

                if (write_index != read_index) {
                    pending[write_index] = pending[read_index];
                }
                write_index++;
            }
            pending.resize(write_index);
        }

        void flush_all() {
            for (uint32_t i = 0; i < pending.size(); i++) {
                _free_pending_delete(pending[i]);
            }
            pending.clear();
        }
    };

private:
    RenderingDevice *rd = nullptr;

    struct BufferSet {
        RID gaussian_buffer;
        RID sort_key_buffer;
        RID sorted_indices_buffer;
        RID fence;
        uint32_t gaussian_count = 0;
        uint32_t visible_count = 0;
        bool in_flight = false;
        bool has_data = false;
        uint64_t in_flight_frame = 0;
        uint32_t in_flight_delay = 0;
        bool debug_cpu_writing = false;
        bool debug_gpu_reading = false;
        RenderingDevice *device = nullptr;

        void reset() {
            gaussian_buffer = RID();
            sort_key_buffer = RID();
            sorted_indices_buffer = RID();
            fence = RID();
            gaussian_count = 0;
            visible_count = 0;
            in_flight = false;
            has_data = false;
            in_flight_frame = 0;
            in_flight_delay = 0;
            debug_cpu_writing = false;
            debug_gpu_reading = false;
            device = nullptr;
        }
    };

    static constexpr uint32_t BUFFER_COUNT = 2;

    BufferSet buffer_sets[BUFFER_COUNT];
    RID uniform_buffer;
    RenderingDevice *uniform_buffer_device = nullptr;
    uint32_t read_index = 0;
    uint32_t write_index = 1;
    uint64_t frame_index = 0;
    bool frame_active = false;
    bool swap_performed_this_frame = false;

    uint32_t max_gaussians = 2000000;
    uint32_t current_count = 0;
    uint32_t current_visible_count = 0;
    bool buffers_created = false;

    // Submission batching (Issue #142)
    RenderingDevice *pending_submission_device = nullptr;
    bool pending_submission_needs_submit = false;
    bool pending_submission_requires_sync = false;

    Error create_buffers();
    void cleanup_buffers();
    Error _create_buffer_set(BufferSet &r_set, uint32_t p_gaussian_size, uint32_t p_sort_key_size, uint32_t p_index_size);
    void _destroy_buffer_set(BufferSet &p_set);
    void _reset_state(bool p_reset_handles = true);
    BufferSet &_get_buffer_set(uint32_t p_index);
    const BufferSet &_get_buffer_set(uint32_t p_index) const;
    void _wait_for_buffer(uint32_t p_index, bool p_block);
    void _force_wait_for_buffer(uint32_t p_index);
    void _mark_buffer_ready(uint32_t p_index);
    bool _acquire_write_buffer(bool p_block);
    bool _begin_frame_internal(bool p_block);
    uint32_t _get_next_index(uint32_t p_index) const { return (p_index + 1) % BUFFER_COUNT; }
    void _queue_submission(RenderingDevice *p_device, bool p_requires_sync);
    void _flush_pending_submission(bool p_block);
    static BufferHandle _make_handle(const BufferSet &p_set, const RID &p_rid, RenderingDevice *p_fallback);

protected:
    static void _bind_methods();

public:
    GPUBufferManager();
    ~GPUBufferManager();

    Error initialize(RenderingDevice *p_rd, uint32_t p_max_gaussians = 2000000);
    Error upload_gaussian_data(const Ref<::GaussianData> &p_data);
    void clear_gaussian_data();

    // Missing methods needed by gaussian_splat_renderer
    Error create_buffers_from_data(const Ref<::GaussianData> &p_data);
    bool is_initialized() const { return buffers_created && rd != nullptr; }
    uint32_t get_buffer_capacity() const { return max_gaussians; }

    void begin_frame();
    void end_frame();
    void swap_buffers();
    bool is_ready_for_update() const;

    RID get_current_read_buffer() const;
    RID get_current_write_buffer() const;

    RID get_gaussian_buffer() const;
    RID get_sort_key_buffer() const;
    RID get_sorted_indices_buffer() const;
    RID get_current_write_sort_key_buffer() const;
    RID get_current_write_sorted_indices_buffer() const;
    BufferHandle get_gaussian_handle() const;
    BufferHandle get_sort_key_handle() const;
    BufferHandle get_sorted_indices_handle() const;
    BufferHandle get_current_read_handle() const;
    BufferHandle get_current_write_handle() const;
    RID get_uniform_buffer() const { return uniform_buffer; }
    
    uint32_t get_gaussian_count() const { return current_count; }
    uint32_t get_visible_count() const { return current_visible_count; }
    float get_memory_usage_mb() const;
    void set_visible_count(uint32_t p_visible);
};

#endif // GPU_BUFFER_MANAGER_H

#ifndef GS_GPU_SORTING_PIPELINE_H
#define GS_GPU_SORTING_PIPELINE_H

#include "gpu_sorting_pipeline_interfaces.h"
#include "render_device_manager.h"
#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/templates/hash_map.h"
#include "servers/rendering/rendering_device.h"
#include "../renderer/gpu_sorter.h"
#include "../renderer/gpu_sorting_config.h"

// Concrete implementation of IGPUSortingPipeline
// Manages GPU sorter lifecycle, sort buffers, and depth compute resources
class GPUSortingPipeline : public RefCounted, public IGPUSortingPipeline {
    GDCLASS(GPUSortingPipeline, RefCounted);

public:
    struct InstancePipelineInputs {
        RID atlas_gaussian_buffer;
        RID quantization_buffer;
        RID instance_buffer;
        RID chunk_meta_buffer;
        RID visible_chunk_buffer;
        RID splat_ref_buffer;
        RID sort_key_buffer;
        RID sort_value_buffer;
        RID counter_buffer;
        RID chunk_dispatch_buffer;
        RID indirect_count_buffer;
        RID instance_count_buffer;
        RenderingDevice *device = nullptr;
        uint32_t visible_chunk_count = 0;
        uint32_t max_visible_chunks = 0;
        uint32_t max_visible_splats = 0;
        uint32_t max_chunk_splats = 0;
    };

    GPUSortingPipeline();
    ~GPUSortingPipeline();

    // IGPUSortingPipeline interface - Lifecycle
    Error initialize(RenderingDevice *p_device, uint32_t p_initial_capacity) override;
    void shutdown() override;
    bool is_initialized() const override { return initialized && rd != nullptr; }
    bool is_ready() const override { return is_initialized() && gpu_sorter.is_valid(); }

    // IGPUSortingPipeline interface - Sorter management
    void rebuild_sorter(uint32_t p_capacity) override;
    void mark_sorter_dirty() override { sorter_needs_rebuild = true; }
    String get_algorithm_name() const override;
    uint32_t get_max_elements() const override;

    // IGPUSortingPipeline interface - Buffer management
    void ensure_buffers(uint32_t p_required_elements) override;
    void release_buffers() override;
    SortBufferHandles get_buffer_handles() const override;
    void set_sort_result_sink(ISortResultSink *p_sink);
    void set_sort_buffer_host_context(ISortBufferHostContext *p_context);
    void ensure_sort_buffers(uint32_t p_required_elements);
    void release_sort_buffers();
    void set_external_sort_indices(RID p_buffer, RenderingDevice *p_device);
    void clear_external_sort_indices();
    RID get_sort_indices_buffer() const { return sort_indices_buffer; }
    void set_forced_sort_algorithm(GPUSorterFactory::SortingAlgorithm p_algorithm);
    GPUSorterFactory::SortingAlgorithm get_forced_sort_algorithm() const { return forced_sort_algorithm; }
    void set_instance_pipeline_inputs(const InstancePipelineInputs &p_inputs);
    void clear_instance_pipeline_inputs();
    uint32_t get_last_instance_visible_splat_count() const { return last_instance_visible_splat_count; }
    void test_set_last_instance_visible_splat_count(uint32_t p_count, uint32_t p_frame_counter = 0) {
        instance_count_readback_state.pending = false;
        instance_count_readback_state.generation++;
        instance_count_readback_state.pending_frame_counter = 0;
        instance_count_readback_state.bootstrap_sync_attempted = true;
        last_instance_visible_splat_count = p_count;
        last_instance_visible_splat_count_valid = true;
        last_instance_visible_splat_count_frame = p_frame_counter;
    }
    String get_last_compute_error() const { return last_compute_error; }

    // IGPUSortingPipeline interface - Depth compute resources
    void ensure_depth_resources(RenderingDevice *p_device) override;
    RID get_depth_compute_shader() const override { return depth_compute_shader; }
    RID get_depth_compute_pipeline() const override { return depth_compute_pipeline; }
    void queue_depth_submission(RenderingDevice *p_device, bool p_requires_wait) override;
    RenderingDevice *get_depth_submission_device() const override { return depth_submission_device; }
    RenderingDevice *get_sort_resource_device() const override { return sort_resource_device ? sort_resource_device : rd; }
    bool populate_gpu_positions(RID p_buffer, uint32_t p_total_gaussians, uint32_t p_visible_splats,
            const Transform3D &p_cam_transform, float *r_position_ptr, bool p_write_distances,
            SortPositionInputs &p_inputs) override;
    bool sort_gaussians_gpu(const Transform3D &p_cam_transform, const SortFrameContext &p_context);

    // IGPUSortingPipeline interface - Sort execution
    SortOperationResult sort(const SortOperationParams &p_params) override;
    SortOperationResult sort_async(const SortOperationParams &p_params) override;
    void wait_for_completion() override;
    float get_last_sort_time_ms() const override;

    // IGPUSortingPipeline interface - Metrics
    SortingMetrics get_metrics() const override;

    // IGPUSortingPipeline interface - State queries
    bool is_sorting_in_progress() const override { return sorting_in_progress; }
    uint64_t get_current_timeline_value() const override { return current_sort_timeline_value; }

    // IGPUSortingPipeline interface - Implementation info
    String get_name() const override { return "GPUSortingPipeline"; }

    // Set the device manager for resource tracking
    void set_device_manager(Ref<RenderDeviceManager> p_device_manager);
    void set_manage_buffers(bool p_manage_buffers) { manage_buffers = p_manage_buffers; }
    bool is_managing_buffers() const { return manage_buffers; }

    // Direct access to underlying sorter (for god class compatibility during transition)
    Ref<IGPUSorter> get_sorter() const { return gpu_sorter; }
    Ref<IGPUSorter> rebuild_sorter_if_needed(RenderingDevice *p_device, uint32_t p_capacity, bool p_needs_rebuild);

    // Statistics
    uint32_t get_sort_buffer_capacity() const { return sort_buffer_capacity; }
    uint32_t get_local_buffer_capacity() const { return local_sort_buffer_capacity; }
    bool are_buffers_external() const { return sort_keys_external || sort_indices_external; }

protected:
    static void _bind_methods();

private:
    // State
    bool initialized = false;
    RenderingDevice *rd = nullptr;
    Ref<RenderDeviceManager> device_manager;

    // GPU sorter
    Ref<IGPUSorter> gpu_sorter;
    bool sorter_needs_rebuild = true;
    GPUSorterFactory::SortingAlgorithm forced_sort_algorithm = GPUSorterFactory::ALGORITHM_AUTO;
    bool sorting_in_progress = false;
    uint64_t current_sort_timeline_value = 0;
    uint64_t last_sort_submission_value = 0;
    bool manage_buffers = true;

    // Sort buffers - explicitly initialized to prevent garbage RID values
    RID sort_keys_buffer = RID();
    RID sort_indices_buffer = RID();
    uint32_t sort_buffer_capacity = 0;
    bool sort_keys_external = false;
    bool sort_indices_external = false;
    RenderingDevice *sort_resource_device = nullptr;

    // Local device buffers (for cross-device sorting)
    RID local_depth_keys_buffer = RID();
    RID local_splat_indices_buffer = RID();
    RID local_visible_indices_buffer = RID();
    uint32_t local_sort_buffer_capacity = 0;

    // Culled position buffer (for depth key generation)
    RID culled_position_buffer = RID();
    uint32_t culled_position_capacity = 0;

    // Depth compute resources - explicitly initialized to prevent garbage RID values
    RID depth_compute_shader = RID();
    RID depth_compute_pipeline = RID();
    RID depth_uniform_set = RID();
    RID instance_count_clamp_shader = RID();
    RID instance_count_clamp_pipeline = RID();
    RID instance_count_uniform_set = RID();
    RID instance_chunk_dispatch_shader = RID();
    RID instance_chunk_dispatch_pipeline = RID();
    RID instance_chunk_dispatch_uniform_set = RID();
    RenderingDevice *depth_submission_device = nullptr;
    RenderingDevice *instance_count_resource_device = nullptr;
    RenderingDevice *instance_chunk_dispatch_resource_device = nullptr;
    bool depth_submission_needs_submit = false;
    bool depth_submission_requires_wait = false;
    bool depth_quantization_enabled = false;
    RID instance_param_buffer = RID();
    RenderingDevice *instance_resource_device = nullptr;

    RID gather_positions_shader = RID();
    RID gather_positions_pipeline = RID();
    RID gather_uniform_set = RID();

    RID remap_compute_shader = RID();
    RID remap_compute_pipeline = RID();
    RID remap_uniform_set = RID();
    RenderingDevice *remap_resource_device = nullptr;

    // CPU-side buffers
    Vector<uint8_t> sort_key_bytes;
    Vector<uint8_t> sort_index_bytes;
    Vector<uint8_t> culled_position_bytes;
    Vector<uint8_t> instance_param_cache;
    bool instance_param_cache_valid = false;

    InstancePipelineInputs instance_inputs;
    bool instance_inputs_valid = false;
    uint32_t last_instance_visible_splat_count = 0;
    bool last_instance_visible_splat_count_valid = false;
    uint32_t last_instance_visible_splat_count_frame = 0;
    String last_compute_error;

    // BUF-3 optimization: Store reference to culler instead of copying data.
    // The original pattern copied indices/distances/importance from culler state
    // into temporary vectors, then copied them back after sorting. This caused
    // 5-20% CPU overhead from triple-copy pattern. Now we store a pointer to
    // the culler and read/write directly from/to its state.
    struct SortReadbackState {
        bool pending = false;
        uint64_t generation = 0;
        uint32_t expected_count = 0;
        Transform3D camera_transform;
        // Snapshot of culler state indices at readback request time.
        // We need these because the culler state may change before async readback completes.
        // Using a single vector instead of three reduces memory pressure.
        Vector<uint32_t> snapshot_indices;
    } sort_readback_state;
    struct InstanceCountReadbackState {
        bool pending = false;
        uint64_t generation = 0;
        uint32_t pending_frame_counter = 0;
        bool bootstrap_sync_attempted = false;
    } instance_count_readback_state;
    ISortResultSink *sort_result_sink = nullptr;
    ISortBufferHostContext *sort_buffer_host_context = nullptr;

    // PERF (#662): cache camera inverse to avoid redundant affine_inverse() calls.
    Transform3D cached_camera_transform;
    Transform3D cached_camera_to_world;
    bool cached_camera_transform_valid = false;

    // Helper methods
    void _track_resource(const RID &p_rid, RenderingDevice *p_device, bool p_owned = true, const char *p_label = nullptr);
    void _forget_resource(const RID &p_rid);
    void _free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid, bool p_is_auto_free = false);
    void ensure_remap_resources(RenderingDevice *p_device);
    void ensure_gather_resources(RenderingDevice *p_device);
    bool _publish_sorted_results(const Vector<uint8_t> &p_sorted_index_bytes);
    void _on_sort_readback(const Vector<uint8_t> &p_data, int64_t p_generation);
    void _on_instance_count_readback(const Vector<uint8_t> &p_data, int64_t p_generation);
    void _ensure_instance_param_buffer(RenderingDevice *p_device);
    void _ensure_instance_count_resources(RenderingDevice *p_device);
    void _ensure_instance_chunk_dispatch_resources(RenderingDevice *p_device);
    bool _sort_instance_pipeline(const Transform3D &p_cam_transform, const SortFrameContext &p_context);
};

#endif // GS_GPU_SORTING_PIPELINE_H

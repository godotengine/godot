#ifndef GS_GPU_SORTING_PIPELINE_INTERFACES_H
#define GS_GPU_SORTING_PIPELINE_INTERFACES_H

#include <cstdint>

#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"
#include "../renderer/gpu_sorter.h"  // For SortKeyConfig and SortingMetrics

class GaussianData;
struct Gaussian;
struct PackedGaussian;

// Sort buffer handles returned by the pipeline
struct SortBufferHandles {
    RID keys_buffer = RID();
    RID indices_buffer = RID();
    uint32_t capacity = 0;
    bool valid = false;
};

struct SortPositionInputs {
    const GaussianData *gaussian_data = nullptr;
    const LocalVector<Vector3> *test_positions = nullptr;
    const LocalVector<Vector3> *test_scales = nullptr;
    bool use_streamed_data = false;
    const LocalVector<Gaussian> *cached_streamed_gaussians = nullptr;
    const HashMap<uint32_t, uint32_t> *cached_streamed_index_lookup = nullptr;
    LocalVector<PackedGaussian> *gpu_gaussian_cache = nullptr;
    RID *gpu_gaussian_cache_buffer = nullptr;
    uint32_t *gpu_gaussian_cache_start = nullptr;
    uint32_t *gpu_gaussian_cache_count = nullptr;
    uint64_t *gpu_gaussian_cache_frame = nullptr;
    bool *gpu_gaussian_cache_valid = nullptr;
    uint64_t frame_counter = 0;
    RenderingDevice *fallback_device = nullptr;
    const LocalVector<uint32_t> *culled_indices = nullptr;
    LocalVector<float> *culled_distances_sq = nullptr;
    float *sort_input_build_time_ms = nullptr;
};

// Parameters for a sort operation
struct SortOperationParams {
    uint32_t element_count = 0;
    RID keys_buffer;
    // Values buffer contains indices sorted by key. When instance pipeline is enabled,
    // values are indices into SplatRefBuffer; otherwise they refer to Gaussian indices.
    RID values_buffer;
    // Optional count buffer (IndirectDispatchLayout with element_count at offset 12).
    // If the sorter supports indirect dispatch, the GPU reads the effective count from
    // this buffer. Otherwise element_count must carry the CPU-known effective count so
    // the pipeline can fall back to the direct-count path.
    RID count_buffer;
    bool async = true;
};

enum class SortOperationErrorCode : uint8_t {
    NONE = 0,
    SORTER_NOT_INITIALIZED,
    INVALID_KEYS_BUFFER,
    INVALID_VALUES_BUFFER,
    INVALID_COUNT_BUFFER,
    INVALID_ELEMENT_COUNT,
    ELEMENT_COUNT_EXCEEDS_CAPACITY,
    UNSUPPORTED_INDIRECT,
    UNSUPPORTED_KEY_FORMAT,
    RESOURCE_DEVICE_UNAVAILABLE,
    SUBMISSION_DEVICE_UNAVAILABLE,
    SORT_SUBMISSION_FAILED,
};

enum class SortRendererFallbackPolicy : uint8_t {
    NONE = 0,
    RETRY_WITH_EXISTING_SORTER,
    USE_SORT_FAILURE_FALLBACK,
};

// Result of a sort operation
struct SortOperationResult {
    bool success = false;
    uint64_t timeline_value = 0;
    float gpu_time_ms = 0.0f;
    String error;
    SortOperationErrorCode error_code = SortOperationErrorCode::NONE;
    SortRendererFallbackPolicy fallback_policy = SortRendererFallbackPolicy::NONE;
};

// Pure abstract interface for GPU sorting pipeline management
// Handles sorter lifecycle, buffer allocation, and depth computation resources
class IGPUSortingPipeline {
public:
    virtual ~IGPUSortingPipeline() = default;

    // Lifecycle
    virtual Error initialize(RenderingDevice *p_device, uint32_t p_initial_capacity) = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;
    virtual bool is_ready() const = 0;

    // Sorter management
    virtual void rebuild_sorter(uint32_t p_capacity) = 0;
    virtual void mark_sorter_dirty() = 0;
    virtual String get_algorithm_name() const = 0;
    virtual uint32_t get_max_elements() const = 0;

    // Buffer management
    virtual void ensure_buffers(uint32_t p_required_elements) = 0;
    virtual void release_buffers() = 0;
    virtual SortBufferHandles get_buffer_handles() const = 0;

    // Depth compute resources
    virtual void ensure_depth_resources(RenderingDevice *p_device) = 0;
    virtual RID get_depth_compute_shader() const = 0;
    virtual RID get_depth_compute_pipeline() const = 0;
    virtual void queue_depth_submission(RenderingDevice *p_device, bool p_requires_wait) = 0;
    virtual RenderingDevice *get_depth_submission_device() const = 0;
    virtual RenderingDevice *get_sort_resource_device() const = 0;
    virtual bool populate_gpu_positions(RID p_buffer, uint32_t p_total_gaussians, uint32_t p_visible_splats,
            const Transform3D &p_world_to_camera_transform, float *r_position_ptr, bool p_write_distances,
            SortPositionInputs &p_inputs) = 0;

    // Sort execution
    virtual SortOperationResult sort(const SortOperationParams &p_params) = 0;
    virtual SortOperationResult sort_async(const SortOperationParams &p_params) = 0;
    virtual void wait_for_completion() = 0;
    virtual float get_last_sort_time_ms() const = 0;

    // Metrics
    virtual SortingMetrics get_metrics() const = 0;

    // State queries
    virtual bool is_sorting_in_progress() const = 0;
    virtual uint64_t get_current_timeline_value() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

#endif // GS_GPU_SORTING_PIPELINE_INTERFACES_H

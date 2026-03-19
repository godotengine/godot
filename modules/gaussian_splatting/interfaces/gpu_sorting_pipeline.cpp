// Phase 10: GPU sorting pipeline extracted from GaussianSplatRenderer
// Manages GPU sorter lifecycle, sort buffers, and depth compute resources

#include "gpu_sorting_pipeline.h"
#include "../core/gs_project_settings.h"
#include "sync_policy.h"
#include "../logger/gs_logger.h"
#include "../logger/gs_debug_trace.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/quantization_config.h"
#include "../renderer/gpu_sorting_config.h"
#include "../renderer/sorting_contract.h"
#include "../renderer/pipeline_io_contracts.h"
#include "../compute/compute_infrastructure.h"
#include "../interfaces/gpu_culler.h"
#include "../compute/depth_compute.glsl.gen.h"
#include "../compute/instance_count_clamp.glsl.gen.h"
#include "../compute/instance_chunk_dispatch.glsl.gen.h"
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <new>

static String _inject_sort_pad_depth_define(const String &p_source) {
    const String define_line = "#define GS_SORT_PAD_DEPTH_VALUE " +
            String::num_scientific(GaussianSplatting::kSortPadDepth) + "\n";

    int version_pos = p_source.find("#version");
    if (version_pos == -1) {
        return define_line + p_source;
    }

    int newline_pos = p_source.find("\n", version_pos);
    if (newline_pos == -1) {
        return p_source + "\n" + define_line;
    }

    String before = p_source.substr(0, newline_pos + 1);
    String after = p_source.substr(newline_pos + 1, p_source.length() - newline_pos - 1);
    return before + define_line + after;
}

static String _inject_quantization_define(const String &p_source, bool p_enabled) {
    if (!p_enabled) {
        return p_source;
    }
    const String define_line = "#define USE_QUANTIZED_GAUSSIANS 1\n";

    int version_pos = p_source.find("#version");
    if (version_pos == -1) {
        return define_line + p_source;
    }

    int newline_pos = p_source.find("\n", version_pos);
    if (newline_pos == -1) {
        return p_source + "\n" + define_line;
    }

    String before = p_source.substr(0, newline_pos + 1);
    String after = p_source.substr(newline_pos + 1, p_source.length() - newline_pos - 1);
    return before + define_line + after;
}

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}

static float _get_float_setting(ProjectSettings *ps, const StringName &name, float fallback) {
    return gs::settings::get_float(ps, name, fallback);
}

static String _get_remap_compute_source() {
    return vformat(
            "#version 450\n"
            "layout(local_size_x = %d, local_size_y = 1, local_size_z = 1) in;\n"
            "\n"
            "layout(set = 0, binding = 0, std430) restrict readonly buffer SortedLocalIndices {\n"
            "    uint indices[];\n"
            "} sorted_local_indices;\n"
            "\n"
            "layout(set = 0, binding = 1, std430) restrict readonly buffer VisibleIndices {\n"
            "    uint indices[];\n"
            "} visible_indices;\n"
            "\n"
            "layout(set = 0, binding = 2, std430) restrict writeonly buffer OutputIndices {\n"
            "    uint indices[];\n"
            "} output_indices;\n"
            "\n"
            "layout(push_constant, std430) uniform Params {\n"
            "    uint count;\n"
            "    uint visible_count;\n"
            "    uint pad0;\n"
            "    uint pad1;\n"
            "} params;\n"
            "\n"
            "void main() {\n"
            "    uint idx = gl_GlobalInvocationID.x;\n"
            "    if (idx >= params.count) {\n"
            "        return;\n"
            "    }\n"
            "    uint local = sorted_local_indices.indices[idx];\n"
            "    if (local >= params.visible_count) {\n"
            "        local = params.visible_count > 0 ? (params.visible_count - 1) : 0;\n"
            "    }\n"
            "    output_indices.indices[idx] = visible_indices.indices[local];\n"
            "}\n",
            GaussianSplatting::kSortWorkgroupSize);
}

static uint32_t _get_sort_key_stride_bytes() {
    SortKeyConfig key_config = SortKeyConfig::from_settings();
    return (key_config.key_bits > 32) ? sizeof(uint32_t) * 2u : sizeof(uint32_t);
}

static SortOperationErrorCode _map_preflight_error(SortPreflightError p_error) {
    switch (p_error) {
        case SortPreflightError::NONE:
            return SortOperationErrorCode::NONE;
        case SortPreflightError::INVALID_KEYS_BUFFER:
            return SortOperationErrorCode::INVALID_KEYS_BUFFER;
        case SortPreflightError::INVALID_VALUES_BUFFER:
            return SortOperationErrorCode::INVALID_VALUES_BUFFER;
        case SortPreflightError::INVALID_COUNT_BUFFER:
            return SortOperationErrorCode::INVALID_COUNT_BUFFER;
        case SortPreflightError::INVALID_ELEMENT_COUNT:
            return SortOperationErrorCode::INVALID_ELEMENT_COUNT;
        case SortPreflightError::ELEMENT_COUNT_EXCEEDS_CAPACITY:
            return SortOperationErrorCode::ELEMENT_COUNT_EXCEEDS_CAPACITY;
        case SortPreflightError::UNSUPPORTED_KEY_FORMAT:
            return SortOperationErrorCode::UNSUPPORTED_KEY_FORMAT;
        case SortPreflightError::RESOURCE_DEVICE_UNAVAILABLE:
            return SortOperationErrorCode::RESOURCE_DEVICE_UNAVAILABLE;
        case SortPreflightError::SUBMISSION_DEVICE_UNAVAILABLE:
            return SortOperationErrorCode::SUBMISSION_DEVICE_UNAVAILABLE;
        default:
            return SortOperationErrorCode::SORT_SUBMISSION_FAILED;
    }
}

static SortRendererFallbackPolicy _fallback_policy_for_error(SortOperationErrorCode p_error) {
    switch (p_error) {
        case SortOperationErrorCode::SORTER_NOT_INITIALIZED:
        case SortOperationErrorCode::SORT_SUBMISSION_FAILED:
            return SortRendererFallbackPolicy::RETRY_WITH_EXISTING_SORTER;
        case SortOperationErrorCode::INVALID_KEYS_BUFFER:
        case SortOperationErrorCode::INVALID_VALUES_BUFFER:
        case SortOperationErrorCode::INVALID_COUNT_BUFFER:
        case SortOperationErrorCode::INVALID_ELEMENT_COUNT:
        case SortOperationErrorCode::ELEMENT_COUNT_EXCEEDS_CAPACITY:
        case SortOperationErrorCode::UNSUPPORTED_INDIRECT:
        case SortOperationErrorCode::UNSUPPORTED_KEY_FORMAT:
        case SortOperationErrorCode::RESOURCE_DEVICE_UNAVAILABLE:
        case SortOperationErrorCode::SUBMISSION_DEVICE_UNAVAILABLE:
            return SortRendererFallbackPolicy::USE_SORT_FAILURE_FALLBACK;
        case SortOperationErrorCode::NONE:
        default:
            return SortRendererFallbackPolicy::NONE;
    }
}

static const char *_sort_algorithm_label(GPUSorterFactory::SortingAlgorithm p_algorithm) {
    switch (p_algorithm) {
        case GPUSorterFactory::ALGORITHM_RADIX:
            return "radix";
        case GPUSorterFactory::ALGORITHM_BITONIC:
            return "bitonic";
        case GPUSorterFactory::ALGORITHM_ONESWEEP:
            return "onesweep";
        case GPUSorterFactory::ALGORITHM_AUTO:
        default:
            return "auto";
	}
}

static uint64_t _get_sort_buffer_byte_limit() {
	// RenderingDevice::storage_buffer_create uses uint32 byte sizes, while Vector::resize takes int.
	return MIN(uint64_t(std::numeric_limits<uint32_t>::max()), uint64_t(std::numeric_limits<int>::max()));
}

static bool _resize_sort_byte_vectors(Vector<uint8_t> &r_key_bytes, Vector<uint8_t> &r_index_bytes, uint32_t p_capacity, uint32_t p_key_stride_bytes, const char *p_context) {
	const uint64_t key_bytes_u64 = uint64_t(p_capacity) * uint64_t(p_key_stride_bytes);
	const uint64_t index_bytes_u64 = uint64_t(p_capacity) * uint64_t(sizeof(uint32_t));
	const uint64_t byte_limit = _get_sort_buffer_byte_limit();
	if (key_bytes_u64 > byte_limit || index_bytes_u64 > byte_limit) {
		GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] %s requested oversized CPU sort buffers (capacity=%s key_stride=%s key_bytes=%s index_bytes=%s limit=%s)",
				String(p_context),
				String::num_uint64(p_capacity),
				String::num_uint64(p_key_stride_bytes),
				String::num_uint64(key_bytes_u64),
				String::num_uint64(index_bytes_u64),
				String::num_uint64(byte_limit)));
		return false;
	}

	const int key_bytes = int(key_bytes_u64);
	const int index_bytes = int(index_bytes_u64);
	if (r_key_bytes.size() != key_bytes) {
		r_key_bytes.resize(key_bytes);
	}
	if (r_index_bytes.size() != index_bytes) {
		r_index_bytes.resize(index_bytes);
	}
	return true;
}

static String _format_stage_failure_with_fallback(const GaussianSplatting::ComputeInfrastructure::StageResult &p_result,
		bool p_allow_cpu_fallback, bool p_allow_retry) {
	GaussianSplatting::ComputeInfrastructure::CapabilityGatePolicy fallback_policy;
	fallback_policy.allow_cpu_fallback = p_allow_cpu_fallback;
	fallback_policy.allow_retry = p_allow_retry;
	GaussianSplatting::ComputeInfrastructure::FallbackDecision fallback =
			GaussianSplatting::ComputeInfrastructure::resolve_fallback(p_result, fallback_policy);
	return GaussianSplatting::ComputeInfrastructure::format_stage_error("GPUSortingPipeline", p_result) +
			vformat(" fallback=%s",
					String(GaussianSplatting::ComputeInfrastructure::fallback_route_name(fallback.route)));
}
	
void GPUSortingPipeline::_bind_methods() {
    // Bind methods for script access if needed
}

GPUSortingPipeline::GPUSortingPipeline() {
}

GPUSortingPipeline::~GPUSortingPipeline() {
    shutdown();
}

Error GPUSortingPipeline::initialize(RenderingDevice *p_device, uint32_t p_initial_capacity) {
    if (!p_device) {
        return ERR_INVALID_PARAMETER;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult capability =
            GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(
                    p_device, "GPUSort.CapabilityGate");
    if (!capability.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(capability, true, false);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return capability.to_error();
    }

    rd = p_device;
    initialized = true;
    last_compute_error = String();

    if (p_initial_capacity > 0) {
        rebuild_sorter(p_initial_capacity);
    }

    return OK;
}

void GPUSortingPipeline::shutdown() {
    release_buffers();

    if (gpu_sorter.is_valid()) {
        gpu_sorter->shutdown();
        gpu_sorter.unref();
    }

    sorter_needs_rebuild = true;
    sorting_in_progress = false;
    current_sort_timeline_value = 0;
    last_sort_submission_value = 0;

    initialized = false;
    rd = nullptr;
    pending_renderer = nullptr;
    sort_readback_state.pending = false;
    sort_readback_state.generation++;
    instance_count_readback_state.pending = false;
    instance_count_readback_state.generation++;
    instance_count_readback_state.pending_frame_counter = 0;
    instance_count_readback_state.bootstrap_sync_attempted = false;
    last_instance_visible_splat_count = 0;
    last_instance_visible_splat_count_valid = false;
    last_instance_visible_splat_count_frame = 0;
    last_compute_error = String();
}

void GPUSortingPipeline::set_device_manager(Ref<RenderDeviceManager> p_device_manager) {
    if (device_manager == p_device_manager) {
        return;
    }

    device_manager = p_device_manager;
    if (!device_manager.is_valid()) {
        return;
    }

    RenderingDevice *sort_device = sort_resource_device ? sort_resource_device : rd;
    RenderingDevice *remap_device = remap_resource_device ? remap_resource_device : sort_device;
    RenderingDevice *instance_device = instance_resource_device ? instance_resource_device : sort_device;
    RenderingDevice *instance_count_device = instance_count_resource_device ? instance_count_resource_device : sort_device;
    RenderingDevice *instance_chunk_device = instance_chunk_dispatch_resource_device ? instance_chunk_dispatch_resource_device : sort_device;

    auto track_with_owner = [&](const RID &p_rid, RenderingDevice *p_owner, bool p_owned, const char *p_label) {
        if (!p_rid.is_valid()) {
            return;
        }
        RenderingDevice *owner = p_owner ? p_owner : rd;
        if (!owner) {
            return;
        }
        _track_resource(p_rid, owner, p_owned, p_label);
    };

    track_with_owner(sort_keys_buffer, sort_device, !sort_keys_external,
            sort_keys_external ? "sort_keys_external" : "sort_keys_buffer");
    track_with_owner(sort_indices_buffer, sort_device, !sort_indices_external,
            sort_indices_external ? "sort_indices_external" : "sort_indices_buffer");

    track_with_owner(culled_position_buffer, sort_device, true, "culled_position_buffer");
    track_with_owner(local_depth_keys_buffer, sort_device, true, "local_depth_keys_buffer");
    track_with_owner(local_splat_indices_buffer, sort_device, true, "local_splat_indices_buffer");
    track_with_owner(local_visible_indices_buffer, sort_device, true, "local_visible_indices_buffer");

    track_with_owner(depth_uniform_set, sort_device, true, "depth_uniform_set");
    track_with_owner(depth_compute_pipeline, sort_device, true, "depth_compute_pipeline");
    track_with_owner(depth_compute_shader, sort_device, true, "depth_compute_shader");

    track_with_owner(gather_uniform_set, sort_device, true, "gather_uniform_set");
    track_with_owner(gather_positions_pipeline, sort_device, true, "gather_positions_pipeline");
    track_with_owner(gather_positions_shader, sort_device, true, "gather_positions_shader");

    track_with_owner(remap_uniform_set, remap_device, true, "remap_uniform_set");
    track_with_owner(remap_compute_pipeline, remap_device, true, "remap_compute_pipeline");
    track_with_owner(remap_compute_shader, remap_device, true, "remap_compute_shader");

    track_with_owner(instance_param_buffer, instance_device, true, "instance_param_buffer");
    track_with_owner(instance_count_uniform_set, instance_count_device, true, "instance_count_uniform_set");
    track_with_owner(instance_count_clamp_pipeline, instance_count_device, true, "instance_count_clamp_pipeline");
    track_with_owner(instance_count_clamp_shader, instance_count_device, true, "instance_count_clamp_shader");
    track_with_owner(instance_chunk_dispatch_uniform_set, instance_chunk_device, true, "instance_chunk_dispatch_uniform_set");
    track_with_owner(instance_chunk_dispatch_pipeline, instance_chunk_device, true, "instance_chunk_dispatch_pipeline");
    track_with_owner(instance_chunk_dispatch_shader, instance_chunk_device, true, "instance_chunk_dispatch_shader");
}

void GPUSortingPipeline::set_forced_sort_algorithm(GPUSorterFactory::SortingAlgorithm p_algorithm) {
    if (forced_sort_algorithm == p_algorithm) {
        return;
    }
    forced_sort_algorithm = p_algorithm;
    sorter_needs_rebuild = true;
    GS_LOG_GPU_SORT_INFO(vformat("[GPUSortingPipeline] Runtime sort algorithm override: %s",
            _sort_algorithm_label(forced_sort_algorithm)));
}

// Helper methods
void GPUSortingPipeline::_track_resource(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label) {
    if (device_manager.is_valid()) {
        device_manager->track_resource(p_rid, p_device, p_owned, p_label);
    }
}

void GPUSortingPipeline::_forget_resource(const RID &p_rid) {
    if (device_manager.is_valid()) {
        device_manager->forget_resource(p_rid);
    }
}

void GPUSortingPipeline::_free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid, bool p_is_auto_free) {
    if (!p_rid.is_valid()) {
        return;
    }
    (void)p_is_auto_free;
    ERR_FAIL_COND_MSG(!device_manager.is_valid(),
            vformat("[GPUSortingPipeline] Cannot free owned RID %s without RenderDeviceManager",
                    String::num_uint64(p_rid.get_id())));
    device_manager->free_owned_resource(p_fallback_device, p_rid);
}

void GPUSortingPipeline::rebuild_sorter(uint32_t p_capacity) {
    if (!rd) {
        sorter_needs_rebuild = false;
        return;
    }

    if (!sorter_needs_rebuild && gpu_sorter.is_valid() && gpu_sorter->get_max_elements() >= p_capacity) {
        return;
    }

    uint32_t capacity = MAX(p_capacity, (uint32_t)1);

    if (gpu_sorter.is_valid()) {
        gpu_sorter->shutdown();
        gpu_sorter.unref();
    }

    SortKeyConfig sort_key_config = SortKeyConfig::from_settings();
    sort_key_config.key_bits = 32;
    sort_key_config.tile_bits = 0;
    sort_key_config.depth_bits = sort_key_config.key_bits;
    sort_key_config.enable_tie_breaker = false;

    Ref<IGPUSorter> new_sorter = GPUSorterFactory::create_sorter(forced_sort_algorithm, rd, capacity, sort_key_config);
    if (!new_sorter.is_valid()) {
        GS_LOG_ERROR_DEFAULT(vformat("[GPUSortingPipeline] Failed to create GPU sorter (requested=%s)",
                _sort_algorithm_label(forced_sort_algorithm)));
        sorter_needs_rebuild = false;
        return;
    }

    gpu_sorter = new_sorter;
    sorter_needs_rebuild = false;

    if (manage_buffers) {
        // Reallocate buffers to match new sorter capacity
        release_buffers();
        ensure_buffers(gpu_sorter->get_max_elements());
    }

    GS_LOG_GPU_SORT_DEBUG(vformat("[GPUSortingPipeline] Sorter rebuilt: algorithm=%s requested=%s capacity=%d",
            gpu_sorter->get_algorithm_name(),
            _sort_algorithm_label(forced_sort_algorithm),
            capacity));
}

Ref<IGPUSorter> GPUSortingPipeline::rebuild_sorter_if_needed(RenderingDevice *p_device, uint32_t p_capacity, bool p_needs_rebuild) {
    if (!p_device) {
        sorter_needs_rebuild = false;
        return gpu_sorter;
    }

    uint32_t capacity = MAX(p_capacity, 1u);
    if (!initialized || rd != p_device) {
        initialize(p_device, capacity);
    }

    if (p_needs_rebuild) {
        sorter_needs_rebuild = true;
    }

    if (!sorter_needs_rebuild && gpu_sorter.is_valid() && gpu_sorter->get_max_elements() >= capacity) {
        return gpu_sorter;
    }

    rebuild_sorter(capacity);
    return gpu_sorter;
}

String GPUSortingPipeline::get_algorithm_name() const {
    if (gpu_sorter.is_valid()) {
        return gpu_sorter->get_algorithm_name();
    }
    return String();
}

uint32_t GPUSortingPipeline::get_max_elements() const {
    if (gpu_sorter.is_valid()) {
        return gpu_sorter->get_max_elements();
    }
    return 0;
}

void GPUSortingPipeline::release_buffers() {
    RenderingDevice *resource_device = sort_resource_device ? sort_resource_device : rd;

    if (sort_keys_buffer.is_valid()) {
        if (!sort_keys_external) {
            _free_owned_resource(resource_device, sort_keys_buffer);
        } else {
            _forget_resource(sort_keys_buffer);
            sort_keys_buffer = RID();
        }
    }
    if (sort_indices_buffer.is_valid()) {
        if (!sort_indices_external) {
            _free_owned_resource(resource_device, sort_indices_buffer);
        } else {
            _forget_resource(sort_indices_buffer);
            sort_indices_buffer = RID();
        }
    }

    _free_owned_resource(resource_device, culled_position_buffer);
    _free_owned_resource(resource_device, depth_uniform_set, true);
    _free_owned_resource(resource_device, depth_compute_pipeline, true);
    _free_owned_resource(resource_device, depth_compute_shader);
    _free_owned_resource(instance_count_resource_device ? instance_count_resource_device : resource_device, instance_count_uniform_set, true);
    _free_owned_resource(instance_count_resource_device ? instance_count_resource_device : resource_device, instance_count_clamp_pipeline, true);
    _free_owned_resource(instance_count_resource_device ? instance_count_resource_device : resource_device, instance_count_clamp_shader);
    _free_owned_resource(instance_chunk_dispatch_resource_device ? instance_chunk_dispatch_resource_device : resource_device, instance_chunk_dispatch_uniform_set, true);
    _free_owned_resource(instance_chunk_dispatch_resource_device ? instance_chunk_dispatch_resource_device : resource_device, instance_chunk_dispatch_pipeline, true);
    _free_owned_resource(instance_chunk_dispatch_resource_device ? instance_chunk_dispatch_resource_device : resource_device, instance_chunk_dispatch_shader);
    _free_owned_resource(instance_resource_device ? instance_resource_device : resource_device, instance_param_buffer);
    _free_owned_resource(resource_device, gather_positions_pipeline, true);
    _free_owned_resource(resource_device, gather_positions_shader);
    _free_owned_resource(resource_device, gather_uniform_set, true);
    _free_owned_resource(resource_device, local_depth_keys_buffer);
    _free_owned_resource(resource_device, local_splat_indices_buffer);
    _free_owned_resource(resource_device, local_visible_indices_buffer);
    _free_owned_resource(remap_resource_device, remap_uniform_set, true);
    _free_owned_resource(remap_resource_device, remap_compute_pipeline, true);
    _free_owned_resource(remap_resource_device, remap_compute_shader);

    sort_buffer_capacity = 0;
    local_sort_buffer_capacity = 0;
    culled_position_capacity = 0;
    sort_key_bytes.clear();
    sort_index_bytes.clear();
    culled_position_bytes.clear();
    instance_param_cache.clear();
    instance_param_cache_valid = false;
    sort_keys_external = false;
    sort_indices_external = false;

    sort_resource_device = nullptr;
    remap_resource_device = nullptr;
    instance_resource_device = nullptr;
    instance_count_resource_device = nullptr;
    instance_chunk_dispatch_resource_device = nullptr;
    instance_count_readback_state.pending = false;
    instance_count_readback_state.generation++;
    instance_count_readback_state.pending_frame_counter = 0;
    instance_count_readback_state.bootstrap_sync_attempted = false;
}

void GPUSortingPipeline::ensure_buffers(uint32_t p_required_elements) {
    if (!manage_buffers) {
        return;
    }
    if (!rd) {
        return;
    }

    // Pad to power of two
    uint32_t padded_required = 1;
    if (p_required_elements > 0) {
        padded_required = 1;
        while (padded_required < p_required_elements) {
            padded_required <<= 1;
            if (padded_required == 0) {
                padded_required = p_required_elements;
                break;
            }
        }
    }

    uint32_t sorter_capacity = 0;
    if (gpu_sorter.is_valid() && gpu_sorter->get_max_elements() > 0) {
        sorter_capacity = gpu_sorter->get_max_elements();
    }

    uint32_t max_elements = padded_required;
    if (sorter_capacity > 0) {
        max_elements = MIN(padded_required, sorter_capacity);
    }

    if (max_elements == 0) {
        max_elements = padded_required > 0 ? padded_required : 1;
    }

    uint32_t desired_capacity = MIN(MAX(padded_required, (uint32_t)1), max_elements);
    if (desired_capacity == 0) {
        desired_capacity = 1;
    }

    // Handle external buffer transitions
    if (sort_keys_external) {
        _forget_resource(sort_keys_buffer);
        sort_keys_buffer = RID();
        sort_keys_external = false;
    }
    if (sort_indices_external) {
        _forget_resource(sort_indices_buffer);
        sort_indices_buffer = RID();
        sort_indices_external = false;
    }

    // Check if existing buffers are sufficient
	if (sort_keys_buffer.is_valid() && sort_indices_buffer.is_valid() && sort_buffer_capacity >= desired_capacity) {
		uint32_t target_capacity = sort_buffer_capacity;
		uint32_t key_stride_bytes = _get_sort_key_stride_bytes();
		if (!_resize_sort_byte_vectors(sort_key_bytes, sort_index_bytes, target_capacity, key_stride_bytes, "ensure_buffers(reuse)")) {
			GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Rejecting unsafe sort buffer resize while reusing existing buffers");
			release_buffers();
			return;
		}
		return;
	}

    // Free old buffers
    if (sort_keys_buffer.is_valid()) {
        _free_owned_resource(nullptr, sort_keys_buffer);
    }
    if (sort_indices_buffer.is_valid()) {
        _free_owned_resource(nullptr, sort_indices_buffer);
    }

	// Allocate new buffers
	sort_buffer_capacity = desired_capacity;
	uint32_t key_stride_bytes = _get_sort_key_stride_bytes();
	if (!_resize_sort_byte_vectors(sort_key_bytes, sort_index_bytes, sort_buffer_capacity, key_stride_bytes, "ensure_buffers(allocate)")) {
		GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Rejecting unsafe sort buffer allocation request");
		release_buffers();
		return;
	}

    // Initialize sort_index_bytes with identity indices [0, 1, 2, ..., N-1]
    // This ensures valid indices before any sort completes
    uint32_t *indices = reinterpret_cast<uint32_t *>(sort_index_bytes.ptrw());
    for (uint32_t i = 0; i < sort_buffer_capacity; i++) {
        indices[i] = i;
    }

    sort_resource_device = rd;
    sort_keys_buffer = rd->storage_buffer_create(sort_key_bytes.size(), sort_key_bytes);
    rd->set_resource_name(sort_keys_buffer, "GS_GPUSortingPipeline_SortKeysBuffer");
    _track_resource(sort_keys_buffer, rd, true, "sort_keys_buffer");
    sort_indices_buffer = rd->storage_buffer_create(sort_index_bytes.size(), sort_index_bytes);
    rd->set_resource_name(sort_indices_buffer, "GS_GPUSortingPipeline_SortIndicesBuffer");
    _track_resource(sort_indices_buffer, rd, true, "sort_indices_buffer");

    if (!sort_keys_buffer.is_valid() || !sort_indices_buffer.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Failed to allocate sorting buffers");
        release_buffers();
    }
}

void GPUSortingPipeline::_apply_sorted_results(GaussianSplatRenderer &p_renderer, const Vector<uint8_t> &p_sorted_index_bytes) {
    if (sort_readback_state.expected_count == 0) {
        return;
    }

    const uint32_t available_splats = sort_readback_state.expected_count;
    const uint32_t expected_bytes = available_splats * sizeof(uint32_t);
    if (static_cast<uint32_t>(p_sorted_index_bytes.size()) < expected_bytes) {
        GS_LOG_WARN_DEFAULT("[GPU Sort] Async readback returned insufficient sorted indices");
        return;
    }

    const uint32_t *sorted_order = reinterpret_cast<const uint32_t *>(p_sorted_index_bytes.ptr());
    auto &subsystem_state = p_renderer.get_subsystem_state();
    auto &sorting_state = p_renderer.get_sorting_state();
    auto &device_state = p_renderer.get_device_state();
    auto &frame_state = p_renderer.get_frame_state();
    auto &performance_state = p_renderer.get_performance_state();

    // BUF-3 optimization: Read from snapshot indices instead of copied arrays.
    // The sorted_order tells us for each output position which input position to read from.
    // We only snapshotted the indices; distances/importance are not needed for sorting.
    const uint32_t snapshot_count = static_cast<uint32_t>(sort_readback_state.snapshot_indices.size());

    // Prepare output arrays in culler state
    GPUCuller::CullingState &cull_state = subsystem_state.gpu_culler->get_state();
    if (static_cast<uint32_t>(cull_state.culled_indices.size()) != available_splats) {
        cull_state.culled_indices.resize(available_splats);
    }

    // BUF-3: Only reorder the indices using the sorted permutation.
    // Distances and importance weights are not used by the rendering path after sorting,
    // so we skip copying them entirely. If they're needed later, they can be recomputed.
    for (uint32_t i = 0; i < available_splats; i++) {
        uint32_t src = sorted_order[i];
        if (src >= snapshot_count) {
            src = snapshot_count > 0 ? (snapshot_count - 1) : 0;
        }
        cull_state.culled_indices[i] = sort_readback_state.snapshot_indices[src];
    }

    // Copy sorted indices to GPU buffer
    sorting_state.sort_index_bytes.resize(expected_bytes);
    uint32_t *final_indices = reinterpret_cast<uint32_t *>(sorting_state.sort_index_bytes.ptrw());
    for (uint32_t i = 0; i < available_splats; i++) {
        final_indices[i] = cull_state.culled_indices[i];
    }
    if (sort_indices_buffer.is_valid() && device_state.rd && !sorting_state.sort_index_bytes.is_empty()) {
        RenderingDevice *target_device = device_manager.is_valid() ?
                device_manager->get_resource_owner(sort_indices_buffer, device_state.rd) : device_state.rd;
        target_device->buffer_update(sort_indices_buffer, 0, sorting_state.sort_index_bytes.size(), sorting_state.sort_index_bytes.ptr());
    }

    sorting_state.sorted_splat_count = available_splats;
    frame_state.visible_splat_count.store(available_splats, std::memory_order_release);
    performance_state.metrics.rendered_splat_count = frame_state.visible_splat_count.load(std::memory_order_acquire);

    // BUF-3: Clear snapshot after use
    sort_readback_state.snapshot_indices.clear();
}

void GPUSortingPipeline::_on_sort_readback(const Vector<uint8_t> &p_data, int64_t p_generation) {
    if (!pending_renderer || !sort_readback_state.pending || p_generation != (int64_t)sort_readback_state.generation) {
        return;
    }

    sort_readback_state.pending = false;
    _apply_sorted_results(*pending_renderer, p_data);
}

void GPUSortingPipeline::_on_instance_count_readback(const Vector<uint8_t> &p_data, int64_t p_generation) {
    if (!instance_count_readback_state.pending ||
            p_generation != (int64_t)instance_count_readback_state.generation) {
        return;
    }
    const uint32_t request_frame = instance_count_readback_state.pending_frame_counter;
    instance_count_readback_state.pending = false;
    instance_count_readback_state.pending_frame_counter = 0;
    if (request_frame < last_instance_visible_splat_count_frame) {
        return;
    }
    if (p_data.size() < static_cast<int>(sizeof(GaussianSplatting::IndirectDispatchLayout))) {
        return;
    }
    const auto *indirect =
            reinterpret_cast<const GaussianSplatting::IndirectDispatchLayout *>(p_data.ptr());
    last_instance_visible_splat_count = indirect->element_count;
    last_instance_visible_splat_count_valid = true;
    last_instance_visible_splat_count_frame = request_frame;
    if (GaussianSplatting::debug_trace_is_enabled()) {
        GaussianSplatting::debug_trace_record_instance_counts(
                indirect->element_count, indirect->unclamped_total, indirect->overflow_flag);
        GaussianSplatting::debug_trace_record_event("sort",
                vformat("InstancePipeline async counts: clamped=%d raw=%d overflow=%d",
                        indirect->element_count, indirect->unclamped_total, indirect->overflow_flag),
                false);
    }
}

SortBufferHandles GPUSortingPipeline::get_buffer_handles() const {
    SortBufferHandles handles;
    handles.keys_buffer = sort_keys_buffer;
    handles.indices_buffer = sort_indices_buffer;
    handles.capacity = sort_buffer_capacity;
    handles.valid = sort_keys_buffer.is_valid() && sort_indices_buffer.is_valid();
    return handles;
}

void GPUSortingPipeline::release_sort_buffers(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }

    GaussianSplatRenderer &renderer = *p_renderer;
    auto &sorting_state = renderer.get_sorting_state();

    if (is_managing_buffers()) {
        release_buffers();
    } else {
        RenderingDevice *resource_device = sort_resource_device ? sort_resource_device : rd;

        if (sort_keys_buffer.is_valid()) {
            if (sort_keys_external) {
                _forget_resource(sort_keys_buffer);
                sort_keys_buffer = RID();
            } else {
                _free_owned_resource(resource_device, sort_keys_buffer);
            }
        }
        if (sort_indices_buffer.is_valid()) {
            if (sort_indices_external) {
                _forget_resource(sort_indices_buffer);
                sort_indices_buffer = RID();
            } else {
                _free_owned_resource(resource_device, sort_indices_buffer);
            }
        }

        _free_owned_resource(resource_device, culled_position_buffer);
        _free_owned_resource(resource_device, local_depth_keys_buffer);
        _free_owned_resource(resource_device, local_splat_indices_buffer);
        _free_owned_resource(resource_device, local_visible_indices_buffer);
        _free_owned_resource(resource_device, depth_uniform_set, true);
        _free_owned_resource(resource_device, depth_compute_pipeline, true);
        _free_owned_resource(resource_device, depth_compute_shader);
        _free_owned_resource(resource_device, gather_uniform_set, true);
        _free_owned_resource(resource_device, gather_positions_pipeline, true);
        _free_owned_resource(resource_device, gather_positions_shader);
        _free_owned_resource(remap_resource_device, remap_uniform_set, true);
        _free_owned_resource(remap_resource_device, remap_compute_pipeline, true);
        _free_owned_resource(remap_resource_device, remap_compute_shader);

        sort_keys_buffer = RID();
        sort_indices_buffer = RID();
        culled_position_buffer = RID();
        local_depth_keys_buffer = RID();
        local_splat_indices_buffer = RID();
        local_visible_indices_buffer = RID();
        depth_uniform_set = RID();
        depth_compute_pipeline = RID();
        depth_compute_shader = RID();
        gather_uniform_set = RID();
        gather_positions_pipeline = RID();
        gather_positions_shader = RID();
        remap_uniform_set = RID();
        remap_compute_pipeline = RID();
        remap_compute_shader = RID();
        sort_buffer_capacity = 0;
        culled_position_capacity = 0;
        local_sort_buffer_capacity = 0;
        sort_key_bytes.clear();
        sort_index_bytes.clear();
        sort_keys_external = false;
        sort_indices_external = false;

        sort_resource_device = nullptr;
        remap_resource_device = nullptr;
    }

    sorting_state.sort_buffer_capacity = 0;
    sorting_state.local_sort_buffer_capacity = 0;
    sorting_state.culled_position_capacity = 0;
    sorting_state.sort_key_bytes.clear();
    sorting_state.sort_index_bytes.clear();
    sorting_state.culled_position_bytes.clear();
    sorting_state.sort_keys_external = false;
    sorting_state.sort_indices_external = false;
    sorting_state.sort_buffers_pipeline_managed = false;
}

void GPUSortingPipeline::set_external_sort_indices(RID p_buffer, RenderingDevice *p_device) {
    RenderingDevice *new_owner = p_device;
    if (p_buffer.is_valid()) {
        if (device_manager.is_valid()) {
            new_owner = device_manager->get_resource_owner(p_buffer, new_owner ? new_owner : rd);
        } else if (!new_owner) {
            new_owner = rd;
        }
        ERR_FAIL_NULL_MSG(new_owner,
                vformat("[GPUSortingPipeline] Missing owner device for external sort index buffer RID %s",
                        String::num_uint64(p_buffer.get_id())));
    }

    if (sort_indices_buffer.is_valid() && sort_indices_buffer == p_buffer && !sort_indices_external) {
        // Keep owned semantics when caller reuses a pipeline-owned buffer RID.
        return;
    }

    if (sort_indices_buffer.is_valid() && sort_indices_buffer != p_buffer) {
        if (sort_indices_external) {
            _forget_resource(sort_indices_buffer);
        } else {
            RenderingDevice *fallback_device = sort_resource_device ? sort_resource_device : (p_device ? p_device : rd);
            _free_owned_resource(fallback_device, sort_indices_buffer);
        }
    }

    sort_indices_buffer = p_buffer;
    sort_indices_external = p_buffer.is_valid();
    if (sort_indices_external) {
        sort_resource_device = new_owner;
        _track_resource(p_buffer, new_owner, false, "external_sort_indices");
    }
}

void GPUSortingPipeline::clear_external_sort_indices() {
    if (sort_indices_external && sort_indices_buffer.is_valid()) {
        _forget_resource(sort_indices_buffer);
    }
    sort_indices_buffer = RID();
    sort_indices_external = false;
}

void GPUSortingPipeline::set_instance_pipeline_inputs(const InstancePipelineInputs &p_inputs) {
    instance_inputs = p_inputs;
    instance_inputs_valid = true;
}

void GPUSortingPipeline::clear_instance_pipeline_inputs() {
    instance_inputs = InstancePipelineInputs();
    instance_inputs_valid = false;
    last_instance_visible_splat_count = 0;
    last_instance_visible_splat_count_valid = false;
    last_instance_visible_splat_count_frame = 0;
    instance_count_readback_state.pending = false;
    instance_count_readback_state.generation++;
    instance_count_readback_state.pending_frame_counter = 0;
    instance_count_readback_state.bootstrap_sync_attempted = false;
}

void GPUSortingPipeline::ensure_sort_buffers(GaussianSplatRenderer *p_renderer, uint32_t p_required_elements) {
    if (!p_renderer) {
        return;
    }

    GaussianSplatRenderer &renderer = *p_renderer;
    auto &sorting_state = renderer.get_sorting_state();
    auto &resource_state = renderer.get_resource_state();
    auto &device_state = renderer.get_device_state();

    if (!renderer.ensure_rendering_device("ensure_sort_buffers")) {
        return;
    }

    DEV_ASSERT(!(sorting_state.sort_keys_external && sorting_state.sort_buffers_pipeline_managed));
    DEV_ASSERT(!(sorting_state.sort_indices_external && sorting_state.sort_buffers_pipeline_managed));
    const bool pipeline_manages_buffers = is_managing_buffers();
    DEV_ASSERT(!sorting_state.sort_buffers_pipeline_managed || pipeline_manages_buffers);

    if (!rd && device_state.rd) {
        rd = device_state.rd;
    }

    GaussianSplatting::SortPaddingInfo padding = GaussianSplatting::get_sort_padding(p_required_elements);
    uint32_t padded_required = padding.padded_elements;

    uint32_t sorter_capacity = 0;
    if (sorting_state.gpu_sorter.is_valid() && sorting_state.gpu_sorter->get_max_elements() > 0) {
        sorter_capacity = sorting_state.gpu_sorter->get_max_elements();
    }

    uint32_t max_elements = padded_required;
    if (sorter_capacity > 0) {
        max_elements = MIN(padded_required, sorter_capacity);
    }

    if (resource_state.buffer_manager.is_valid() && resource_state.buffer_manager_initialized) {
        uint32_t manager_capacity = resource_state.buffer_manager->get_buffer_capacity();
        if (manager_capacity > 0) {
            if (max_elements == 0) {
                max_elements = manager_capacity;
            } else {
                max_elements = MIN(max_elements, manager_capacity);
            }
        }

        GPUBufferManager::BufferHandle manager_keys = resource_state.buffer_manager->get_sort_key_handle();
        GPUBufferManager::BufferHandle manager_indices = resource_state.buffer_manager->get_sorted_indices_handle();
        if (manager_keys.is_valid() && manager_indices.is_valid() && manager_keys.device == manager_indices.device) {
            if (sort_keys_buffer.is_valid() || sort_indices_buffer.is_valid()) {
                release_buffers();
            }

            sort_resource_device = manager_keys.device;
            sort_keys_buffer = manager_keys.buffer;
            sort_indices_buffer = manager_indices.buffer;
            sort_keys_external = true;
            sort_indices_external = true;
            _track_resource(sort_keys_buffer, sort_resource_device, false, "sort_keys_external");
            _track_resource(sort_indices_buffer, sort_resource_device, false, "sort_indices_external");

			sort_buffer_capacity = manager_capacity > 0 ? manager_capacity : padded_required;
			if (sort_buffer_capacity == 0) {
				sort_buffer_capacity = 1;
			}
			uint32_t cpu_capacity = MIN(padded_required, sort_buffer_capacity);
			uint32_t key_stride_bytes = _get_sort_key_stride_bytes();
			if (!_resize_sort_byte_vectors(sort_key_bytes, sort_index_bytes, cpu_capacity, key_stride_bytes, "ensure_sort_buffers(external/local)")) {
				GS_LOG_WARN_DEFAULT("[GPU Sort] Failed to size local sort CPU buffers for external sort handles");
				release_sort_buffers(&renderer);
				return;
			}
			if (!_resize_sort_byte_vectors(sorting_state.sort_key_bytes, sorting_state.sort_index_bytes, cpu_capacity, key_stride_bytes, "ensure_sort_buffers(external/state)")) {
				GS_LOG_WARN_DEFAULT("[GPU Sort] Failed to size renderer sort CPU buffers for external sort handles");
				release_sort_buffers(&renderer);
				return;
			}

            sorting_state.sort_keys_external = true;
            sorting_state.sort_indices_external = true;
            sorting_state.sort_buffers_pipeline_managed = false;
            sorting_state.sort_buffer_capacity = sort_buffer_capacity;
            return;
        } else {
            GS_LOG_WARN_DEFAULT("[GPU Sort] GPU buffer manager returned invalid sort buffers; using local buffers instead");
        }
    }

    if (max_elements == 0) {
        max_elements = padded_required > 0 ? padded_required : 1;
    }

    uint32_t desired_capacity = MIN(MAX(padded_required, (uint32_t)1), max_elements);
    if (desired_capacity == 0) {
        desired_capacity = 1;
    }

    if (sort_keys_external || sort_indices_external) {
        RenderingDevice *fallback_device = sort_resource_device ? sort_resource_device : rd;
        if (sort_keys_buffer.is_valid()) {
            if (sort_keys_external) {
                _forget_resource(sort_keys_buffer);
                sort_keys_buffer = RID();
            } else {
                _free_owned_resource(fallback_device, sort_keys_buffer);
            }
        }
        if (sort_indices_buffer.is_valid()) {
            if (sort_indices_external) {
                _forget_resource(sort_indices_buffer);
                sort_indices_buffer = RID();
            } else {
                _free_owned_resource(fallback_device, sort_indices_buffer);
            }
        }
        sort_keys_external = false;
        sort_indices_external = false;
    }

    if (pipeline_manages_buffers) {
        if (!is_initialized() && rd) {
            initialize(rd, desired_capacity);
        }
        ensure_buffers(desired_capacity);
    } else if (rd) {
        ensure_buffers(desired_capacity);
    }

    if (!sort_keys_buffer.is_valid() || !sort_indices_buffer.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPU Sort] Failed to allocate sorting buffers");
        release_sort_buffers(&renderer);
        return;
    }

	sorting_state.sort_keys_external = false;
	sorting_state.sort_indices_external = false;
	sorting_state.sort_buffers_pipeline_managed = pipeline_manages_buffers;
	sorting_state.sort_buffer_capacity = sort_buffer_capacity > 0 ? sort_buffer_capacity : desired_capacity;

	uint32_t cpu_capacity = MIN(padded_required, sorting_state.sort_buffer_capacity);
	uint32_t key_stride_bytes = _get_sort_key_stride_bytes();
	if (!_resize_sort_byte_vectors(sort_key_bytes, sort_index_bytes, cpu_capacity, key_stride_bytes, "ensure_sort_buffers(local/local)")) {
		GS_LOG_WARN_DEFAULT("[GPU Sort] Failed to size local sort CPU buffers");
		release_sort_buffers(&renderer);
		return;
	}
	if (!_resize_sort_byte_vectors(sorting_state.sort_key_bytes, sorting_state.sort_index_bytes, cpu_capacity, key_stride_bytes, "ensure_sort_buffers(local/state)")) {
		GS_LOG_WARN_DEFAULT("[GPU Sort] Failed to size renderer sort CPU buffers");
		release_sort_buffers(&renderer);
		return;
	}
}

void GPUSortingPipeline::ensure_depth_resources(RenderingDevice *p_device) {
    RenderingDevice *device = p_device ? p_device : rd;
    if (!device) {
        return;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult capability =
            GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(
                    device, "GPUSort.Depth.CapabilityGate");
    if (!capability.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(capability, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }

    const bool quantized_storage = g_quantization_config.per_chunk_quantization;
    if (depth_compute_shader.is_valid() && quantized_storage != depth_quantization_enabled) {
        RenderingDevice *resource_device = sort_resource_device ? sort_resource_device : device;
        _free_owned_resource(resource_device, depth_uniform_set, true);
        _free_owned_resource(resource_device, depth_compute_pipeline, true);
        _free_owned_resource(resource_device, depth_compute_shader);
        depth_uniform_set = RID();
        depth_compute_pipeline = RID();
        depth_compute_shader = RID();
    }
    depth_quantization_enabled = quantized_storage;

    if (!depth_compute_shader.is_valid()) {
        static DepthComputeShaderRD depth_shader_source;
        static std::once_flag depth_shader_once;

        std::call_once(depth_shader_once, []() {
            depth_shader_source.~DepthComputeShaderRD();
            new (&depth_shader_source) DepthComputeShaderRD();
            Vector<String> versions;
            versions.push_back("");
            depth_shader_source.initialize(versions);
        });

        RID depth_shader_version = depth_shader_source.version_create();
        if (!depth_shader_version.is_valid()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Failed to create depth compute shader version");
            return;
        }

        RS::ShaderNativeSourceCode native_source = depth_shader_source.version_get_native_source_code(depth_shader_version);
        depth_shader_source.version_free(depth_shader_version);

        if (native_source.versions.is_empty() || native_source.versions[0].stages.is_empty()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Embedded depth compute shader missing source");
            return;
        }

        String compute_source;
        const RS::ShaderNativeSourceCode::Version &version = native_source.versions[0];
        for (int i = 0; i < version.stages.size(); i++) {
            if (version.stages[i].name == "compute") {
                compute_source = version.stages[i].code;
                break;
            }
        }

        if (compute_source.is_empty()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Embedded depth compute shader missing compute stage");
            return;
        }

        compute_source = _inject_sort_pad_depth_define(compute_source);
        compute_source = _inject_quantization_define(compute_source, quantized_storage);

        GaussianSplatting::ComputeInfrastructure::StageResult compile_result =
                GaussianSplatting::ComputeInfrastructure::compile_compute_shader_from_source(
                        device, "GPUSort.Depth.ShaderCompile", compute_source,
                        "GaussianDepthCompute", depth_compute_shader);
        if (!compile_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(compile_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }

        _track_resource(depth_compute_shader, device, true, "depth_compute_shader");
        sort_resource_device = device;
    }

    if (!depth_compute_pipeline.is_valid() && depth_compute_shader.is_valid()) {
        GaussianSplatting::ComputeInfrastructure::StageResult pipeline_result =
                GaussianSplatting::ComputeInfrastructure::create_pipeline_checked(
                        device, depth_compute_shader, "GPUSort.Depth.PipelineCreate", depth_compute_pipeline);
        if (!pipeline_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(pipeline_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }
        _track_resource(depth_compute_pipeline, device, true, "depth_compute_pipeline");
        sort_resource_device = device;
    }

    last_compute_error = String();
}

void GPUSortingPipeline::_ensure_instance_param_buffer(RenderingDevice *p_device) {
    RenderingDevice *device = p_device ? p_device : rd;
    if (!device) {
        return;
    }

    if (instance_resource_device && instance_resource_device != device) {
        _free_owned_resource(instance_resource_device, instance_param_buffer);
        instance_param_buffer = RID();
        instance_resource_device = nullptr;
        instance_param_cache.clear();
        instance_param_cache_valid = false;
    }

    if (!instance_param_buffer.is_valid()) {
        Vector<uint8_t> param_bytes_init;
        param_bytes_init.resize(sizeof(InstanceDepthParamsGPU));
        instance_param_buffer = device->uniform_buffer_create(param_bytes_init.size(), param_bytes_init);
        if (instance_param_buffer.is_valid()) {
            device->set_resource_name(instance_param_buffer, "GS_InstanceDepthParamsBuffer");
            _track_resource(instance_param_buffer, device, true, "instance_param_buffer");
        }
        instance_param_cache.clear();
        instance_param_cache_valid = false;
    }

    instance_resource_device = device;
}

void GPUSortingPipeline::_ensure_instance_count_resources(RenderingDevice *p_device) {
    RenderingDevice *device = p_device ? p_device : rd;
    if (!device) {
        return;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult capability =
            GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(
                    device, "GPUSort.InstanceCount.CapabilityGate");
    if (!capability.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(capability, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }

    if (instance_count_resource_device && instance_count_resource_device != device) {
        _free_owned_resource(instance_count_resource_device, instance_count_uniform_set, true);
        _free_owned_resource(instance_count_resource_device, instance_count_clamp_pipeline, true);
        _free_owned_resource(instance_count_resource_device, instance_count_clamp_shader);
        instance_count_uniform_set = RID();
        instance_count_clamp_pipeline = RID();
        instance_count_clamp_shader = RID();
        instance_count_resource_device = nullptr;
    }

    if (!instance_count_clamp_shader.is_valid()) {
        static InstanceCountClampShaderRD clamp_shader_source;
        static std::once_flag clamp_shader_once;

        std::call_once(clamp_shader_once, []() {
            clamp_shader_source.~InstanceCountClampShaderRD();
            new (&clamp_shader_source) InstanceCountClampShaderRD();
            Vector<String> versions;
            versions.push_back("");
            clamp_shader_source.initialize(versions);
        });

        RID clamp_shader_version = clamp_shader_source.version_create();
        if (!clamp_shader_version.is_valid()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Failed to create instance count clamp shader version");
            return;
        }

        RS::ShaderNativeSourceCode native_source = clamp_shader_source.version_get_native_source_code(clamp_shader_version);
        clamp_shader_source.version_free(clamp_shader_version);

        if (native_source.versions.is_empty() || native_source.versions[0].stages.is_empty()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Embedded instance count clamp shader missing source");
            return;
        }

        String compute_source;
        const RS::ShaderNativeSourceCode::Version &version = native_source.versions[0];
        for (int i = 0; i < version.stages.size(); i++) {
            if (version.stages[i].name == "compute") {
                compute_source = version.stages[i].code;
                break;
            }
        }

        if (compute_source.is_empty()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Embedded instance count clamp shader missing compute stage");
            return;
        }

        GaussianSplatting::ComputeInfrastructure::StageResult compile_result =
                GaussianSplatting::ComputeInfrastructure::compile_compute_shader_from_source(
                        device, "GPUSort.InstanceCount.ShaderCompile", compute_source,
                        "GaussianInstanceCountClamp", instance_count_clamp_shader);
        if (!compile_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(compile_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }

        _track_resource(instance_count_clamp_shader, device, true, "instance_count_clamp_shader");
        instance_count_resource_device = device;
    }

    if (!instance_count_clamp_pipeline.is_valid() && instance_count_clamp_shader.is_valid()) {
        GaussianSplatting::ComputeInfrastructure::StageResult pipeline_result =
                GaussianSplatting::ComputeInfrastructure::create_pipeline_checked(
                        device, instance_count_clamp_shader,
                        "GPUSort.InstanceCount.PipelineCreate", instance_count_clamp_pipeline);
        if (!pipeline_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(pipeline_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }
        _track_resource(instance_count_clamp_pipeline, device, true, "instance_count_clamp_pipeline");
        instance_count_resource_device = device;
    }

    last_compute_error = String();
}

void GPUSortingPipeline::_ensure_instance_chunk_dispatch_resources(RenderingDevice *p_device) {
    RenderingDevice *device = p_device ? p_device : rd;
    if (!device) {
        return;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult capability =
            GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(
                    device, "GPUSort.InstanceDispatch.CapabilityGate");
    if (!capability.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(capability, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }

    if (instance_chunk_dispatch_resource_device && instance_chunk_dispatch_resource_device != device) {
        _free_owned_resource(instance_chunk_dispatch_resource_device, instance_chunk_dispatch_uniform_set, true);
        _free_owned_resource(instance_chunk_dispatch_resource_device, instance_chunk_dispatch_pipeline, true);
        _free_owned_resource(instance_chunk_dispatch_resource_device, instance_chunk_dispatch_shader);
        instance_chunk_dispatch_uniform_set = RID();
        instance_chunk_dispatch_pipeline = RID();
        instance_chunk_dispatch_shader = RID();
        instance_chunk_dispatch_resource_device = nullptr;
    }

    if (!instance_chunk_dispatch_shader.is_valid()) {
        static InstanceChunkDispatchShaderRD dispatch_shader_source;
        static std::once_flag dispatch_shader_once;

        std::call_once(dispatch_shader_once, []() {
            dispatch_shader_source.~InstanceChunkDispatchShaderRD();
            new (&dispatch_shader_source) InstanceChunkDispatchShaderRD();
            Vector<String> versions;
            versions.push_back("");
            dispatch_shader_source.initialize(versions);
        });

        RID dispatch_shader_version = dispatch_shader_source.version_create();
        if (!dispatch_shader_version.is_valid()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Failed to create instance chunk dispatch shader version");
            return;
        }

        RS::ShaderNativeSourceCode native_source = dispatch_shader_source.version_get_native_source_code(dispatch_shader_version);
        dispatch_shader_source.version_free(dispatch_shader_version);

        if (native_source.versions.is_empty() || native_source.versions[0].stages.is_empty()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Embedded instance chunk dispatch shader missing source");
            return;
        }

        String compute_source;
        const RS::ShaderNativeSourceCode::Version &version = native_source.versions[0];
        for (int i = 0; i < version.stages.size(); i++) {
            if (version.stages[i].name == "compute") {
                compute_source = version.stages[i].code;
                break;
            }
        }

        if (compute_source.is_empty()) {
            GS_LOG_WARN_DEFAULT("[GPUSortingPipeline] Embedded instance chunk dispatch shader missing compute stage");
            return;
        }

        GaussianSplatting::ComputeInfrastructure::StageResult compile_result =
                GaussianSplatting::ComputeInfrastructure::compile_compute_shader_from_source(
                        device, "GPUSort.InstanceDispatch.ShaderCompile", compute_source,
                        "GaussianInstanceChunkDispatch", instance_chunk_dispatch_shader);
        if (!compile_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(compile_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }

        _track_resource(instance_chunk_dispatch_shader, device, true, "instance_chunk_dispatch_shader");
        instance_chunk_dispatch_resource_device = device;
    }

    if (!instance_chunk_dispatch_pipeline.is_valid() && instance_chunk_dispatch_shader.is_valid()) {
        GaussianSplatting::ComputeInfrastructure::StageResult pipeline_result =
                GaussianSplatting::ComputeInfrastructure::create_pipeline_checked(
                        device, instance_chunk_dispatch_shader,
                        "GPUSort.InstanceDispatch.PipelineCreate", instance_chunk_dispatch_pipeline);
        if (!pipeline_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(pipeline_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }
        _track_resource(instance_chunk_dispatch_pipeline, device, true, "instance_chunk_dispatch_pipeline");
        instance_chunk_dispatch_resource_device = device;
    }

    last_compute_error = String();
}

void GPUSortingPipeline::ensure_remap_resources(RenderingDevice *p_device) {
    RenderingDevice *device = p_device ? p_device : rd;
    if (!device) {
        return;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult capability =
            GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(
                    device, "GPUSort.Remap.CapabilityGate");
    if (!capability.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(capability, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }

    if (remap_resource_device && remap_resource_device != device) {
        _free_owned_resource(remap_resource_device, remap_uniform_set, true);
        _free_owned_resource(remap_resource_device, remap_compute_pipeline, true);
        _free_owned_resource(remap_resource_device, remap_compute_shader);
        remap_resource_device = nullptr;
    }

    if (!remap_compute_shader.is_valid()) {
        String remap_source = _get_remap_compute_source();
        GaussianSplatting::ComputeInfrastructure::StageResult compile_result =
                GaussianSplatting::ComputeInfrastructure::compile_compute_shader_from_source(
                        device, "GPUSort.Remap.ShaderCompile", remap_source,
                        "GaussianSortRemap", remap_compute_shader);
        if (!compile_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(compile_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }

        _track_resource(remap_compute_shader, device, true, "remap_compute_shader");
    }

    if (!remap_compute_pipeline.is_valid() && remap_compute_shader.is_valid()) {
        GaussianSplatting::ComputeInfrastructure::StageResult pipeline_result =
                GaussianSplatting::ComputeInfrastructure::create_pipeline_checked(
                        device, remap_compute_shader, "GPUSort.Remap.PipelineCreate", remap_compute_pipeline);
        if (!pipeline_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(pipeline_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }
        _track_resource(remap_compute_pipeline, device, true, "remap_compute_pipeline");
    }

    remap_resource_device = device;
    last_compute_error = String();
}

void GPUSortingPipeline::ensure_gather_resources(RenderingDevice *p_device) {
    RenderingDevice *device = p_device ? p_device : rd;
    if (!device) {
        return;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult capability =
            GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(
                    device, "GPUSort.Gather.CapabilityGate");
    if (!capability.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(capability, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }

    if (!gather_positions_shader.is_valid()) {
        String gather_source = vformat(R"(
#version 450

#define WORKGROUP_SIZE %d

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

struct Gaussian {
    vec3 position;
    float opacity;

    vec3 scale;
    float area;

    vec4 rotation;

    vec4 sh_dc;
    float sh_encoded[12];

    vec3 normal;
    float stroke_age;

    vec2 brush_axes;
    uint painterly_meta;
    uint sh_metadata;
};

layout(set = 0, binding = 0, std430) readonly buffer GaussianBuffer {
    Gaussian gaussians[];
} gaussian_buffer;

layout(set = 0, binding = 1, std430) readonly buffer VisibleIndices {
    uint indices[];
} visible_indices;

layout(set = 0, binding = 2, std430) writeonly buffer PositionBuffer {
    vec4 positions[];
} position_buffer;

layout(push_constant) uniform PushConstants {
    uint count;
    uint total_gaussians;
    uint pad0;
    uint pad1;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.count) {
        return;
    }

    uint gaussian_idx = visible_indices.indices[idx];
    if (gaussian_idx >= params.total_gaussians) {
        gaussian_idx = params.total_gaussians > 0u ? (params.total_gaussians - 1u) : 0u;
    }

    Gaussian g = gaussian_buffer.gaussians[gaussian_idx];
    vec3 scale = g.scale;
    float radius = max(max(scale.x, scale.y), scale.z);
    if (radius <= 0.0) {
        radius = 1.0;
    }
    position_buffer.positions[idx] = vec4(g.position, radius);
}
        )", GaussianSplatting::kSortWorkgroupSize);

        GaussianSplatting::ComputeInfrastructure::StageResult compile_result =
                GaussianSplatting::ComputeInfrastructure::compile_compute_shader_from_source(
                        device, "GPUSort.Gather.ShaderCompile", gather_source,
                        "GaussianGatherVisible", gather_positions_shader);
        if (!compile_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(compile_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }

        _track_resource(gather_positions_shader, device, true, "gather_positions_shader");
        sort_resource_device = device;
    }

    if (!gather_positions_pipeline.is_valid() && gather_positions_shader.is_valid()) {
        GaussianSplatting::ComputeInfrastructure::StageResult pipeline_result =
                GaussianSplatting::ComputeInfrastructure::create_pipeline_checked(
                        device, gather_positions_shader, "GPUSort.Gather.PipelineCreate", gather_positions_pipeline);
        if (!pipeline_result.ok()) {
            last_compute_error = _format_stage_failure_with_fallback(pipeline_result, true, true);
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return;
        }
        _track_resource(gather_positions_pipeline, device, true, "gather_positions_pipeline");
        sort_resource_device = device;
    }

    last_compute_error = String();
}

void GPUSortingPipeline::queue_depth_submission(RenderingDevice *p_device, bool p_requires_wait) {
    depth_submission_device = p_device;
    if (!p_device) {
        return;
    }
    if (p_requires_wait) {
        gs_device_utils::safe_submit_and_sync(p_device);
    } else {
        gs_device_utils::safe_submit(p_device);
    }
    depth_submission_needs_submit = false;
    depth_submission_requires_wait = false;
}

bool GPUSortingPipeline::populate_gpu_positions(RID p_buffer, uint32_t p_total_gaussians, uint32_t p_visible_splats,
        const Transform3D &p_cam_transform, float *r_position_ptr, bool p_write_distances,
        SortPositionInputs &p_inputs) {
    if (p_visible_splats == 0) {
        return false;
    }

    RenderingDevice *fallback_device = p_inputs.fallback_device ? p_inputs.fallback_device : rd;
    if (!fallback_device && device_manager.is_valid()) {
        fallback_device = device_manager->get_main_device();
    }
    if (!fallback_device) {
        return false;
    }

    const uint64_t profiling_start = OS::get_singleton()->get_ticks_usec();
    auto accumulate_build_time = [&]() {
        const uint64_t profiling_end = OS::get_singleton()->get_ticks_usec();
        if (p_inputs.sort_input_build_time_ms) {
            *p_inputs.sort_input_build_time_ms += (profiling_end - profiling_start) / 1000.0f;
        }
    };

    auto get_source_index = [&](uint32_t p_index) -> uint32_t {
        if (p_inputs.culled_indices && p_inputs.culled_indices->size() > p_index) {
            return (*p_inputs.culled_indices)[p_index];
        }
        return p_index;
    };

    Vector3 camera_pos;
    if (p_write_distances) {
        if (!p_inputs.culled_distances_sq) {
            return false;
        }
        p_inputs.culled_distances_sq->resize(p_visible_splats);
        if (!cached_camera_transform_valid || !cached_camera_transform.is_equal_approx(p_cam_transform)) {
            cached_camera_transform = p_cam_transform;
            cached_camera_to_world = p_cam_transform.affine_inverse();
            cached_camera_transform_valid = true;
        }
        camera_pos = cached_camera_to_world.origin;
    }

    // When streaming is active, culled_indices contains BUFFER-SPACE indices,
    // not SOURCE indices. The cpu_gaussians array is indexed by SOURCE indices,
    // so we must skip this path and use the streamed data lookup instead.
    const LocalVector<Gaussian> *cpu_gaussians = nullptr;
    if (!p_inputs.use_streamed_data && p_inputs.gaussian_data) {
        const LocalVector<Gaussian> &gaussians = p_inputs.gaussian_data->get_gaussian_storage();
        if ((uint32_t)gaussians.size() >= p_total_gaussians) {
            cpu_gaussians = &gaussians;
        }
    }

    if (cpu_gaussians) {
        for (uint32_t i = 0; i < p_visible_splats; i++) {
            uint32_t source_index = get_source_index(i);
            if (source_index >= p_total_gaussians) {
                return false;
            }

            const Gaussian &g = (*cpu_gaussians)[source_index];
            Vector3 position = g.position;
            float radius = MAX(MAX(g.scale.x, g.scale.y), g.scale.z);
            if (radius <= 0.0f) {
                radius = 1.0f;
            }

            if (r_position_ptr) {
                r_position_ptr[i * 4 + 0] = position.x;
                r_position_ptr[i * 4 + 1] = position.y;
                r_position_ptr[i * 4 + 2] = position.z;
                r_position_ptr[i * 4 + 3] = radius;
            }

            if (p_write_distances) {
                (*p_inputs.culled_distances_sq)[i] = position.distance_squared_to(camera_pos);
            }
        }

        accumulate_build_time();
        return true;
    }

    if (p_inputs.use_streamed_data && p_inputs.cached_streamed_gaussians && !p_inputs.cached_streamed_gaussians->is_empty()) {
        auto get_streamed_gaussian = [&](uint32_t p_index) -> const Gaussian * {
            if (!p_inputs.cached_streamed_index_lookup) {
                return nullptr;
            }
            if (const uint32_t *offset = p_inputs.cached_streamed_index_lookup->getptr(p_index)) {
                uint32_t position = *offset;
                if (position < (uint32_t)p_inputs.cached_streamed_gaussians->size()) {
                    return &(*p_inputs.cached_streamed_gaussians)[position];
                }
            }
            return nullptr;
        };

        for (uint32_t i = 0; i < p_visible_splats; i++) {
            uint32_t source_index = get_source_index(i);
            const Gaussian *gaussian = get_streamed_gaussian(source_index);
            if (!gaussian) {
                return false;
            }

            Vector3 position = gaussian->position;
            float radius = MAX(MAX(gaussian->scale.x, gaussian->scale.y), gaussian->scale.z);
            if (radius <= 0.0f) {
                radius = 1.0f;
            }

            if (r_position_ptr) {
                r_position_ptr[i * 4 + 0] = position.x;
                r_position_ptr[i * 4 + 1] = position.y;
                r_position_ptr[i * 4 + 2] = position.z;
                r_position_ptr[i * 4 + 3] = radius;
            }

            if (p_write_distances) {
                (*p_inputs.culled_distances_sq)[i] = position.distance_squared_to(camera_pos);
            }
        }

        accumulate_build_time();
        return true;
    }

    if (!p_buffer.is_valid()) {
        return false;
    }

    if (p_inputs.test_positions && !p_inputs.test_positions->is_empty() &&
            (uint32_t)p_inputs.test_positions->size() >= p_total_gaussians) {
        for (uint32_t i = 0; i < p_visible_splats; i++) {
            uint32_t source_index = get_source_index(i);
            if (source_index >= p_total_gaussians) {
                return false;
            }

            Vector3 position = (*p_inputs.test_positions)[source_index];
            float radius = 1.0f;
            if (p_inputs.test_scales && (uint32_t)p_inputs.test_scales->size() > source_index) {
                const Vector3 &scale = (*p_inputs.test_scales)[source_index];
                radius = MAX(MAX(scale.x, scale.y), scale.z);
            }
            if (radius <= 0.0f) {
                radius = 1.0f;
            }

            if (r_position_ptr) {
                r_position_ptr[i * 4 + 0] = position.x;
                r_position_ptr[i * 4 + 1] = position.y;
                r_position_ptr[i * 4 + 2] = position.z;
                r_position_ptr[i * 4 + 3] = radius;
            }

            if (p_write_distances) {
                (*p_inputs.culled_distances_sq)[i] = position.distance_squared_to(camera_pos);
            }
        }

        accumulate_build_time();
        return true;
    }

    if (p_visible_splats == 0) {
        accumulate_build_time();
        return true;
    }

    Vector<uint32_t> source_indices;
    source_indices.resize(p_visible_splats);
    uint32_t *source_indices_ptr = source_indices.ptrw();

    uint32_t min_index = UINT32_MAX;
    uint32_t max_index = 0;
    for (uint32_t i = 0; i < p_visible_splats; i++) {
        uint32_t source_index = get_source_index(i);
        if (source_index >= p_total_gaussians) {
            return false;
        }

        source_indices_ptr[i] = source_index;
        min_index = MIN(min_index, source_index);
        max_index = MAX(max_index, source_index);
    }

    if (min_index == UINT32_MAX) {
        return false;
    }

    const uint32_t range_start = min_index;
    const uint32_t range_count = max_index - min_index + 1;
    const bool indices_sparse = (range_count > p_visible_splats * 4);

    if (!indices_sparse) {
        bool cache_ready = false;
        if (p_inputs.gpu_gaussian_cache_valid && p_inputs.gpu_gaussian_cache_buffer &&
                p_inputs.gpu_gaussian_cache_start && p_inputs.gpu_gaussian_cache_count && p_inputs.gpu_gaussian_cache_frame) {
            cache_ready = *p_inputs.gpu_gaussian_cache_valid &&
                    *p_inputs.gpu_gaussian_cache_buffer == p_buffer &&
                    *p_inputs.gpu_gaussian_cache_start == range_start &&
                    *p_inputs.gpu_gaussian_cache_count >= range_count;
        }

        if (!cache_ready) {
            const uint64_t byte_offset = uint64_t(range_start) * sizeof(PackedGaussian);
            const uint64_t byte_size = uint64_t(range_count) * sizeof(PackedGaussian);
            RenderingDevice *buffer_device = device_manager.is_valid() ?
                    device_manager->get_resource_owner(p_buffer, fallback_device) : fallback_device;
            GaussianSplatManager::ScopedSubmissionLock buffer_lock;
            if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
                buffer_device = manager->acquire_submission_device(buffer_device, buffer_lock);
            }
            Vector<uint8_t> gpu_bytes = buffer_device ? buffer_device->buffer_get_data(p_buffer, byte_offset, byte_size) : Vector<uint8_t>();
            if ((uint64_t)gpu_bytes.size() != byte_size) {
                if (p_inputs.gpu_gaussian_cache_valid) {
                    *p_inputs.gpu_gaussian_cache_valid = false;
                }
                return false;
            }

            if (p_inputs.gpu_gaussian_cache) {
                p_inputs.gpu_gaussian_cache->resize(range_count);
                memcpy(p_inputs.gpu_gaussian_cache->ptr(), gpu_bytes.ptr(), byte_size);
            } else {
                return false;
            }

            if (p_inputs.gpu_gaussian_cache_buffer) {
                *p_inputs.gpu_gaussian_cache_buffer = p_buffer;
            }
            if (p_inputs.gpu_gaussian_cache_start) {
                *p_inputs.gpu_gaussian_cache_start = range_start;
            }
            if (p_inputs.gpu_gaussian_cache_count) {
                *p_inputs.gpu_gaussian_cache_count = range_count;
            }
            if (p_inputs.gpu_gaussian_cache_frame) {
                *p_inputs.gpu_gaussian_cache_frame = p_inputs.frame_counter;
            }
            if (p_inputs.gpu_gaussian_cache_valid) {
                *p_inputs.gpu_gaussian_cache_valid = true;
            }
        }

        if (!p_inputs.gpu_gaussian_cache) {
            return false;
        }

        for (uint32_t i = 0; i < p_visible_splats; i++) {
            uint32_t source_index = source_indices_ptr[i];

            const PackedGaussian &packed_gaussian = (*p_inputs.gpu_gaussian_cache)[source_index - range_start];
            Vector3 position = Vector3(packed_gaussian.position[0], packed_gaussian.position[1], packed_gaussian.position[2]);
            float radius = MAX(MAX(packed_gaussian.scale[0], packed_gaussian.scale[1]), packed_gaussian.scale[2]);
            if (radius <= 0.0f) {
                radius = 1.0f;
            }

            if (r_position_ptr) {
                r_position_ptr[i * 4 + 0] = position.x;
                r_position_ptr[i * 4 + 1] = position.y;
                r_position_ptr[i * 4 + 2] = position.z;
                r_position_ptr[i * 4 + 3] = radius;
            }

            if (p_write_distances) {
                (*p_inputs.culled_distances_sq)[i] = position.distance_squared_to(camera_pos);
            }
        }
    } else {
        if (p_inputs.gpu_gaussian_cache_valid) {
            *p_inputs.gpu_gaussian_cache_valid = false;
        }

        Vector<uint32_t> sorted_indices = source_indices;
        sorted_indices.sort();

        Vector<uint32_t> unique_indices;
        const int sorted_count = sorted_indices.size();
        for (int i = 0; i < sorted_count; i++) {
            uint32_t idx = sorted_indices[i];
            if (unique_indices.is_empty() || unique_indices[unique_indices.size() - 1] != idx) {
                unique_indices.push_back(idx);
            }
        }

        HashMap<uint32_t, PackedGaussian> fetched_gaussians;
        fetched_gaussians.reserve(unique_indices.size());
        Vector<PackedGaussian> run_data;

        auto fetch_run = [&](uint32_t p_start, uint32_t p_end) -> bool {
            const uint32_t count = p_end - p_start + 1;
            const uint64_t byte_offset = uint64_t(p_start) * sizeof(PackedGaussian);
            const uint64_t byte_size = uint64_t(count) * sizeof(PackedGaussian);
            RenderingDevice *buffer_device = device_manager.is_valid() ?
                    device_manager->get_resource_owner(p_buffer, fallback_device) : fallback_device;
            GaussianSplatManager::ScopedSubmissionLock buffer_lock;
            if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
                buffer_device = manager->acquire_submission_device(buffer_device, buffer_lock);
            }
            Vector<uint8_t> gpu_bytes = buffer_device ? buffer_device->buffer_get_data(p_buffer, byte_offset, byte_size) : Vector<uint8_t>();
            if ((uint64_t)gpu_bytes.size() != byte_size) {
                return false;
            }

            run_data.resize(count);
            memcpy(run_data.ptrw(), gpu_bytes.ptr(), byte_size);
            for (uint32_t j = 0; j < count; j++) {
                fetched_gaussians.insert(p_start + j, run_data[j]);
            }
            return true;
        };

        if (unique_indices.is_empty()) {
            return false;
        }

        uint32_t run_start = unique_indices[0];
        uint32_t run_end = run_start;
        const int unique_count = unique_indices.size();
        for (int i = 1; i < unique_count; i++) {
            uint32_t idx = unique_indices[i];
            if (idx != run_end + 1) {
                if (!fetch_run(run_start, run_end)) {
                    return false;
                }
                run_start = idx;
            }
            run_end = idx;
        }

        if (!fetch_run(run_start, run_end)) {
            return false;
        }

        for (uint32_t i = 0; i < p_visible_splats; i++) {
            uint32_t source_index = source_indices_ptr[i];
            PackedGaussian *packed_ptr = fetched_gaussians.getptr(source_index);
            if (!packed_ptr) {
                return false;
            }

            const PackedGaussian &packed_gaussian = *packed_ptr;
            Vector3 position = Vector3(packed_gaussian.position[0], packed_gaussian.position[1], packed_gaussian.position[2]);
            float radius = MAX(MAX(packed_gaussian.scale[0], packed_gaussian.scale[1]), packed_gaussian.scale[2]);
            if (radius <= 0.0f) {
                radius = 1.0f;
            }

            if (r_position_ptr) {
                r_position_ptr[i * 4 + 0] = position.x;
                r_position_ptr[i * 4 + 1] = position.y;
                r_position_ptr[i * 4 + 2] = position.z;
                r_position_ptr[i * 4 + 3] = radius;
            }

            if (p_write_distances) {
                (*p_inputs.culled_distances_sq)[i] = position.distance_squared_to(camera_pos);
            }
        }
    }

    accumulate_build_time();
    return true;
}

bool GPUSortingPipeline::_sort_instance_pipeline(GaussianSplatRenderer &renderer, const Transform3D &p_cam_transform) {
    const bool trace_enabled = GaussianSplatting::debug_trace_is_enabled();
    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                "InstancePipeline ENTER: instance_inputs_valid=" + String(instance_inputs_valid ? "YES" : "NO"),
                false);
    }
    if (!instance_inputs_valid) {
        if (trace_enabled) {
            GaussianSplatting::debug_trace_record_event("sort",
                    "InstancePipeline EARLY EXIT: instance_inputs_valid=false",
                    true);
        }
        return false;
    }

    InstancePipelineInputs inputs = instance_inputs;
    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                vformat("InstancePipeline inputs: vis_chunk=%d max_vis_chunks=%d max_chunk_splats=%d max_vis_splats=%d",
                        inputs.visible_chunk_count, inputs.max_visible_chunks, inputs.max_chunk_splats, inputs.max_visible_splats),
                false);
    }
    const bool quantized_storage = g_quantization_config.per_chunk_quantization;
    RenderingDevice *compute_rd = inputs.device ? inputs.device : renderer.get_device_state().rd;
    if (!compute_rd) {
        return false;
    }

    if (!renderer.ensure_rendering_device("_sort_instance_pipeline")) {
        return false;
    }

    auto &sorting_state = renderer.get_sorting_state();
    auto &frame_state = renderer.get_frame_state();
    auto &performance_state = renderer.get_performance_state();
    Ref<GPUCuller> gpu_culler = renderer.get_subsystem_state().gpu_culler;
    if (gpu_culler.is_valid()) {
        gpu_culler->update_culling_settings();
        gpu_culler->update_lod_cache();
    }
    if (sorting_state.sorter_needs_rebuild) {
        renderer.refresh_gpu_sorter("_sort_instance_pipeline");
    }
    // Instance pipeline uses the GPUSortingPipeline's own gpu_sorter (rebuilt
    // to full multi-instance capacity), NOT sorting_state.gpu_sorter (which is
    // sized for single-instance perf_settings.max_splats).
    if (!gpu_sorter.is_valid()) {
        // Fallback: also accept sorting_state.gpu_sorter for the validity check
        // (it may have been rebuilt via refresh_gpu_sorter above).
        if (!sorting_state.gpu_sorter.is_valid()) {
            return false;
        }
    }

    ensure_depth_resources(compute_rd);
    if (!depth_compute_shader.is_valid() || !depth_compute_pipeline.is_valid()) {
        return false;
    }

    _ensure_instance_param_buffer(compute_rd);
    if (!instance_param_buffer.is_valid()) {
        return false;
    }

    _ensure_instance_count_resources(compute_rd);
    if (!instance_count_clamp_shader.is_valid() || !instance_count_clamp_pipeline.is_valid()) {
        return false;
    }

    _ensure_instance_chunk_dispatch_resources(compute_rd);
    if (!instance_chunk_dispatch_shader.is_valid() || !instance_chunk_dispatch_pipeline.is_valid()) {
        return false;
    }

    if (!inputs.atlas_gaussian_buffer.is_valid() || (quantized_storage && !inputs.quantization_buffer.is_valid()) ||
            !inputs.instance_buffer.is_valid() || !inputs.chunk_meta_buffer.is_valid() ||
            !inputs.visible_chunk_buffer.is_valid() || !inputs.splat_ref_buffer.is_valid() ||
            !inputs.sort_key_buffer.is_valid() || !inputs.sort_value_buffer.is_valid() ||
            !inputs.counter_buffer.is_valid() || !inputs.chunk_dispatch_buffer.is_valid() ||
            !inputs.indirect_count_buffer.is_valid() || !inputs.instance_count_buffer.is_valid()) {
        return false;
    }

    uint32_t max_visible_splats = inputs.max_visible_splats;
    // Use the pipeline's own gpu_sorter capacity (rebuilt to full multi-instance
    // budget), not sorting_state.gpu_sorter (single-instance perf_settings.max_splats).
    const uint32_t sorter_capacity = gpu_sorter.is_valid()
            ? gpu_sorter->get_max_elements()
            : (sorting_state.gpu_sorter.is_valid() ? sorting_state.gpu_sorter->get_max_elements() : 0);
    if (sorter_capacity > 0 && max_visible_splats > sorter_capacity) {
        max_visible_splats = sorter_capacity;
    }
    if (inputs.max_visible_chunks == 0 || inputs.max_chunk_splats == 0 || max_visible_splats == 0) {
        if (trace_enabled) {
            GaussianSplatting::debug_trace_record_event("sort",
                    vformat("InstancePipeline EARLY EXIT: max_vis_chunks=%d max_chunk_splats=%d max_vis_splats=%d",
                            inputs.max_visible_chunks, inputs.max_chunk_splats, max_visible_splats),
                    true);
        }
        last_instance_visible_splat_count = 0;
        sorting_state.sorted_splat_count = 0;
        frame_state.visible_splat_count.store(0, std::memory_order_release);
        performance_state.metrics.sort_submission_time_ms = 0.0f;
        performance_state.metrics.sort_wait_time_ms = 0.0f;
        return true;
    }
    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                "InstancePipeline proceeding to depth compute dispatch",
                false);
    }

    // Only change buffer ownership if the input buffers differ from what we
    // already hold. When ensure_buffers() creates owned buffers and the
    // renderer reads them back via get_buffer_handles(), the same RIDs arrive
    // here as "external" inputs. Marking owned buffers as external causes
    // ensure_buffers() to _forget (not free) them on the next frame and
    // allocate new ones -- leaking ~2 StorageBuffer RIDs per frame.
    if (inputs.sort_value_buffer != sort_indices_buffer) {
        set_external_sort_indices(inputs.sort_value_buffer, compute_rd);
    }
    if (inputs.sort_key_buffer != sort_keys_buffer) {
        RenderingDevice *new_key_owner = compute_rd;
        if (inputs.sort_key_buffer.is_valid()) {
            if (device_manager.is_valid()) {
                new_key_owner = device_manager->get_resource_owner(inputs.sort_key_buffer, new_key_owner ? new_key_owner : rd);
            } else if (!new_key_owner) {
                new_key_owner = rd;
            }
            ERR_FAIL_NULL_V_MSG(new_key_owner, false,
                    vformat("[GPUSortingPipeline] Missing owner device for external sort key buffer RID %s",
                            String::num_uint64(inputs.sort_key_buffer.get_id())));
        }

        if (sort_keys_buffer.is_valid() && sort_keys_buffer == inputs.sort_key_buffer && !sort_keys_external) {
            // Reusing the pipeline-owned key buffer; do not flip ownership role.
        } else if (sort_keys_buffer.is_valid()) {
            if (sort_keys_external) {
                _forget_resource(sort_keys_buffer);
                sort_keys_buffer = RID();
            } else {
                RenderingDevice *fallback_device = sort_resource_device ? sort_resource_device : compute_rd;
                _free_owned_resource(fallback_device, sort_keys_buffer);
            }
        }

        if (!(sort_keys_buffer.is_valid() && sort_keys_buffer == inputs.sort_key_buffer && !sort_keys_external)) {
            sort_keys_buffer = inputs.sort_key_buffer;
            sort_keys_external = inputs.sort_key_buffer.is_valid();
            if (sort_keys_external) {
                _track_resource(inputs.sort_key_buffer, new_key_owner, false, "external_sort_keys");
                sort_resource_device = new_key_owner;
            }
        }
    }
    if (sort_keys_external && sort_indices_external && sort_keys_buffer.is_valid() && sort_indices_buffer.is_valid() &&
            device_manager.is_valid()) {
        RenderingDevice *key_owner = device_manager->get_resource_owner(sort_keys_buffer, compute_rd ? compute_rd : rd);
        RenderingDevice *index_owner = device_manager->get_resource_owner(sort_indices_buffer, compute_rd ? compute_rd : rd);
        ERR_FAIL_NULL_V_MSG(key_owner, false, "[GPUSortingPipeline] Missing owner for external sort key buffer");
        ERR_FAIL_NULL_V_MSG(index_owner, false, "[GPUSortingPipeline] Missing owner for external sort index buffer");
        ERR_FAIL_COND_V_MSG(key_owner != index_owner, false,
                "[GPUSortingPipeline] External sort key/index buffers must share the same owner device");
        sort_resource_device = key_owner;
    }
    sorting_state.sort_keys_external = sort_keys_external;
    sorting_state.sort_indices_external = sort_indices_external;
    sorting_state.sort_buffers_pipeline_managed = false;
    sorting_state.sort_buffer_capacity = max_visible_splats;

    if (depth_uniform_set.is_valid()) {
        _free_owned_resource(sort_resource_device ? sort_resource_device : compute_rd, depth_uniform_set, true);
    }
    if (instance_count_uniform_set.is_valid()) {
        _free_owned_resource(instance_count_resource_device ? instance_count_resource_device : compute_rd, instance_count_uniform_set, true);
    }
    if (instance_chunk_dispatch_uniform_set.is_valid()) {
        _free_owned_resource(instance_chunk_dispatch_resource_device ? instance_chunk_dispatch_resource_device : compute_rd,
                instance_chunk_dispatch_uniform_set, true);
    }

    uint32_t group_x = (inputs.max_chunk_splats + (GaussianSplatting::kSortWorkgroupSize - 1)) /
            GaussianSplatting::kSortWorkgroupSize;
    if (group_x == 0) {
        group_x = 1;
    }

    InstanceDepthParamsGPU params = {};
    const Basis &basis = p_cam_transform.basis;
    const Vector3 &origin = p_cam_transform.origin;
    params.view_matrix[0] = basis[0][0];
    params.view_matrix[1] = basis[1][0];
    params.view_matrix[2] = basis[2][0];
    params.view_matrix[3] = 0.0f;
    params.view_matrix[4] = basis[0][1];
    params.view_matrix[5] = basis[1][1];
    params.view_matrix[6] = basis[2][1];
    params.view_matrix[7] = 0.0f;
    params.view_matrix[8] = basis[0][2];
    params.view_matrix[9] = basis[1][2];
    params.view_matrix[10] = basis[2][2];
    params.view_matrix[11] = 0.0f;
    params.view_matrix[12] = origin.x;
    params.view_matrix[13] = origin.y;
    params.view_matrix[14] = origin.z;
    params.view_matrix[15] = 1.0f;
    // Buffer capacity used as structural guard in depth_compute.glsl.
    // Actual dispatch is GPU-driven via instance_chunk_dispatch.glsl.
    params.visible_chunk_count = MIN(inputs.visible_chunk_count, inputs.max_visible_chunks);
    params.max_visible_splats = max_visible_splats;
    params.pad0 = group_x;
    params.pad1 = 0u;
    params.wind_dir_strength[0] = 1.0f;
    params.wind_dir_strength[1] = 0.0f;
    params.wind_dir_strength[2] = 0.0f;
    params.wind_dir_strength[3] = 0.0f;
    params.wind_time_config[0] = 0.0f;
    params.wind_time_config[1] = 1.0f;
    params.wind_time_config[2] = 0.1f;
    params.wind_time_config[3] = 0.0f;
    params.effector_sphere[0] = 0.0f;
    params.effector_sphere[1] = 0.0f;
    params.effector_sphere[2] = 0.0f;
    params.effector_sphere[3] = 0.0f;
    params.effector_config[0] = 0.0f;
    params.effector_config[1] = 0.0f;
    params.effector_config[2] = 2.0f;
    params.effector_config[3] = 2.0f; // Default frequency 2 Hz
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        static const StringName wind_enabled_path("rendering/gaussian_splatting/animation/wind_enabled");
        static const StringName wind_direction_x_path("rendering/gaussian_splatting/animation/wind_direction_x");
        static const StringName wind_direction_y_path("rendering/gaussian_splatting/animation/wind_direction_y");
        static const StringName wind_direction_z_path("rendering/gaussian_splatting/animation/wind_direction_z");
        static const StringName wind_strength_path("rendering/gaussian_splatting/animation/wind_strength");
        static const StringName wind_frequency_path("rendering/gaussian_splatting/animation/wind_frequency");
        static const StringName wind_spatial_frequency_path("rendering/gaussian_splatting/animation/wind_spatial_frequency");
        static const StringName wind_time_scale_path("rendering/gaussian_splatting/animation/wind_time_scale");
        static const StringName max_effectors_path("rendering/gaussian_splatting/effects/max_effectors");
        static const StringName sphere_effector_enabled_path("rendering/gaussian_splatting/effects/sphere_effector_enabled");
        static const StringName sphere_effector_center_x_path("rendering/gaussian_splatting/effects/sphere_effector_center_x");
        static const StringName sphere_effector_center_y_path("rendering/gaussian_splatting/effects/sphere_effector_center_y");
        static const StringName sphere_effector_center_z_path("rendering/gaussian_splatting/effects/sphere_effector_center_z");
        static const StringName sphere_effector_radius_path("rendering/gaussian_splatting/effects/sphere_effector_radius");
        static const StringName sphere_effector_strength_path("rendering/gaussian_splatting/effects/sphere_effector_strength");
        static const StringName sphere_effector_falloff_path("rendering/gaussian_splatting/effects/sphere_effector_falloff");
        static const StringName sphere_effector_frequency_path("rendering/gaussian_splatting/effects/sphere_effector_frequency");

        const bool wind_enabled = _get_bool_setting(ps, wind_enabled_path, false);
        const float wind_time_scale = MAX(_get_float_setting(ps, wind_time_scale_path, 1.0f), 0.0f);
        params.wind_dir_strength[0] = _get_float_setting(ps, wind_direction_x_path, 1.0f);
        params.wind_dir_strength[1] = _get_float_setting(ps, wind_direction_y_path, 0.0f);
        params.wind_dir_strength[2] = _get_float_setting(ps, wind_direction_z_path, 0.0f);
        params.wind_dir_strength[3] = MAX(_get_float_setting(ps, wind_strength_path, 0.0f), 0.0f);
        params.wind_time_config[0] = float(double(renderer.get_frame_state().frame_counter) * (1.0 / 60.0) *
                double(wind_time_scale));
        params.wind_time_config[1] = MAX(_get_float_setting(ps, wind_frequency_path, 1.0f), 0.0f);
        params.wind_time_config[2] = _get_float_setting(ps, wind_spatial_frequency_path, 0.1f);
        params.wind_time_config[3] = wind_enabled ? 1.0f : 0.0f;

        const int max_effectors = CLAMP((int)_get_float_setting(ps, max_effectors_path, 1.0f), 0, 1);
        const bool sphere_effective_enabled = max_effectors > 0 &&
                _get_bool_setting(ps, sphere_effector_enabled_path, false);
        params.effector_sphere[0] = _get_float_setting(ps, sphere_effector_center_x_path, 0.0f);
        params.effector_sphere[1] = _get_float_setting(ps, sphere_effector_center_y_path, 0.0f);
        params.effector_sphere[2] = _get_float_setting(ps, sphere_effector_center_z_path, 0.0f);
        params.effector_sphere[3] = MAX(_get_float_setting(ps, sphere_effector_radius_path, 0.0f), 0.0f);
        params.effector_config[0] = sphere_effective_enabled ? 1.0f : 0.0f;
        params.effector_config[1] = _get_float_setting(ps, sphere_effector_strength_path, 0.0f);
        params.effector_config[2] = MAX(_get_float_setting(ps, sphere_effector_falloff_path, 2.0f), 0.001f);
        params.effector_config[3] = MAX(_get_float_setting(ps, sphere_effector_frequency_path, 2.0f), 0.1f);
    }

    Projection projection = renderer.get_view_state().last_camera_projection;
    if (projection.get_z_far() <= projection.get_z_near()) {
        projection.set_perspective(60.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    }
    const Transform3D camera_transform = p_cam_transform.affine_inverse();
    const bool orthographic = projection.is_orthogonal();

    float tiny_splat_screen_radius = 0.0f;
    float min_screen_threshold = 0.0f;
    float max_distance_sq = 0.0f;
    float radius_multiplier = 1.0f;
    float frustum_plane_slack = 1.0f;
    float enable_frustum = 0.0f;
    Size2i viewport_size = Size2i(1280, 720);
    Vector<Plane> frustum_planes;
    if (gpu_culler.is_valid()) {
        const GPUCuller::CullingConfig &cull_config = gpu_culler->get_config();
        const GPUCuller::CullingState &cull_state = gpu_culler->get_state();
        viewport_size = cull_config.last_cull_viewport_size;
        tiny_splat_screen_radius = MAX(cull_state.tiny_splat_screen_radius_px, 0.0f);
        min_screen_threshold = cull_config.lod_enabled ? MAX(cull_config.lod_cached_min_screen_threshold, 0.0f) : 0.0f;
        max_distance_sq = cull_config.lod_enabled ? MAX(cull_config.lod_cached_max_distance_sq, 0.0f) : 0.0f;
        radius_multiplier = MAX(cull_config.cull_radius_multiplier, 0.0f);
        frustum_plane_slack = MAX(cull_config.cull_frustum_plane_slack, 1.0f);
        enable_frustum = cull_config.frustum_culling ? 1.0f : 0.0f;
        if (cull_config.frustum_culling) {
            frustum_planes = projection.get_projection_planes(camera_transform);
        }
    }
    if (viewport_size.x <= 0 || viewport_size.y <= 0) {
        const Size2i manual_override = renderer.get_view_state().manual_viewport_override;
        if (manual_override.x > 0 && manual_override.y > 0) {
            viewport_size = manual_override;
        } else {
            viewport_size = Size2i(1280, 720);
        }
    }
    const float viewport_height = viewport_size.y > 0 ? float(viewport_size.y) : 720.0f;
    float vertical_scale = Math::abs(projection[1][1]);
    if (vertical_scale <= 0.0f) {
        vertical_scale = 1.0f;
    }
    const float pixel_scale_y = vertical_scale * (viewport_height * 0.5f);

    for (uint32_t i = 0; i < InstanceCullParamsGPU::kFrustumPlaneCount; i++) {
        if (i < uint32_t(frustum_planes.size())) {
            const Plane &plane = frustum_planes[int(i)];
            params.frustum_planes[i][0] = plane.normal.x;
            params.frustum_planes[i][1] = plane.normal.y;
            params.frustum_planes[i][2] = plane.normal.z;
            params.frustum_planes[i][3] = plane.d;
        } else {
            params.frustum_planes[i][0] = 0.0f;
            params.frustum_planes[i][1] = 0.0f;
            params.frustum_planes[i][2] = 0.0f;
            params.frustum_planes[i][3] = 0.0f;
        }
    }
    params.camera_position_ortho[0] = camera_transform.origin.x;
    params.camera_position_ortho[1] = camera_transform.origin.y;
    params.camera_position_ortho[2] = camera_transform.origin.z;
    params.camera_position_ortho[3] = orthographic ? 1.0f : 0.0f;
    params.cull_screen_distance[0] = pixel_scale_y;
    params.cull_screen_distance[1] = tiny_splat_screen_radius;
    params.cull_screen_distance[2] = min_screen_threshold;
    params.cull_screen_distance[3] = max_distance_sq;
    params.cull_frustum_radius[0] = radius_multiplier;
    params.cull_frustum_radius[1] = frustum_plane_slack;
    params.cull_frustum_radius[2] = enable_frustum;
    params.cull_frustum_radius[3] = 0.0f;

    Vector<uint8_t> param_bytes;
    param_bytes.resize(sizeof(InstanceDepthParamsGPU));
    std::memcpy(param_bytes.ptrw(), &params, sizeof(InstanceDepthParamsGPU));
    bool param_dirty = !instance_param_cache_valid ||
            instance_param_cache.size() != param_bytes.size() ||
            std::memcmp(instance_param_cache.ptr(), param_bytes.ptr(), param_bytes.size()) != 0;
    if (param_dirty) {
        compute_rd->buffer_update(instance_param_buffer, 0, param_bytes.size(), param_bytes.ptr());
        instance_param_cache = param_bytes;
        instance_param_cache_valid = true;
    }

    auto append_binding = [&](Vector<GaussianSplatting::ComputeInfrastructure::UniformBindingContract> &r_bindings,
                                  RenderingDevice::UniformType p_type, uint32_t p_binding, const RID &p_resource,
                                  const char *p_label) {
        GaussianSplatting::ComputeInfrastructure::UniformBindingContract contract;
        contract.type = p_type;
        contract.binding = p_binding;
        contract.resource = p_resource;
        contract.label = p_label;
        r_bindings.push_back(contract);
    };

    Vector<GaussianSplatting::ComputeInfrastructure::UniformBindingContract> depth_bindings;
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, inputs.atlas_gaussian_buffer, "atlas_gaussian_buffer");
    if (quantized_storage) {
        append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, inputs.quantization_buffer, "quantization_buffer");
    }
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, inputs.instance_buffer, "instance_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, inputs.chunk_meta_buffer, "chunk_meta_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, inputs.visible_chunk_buffer, "visible_chunk_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, inputs.splat_ref_buffer, "splat_ref_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, inputs.sort_key_buffer, "sort_key_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 7, inputs.sort_value_buffer, "sort_value_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, inputs.counter_buffer, "counter_buffer");
    append_binding(depth_bindings, RD::UNIFORM_TYPE_UNIFORM_BUFFER, 9, instance_param_buffer, "instance_param_buffer");

    GaussianSplatting::ComputeInfrastructure::StageValidationHarness stage_validation_harness;
    GaussianSplatting::ComputeInfrastructure::StageValidationInput stage_validation_input;
    stage_validation_input.stage_name = "GPUSort.InstanceDepth.UniformContract";
    stage_validation_input.bindings = depth_bindings;
    GaussianSplatting::ComputeInfrastructure::StageResult stage_validation_result =
            stage_validation_harness.validate(stage_validation_input);
    if (!stage_validation_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error(
                "GPUSortingPipeline", stage_validation_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    sort_resource_device = compute_rd;
    GaussianSplatting::ComputeInfrastructure::StageResult depth_uniform_result =
            GaussianSplatting::ComputeInfrastructure::create_uniform_set_checked(
                    compute_rd, depth_compute_shader, 0, depth_bindings,
                    "GPUSort.InstanceDepth.UniformSet",
                    "GS_GPUSortingPipeline_InstanceDepthUniformSet", depth_uniform_set);
    if (!depth_uniform_result.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(depth_uniform_result, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    _track_resource(depth_uniform_set, compute_rd, true, "instance_depth_uniform_set");

    Vector<GaussianSplatting::ComputeInfrastructure::UniformBindingContract> chunk_dispatch_bindings;
    append_binding(chunk_dispatch_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, inputs.counter_buffer, "counter_buffer");
    append_binding(chunk_dispatch_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, inputs.chunk_dispatch_buffer, "chunk_dispatch_buffer");
    append_binding(chunk_dispatch_bindings, RD::UNIFORM_TYPE_UNIFORM_BUFFER, 2, instance_param_buffer, "instance_param_buffer");

    stage_validation_input = GaussianSplatting::ComputeInfrastructure::StageValidationInput();
    stage_validation_input.stage_name = "GPUSort.InstanceDispatch.UniformContract";
    stage_validation_input.bindings = chunk_dispatch_bindings;
    stage_validation_result = stage_validation_harness.validate(stage_validation_input);
    if (!stage_validation_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error(
                "GPUSortingPipeline", stage_validation_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    instance_chunk_dispatch_resource_device = compute_rd;
    GaussianSplatting::ComputeInfrastructure::StageResult chunk_dispatch_uniform_result =
            GaussianSplatting::ComputeInfrastructure::create_uniform_set_checked(
                    compute_rd, instance_chunk_dispatch_shader, 0, chunk_dispatch_bindings,
                    "GPUSort.InstanceDispatch.UniformSet",
                    "GS_GPUSortingPipeline_InstanceChunkDispatchSet", instance_chunk_dispatch_uniform_set);
    if (!chunk_dispatch_uniform_result.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(chunk_dispatch_uniform_result, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    _track_resource(instance_chunk_dispatch_uniform_set, compute_rd, true, "instance_chunk_dispatch_uniform_set");

    Vector<GaussianSplatting::ComputeInfrastructure::UniformBindingContract> clamp_bindings;
    append_binding(clamp_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, inputs.counter_buffer, "counter_buffer");
    append_binding(clamp_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, inputs.indirect_count_buffer, "indirect_count_buffer");
    append_binding(clamp_bindings, RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, inputs.instance_count_buffer, "instance_count_buffer");
    append_binding(clamp_bindings, RD::UNIFORM_TYPE_UNIFORM_BUFFER, 2, instance_param_buffer, "instance_param_buffer");

    stage_validation_input = GaussianSplatting::ComputeInfrastructure::StageValidationInput();
    stage_validation_input.stage_name = "GPUSort.InstanceCount.UniformContract";
    stage_validation_input.bindings = clamp_bindings;
    stage_validation_result = stage_validation_harness.validate(stage_validation_input);
    if (!stage_validation_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error(
                "GPUSortingPipeline", stage_validation_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    instance_count_resource_device = compute_rd;
    GaussianSplatting::ComputeInfrastructure::StageResult clamp_uniform_result =
            GaussianSplatting::ComputeInfrastructure::create_uniform_set_checked(
                    compute_rd, instance_count_clamp_shader, 0, clamp_bindings,
                    "GPUSort.InstanceCount.UniformSet",
                    "GS_GPUSortingPipeline_InstanceCountClampSet", instance_count_uniform_set);
    if (!clamp_uniform_result.ok()) {
        last_compute_error = _format_stage_failure_with_fallback(clamp_uniform_result, true, true);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    _track_resource(instance_count_uniform_set, compute_rd, true, "instance_count_clamp_uniform_set");

    stage_validation_input = GaussianSplatting::ComputeInfrastructure::StageValidationInput();
    stage_validation_input.stage_name = "GPUSort.InstanceDispatch.Dispatch";
    stage_validation_input.bindings = chunk_dispatch_bindings;
    stage_validation_input.validate_dispatch = true;
    stage_validation_input.dispatch_x = 1;
    stage_validation_input.dispatch_y = 1;
    stage_validation_input.dispatch_z = 1;
    stage_validation_result = stage_validation_harness.validate(stage_validation_input);
    if (!stage_validation_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error(
                "GPUSortingPipeline", stage_validation_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    RD::ComputeListID compute_list = compute_rd->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        last_compute_error = "[GPU Sort] Failed to begin instance chunk dispatch compute list";
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    compute_rd->compute_list_bind_compute_pipeline(compute_list, instance_chunk_dispatch_pipeline);
    compute_rd->compute_list_bind_uniform_set(compute_list, instance_chunk_dispatch_uniform_set, 0);
    compute_rd->compute_list_dispatch(compute_list, 1, 1, 1);
    compute_rd->compute_list_add_barrier(compute_list);
    compute_rd->compute_list_end();

    compute_list = compute_rd->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        last_compute_error = "[GPU Sort] Failed to begin instance depth compute list";
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    compute_rd->compute_list_bind_compute_pipeline(compute_list, depth_compute_pipeline);
    compute_rd->compute_list_bind_uniform_set(compute_list, depth_uniform_set, 0);
    compute_rd->compute_list_dispatch_indirect(compute_list, inputs.chunk_dispatch_buffer, 0);
    compute_rd->compute_list_add_barrier(compute_list);
    compute_rd->compute_list_end();

    stage_validation_input = GaussianSplatting::ComputeInfrastructure::StageValidationInput();
    stage_validation_input.stage_name = "GPUSort.InstanceCount.Dispatch";
    stage_validation_input.bindings = clamp_bindings;
    stage_validation_input.validate_dispatch = true;
    stage_validation_input.dispatch_x = 1;
    stage_validation_input.dispatch_y = 1;
    stage_validation_input.dispatch_z = 1;
    stage_validation_result = stage_validation_harness.validate(stage_validation_input);
    if (!stage_validation_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error(
                "GPUSortingPipeline", stage_validation_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    compute_list = compute_rd->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        last_compute_error = "[GPU Sort] Failed to begin instance count clamp compute list";
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    compute_rd->compute_list_bind_compute_pipeline(compute_list, instance_count_clamp_pipeline);
    compute_rd->compute_list_bind_uniform_set(compute_list, instance_count_uniform_set, 0);
    compute_rd->compute_list_dispatch(compute_list, 1, 1, 1);
    compute_rd->compute_list_add_barrier(compute_list);
    compute_rd->compute_list_end();

    constexpr uint32_t kInstanceCountAsyncMaxAgeFrames = 8u;
    constexpr uint32_t kInstanceCountAsyncResetAgeFrames = 45u;
    uint32_t pending_count_age_frames = 0;
    if (instance_count_readback_state.pending &&
            frame_state.frame_counter >= instance_count_readback_state.pending_frame_counter) {
        pending_count_age_frames = frame_state.frame_counter - instance_count_readback_state.pending_frame_counter;
    }

    RenderingDevice *sort_submission_device = nullptr;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        sort_submission_device = manager->get_shared_submission_device();
    }
    const bool cross_device_sort_submission =
            sort_submission_device && sort_submission_device != compute_rd;
    if (cross_device_sort_submission) {
        // Depth/count writes are produced on compute_rd but consumed by the shared submission
        // device in sort_async(). Always flush+sync before cross-device submission to avoid
        // consuming stale or incomplete buffers.
        gs_device_utils::safe_submit_and_sync(compute_rd);
    } else {
        gs_device_utils::safe_submit(compute_rd);
    }

    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                vformat("InstancePipeline depth+clamp dispatched (max_visible_splats=%d group_x=%d)",
                        max_visible_splats, group_x),
                false);
    }

    SortOperationParams sort_params;
    sort_params.element_count = max_visible_splats;
    sort_params.keys_buffer = inputs.sort_key_buffer;
    sort_params.values_buffer = inputs.sort_value_buffer;
    sort_params.count_buffer = inputs.instance_count_buffer;
    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                "InstancePipeline calling sort_async() with instance count buffer",
                false);
    }
    SortOperationResult sort_result = sort_async(sort_params);
    if (!sort_result.success) {
        GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Instance pipeline sort failed: %s", sort_result.error));
        if (trace_enabled) {
            GaussianSplatting::debug_trace_record_event("sort",
                    vformat("InstancePipeline SORT FAILED: %s", sort_result.error),
                    true);
        }
        return false;
    }
    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                vformat("InstancePipeline sort SUCCESS: time=%.2fms", sort_result.gpu_time_ms),
                false);
    }

    const bool debug_sync_requested =
            g_gpu_sorting_config.enable_prefix_readback || g_gpu_sorting_config.debug_validate_prefix;
    const gs_sort_policy::ReadbackPolicy readback_policy =
            gs_sort_policy::resolve_readback_policy(debug_sync_requested,
                    g_gpu_sorting_config.profiling_preserve_gpu_timestamps);

    // Resolve visible count with policy-driven async/sync behavior.
    // Bootstrap may use one synchronous sample, while steady-state always
    // consumes async-published counts to avoid frame-time stalls.
    const uint32_t safe_max = (sort_buffer_capacity > 0)
            ? MIN(max_visible_splats, sort_buffer_capacity)
            : max_visible_splats;
    // Bound reused async counts by the current-frame visible chunk budget so we
    // do not render stale tails from prior high-visibility frames.
    const uint32_t visible_chunk_budget = MIN(inputs.visible_chunk_count, inputs.max_visible_chunks);
    uint32_t current_frame_safe_max = safe_max;
    if (visible_chunk_budget == 0 || inputs.max_chunk_splats == 0) {
        current_frame_safe_max = 0;
    } else {
        const uint64_t chunk_budget_splats_u64 = uint64_t(visible_chunk_budget) * uint64_t(inputs.max_chunk_splats);
        const uint32_t chunk_budget_splats = chunk_budget_splats_u64 > uint64_t(UINT32_MAX)
                ? UINT32_MAX
                : uint32_t(chunk_budget_splats_u64);
        current_frame_safe_max = MIN(current_frame_safe_max, chunk_budget_splats);
    }

    // Bootstrap with one synchronous sample so startup/reset frames do not
    // publish zero visible splats before async callbacks arrive.
    if (!last_instance_visible_splat_count_valid &&
            !instance_count_readback_state.bootstrap_sync_attempted &&
            readback_policy.allow_sync_bootstrap &&
            compute_rd &&
            inputs.instance_count_buffer.is_valid()) {
        instance_count_readback_state.bootstrap_sync_attempted = true;
        performance_state.metrics.instance_sort_sync_fallback_count++;
        Vector<uint8_t> bootstrap_count_data = compute_rd->buffer_get_data(
                inputs.instance_count_buffer, 0, sizeof(GaussianSplatting::IndirectDispatchLayout));
        if (bootstrap_count_data.size() >= static_cast<int>(sizeof(GaussianSplatting::IndirectDispatchLayout))) {
            const auto *indirect =
                    reinterpret_cast<const GaussianSplatting::IndirectDispatchLayout *>(bootstrap_count_data.ptr());
            last_instance_visible_splat_count = indirect->element_count;
            last_instance_visible_splat_count_valid = true;
            last_instance_visible_splat_count_frame = frame_state.frame_counter;
        }
    }

    const bool stale_pending_readback =
            instance_count_readback_state.pending &&
            frame_state.frame_counter >= instance_count_readback_state.pending_frame_counter &&
            pending_count_age_frames > kInstanceCountAsyncMaxAgeFrames;
    if (stale_pending_readback) {
        // Once pending async readback is stale, recover immediately with a sync snapshot
        // so visibility changes are not masked until the much later hard-reset window.
        if (compute_rd && inputs.instance_count_buffer.is_valid() && readback_policy.allow_sync_pending_readback) {
            performance_state.metrics.instance_sort_sync_fallback_count++;
            const uint32_t stale_age_frames = pending_count_age_frames;
            // Invalidate the stale async request so its callback cannot overwrite the
            // refreshed count and so we can immediately enqueue a fresh async readback.
            instance_count_readback_state.pending = false;
            instance_count_readback_state.pending_frame_counter = 0;
            instance_count_readback_state.generation++;

            Vector<uint8_t> stale_count_data = compute_rd->buffer_get_data(
                    inputs.instance_count_buffer, 0, sizeof(GaussianSplatting::IndirectDispatchLayout));
            if (stale_count_data.size() >= static_cast<int>(sizeof(GaussianSplatting::IndirectDispatchLayout))) {
                const auto *indirect =
                        reinterpret_cast<const GaussianSplatting::IndirectDispatchLayout *>(stale_count_data.ptr());
                last_instance_visible_splat_count = indirect->element_count;
                last_instance_visible_splat_count_valid = true;
                last_instance_visible_splat_count_frame = frame_state.frame_counter;
            } else {
                static uint32_t stale_readback_log_counter = 0;
                stale_readback_log_counter++;
                if (stale_readback_log_counter == 1 || (stale_readback_log_counter % 120u) == 0u) {
                    GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Stale instance-count async readback recovery failed (age_frames=%d, bytes=%d)",
                            int(stale_age_frames),
                            stale_count_data.size()));
                }
            }
        } else if (pending_count_age_frames > kInstanceCountAsyncResetAgeFrames) {
            // Extremely stale async requests are dropped to re-arm a fresh async readback
            // without introducing a blocking sync in strict async modes.
            instance_count_readback_state.pending = false;
            instance_count_readback_state.pending_frame_counter = 0;
            instance_count_readback_state.generation++;
            static uint32_t stale_reset_log_counter = 0;
            stale_reset_log_counter++;
            if (stale_reset_log_counter == 1 || (stale_reset_log_counter % 240u) == 0u) {
                GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Dropping stale async count readback without sync (age_frames=%d, policy=%s)",
                        int(pending_count_age_frames),
                        gs_sort_policy::mode_name(readback_policy.mode)));
            }
        }
    }

    uint32_t resolved_visible = 0;
    if (last_instance_visible_splat_count_valid) {
        resolved_visible = MIN(last_instance_visible_splat_count, current_frame_safe_max);
    }

    if (compute_rd && inputs.instance_count_buffer.is_valid() && !instance_count_readback_state.pending) {
        instance_count_readback_state.pending = true;
        instance_count_readback_state.generation++;
        instance_count_readback_state.pending_frame_counter = frame_state.frame_counter;
        const int64_t readback_generation = int64_t(instance_count_readback_state.generation);
        Callable count_callback =
                callable_mp(this, &GPUSortingPipeline::_on_instance_count_readback).bind(readback_generation);
        Error count_err = compute_rd->buffer_get_data_async(inputs.instance_count_buffer, count_callback, 0,
                sizeof(GaussianSplatting::IndirectDispatchLayout));
        if (count_err != OK) {
            instance_count_readback_state.pending = false;
            instance_count_readback_state.pending_frame_counter = 0;
            bool enqueue_sync_fallback_used = false;
            if (compute_rd && inputs.instance_count_buffer.is_valid() && readback_policy.allow_sync_enqueue_fallback) {
                // Preserve count freshness when async enqueue fails, while honoring strict
                // async/timestamp-preserving policy modes that disallow blocking fallbacks.
                performance_state.metrics.instance_sort_sync_fallback_count++;
                Vector<uint8_t> sync_count_data = compute_rd->buffer_get_data(
                        inputs.instance_count_buffer, 0, sizeof(GaussianSplatting::IndirectDispatchLayout));
                if (sync_count_data.size() >= static_cast<int>(sizeof(GaussianSplatting::IndirectDispatchLayout))) {
                    const auto *indirect =
                            reinterpret_cast<const GaussianSplatting::IndirectDispatchLayout *>(sync_count_data.ptr());
                    last_instance_visible_splat_count = indirect->element_count;
                    last_instance_visible_splat_count_valid = true;
                    last_instance_visible_splat_count_frame = frame_state.frame_counter;
                    resolved_visible = MIN(indirect->element_count, current_frame_safe_max);
                    enqueue_sync_fallback_used = true;
                }
            }
            GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Instance pipeline async count readback enqueue failed: err=%d", int(count_err)));
            if (trace_enabled) {
                GaussianSplatting::debug_trace_record_event("sort",
                        vformat("InstancePipeline async count readback failed: err=%d policy=%s sync_fallback=%s visible=%d",
                                int(count_err),
                                gs_sort_policy::mode_name(readback_policy.mode),
                                enqueue_sync_fallback_used ? "YES" : "NO",
                                int(resolved_visible)),
                        true);
            }
        } else if (!compute_rd->is_main_rendering_device()) {
            // Ensure async readback requests are flushed on worker rendering devices.
            compute_rd->submit();
        }
    }
    sorting_state.sorted_splat_count = resolved_visible;
    frame_state.visible_splat_count.store(resolved_visible, std::memory_order_release);
    if (trace_enabled) {
        GaussianSplatting::debug_trace_record_event("sort",
                vformat("InstancePipeline COMPLETE: sorted_budget=%d",
                        sorting_state.sorted_splat_count),
                false);
    }
    frame_state.sort_time_ms = sort_result.gpu_time_ms;
    performance_state.metrics.sort_submission_time_ms = sort_result.gpu_time_ms;
    performance_state.metrics.sort_wait_time_ms = 0.0f;
    performance_state.metrics.async_sort_used = true;
    performance_state.metrics.async_sort_waited = false;
    performance_state.metrics.async_overlap_efficiency = 0.0f;
    last_compute_error = String();
    return true;
}

bool GPUSortingPipeline::sort_gaussians_gpu(GaussianSplatRenderer *p_renderer, const Transform3D &p_cam_transform) {
    if (!p_renderer) {
        return false;
    }

    GaussianSplatRenderer &renderer = *p_renderer;
    if (GaussianSplatting::debug_trace_is_enabled()) {
        GaussianSplatting::debug_trace_record_event("sort",
                vformat("SortGaussiansGPU ENTER: inst_pipe=YES inst_inputs_valid=%s",
                        instance_inputs_valid ? "YES" : "NO"),
                false);
    }
    if (instance_inputs_valid) {
        if (_sort_instance_pipeline(renderer, p_cam_transform)) {
            return true;
        }
    }
    // Legacy GPU depth sort shader path has been retired alongside instance-only compute shaders.
    // Keep the legacy CPU sort fallback in RenderSortingOrchestrator.
    return false;
}

SortOperationResult GPUSortingPipeline::sort(const SortOperationParams &p_params) {
    SortOperationResult result;

    if (!gpu_sorter.is_valid()) {
        result.error_code = SortOperationErrorCode::SORTER_NOT_INITIALIZED;
        result.fallback_policy = _fallback_policy_for_error(result.error_code);
        result.error = "Sorter not initialized";
        return result;
    }

    if (!p_params.keys_buffer.is_valid() || !p_params.values_buffer.is_valid()) {
        result.error_code = !p_params.keys_buffer.is_valid()
                ? SortOperationErrorCode::INVALID_KEYS_BUFFER
                : SortOperationErrorCode::INVALID_VALUES_BUFFER;
        result.fallback_policy = _fallback_policy_for_error(result.error_code);
        result.error = "Invalid buffer handles";
        return result;
    }

    Error err = OK;
    if (p_params.count_buffer.is_valid()) {
        if (!gpu_sorter->supports_indirect()) {
            result.error_code = SortOperationErrorCode::UNSUPPORTED_INDIRECT;
            result.fallback_policy = _fallback_policy_for_error(result.error_code);
            result.error = "Sorter does not support indirect count";
            return result;
        }
        err = gpu_sorter->sort_indirect(p_params.keys_buffer, p_params.values_buffer, p_params.count_buffer);
    } else {
        if (p_params.element_count == 0) {
            result.error_code = SortOperationErrorCode::INVALID_ELEMENT_COUNT;
            result.fallback_policy = _fallback_policy_for_error(result.error_code);
            result.error = "Invalid element count";
            return result;
        }
        err = gpu_sorter->sort(p_params.keys_buffer, p_params.values_buffer, p_params.element_count);
    }
    if (err != OK) {
        result.error_code = _map_preflight_error(gpu_sorter->get_last_preflight_error());
        if (result.error_code == SortOperationErrorCode::NONE) {
            result.error_code = SortOperationErrorCode::SORT_SUBMISSION_FAILED;
        }
        result.fallback_policy = _fallback_policy_for_error(result.error_code);
        result.error = vformat("Sort failed with error %d", (int)err);
        return result;
    }

    gpu_sorter->wait_for_completion();
    result.success = true;
    result.error_code = SortOperationErrorCode::NONE;
    result.fallback_policy = SortRendererFallbackPolicy::NONE;
    result.gpu_time_ms = gpu_sorter->get_last_sort_time_ms();
    return result;
}

SortOperationResult GPUSortingPipeline::sort_async(const SortOperationParams &p_params) {
    SortOperationResult result;

    if (!gpu_sorter.is_valid()) {
        result.error_code = SortOperationErrorCode::SORTER_NOT_INITIALIZED;
        result.fallback_policy = _fallback_policy_for_error(result.error_code);
        result.error = "Sorter not initialized";
        return result;
    }

    if (!p_params.keys_buffer.is_valid() || !p_params.values_buffer.is_valid()) {
        result.error_code = !p_params.keys_buffer.is_valid()
                ? SortOperationErrorCode::INVALID_KEYS_BUFFER
                : SortOperationErrorCode::INVALID_VALUES_BUFFER;
        result.fallback_policy = _fallback_policy_for_error(result.error_code);
        result.error = "Invalid buffer handles";
        return result;
    }

    sorting_in_progress = true;
    if (p_params.count_buffer.is_valid()) {
        if (!gpu_sorter->supports_indirect()) {
            result.error_code = SortOperationErrorCode::UNSUPPORTED_INDIRECT;
            result.fallback_policy = _fallback_policy_for_error(result.error_code);
            result.error = "Sorter does not support indirect count";
            sorting_in_progress = false;
            return result;
        }
        result.timeline_value = gpu_sorter->sort_indirect_async(p_params.keys_buffer, p_params.values_buffer, p_params.count_buffer);
    } else {
        if (p_params.element_count == 0) {
            result.error_code = SortOperationErrorCode::INVALID_ELEMENT_COUNT;
            result.fallback_policy = _fallback_policy_for_error(result.error_code);
            result.error = "Invalid element count";
            sorting_in_progress = false;
            return result;
        }
        result.timeline_value = gpu_sorter->sort_async(p_params.keys_buffer, p_params.values_buffer, p_params.element_count);
    }
    current_sort_timeline_value = result.timeline_value;
    last_sort_submission_value = result.timeline_value;

    if (result.timeline_value == 0) {
        result.error_code = _map_preflight_error(gpu_sorter->get_last_preflight_error());
        if (result.error_code == SortOperationErrorCode::NONE) {
            result.error_code = SortOperationErrorCode::SORT_SUBMISSION_FAILED;
        }
        result.fallback_policy = _fallback_policy_for_error(result.error_code);
        result.error = "Async sort submission did not produce a valid timeline value";
        sorting_in_progress = false;
        return result;
    }

    result.success = true;
    result.error_code = SortOperationErrorCode::NONE;
    result.fallback_policy = SortRendererFallbackPolicy::NONE;
    result.gpu_time_ms = gpu_sorter->get_last_sort_time_ms();
    return result;
}

void GPUSortingPipeline::wait_for_completion() {
    if (gpu_sorter.is_valid()) {
        gpu_sorter->wait_for_completion();
    }
    sorting_in_progress = false;
}

float GPUSortingPipeline::get_last_sort_time_ms() const {
    if (gpu_sorter.is_valid()) {
        return gpu_sorter->get_last_sort_time_ms();
    }
    return 0.0f;
}

SortingMetrics GPUSortingPipeline::get_metrics() const {
    if (gpu_sorter.is_valid()) {
        return gpu_sorter->get_metrics();
    }
    return SortingMetrics();
}

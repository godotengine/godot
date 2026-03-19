/**
 * GPU Radix Sort Implementation for Gaussian Splatting
 * =====================================================
 *
 * ALGORITHM OVERVIEW
 * ------------------
 * This file implements GPU-accelerated radix sorting for depth-ordering Gaussian splats.
 * Gaussian splatting requires back-to-front rendering order for correct alpha blending,
 * making efficient sorting critical for real-time performance.
 *
 * RADIX SORT ALGORITHM
 * --------------------
 * The implementation uses a multi-pass radix sort processing 8-bit digit groups:
 *
 * 1. HISTOGRAM PASS: Count occurrences of each digit value (0-255) across all keys.
 *    Each workgroup builds a local histogram, then atomically accumulates to global.
 *
 * 2. WORKGROUP PREFIX PASS: Compute exclusive prefix sums within each workgroup histogram.
 *    This determines where each workgroup elements should be placed in the output.
 *
 * 3. BIN PREFIX PASS: Compute global prefix sums across all bins (digit values).
 *    Single-workgroup dispatch that scans across the 256 histogram bins.
 *
 * 4. SCATTER PASS: Redistribute elements to their sorted positions.
 *    Each thread reads its key, extracts the current digit, looks up the destination
 *    offset from the prefix sums, and writes to the output buffer.
 *
 * For 64-bit sort keys (tile+depth composite keys), 8 passes are required.
 *
 * WHY 8-BIT DIGITS?
 * -----------------
 * - 256 bins fit efficiently in shared memory for histogram accumulation
 * - Good balance between pass count and memory bandwidth
 * - Matches GPU architecture for coalesced memory access
 *
 * SORT KEY COMPOSITION
 * --------------------
 * Sort keys encode both tile ID and depth for tile-based rasterization:
 *   - High bits: Tile ID (configurable via tile_bits)
 *   - Low bits: Depth value (configurable via depth_bits)
 *
 * BUFFER LAYOUT
 * -------------
 * - keys_buffer: Input/output sort keys (ping-pong between passes)
 * - values_buffer: Payload indices (sorted alongside keys)
 * - histogram_buffer: Per-workgroup histograms for all passes
 * - indirect_count_buffer: Dispatch parameters for GPU-driven execution
 *
 * ALGORITHM VARIANTS
 * ------------------
 * - RadixSort: Standard multi-pass radix sort (default, most compatible)
 * - OneSweepSort: Single-pass variant for smaller datasets
 * - BitonicSort: Comparison-based fallback for very small arrays
 *
 * PERFORMANCE: Workgroup size tuned for GPU occupancy, shared memory histograms,
 * ping-pong buffers, and async execution via timeline semaphores.
 */

#include "gpu_sorter.h"
#include "gpu_sorting_config.h"
#include "pipeline_io_contracts.h"
#include "gpu_debug_utils.h"
#include "resource_owner_mismatch_contract.h"
#include "core/error/error_macros.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "servers/rendering/rendering_device.h"

#include "../core/gaussian_splat_manager.h"
#include "../logger/gs_logger.h"
#include "../interfaces/sync_policy.h"

using GaussianSplatting::PassColors;
using GaussianSplatting::ScopedGpuMarker;
using GaussianSplatting::ScopedGpuMarkerEx;
#include <algorithm>
#include <cstdint>

#ifndef VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
#define VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT 0x00000800
#endif

using ComputeListID = RenderingDevice::ComputeListID;

namespace {

static RenderingDevice *_acquire_submission_device(RenderingDevice *candidate, GaussianSplatManager::ScopedSubmissionLock &lock) {
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        return manager->acquire_submission_device(candidate, lock);
    }
    WARN_PRINT_ONCE("[GPU Sort] GaussianSplatManager unavailable; cannot acquire submission device");
    return nullptr;
}

static void _free_uniform_sets(RenderingDevice *p_owner, LocalVector<RID> &p_sets) {
    if (!p_owner) {
        p_sets.clear();
        return;
    }
    for (const RID &rid : p_sets) {
        if (rid.is_valid() && p_owner->uniform_set_is_valid(rid)) {
            p_owner->free(rid);
        }
    }
    p_sets.clear();
}

} // namespace

static const char *_sort_preflight_error_name(SortPreflightError p_error) {
    switch (p_error) {
        case SortPreflightError::NONE:
            return "none";
        case SortPreflightError::INVALID_KEYS_BUFFER:
            return "invalid_keys_buffer";
        case SortPreflightError::INVALID_VALUES_BUFFER:
            return "invalid_values_buffer";
        case SortPreflightError::INVALID_COUNT_BUFFER:
            return "invalid_count_buffer";
        case SortPreflightError::INVALID_ELEMENT_COUNT:
            return "invalid_element_count";
        case SortPreflightError::ELEMENT_COUNT_EXCEEDS_CAPACITY:
            return "element_count_exceeds_capacity";
        case SortPreflightError::UNSUPPORTED_KEY_FORMAT:
            return "unsupported_key_format";
        case SortPreflightError::RESOURCE_DEVICE_UNAVAILABLE:
            return "resource_device_unavailable";
        case SortPreflightError::SUBMISSION_DEVICE_UNAVAILABLE:
            return "submission_device_unavailable";
        default:
            return "unknown";
    }
}

// Static helper function for creating compute shaders with new API
static RID create_compute_shader_from_spirv(RenderingDevice *rd, const String &source) {
    ERR_FAIL_NULL_V(rd, RID());

    String compile_error;
    Vector<uint8_t> spirv_data = rd->shader_compile_spirv_from_source(
            RD::SHADER_STAGE_COMPUTE, source, RenderingDevice::SHADER_LANGUAGE_GLSL, &compile_error);
    ERR_FAIL_COND_V_MSG(spirv_data.is_empty(), RID(), compile_error.is_empty() ? "Failed to compile compute shader source" : compile_error);

    // Create ShaderStageSPIRVData vector for new API
    Vector<RenderingDevice::ShaderStageSPIRVData> spirv_stages;
    RenderingDevice::ShaderStageSPIRVData stage_data;
    stage_data.shader_stage = RD::SHADER_STAGE_COMPUTE;
    stage_data.spirv = spirv_data;
    spirv_stages.push_back(stage_data);

    return rd->shader_create_from_spirv(spirv_stages);
}

static RenderingDevice *get_submission_device() {
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (manager) {
        return manager->get_shared_submission_device();
    }
    WARN_PRINT_ONCE("[GPU Sort] GaussianSplatManager unavailable; no shared submission device available");
    return nullptr;
}

static Error _init_sorter_devices(RenderingDevice *p_rd, RenderingDevice *&r_local_rd, RenderingDevice *&r_resource_rd) {
    r_local_rd = get_submission_device();
    if (!r_local_rd) {
        return ERR_CANT_CREATE;
    }
    r_resource_rd = p_rd;
    if (!r_resource_rd) {
        return ERR_CANT_CREATE;
    }
    return OK;
}

template <typename VariantT>
static void _record_wg_prefix_pass(RenderingDevice *command_rd, ComputeListID compute_list, const VariantT *variant,
        RID uniform_set, uint32_t histogram_offset, uint32_t workgroup_stride, uint32_t bin_offset, const char *p_label) {
    ScopedGpuMarkerEx prefix_marker(command_rd, p_label, PassColors::SORTING);
    command_rd->compute_list_bind_compute_pipeline(compute_list, variant->wg_prefix_pipeline);
    command_rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

    struct WgPrefixParams {
        uint32_t histogram_offset;
        uint32_t workgroup_stride;
        uint32_t bin_offset;
        uint32_t pad0;
    } wg_prefix_params = { histogram_offset, workgroup_stride, bin_offset, 0u };

    command_rd->compute_list_set_push_constant(compute_list, &wg_prefix_params, sizeof(wg_prefix_params));
    command_rd->compute_list_dispatch(compute_list, variant->radix_size, 1, 1);
}

template <typename VariantT>
static void _record_bin_prefix_pass(RenderingDevice *command_rd, ComputeListID compute_list, const VariantT *variant,
        RID uniform_set, uint32_t bin_offset, const char *p_label) {
    ScopedGpuMarkerEx bin_prefix_marker(command_rd, p_label, PassColors::SORTING);
    command_rd->compute_list_bind_compute_pipeline(compute_list, variant->bin_prefix_pipeline);
    command_rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

    struct BinPrefixParams {
        uint32_t bin_offset;
        uint32_t pad0;
        uint32_t pad1;
        uint32_t pad2;
    } bin_prefix_params = { bin_offset, 0, 0, 0 };

    command_rd->compute_list_set_push_constant(compute_list, &bin_prefix_params, sizeof(bin_prefix_params));
    command_rd->compute_list_dispatch(compute_list, 1, 1, 1);
}

template <typename VariantT>
static void _record_scatter_pass(RenderingDevice *command_rd, ComputeListID compute_list, const VariantT *variant,
        RID uniform_set, uint32_t bit_shift, uint32_t histogram_offset, uint32_t workgroup_stride, uint32_t bin_offset,
        uint32_t workgroups, bool p_use_indirect_dispatch, RID p_dispatch_args_buffer, const char *p_label) {
    ScopedGpuMarkerEx scatter_marker(command_rd, p_label, PassColors::SORTING);
    command_rd->compute_list_bind_compute_pipeline(compute_list, variant->scatter_pipeline);
    command_rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

    // Struct padded to 32 bytes for SPIR-V alignment
    struct ScatterParams {
        uint32_t bit_shift;
        uint32_t histogram_offset;
        uint32_t workgroup_stride;
        uint32_t bin_offset;
        uint32_t wg_prefix_offset;
        uint32_t workgroup_count;
        uint32_t _pad0;
        uint32_t _pad1;
    } scatter_params = { bit_shift, histogram_offset, workgroup_stride, bin_offset, histogram_offset, workgroups, 0, 0 };

    command_rd->compute_list_set_push_constant(compute_list, &scatter_params, sizeof(scatter_params));
    if (p_use_indirect_dispatch) {
        command_rd->compute_list_dispatch_indirect(compute_list, p_dispatch_args_buffer, 0);
    } else {
        command_rd->compute_list_dispatch(compute_list, workgroups, 1, 1);
    }
}

static bool _device_is_active(RenderingDevice *device) {
    if (!device) {
        return false;
    }
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        RenderingDevice *shared = manager->get_shared_submission_device();
        if (shared && shared == device) {
            return true;
        }
        RenderingDevice *primary = manager->get_primary_rendering_device();
        if (primary && primary == device) {
            return true;
        }
    }
    if (RenderingDevice *singleton = RenderingDevice::get_singleton()) {
        return singleton == device;
    }
    // During shutdown both singletons may already be destroyed while the
    // device pointer is still alive.  Trust the stored pointer so that
    // shutdown() can free its GPU resources instead of leaking them.
    return true;
}

// ISSUE-010: Generation-safe variant that validates the device pointer hasn't
// gone stale before attempting to free uniform sets on it.
static void _free_uniform_sets_safe(RenderingDevice *p_owner, uint64_t p_owner_generation,
        RenderingDevice *p_fallback, LocalVector<RID> &p_sets) {
    if (p_sets.is_empty()) {
        return;
    }
    // Validate stored owner is still the same device instance.
    RenderingDevice *effective_owner = nullptr;
    if (p_owner && ResourceOwnerMismatchContract::is_device_generation_valid(p_owner, p_owner_generation)) {
        effective_owner = p_owner;
    } else if (p_fallback && _device_is_active(p_fallback)) {
        effective_owner = p_fallback;
    }
    // If no valid owner available, just clear RIDs without freeing.
    // The owning device already cleaned up its resources on destruction.
    _free_uniform_sets(effective_owner, p_sets);
}

static constexpr uint32_t AUTO_SMALL_ELEMENT_THRESHOLD = 32768u;
static constexpr uint32_t AUTO_LARGE_ELEMENT_THRESHOLD = 1048576u;
static constexpr uint32_t MIN_STORAGE_BUFFERS_PER_SET = 7u;
static constexpr uint32_t MIN_BOUND_UNIFORM_SETS = 1u;

struct ComputeCapabilityProbe {
    bool valid = false;
    uint64_t workgroup_size_x = 0;
    uint64_t workgroup_invocations = 0;
    uint64_t workgroup_count_x = 0;
    uint64_t storage_buffers_per_set = 0;
    uint64_t bound_uniform_sets = 0;
    uint64_t shared_memory_bytes = 0;
    uint64_t subgroup_ops = 0;
    uint64_t subgroup_stages = 0;
};

struct AlgorithmProbe {
    bool supported = false;
    bool supports_indirect = false;
    SorterCapabilities capabilities;
};

static AlgorithmProbe _probe_algorithm(GPUSorterFactory::SortingAlgorithm algorithm, RenderingDevice *rd);

static ComputeCapabilityProbe _probe_compute_capabilities(RenderingDevice *rd) {
    ComputeCapabilityProbe probe;
    if (!rd) {
        return probe;
    }

    probe.valid = true;
    probe.workgroup_size_x = rd->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X);
    probe.workgroup_invocations = rd->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS);
    probe.workgroup_count_x = rd->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);
    probe.storage_buffers_per_set = rd->limit_get(RenderingDevice::LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET);
    probe.bound_uniform_sets = rd->limit_get(RenderingDevice::LIMIT_MAX_BOUND_UNIFORM_SETS);
    probe.shared_memory_bytes = rd->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE);
    probe.subgroup_ops = rd->limit_get(RenderingDevice::LIMIT_SUBGROUP_OPERATIONS);
    probe.subgroup_stages = rd->limit_get(RenderingDevice::LIMIT_SUBGROUP_IN_SHADERS);
    return probe;
}

static bool _subgroup_prefix_forced_off() {
    return g_gpu_sorting_config.subgroup_prefix_mode == GPUSortingConfig::SUBGROUP_PREFIX_FORCE_OFF;
}

static bool _supports_required_subgroups(const ComputeCapabilityProbe &probe) {
    if (_subgroup_prefix_forced_off()) {
        return false;
    }
    bool has_basic = (probe.subgroup_ops & RenderingDevice::SUBGROUP_BASIC_BIT) != 0;
    bool has_ballot = (probe.subgroup_ops & RenderingDevice::SUBGROUP_BALLOT_BIT) != 0;
    bool has_compute_stage = (probe.subgroup_stages & RenderingDevice::SHADER_STAGE_COMPUTE_BIT) != 0;
    return has_basic && has_ballot && has_compute_stage;
}

static bool _supports_compute_profile(const ComputeCapabilityProbe &probe, uint32_t required_workgroup_size,
        uint32_t required_storage_buffers, uint32_t required_shared_memory_bytes) {
    if (!probe.valid) {
        return false;
    }
    if (probe.workgroup_size_x == 0 || probe.workgroup_invocations == 0 || probe.workgroup_count_x == 0) {
        return false;
    }
    if (probe.storage_buffers_per_set < required_storage_buffers || probe.bound_uniform_sets < MIN_BOUND_UNIFORM_SETS) {
        return false;
    }
    if (probe.shared_memory_bytes < required_shared_memory_bytes) {
        return false;
    }
    return probe.workgroup_size_x >= required_workgroup_size && probe.workgroup_invocations >= required_workgroup_size;
}

static GPUSorterFactory::PolicyProbe _to_policy_probe(const AlgorithmProbe &probe) {
    GPUSorterFactory::PolicyProbe policy_probe;
    policy_probe.supported = probe.supported;
    policy_probe.supports_indirect = probe.supports_indirect;
    policy_probe.supports_64bit_keys = probe.capabilities.supports_64bit_keys;
    return policy_probe;
}

static bool _algorithm_meets_requirements(const GPUSorterFactory::PolicyProbe &probe, bool require_indirect, bool require_64bit_keys) {
    if (!probe.supported) {
        return false;
    }
    if (require_indirect && !probe.supports_indirect) {
        return false;
    }
    if (require_64bit_keys && !probe.supports_64bit_keys) {
        return false;
    }
    return true;
}

static GPUSorterFactory::SortingAlgorithm _preferred_auto_algorithm(uint32_t element_count, const SortKeyConfig &key_config) {
    if (key_config.require_stable || key_config.key_bits > 32 || key_config.enable_tie_breaker) {
        return GPUSorterFactory::ALGORITHM_RADIX;
    }
    if (element_count <= AUTO_SMALL_ELEMENT_THRESHOLD) {
        return GPUSorterFactory::ALGORITHM_BITONIC;
    }
    if (element_count >= AUTO_LARGE_ELEMENT_THRESHOLD) {
        return GPUSorterFactory::ALGORITHM_ONESWEEP;
    }
    return GPUSorterFactory::ALGORITHM_RADIX;
}

static String _failure_reason_for_algorithm(const char *label, const GPUSorterFactory::PolicyProbe &probe,
        bool require_indirect, bool require_64bit_keys) {
    if (!probe.supported) {
        return vformat("algorithm=%s reason=unsupported", label);
    }
    if (require_indirect && !probe.supports_indirect) {
        return vformat("algorithm=%s reason=missing_indirect", label);
    }
    if (require_64bit_keys && !probe.supports_64bit_keys) {
        return vformat("algorithm=%s reason=missing_64bit", label);
    }
    return vformat("algorithm=%s reason=unknown", label);
}

static String _join_policy_fallback_reason(const char *preferred_label, const String &preferred_failure,
        const char *selected_label) {
    return vformat("type=auto preferred=%s selected=%s failure={%s}", preferred_label, selected_label, preferred_failure);
}

SortKeyConfig SortKeyConfig::from_settings() {
    static constexpr uint32_t MIN_DEPTH_BITS = 8;

    SortKeyConfig cfg;
    // Instance pipeline requires 64-bit sort keys.
    cfg.key_bits = 64;
    cfg.tile_bits = g_gpu_sorting_config.tile_bits;
    cfg.depth_bits = g_gpu_sorting_config.depth_bits;
    cfg.enable_tie_breaker = g_gpu_sorting_config.enable_tie_breaker;
    // Clamp to sane defaults if project settings are misconfigured.
    if (cfg.tile_bits > cfg.key_bits) {
        cfg.tile_bits = cfg.key_bits;
    }
    if (cfg.depth_bits > cfg.key_bits) {
        cfg.depth_bits = cfg.key_bits;
    }
    if (cfg.tile_bits + cfg.depth_bits > cfg.key_bits) {
        if (cfg.tile_bits >= cfg.key_bits) {
            cfg.tile_bits = cfg.key_bits;
            cfg.depth_bits = 0;
        } else {
            cfg.depth_bits = cfg.key_bits - cfg.tile_bits;
        }
    }
    // Validate non-zero: tile_bits + depth_bits == 0 would leave the scatter
    // pass with undefined bit offsets. Ensure at least MIN_DEPTH_BITS depth
    // bits so that sorting produces a meaningful ordering.
    if (cfg.tile_bits + cfg.depth_bits == 0) {
        WARN_PRINT("SortKeyConfig: tile_bits + depth_bits == 0 is invalid; clamping depth_bits to minimum viable configuration.");
        cfg.depth_bits = MIN_DEPTH_BITS;
    }
    return cfg;
}

void SortingMetricsCollector::record_sort(uint32_t element_count, float time_ms, bool used_gpu) {
    (void)used_gpu;
    metrics.last_sort_time_ms = time_ms;
    metrics.total_sorts++;
    if (element_count > 0) {
        metrics.total_elements_sorted += element_count;
    }
    if (metrics.total_sorts == 1) {
        metrics.avg_sort_time_ms = time_ms;
    } else {
        metrics.avg_sort_time_ms = ((metrics.avg_sort_time_ms * (metrics.total_sorts - 1)) + time_ms) / metrics.total_sorts;
    }
    if (time_ms > metrics.peak_sort_time_ms) {
        metrics.peak_sort_time_ms = time_ms;
    }
}

void SortingMetricsCollector::record_async_sort(uint32_t element_count, float time_ms) {
    record_sort(element_count, time_ms, true);
    metrics.async_sorts++;
}

void SortingMetricsCollector::record_fallback(const String &reason) {
    String reason_label = reason.strip_edges();
    if (reason_label.is_empty()) {
        reason_label = "type=unknown reason=unspecified";
    }
    metrics.fallback_events++;
    metrics.last_fallback_reason = reason_label;
    int current_count = int(metrics.fallback_reason_counts.get(reason_label, 0));
    metrics.fallback_reason_counts[reason_label] = current_count + 1;
}

static bool device_supports_workgroup(RenderingDevice *rd, uint32_t required_size) {
    const ComputeCapabilityProbe probe = _probe_compute_capabilities(rd);
    return _supports_compute_profile(probe, required_size, 1u, sizeof(uint32_t));
}

static GPUSorterFactory::SortingAlgorithm select_sort_algorithm(uint32_t element_count, const SortKeyConfig &key_config, RenderingDevice *rd = nullptr) {
    const AlgorithmProbe radix_probe = _probe_algorithm(GPUSorterFactory::ALGORITHM_RADIX, rd);
    const AlgorithmProbe bitonic_probe = _probe_algorithm(GPUSorterFactory::ALGORITHM_BITONIC, rd);
    const AlgorithmProbe onesweep_probe = _probe_algorithm(GPUSorterFactory::ALGORITHM_ONESWEEP, rd);

    GPUSorterFactory::PolicyDecision decision = GPUSorterFactory::evaluate_auto_policy(element_count, key_config,
            _to_policy_probe(radix_probe), _to_policy_probe(bitonic_probe), _to_policy_probe(onesweep_probe),
            false, key_config.key_bits > 32);
    if (!decision.fallback_reason.is_empty()) {
        GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] AUTO policy fallback: %s", decision.fallback_reason));
    }
    return decision.selected_algorithm;
}

static const char *_algorithm_name_label(GPUSorterFactory::SortingAlgorithm p_algorithm) {
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

// BitonicSort Implementation

BitonicSort::BitonicSort() {
    // Initialize RIDs to prevent garbage values that might pass is_valid() checks
    bitonic_shader = RID();
    bitonic_pipeline = RID();
    uniform_set = RID();
}

BitonicSort::~BitonicSort() {
    shutdown();
}

void BitonicSort::_bind_methods() {
    // Bind methods for GDScript if needed
}

uint32_t BitonicSort::next_power_of_two(uint32_t n) const {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

Error BitonicSort::initialize(RenderingDevice *p_rd, uint32_t p_max_elements) {
    ERR_FAIL_COND_V(p_rd == nullptr, ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(p_max_elements == 0, ERR_INVALID_PARAMETER);

    // Clean up any existing resources before re-initializing
    shutdown();

    rd = p_rd;

    local_rd = get_submission_device();
    ERR_FAIL_COND_V_MSG(local_rd == nullptr, ERR_CANT_CREATE, "Failed to acquire shared submission rendering device");
    max_elements = next_power_of_two(p_max_elements); // Bitonic needs power of 2

    RenderingDevice *command_rd = local_rd;
    ERR_FAIL_NULL_V_MSG(command_rd, ERR_CANT_CREATE, "Rendering device unavailable for BitonicSort initialization");

    // All shader, pipeline, and buffer resources must live on the same
    // RenderingDevice that records the compute work. Use the submission
    // device returned by GaussianSplatManager instead of the graphics device
    // pointer provided during construction.
    pipeline_device = command_rd;
    resource_device = command_rd;
    resource_device_generation = command_rd->get_device_instance_id();

    if (!device_supports_workgroup(command_rd, WORKGROUP_SIZE)) {
        GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] BitonicSort requires %d-thread workgroups; falling back to CPU sorter",
                WORKGROUP_SIZE));
        return ERR_UNAVAILABLE;
    }

    // Load compute shader using helper function
    String bitonic_source = vformat(R"(
#version 450

layout(local_size_x = %d) in;

layout(set = 0, binding = 0, std430) restrict buffer KeyBuffer {
    float keys[];
} key_buffer;

layout(set = 0, binding = 1, std430) restrict buffer ValueBuffer {
    uint values[];
} value_buffer;

layout(push_constant, std430) uniform Params {
    uint stage;
    uint pass_in_stage;
    uint num_elements;
    uint block_size;
} params;

void compare_and_swap(uint i, uint j, bool ascending) {
    float key_i = key_buffer.keys[i];
    float key_j = key_buffer.keys[j];
    uint val_i = value_buffer.values[i];
    uint val_j = value_buffer.values[j];

    bool should_swap = ascending ? (key_i > key_j) : (key_i < key_j);

    key_buffer.keys[i] = should_swap ? key_j : key_i;
    key_buffer.keys[j] = should_swap ? key_i : key_j;
    value_buffer.values[i] = should_swap ? val_j : val_i;
    value_buffer.values[j] = should_swap ? val_i : val_j;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;

    uint half_block_size = params.block_size >> 1;
    uint block_id = gid / half_block_size;
    uint thread_in_block = gid %% half_block_size;

    bool ascending = ((block_id & 1) == 0);

    uint stride = 1u << (params.stage - params.pass_in_stage);
    uint offset = thread_in_block * stride * 2;
    uint block_start = (block_id / 2) * params.block_size;

    uint i = block_start + offset;
    uint j = i + stride;

    if (j >= params.num_elements) return;

    compare_and_swap(i, j, ascending);
}
    )",
            WORKGROUP_SIZE);
    bitonic_shader = create_compute_shader_from_spirv(resource_device, bitonic_source);
    ERR_FAIL_COND_V(!bitonic_shader.is_valid(), ERR_CANT_CREATE);
    
    // Create compute pipeline
    bitonic_pipeline = resource_device->compute_pipeline_create(bitonic_shader);
    ERR_FAIL_COND_V(!bitonic_pipeline.is_valid(), ERR_CANT_CREATE);

    // Timeline synchronization is driven by timeline_value; no device-level semaphore setup required.
    return OK;
}

// ======================================================================
// WORK BATCHING OPTIMIZATION (Issue #108)
// ======================================================================

bool BitonicSort::_requires_synchronization(uint32_t stage, uint32_t pass, uint32_t num_elements) const {
    // Determine if synchronization is required between passes
    // Strategy: Only sync when data dependencies require it

    // Always sync on final pass of each stage
    if (pass == 1) {
        return true;
    }

    // Sync for large workloads to prevent GPU timeout
    if (num_elements > 2000000) { // > 2M elements
        return true;
    }

    // Sync at power-of-2 intervals for memory coherency
    uint32_t stage_power = 1u << stage;
    if ((pass & (stage_power - 1)) == 0) {
        return true;
    }

    // For smaller workloads, batch more aggressively
    return false;
}

void BitonicSort::shutdown() {
    // Check if resource_device is still valid by comparing with the current shared device
    RenderingDevice *current_shared = get_submission_device();
    // During shutdown the manager singleton may be gone; trust resource_device.
    bool device_still_valid = (resource_device != nullptr) &&
            (current_shared == nullptr || resource_device == current_shared);

    // ISSUE-010: Also validate device generation to detect recycled/stale pointers.
    if (device_still_valid && resource_device_generation != 0) {
        device_still_valid = ResourceOwnerMismatchContract::is_device_generation_valid(
                resource_device, resource_device_generation);
    }

    if (device_still_valid) {
        // Device is still valid - properly free resources
#define SAFE_FREE(buf) \
    do { \
        if (buf.is_valid()) { \
            resource_device->free(buf); \
            buf = RID(); \
        } \
    } while (0)

        SAFE_FREE(bitonic_pipeline);
        SAFE_FREE(bitonic_shader);
        SAFE_FREE(uniform_set);

#undef SAFE_FREE
    } else {
        // Device may be gone or different - just invalidate RIDs without freeing
        // Resources will be cleaned up when the owning device is destroyed
        bitonic_shader = RID();
        bitonic_pipeline = RID();
        uniform_set = RID();
    }

    // Clear device pointers and generations
    uniform_owner = nullptr;
    uniform_owner_generation = 0;
    resource_device = nullptr;
    resource_device_generation = 0;
    pipeline_device = nullptr;
    local_rd = nullptr;
    rd = nullptr;
}
void BitonicSort::dispatch_bitonic_pass(RenderingDevice *p_rd, RenderingDevice::ComputeListID p_command_list, uint32_t stage, uint32_t pass, uint32_t num_elements) {
    ERR_FAIL_NULL(p_rd);

    BitonicParams params;
    params.stage = stage;
    params.pass_in_stage = pass;
    params.num_elements = num_elements;
    params.block_size = 1u << stage;

    // Update push constants
    p_rd->compute_list_set_push_constant(p_command_list, &params, sizeof(BitonicParams));

    // Calculate dispatch size
    uint32_t num_threads = (num_elements + 1) / 2; // Each thread handles a pair
    uint32_t num_groups = (num_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    
    // Dispatch compute
    p_rd->compute_list_dispatch(p_command_list, num_groups, 1, 1);
}

Error BitonicSort::sort(RID keys_buffer, RID values_buffer, uint32_t count) {
    ERR_FAIL_COND_V(!keys_buffer.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!values_buffer.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(count == 0 || count > max_elements, ERR_INVALID_PARAMETER);

    ERR_FAIL_NULL_V_MSG(pipeline_device, ERR_CANT_CREATE, "Pipeline rendering device unavailable for BitonicSort::sort");

    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *compute_rd = _acquire_submission_device(pipeline_device, submission_lock);
    // compute_rd is the submission RenderingDevice that records and submits
    // the bitonic sort compute list. All shader, pipeline, and uniform state
    // was created against this same device during initialization.
    ERR_FAIL_NULL_V_MSG(compute_rd, ERR_CANT_CREATE, "Rendering device unavailable for BitonicSort::sort");

    // Capture GPU timestamp at start (for actual GPU execution time when available)
    String timestamp_start = vformat("BitonicSort_%d_Start", metrics_collector.total_sorts());
    rd->capture_timestamp(timestamp_start);
    uint64_t cpu_start = OS::get_singleton()->get_ticks_usec();

    // CPU-side recording time as fallback if GPU timestamps are unavailable.
    start_cpu_record_timing();

    // Pad to power of 2 if needed
    uint32_t padded_count = next_power_of_two(count);
    
    // Create uniform set for buffers
    Vector<RD::Uniform> uniforms;
    
    RD::Uniform key_uniform;
    key_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    key_uniform.binding = 0;
    key_uniform.append_id(keys_buffer);
    uniforms.push_back(key_uniform);
    
    RD::Uniform value_uniform;
    value_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    value_uniform.binding = 1;
    value_uniform.append_id(values_buffer);
    uniforms.push_back(value_uniform);
    
    // Dont manually free uniform sets - they auto-free when buffer dependencies are freed
    // (Godot PR 103113). Manual frees cause "invalid ID" if sets were already auto-freed.
    uniform_set = RID();
    uniform_owner = nullptr;
    uniform_owner_generation = 0;
    RenderingDevice *uniform_device = resource_device;
    ERR_FAIL_NULL_V_MSG(uniform_device, ERR_CANT_CREATE, "Rendering device unavailable for BitonicSort uniform set");
    uniform_set = uniform_device->uniform_set_create(uniforms, bitonic_shader, 0);
    if (uniform_set.is_valid()) {
        uniform_device->set_resource_name(uniform_set, "GS_BitonicSort_UniformSet");
    }
    uniform_owner = uniform_device;
    uniform_owner_generation = uniform_device->get_device_instance_id();

    // Begin compute list
    ComputeListID compute_list = compute_rd->compute_list_begin();
    compute_rd->compute_list_bind_compute_pipeline(compute_list, bitonic_pipeline);
    compute_rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
    
    // Bitonic sort stages
    uint32_t num_stages = 0;
    uint32_t temp = padded_count;
    while (temp > 1) {
        temp >>= 1;
        num_stages++;
    }
    
    // Execute all stages and passes with work batching optimization (Issue #108)
    uint32_t dispatches_batched = 0;
    const uint32_t MAX_BATCH_DISPATCHES = 8; // Batch small dispatches

    for (uint32_t stage = 1; stage <= num_stages; stage++) {
        for (uint32_t pass = stage; pass > 0; pass--) {
            dispatch_bitonic_pass(compute_rd, compute_list, stage, pass, padded_count);
            dispatches_batched++;

            // Smart barrier insertion - reduce barriers for better GPU utilization
            bool needs_barrier = (pass > 1) &&
                               ((dispatches_batched % MAX_BATCH_DISPATCHES) == 0 ||
                                _requires_synchronization(stage, pass, padded_count));

            if (needs_barrier) {
                compute_rd->compute_list_add_barrier(compute_list);
                dispatches_batched = 0; // Reset batch counter
            }
        }

        // Barrier between stages (always required for correctness)
        if (stage < num_stages) {
            compute_rd->compute_list_add_barrier(compute_list);
            dispatches_batched = 0;
        }
    }
    
    // End compute list and submit
    compute_rd->compute_list_end();

    // Capture GPU timestamp at end
    String timestamp_end = vformat("BitonicSort_%d_End", metrics_collector.total_sorts());
    rd->capture_timestamp(timestamp_end);

    gs_device_utils::safe_submit(compute_rd);

    current_sort_value = ++timeline_value;
    is_sorting = false;
    float wait_time_ms = 0.0f;

    // Get actual GPU time from timestamps
    float gpu_time_ms = 0.0f;
    uint32_t timestamp_count = rd->get_captured_timestamps_count();
    if (timestamp_count >= 2) {
        uint64_t gpu_start = rd->get_captured_timestamp_gpu_time(timestamp_count - 2);
        uint64_t gpu_end = rd->get_captured_timestamp_gpu_time(timestamp_count - 1);
        gpu_time_ms = (gpu_end - gpu_start) / 1000000.0f;

        uint64_t cpu_end = OS::get_singleton()->get_ticks_usec();
        float cpu_time_ms = (cpu_end - cpu_start) / 1000.0f;

        GS_LOG_INFO_DEFAULT(vformat("[MEASURED] BitonicSort %d elements: GPU %.2fms, CPU %.2fms",
                          count, gpu_time_ms, cpu_time_ms));
    }

    // Prefer actual GPU timestamp when available; fall back to CPU recording time.
    // Note: end_cpu_record_timing() returns CPU wall-clock time, not GPU execution time.
    float sort_time = gpu_time_ms > 0.0f ? gpu_time_ms : end_cpu_record_timing();
    if (gpu_time_ms <= 0.0f) {
        sort_time = MAX(sort_time, wait_time_ms);
    }
    metrics_collector.record_sort(count, sort_time, true);
    
    // Calculate bandwidth utilization (rough estimate)
    // Each pass reads and writes all elements once
    // BitonicSort uses 64-bit keys (8 bytes) + 32-bit values (4 bytes) = 12 bytes per element
    uint32_t total_passes = (num_stages * (num_stages + 1)) / 2;
    constexpr uint32_t key_stride_bytes = 8; // 64-bit keys
    uint64_t bytes_transferred = uint64_t(count) * (uint64_t(key_stride_bytes) + sizeof(uint32_t)) * 2 * total_passes;
    float bandwidth_gbps = (bytes_transferred / 1e9) / (sort_time / 1000.0f);
    metrics_collector.set_bandwidth_utilization(bandwidth_gbps / 500.0f * 100.0f); // Assume 500 GB/s theoretical max
    
    is_sorting = false;
    
    return OK;
}

uint64_t BitonicSort::sort_async(RID keys_buffer, RID values_buffer, uint32_t count) {
    Error err = sort(keys_buffer, values_buffer, count);
    if (err != OK) {
        return 0;
    }

    metrics_collector.set_async_speedup(1.0f);
    current_sort_value = 0;
    is_sorting = false;
    return ++timeline_value;
}

bool BitonicSort::is_ready() const {
    return !is_sorting.load();
}

void BitonicSort::wait_for_completion() {
    if (!is_sorting.load()) {
        return;
    }

    if (local_rd) {
        GaussianSplatManager::ScopedSubmissionLock submission_lock;
        if (RenderingDevice *submission_rd = _acquire_submission_device(local_rd, submission_lock)) {
            gs_device_utils::safe_submit_and_sync(submission_rd);
        }
    }

    current_sort_value = 0;
    is_sorting = false;
}

// GPUSorterFactory Implementation

Ref<IGPUSorter> GPUSorterFactory::create_sorter(SortingAlgorithm algorithm, RenderingDevice *rd, uint32_t max_elements,
        const SortKeyConfig &p_key_config) {
    SortKeyConfig key_config = p_key_config;
    if (key_config.key_bits != 64) {
        key_config.key_bits = 64;
    }
    if (key_config.tile_bits > key_config.key_bits) {
        key_config.tile_bits = key_config.key_bits;
    }
    if (key_config.depth_bits > key_config.key_bits) {
        key_config.depth_bits = key_config.key_bits;
    }
    if (key_config.tile_bits + key_config.depth_bits > key_config.key_bits) {
        if (key_config.tile_bits >= key_config.key_bits) {
            key_config.tile_bits = key_config.key_bits;
            key_config.depth_bits = 0;
        } else {
            key_config.depth_bits = key_config.key_bits - key_config.tile_bits;
        }
    }

    const AlgorithmProbe radix_runtime_probe = _probe_algorithm(ALGORITHM_RADIX, rd);
    const AlgorithmProbe bitonic_runtime_probe = _probe_algorithm(ALGORITHM_BITONIC, rd);
    const AlgorithmProbe onesweep_runtime_probe = _probe_algorithm(ALGORITHM_ONESWEEP, rd);

    // The instance sorting pipeline always uses GPU-driven count buffers and 64-bit sort keys.
    // Force selection onto an algorithm that satisfies those capabilities.
    const bool requires_indirect = true;
    const bool requires_64bit_keys = key_config.key_bits > 32;

    SortingAlgorithm resolved_algorithm = algorithm;
    String fallback_reason;

    auto get_policy_probe = [&](SortingAlgorithm p_algorithm) -> PolicyProbe {
        switch (p_algorithm) {
            case ALGORITHM_RADIX:
                return _to_policy_probe(radix_runtime_probe);
            case ALGORITHM_BITONIC:
                return _to_policy_probe(bitonic_runtime_probe);
            case ALGORITHM_ONESWEEP:
                return _to_policy_probe(onesweep_runtime_probe);
            case ALGORITHM_AUTO:
            default:
                return PolicyProbe();
        }
    };

    if (algorithm == ALGORITHM_AUTO) {
        PolicyDecision auto_decision = evaluate_auto_policy(max_elements, key_config,
                get_policy_probe(ALGORITHM_RADIX), get_policy_probe(ALGORITHM_BITONIC), get_policy_probe(ALGORITHM_ONESWEEP),
                requires_indirect, requires_64bit_keys);
        resolved_algorithm = auto_decision.selected_algorithm;
        fallback_reason = auto_decision.fallback_reason;
        if (!fallback_reason.is_empty()) {
            GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] AUTO policy selected '%s' with fallback diagnostics: %s",
                    _algorithm_name_label(resolved_algorithm), fallback_reason));
        }
    }

    auto algorithm_meets_requirements = [&](SortingAlgorithm p_algorithm) {
        return _algorithm_meets_requirements(get_policy_probe(p_algorithm), requires_indirect, requires_64bit_keys);
    };

    if (!algorithm_meets_requirements(resolved_algorithm)) {
        const SortingAlgorithm requested_algorithm = resolved_algorithm;
        const PolicyProbe requested_probe = get_policy_probe(requested_algorithm);
        const String requested_failure = _failure_reason_for_algorithm(_algorithm_name_label(requested_algorithm), requested_probe,
                requires_indirect, requires_64bit_keys);
        if (requested_algorithm != ALGORITHM_RADIX && algorithm_meets_requirements(ALGORITHM_RADIX)) {
            fallback_reason = vformat("type=requested preferred=%s selected=radix failure={%s}",
                    _algorithm_name_label(requested_algorithm), requested_failure);
            GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Requested sorter '%s' does not satisfy instance pipeline requirements; falling back to radix (%s)",
                    _algorithm_name_label(requested_algorithm), requested_failure));
            resolved_algorithm = ALGORITHM_RADIX;
        } else {
            GS_LOG_ERROR_DEFAULT(vformat("[GPU Sort] Requested sorter '%s' does not satisfy instance pipeline requirements and no valid fallback is available (%s)",
                    _algorithm_name_label(requested_algorithm), requested_failure));
            return Ref<IGPUSorter>();
        }
    }

    // Check if selected algorithm is supported before instantiation
    if (!get_policy_probe(resolved_algorithm).supported) {
        GS_LOG_ERROR_DEFAULT(vformat("[GPU Sort] Requested sorter '%d' not supported; GPU sorting unavailable",
                int(resolved_algorithm)));
        return Ref<IGPUSorter>();
    }

    Ref<IGPUSorter> sorter;
    switch (resolved_algorithm) {
        case ALGORITHM_BITONIC:
            sorter = memnew(BitonicSort);
            break;
        case ALGORITHM_ONESWEEP:
            sorter = memnew(OneSweepSort);
            break;
        case ALGORITHM_RADIX:
        case ALGORITHM_AUTO:
        default:
            sorter = memnew(RadixSort);
            break;
    }

    if (sorter.is_valid()) {
        sorter->set_key_config(key_config);
        Error err = sorter->initialize(rd, max_elements);
        if (err != OK) {
            String name = sorter->get_algorithm_name();
            print_error("Failed to initialize GPU sorter (" + name + "): " + itos(err));
            sorter.unref();
        } else if (!fallback_reason.is_empty()) {
            sorter->record_fallback_reason(fallback_reason);
        }
    }

    if (sorter.is_valid()) {
        GS_LOG_INFO_DEFAULT(vformat("[GPU Sort] Using %s for %d elements", sorter->get_algorithm_name(), max_elements));
    }

    return sorter;
}

GPUSorterFactory::SortingAlgorithm GPUSorterFactory::get_best_algorithm_for_size(uint32_t element_count,
        const SortKeyConfig &key_config) {
    return select_sort_algorithm(element_count, key_config, nullptr);
}

GPUSorterFactory::SortingAlgorithm GPUSorterFactory::get_best_algorithm_for_size(uint32_t element_count,
        const SortKeyConfig &key_config, RenderingDevice *rd) {
    return select_sort_algorithm(element_count, key_config, rd);
}

GPUSorterFactory::PolicyDecision GPUSorterFactory::evaluate_auto_policy(uint32_t element_count, const SortKeyConfig &key_config,
        const PolicyProbe &radix_probe, const PolicyProbe &bitonic_probe, const PolicyProbe &onesweep_probe,
        bool require_indirect, bool require_64bit_keys) {
    PolicyDecision decision;
    decision.preferred_algorithm = _preferred_auto_algorithm(element_count, key_config);

    auto get_probe = [&](SortingAlgorithm p_algorithm) -> const PolicyProbe & {
        switch (p_algorithm) {
            case ALGORITHM_BITONIC:
                return bitonic_probe;
            case ALGORITHM_ONESWEEP:
                return onesweep_probe;
            case ALGORITHM_RADIX:
            case ALGORITHM_AUTO:
            default:
                return radix_probe;
        }
    };

    const SortingAlgorithm preferred_algorithm = decision.preferred_algorithm;
    const PolicyProbe &preferred_probe = get_probe(preferred_algorithm);
    const char *preferred_label = _algorithm_name_label(preferred_algorithm);
    const String preferred_failure = _failure_reason_for_algorithm(preferred_label, preferred_probe, require_indirect, require_64bit_keys);

    if (_algorithm_meets_requirements(preferred_probe, require_indirect, require_64bit_keys)) {
        decision.selected_algorithm = preferred_algorithm;
        return decision;
    }

    struct Candidate {
        SortingAlgorithm algorithm;
        const PolicyProbe *probe;
    };

    Candidate candidates[3];
    switch (preferred_algorithm) {
        case ALGORITHM_BITONIC:
            candidates[0] = { ALGORITHM_RADIX, &radix_probe };
            candidates[1] = { ALGORITHM_ONESWEEP, &onesweep_probe };
            candidates[2] = { ALGORITHM_BITONIC, &bitonic_probe };
            break;
        case ALGORITHM_ONESWEEP:
            candidates[0] = { ALGORITHM_RADIX, &radix_probe };
            candidates[1] = { ALGORITHM_BITONIC, &bitonic_probe };
            candidates[2] = { ALGORITHM_ONESWEEP, &onesweep_probe };
            break;
        case ALGORITHM_RADIX:
        case ALGORITHM_AUTO:
        default:
            candidates[0] = { ALGORITHM_BITONIC, &bitonic_probe };
            candidates[1] = { ALGORITHM_ONESWEEP, &onesweep_probe };
            candidates[2] = { ALGORITHM_RADIX, &radix_probe };
            break;
    }

    for (const Candidate &candidate : candidates) {
        if (candidate.algorithm == preferred_algorithm) {
            continue;
        }
        if (_algorithm_meets_requirements(*candidate.probe, require_indirect, require_64bit_keys)) {
            decision.selected_algorithm = candidate.algorithm;
            decision.fallback_reason = _join_policy_fallback_reason(preferred_label, preferred_failure,
                    _algorithm_name_label(candidate.algorithm));
            return decision;
        }
    }

    decision.selected_algorithm = preferred_algorithm;
    decision.fallback_reason = vformat("type=auto preferred=%s selected=none failure={%s}",
            preferred_label, preferred_failure);
    return decision;
}

// RadixSort Implementation

RadixSort::RadixSort() {
    current_sort_value = 0;
    key_config = SortKeyConfig::from_settings();
}

RadixSort::~RadixSort() {
    shutdown();
}

void RadixSort::_bind_methods() {
    // Bind methods for GDScript if needed
}

void RadixSort::_reset_pass_uniform_sets(PassUniformSets &p_sets) {
    p_sets.histogram_even = RID();
    p_sets.histogram_odd = RID();
    p_sets.wg_prefix = RID();
    p_sets.bin_prefix = RID();
    p_sets.scatter_even = RID();
    p_sets.scatter_odd = RID();
}

Error RadixSort::_create_pass_uniform_sets(RenderingDevice *resource_rd, const RadixVariant *variant, RID keys_buffer, RID values_buffer,
        RID count_buffer, const String &label_prefix, PassUniformSets &r_sets) {
    ERR_FAIL_NULL_V(resource_rd, ERR_CANT_CREATE);
    ERR_FAIL_NULL_V(variant, ERR_CANT_CREATE);
    ERR_FAIL_COND_V(!keys_buffer.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!values_buffer.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!count_buffer.is_valid(), ERR_INVALID_PARAMETER);

    _reset_pass_uniform_sets(r_sets);
    uniform_owner = resource_rd;
    uniform_owner_generation = resource_rd->get_device_instance_id();
    uniform_sets.clear();

    auto create_histogram_set = [&](bool use_primary, RID &r_set, const char *suffix) -> Error {
        RID current_keys = use_primary ? keys_buffer : temp_keys_buffer;
        Vector<RD::Uniform> histogram_uniforms;
        histogram_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, current_keys));
        histogram_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, histogram_buffer));
        histogram_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, count_buffer));
        r_set = resource_rd->uniform_set_create(histogram_uniforms, variant->histogram_shader, 0);
        if (!r_set.is_valid()) {
            return ERR_CANT_CREATE;
        }
        resource_rd->set_resource_name(r_set, vformat("%s_HistogramSet_%s", label_prefix, suffix));
        uniform_sets.push_back(r_set);
        return OK;
    };

    auto create_scatter_set = [&](bool use_primary, RID &r_set, const char *suffix) -> Error {
        RID current_keys = use_primary ? keys_buffer : temp_keys_buffer;
        RID current_values = use_primary ? values_buffer : temp_values_buffer;
        RID next_keys = use_primary ? temp_keys_buffer : keys_buffer;
        RID next_values = use_primary ? temp_values_buffer : values_buffer;
        Vector<RD::Uniform> scatter_uniforms;
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, current_keys));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, next_keys));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, current_values));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, next_values));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, wg_prefix_buffer));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, bin_prefix_buffer));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, count_buffer));
        r_set = resource_rd->uniform_set_create(scatter_uniforms, variant->scatter_shader, 0);
        if (!r_set.is_valid()) {
            return ERR_CANT_CREATE;
        }
        resource_rd->set_resource_name(r_set, vformat("%s_ScatterSet_%s", label_prefix, suffix));
        uniform_sets.push_back(r_set);
        return OK;
    };

    Vector<RD::Uniform> wg_prefix_uniforms;
    wg_prefix_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, histogram_buffer));
    wg_prefix_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, wg_prefix_buffer));
    wg_prefix_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, bin_counts_buffer));
    wg_prefix_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, count_buffer));
    r_sets.wg_prefix = resource_rd->uniform_set_create(wg_prefix_uniforms, variant->wg_prefix_shader, 0);
    if (!r_sets.wg_prefix.is_valid()) {
        _free_uniform_sets(uniform_owner ? uniform_owner : resource_rd, uniform_sets);
        uniform_owner = nullptr;
        return ERR_CANT_CREATE;
    }
    resource_rd->set_resource_name(r_sets.wg_prefix, vformat("%s_WgPrefixSet", label_prefix));
    uniform_sets.push_back(r_sets.wg_prefix);

    Vector<RD::Uniform> bin_prefix_uniforms;
    bin_prefix_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, bin_counts_buffer));
    bin_prefix_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, bin_prefix_buffer));
    r_sets.bin_prefix = resource_rd->uniform_set_create(bin_prefix_uniforms, variant->bin_prefix_shader, 0);
    if (!r_sets.bin_prefix.is_valid()) {
        _free_uniform_sets(uniform_owner ? uniform_owner : resource_rd, uniform_sets);
        uniform_owner = nullptr;
        return ERR_CANT_CREATE;
    }
    resource_rd->set_resource_name(r_sets.bin_prefix, vformat("%s_BinPrefixSet", label_prefix));
    uniform_sets.push_back(r_sets.bin_prefix);

    if (create_histogram_set(true, r_sets.histogram_even, "Even") != OK ||
            create_histogram_set(false, r_sets.histogram_odd, "Odd") != OK ||
            create_scatter_set(true, r_sets.scatter_even, "Even") != OK ||
            create_scatter_set(false, r_sets.scatter_odd, "Odd") != OK) {
        _free_uniform_sets(uniform_owner ? uniform_owner : resource_rd, uniform_sets);
        uniform_owner = nullptr;
        _reset_pass_uniform_sets(r_sets);
        return ERR_CANT_CREATE;
    }

    return OK;
}

SortPreflightResult RadixSort::_validate_sort_preflight(RID keys_buffer, RID values_buffer, uint32_t count, RID count_buffer, bool p_indirect) const {
    SortPreflightResult result;

    if (!keys_buffer.is_valid()) {
        result.code = SortPreflightError::INVALID_KEYS_BUFFER;
        result.error = ERR_INVALID_PARAMETER;
        result.message = "keys buffer RID is invalid";
        return result;
    }
    if (!values_buffer.is_valid()) {
        result.code = SortPreflightError::INVALID_VALUES_BUFFER;
        result.error = ERR_INVALID_PARAMETER;
        result.message = "values buffer RID is invalid";
        return result;
    }
    if (p_indirect) {
        if (!count_buffer.is_valid()) {
            result.code = SortPreflightError::INVALID_COUNT_BUFFER;
            result.error = ERR_INVALID_PARAMETER;
            result.message = "indirect count buffer RID is invalid";
            return result;
        }
    } else {
        if (count == 0) {
            result.code = SortPreflightError::INVALID_ELEMENT_COUNT;
            result.error = ERR_INVALID_PARAMETER;
            result.message = "element count must be greater than zero";
            return result;
        }
        if (count > max_elements) {
            result.code = SortPreflightError::ELEMENT_COUNT_EXCEEDS_CAPACITY;
            result.error = ERR_INVALID_PARAMETER;
            result.message = vformat("element count %u exceeds sorter capacity %u", count, max_elements);
            return result;
        }
    }

    if (key_config.key_bits != 32 && key_config.key_bits != 64) {
        result.code = SortPreflightError::UNSUPPORTED_KEY_FORMAT;
        result.error = ERR_INVALID_PARAMETER;
        result.message = vformat("unsupported key format: key_bits=%u", key_config.key_bits);
        return result;
    }

    if (!resource_device) {
        result.code = SortPreflightError::RESOURCE_DEVICE_UNAVAILABLE;
        result.error = ERR_CANT_CREATE;
        result.message = "primary rendering device unavailable";
        return result;
    }

    if (!local_rd && GaussianSplatManager::get_singleton() == nullptr) {
        result.code = SortPreflightError::SUBMISSION_DEVICE_UNAVAILABLE;
        result.error = ERR_CANT_CREATE;
        result.message = "submission rendering device unavailable";
        return result;
    }

    result.code = SortPreflightError::NONE;
    result.error = OK;
    return result;
}

uint64_t RadixSort::_sort_async_internal(RID keys_buffer, RID values_buffer, uint32_t count, RID p_wait_semaphore,
        uint64_t p_wait_value, RID p_signal_semaphore, uint64_t p_signal_value_override) {
    SortPreflightResult preflight = _validate_sort_preflight(keys_buffer, values_buffer, count, RID(), false);
    if (!preflight.is_ok()) {
        last_preflight_error = preflight.code;
        GS_LOG_ERROR_DEFAULT(vformat("RadixSort::sort_async preflight failed (%s): %s",
                _sort_preflight_error_name(preflight.code), preflight.message));
        return 0;
    }
    last_preflight_error = SortPreflightError::NONE;

    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *compute_rd = _acquire_submission_device(local_rd, submission_lock);
    ERR_FAIL_NULL_V_MSG(compute_rd, 0, "Local rendering device unavailable for RadixSort::sort_async");

    RenderingDevice *resource_rd = resource_device;
    ERR_FAIL_NULL_V_MSG(resource_rd, 0, "Primary rendering device unavailable for RadixSort::sort_async");

    const RadixVariant *variant = select_variant(count);
    ERR_FAIL_NULL_V_MSG(variant, 0, "Radix sort variant not initialized");

    if (is_sorting.load()) {
        GS_LOG_ERROR_DEFAULT("RadixSort::sort_async called while a sort is already in flight");
        return 0;
    }

    // ISSUE-010: Use generation-safe cleanup for prior uniform sets.
    _free_uniform_sets_safe(uniform_owner, uniform_owner_generation, resource_rd, uniform_sets);

    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    is_sorting = true;

    uint32_t num_passes = variant->num_passes;
    uint32_t workgroups = get_workgroup_count(count);

    // Write dispatch header to staging buffer for GPU-driven shaders (see IndirectDispatchLayout).
    struct IndirectDispatchHeader {
        uint32_t dispatch_x;
        uint32_t dispatch_y;
        uint32_t dispatch_z;
        uint32_t element_count;
    } header = { workgroups, 1u, 1u, count };
    resource_rd->buffer_update(indirect_count_buffer, 0, sizeof(header), &header);

    PassUniformSets pass_uniform_sets;
    if (_create_pass_uniform_sets(resource_rd, variant, keys_buffer, values_buffer, indirect_count_buffer, "GS_RadixSort", pass_uniform_sets) != OK) {
        GS_LOG_ERROR_DEFAULT("RadixSort::sort_async failed to create reused pass uniform sets");
        is_sorting = false;
        return 0;
    }

    auto record_commands = [this, variant, pass_uniform_sets, num_passes, workgroups](RenderingDevice *command_rd, ComputeListID compute_list) {
	    for (uint32_t pass = 0; pass < num_passes; ++pass) {
	        bool use_primary = (pass & 1) == 0;
            RID histogram_set = use_primary ? pass_uniform_sets.histogram_even : pass_uniform_sets.histogram_odd;
            RID scatter_set = use_primary ? pass_uniform_sets.scatter_even : pass_uniform_sets.scatter_odd;
            uint32_t histogram_offset = pass * histogram_stride;
            uint32_t bin_offset = pass * variant->radix_size;

            // GPU Debug: Histogram pass
            {
                ScopedGpuMarkerEx histogram_marker(command_rd, "GS_RadixSort_Histogram", PassColors::SORTING);
                command_rd->compute_list_bind_compute_pipeline(compute_list, variant->histogram_pipeline);
                command_rd->compute_list_bind_uniform_set(compute_list, histogram_set, 0);

                // num_keys is read from indirect buffer (binding 2), not push constants
                // Struct padded to 16 bytes for SPIR-V alignment
                struct HistogramParams {
                    uint32_t bit_shift;
                    uint32_t histogram_offset;
                    uint32_t workgroup_stride;
                    uint32_t _pad0;
                } hist_params = { pass * variant->radix_bits, histogram_offset, workgroup_stride, 0 };

                command_rd->compute_list_set_push_constant(compute_list, &hist_params, sizeof(hist_params));
                command_rd->compute_list_dispatch(compute_list, workgroups, 1, 1);
            }
            command_rd->compute_list_add_barrier(compute_list);

            // GPU Debug: Workgroup prefix sum pass
            {
                _record_wg_prefix_pass(command_rd, compute_list, variant, pass_uniform_sets.wg_prefix,
                        histogram_offset, workgroup_stride, bin_offset, "GS_RadixSort_WGPrefix");
            }
            command_rd->compute_list_add_barrier(compute_list);

            // GPU Debug: Bin prefix sum pass
            {
                _record_bin_prefix_pass(command_rd, compute_list, variant, pass_uniform_sets.bin_prefix, bin_offset,
                        "GS_RadixSort_BinPrefix");
            }
            command_rd->compute_list_add_barrier(compute_list);

            // GPU Debug: Scatter pass
            {
                _record_scatter_pass(command_rd, compute_list, variant, scatter_set,
                        pass * variant->radix_bits, histogram_offset, workgroup_stride, bin_offset, workgroups, false, RID(),
                        "GS_RadixSort_Scatter");
            }

            if (pass + 1 < num_passes) {
                command_rd->compute_list_add_barrier(compute_list);
            }
        }

        ERR_FAIL_COND_MSG((num_passes & 1) != 0,
                "RadixSort::sort_async assumes an even number of passes so results stay in the source buffer.");
    };

    uint64_t submit_start = OS::get_singleton()->get_ticks_usec();

    ScopedGpuMarkerEx sort_marker(compute_rd, "GS_RadixSort", PassColors::SORTING);
    ComputeListID compute_list = compute_rd->compute_list_begin();
    record_commands(compute_rd, compute_list);
    compute_rd->compute_list_end();
    gs_device_utils::safe_submit(compute_rd);

    current_sort_value = ++timeline_value;

    float cpu_submit_time_ms = (OS::get_singleton()->get_ticks_usec() - submit_start) / 1000.0f;
    metrics_collector.record_async_sort(count, cpu_submit_time_ms);

    // ISSUE-010: Use generation-safe cleanup.
    _free_uniform_sets_safe(uniform_owner, uniform_owner_generation, resource_rd, uniform_sets);
    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    is_sorting = false;
    return current_sort_value;
}

Error RadixSort::create_variant(RenderingDevice *device, uint32_t radix_bits) {
    if (!device) {
        return ERR_CANT_CREATE;
    }
    if (radix_bits == 0) {
        return ERR_INVALID_PARAMETER;
    }

    for (const RadixVariant &existing : variants) {
        if (existing.radix_bits == radix_bits) {
            return OK;
        }
    }

    RadixVariant variant;
    variant.radix_bits = radix_bits;
    variant.radix_size = 1u << radix_bits;
    variant.num_passes = MAX<uint32_t>(1, (key_config.key_bits + radix_bits - 1u) / radix_bits);
    auto cleanup_variant = [&](RadixVariant &p_variant) {
        if (!device) {
            p_variant = RadixVariant();
            return;
        }
        if (p_variant.scatter_pipeline.is_valid() && device->compute_pipeline_is_valid(p_variant.scatter_pipeline)) {
            device->free(p_variant.scatter_pipeline);
        }
        if (p_variant.wg_prefix_pipeline.is_valid() && device->compute_pipeline_is_valid(p_variant.wg_prefix_pipeline)) {
            device->free(p_variant.wg_prefix_pipeline);
        }
        if (p_variant.bin_prefix_pipeline.is_valid() && device->compute_pipeline_is_valid(p_variant.bin_prefix_pipeline)) {
            device->free(p_variant.bin_prefix_pipeline);
        }
        if (p_variant.histogram_pipeline.is_valid() && device->compute_pipeline_is_valid(p_variant.histogram_pipeline)) {
            device->free(p_variant.histogram_pipeline);
        }
        if (p_variant.scatter_shader.is_valid()) {
            device->free(p_variant.scatter_shader);
        }
        if (p_variant.wg_prefix_shader.is_valid()) {
            device->free(p_variant.wg_prefix_shader);
        }
        if (p_variant.bin_prefix_shader.is_valid()) {
            device->free(p_variant.bin_prefix_shader);
        }
        if (p_variant.histogram_shader.is_valid()) {
            device->free(p_variant.histogram_shader);
        }
        p_variant = RadixVariant();
    };

    String key_type = use_64bit_keys ? "uvec2" : "uint";
    String key_helper = use_64bit_keys
            ? String(R"(
uint get_radix(uvec2 key, uint shift) {
    uint lo = key.x;
    uint hi = key.y;
    if (shift >= 32u) {
        return (hi >> (shift - 32u)) & (RADIX_SIZE - 1u);
    }
    uint bits = lo >> shift;
    if (shift > 32u - RADIX_BITS) {
        bits |= hi << (32u - shift);
    }
    return bits & (RADIX_SIZE - 1u);
}
)")
            : String(R"(
uint get_radix(uint key, uint shift) {
    return (key >> shift) & (RADIX_SIZE - 1u);
}
)");

    String key_read_hist = use_64bit_keys
            ? String("            uvec2 key = keys_in.keys[idx];\n            radix = get_radix(key, params.bit_shift);")
            : String("            uint key = keys_in.keys[idx];\n            radix = get_radix(key, params.bit_shift);");

    String subgroup_preamble = subgroups_available
            ? String("#extension GL_KHR_shader_subgroup_basic : enable\n"
                     "#extension GL_KHR_shader_subgroup_ballot : enable\n"
                     "#define GS_ENABLE_SUBGROUPS 1\n")
            : String("#define GS_ENABLE_SUBGROUPS 0\n");

    String histogram_update = String(R"(
#if GS_ENABLE_SUBGROUPS
        for (uint r = 0u; r < RADIX_SIZE; ++r) {
            uvec4 mask = subgroupBallot(valid && radix == r);
            uint count = subgroupBallotBitCount(mask);
            if (count > 0u) {
                uint leader = subgroupBallotFindLSB(mask);
                if (gl_SubgroupInvocationID == leader) {
                    atomicAdd(local_histogram[r], count);
                }
            }
        }
#else
        if (valid) {
            atomicAdd(local_histogram[radix], 1);
        }
#endif
)");

    String scatter_bin_update = String(R"(
#if GS_ENABLE_SUBGROUPS
        if (gl_SubgroupSize >= 32u) {
            uint subgroup_words = (gl_SubgroupSize + 31u) / 32u;
            uint subgroup_index = lid / gl_SubgroupSize;
            uint base_word = subgroup_index * subgroup_words;
            for (uint r = 0u; r < RADIX_SIZE; ++r) {
                uvec4 mask = subgroupBallot(valid && radix == r);
                uint count = subgroupBallotBitCount(mask);
                if (count == 0u) {
                    continue;
                }
                uint leader = subgroupBallotFindLSB(mask);
                if (gl_SubgroupInvocationID == leader) {
                    for (uint w = 0u; w < subgroup_words; ++w) {
                        uint bits = mask[w];
                        if (bits != 0u) {
                            atomicOr(bin_masks[r * MASK_WORDS + base_word + w], bits);
                        }
                    }
                }
            }
        } else if (valid) {
            uint word_index = lid >> 5u;
            uint bit_mask = 1u << (lid & 31u);
            atomicOr(bin_masks[radix * MASK_WORDS + word_index], bit_mask);
        }
#else
        if (valid) {
            uint word_index = lid >> 5u;
            uint bit_mask = 1u << (lid & 31u);
            atomicOr(bin_masks[radix * MASK_WORDS + word_index], bit_mask);
        }
#endif
)");

    // Pass 1: per-workgroup histogram (no cross-WG atomics)
    // GPU-driven: reads num_keys from indirect buffer at binding 2
    String histogram_source = vformat(R"(
#version 450

#define RADIX_BITS %d
#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d

%s

layout(local_size_x = WORKGROUP_SIZE) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    %s keys[];
} keys_in;

layout(set = 0, binding = 1, std430) restrict buffer HistogramBuffer {
    uint histogram[];
} hist;

// GPU-driven indirect buffer layout matches IndirectDispatchLayout (element_count after dispatch_xyz).
layout(set = 0, binding = 2, std430) restrict readonly buffer IndirectCount {
    uint dispatch_xyz[3];
    uint element_count;
} indirect;

layout(push_constant) uniform PushConstants {
    uint bit_shift;
    uint histogram_offset;
    uint workgroup_stride;
    uint _pad0;
} params;

shared uint local_histogram[RADIX_SIZE];

%s

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint num_keys = indirect.element_count;

    if (tid < RADIX_SIZE) {
        local_histogram[tid] = 0;
    }
    barrier();

    uint total_threads = gl_NumWorkGroups.x * WORKGROUP_SIZE;
    uint keys_per_thread = (num_keys + total_threads - 1) / total_threads;

    for (uint i = 0; i < keys_per_thread; ++i) {
        uint idx = gid + i * total_threads;
        bool valid = idx < num_keys;
        uint radix = 0u;
        if (valid) {
%s
        }
%s
    }
    barrier();

    if (tid < RADIX_SIZE) {
        uint base = params.histogram_offset + gl_WorkGroupID.x * params.workgroup_stride;
        hist.histogram[base + tid] = local_histogram[tid];
    }
}
        )",
            variant.radix_bits,
            variant.radix_size,
            workgroup_size,
            subgroup_preamble,
            key_type,
            key_helper,
            key_read_hist,
            histogram_update);

    RID histogram_shader_file = create_compute_shader_from_spirv(device, histogram_source);
    if (!histogram_shader_file.is_valid()) {
        return ERR_CANT_CREATE;
    }
    variant.histogram_shader = histogram_shader_file;

    // Pass 2: workgroup prefix per bin + total count per bin
    String wg_prefix_source = vformat(R"(
#version 450

#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d

layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer HistogramBuffer {
    uint histogram[];
} hist;

layout(set = 0, binding = 1, std430) restrict buffer WorkgroupPrefixBuffer {
    uint wg_prefix[];
} wg_prefix_buf;

layout(set = 0, binding = 2, std430) restrict buffer BinCountBuffer {
    uint bin_counts[];
} bin_counts_buf;

// GPU-driven indirect buffer layout matches IndirectDispatchLayout (element_count after dispatch_xyz).
layout(set = 0, binding = 3, std430) restrict readonly buffer IndirectCount {
    uint dispatch_xyz[3];
    uint element_count;
} indirect;

layout(push_constant) uniform PushConstants {
    uint histogram_offset;
    uint workgroup_stride;
    uint bin_offset;
    uint pad0;
} params;

void main() {
    uint bin = gl_WorkGroupID.x;
    if (bin >= RADIX_SIZE) {
        return;
    }

    // Derive workgroup count from element_count; dispatch_xyz may be unrelated to radix sort.
    uint count = indirect.element_count;
    uint workgroup_count = (count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    if (workgroup_count == 0u) {
        workgroup_count = 1u;
    }

    uint prefix = 0;
    for (uint wg = 0; wg < workgroup_count; ++wg) {
        uint idx = params.histogram_offset + wg * params.workgroup_stride + bin;
        uint count = hist.histogram[idx];
        wg_prefix_buf.wg_prefix[idx] = prefix;
        prefix += count;
    }

    bin_counts_buf.bin_counts[params.bin_offset + bin] = prefix;
}
        )",
            variant.radix_size,
            workgroup_size);

    RID wg_prefix_shader_file = create_compute_shader_from_spirv(device, wg_prefix_source);
    if (!wg_prefix_shader_file.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }
    variant.wg_prefix_shader = wg_prefix_shader_file;

    // Pass 3: exclusive prefix of bin totals (global base per bin)
    String bin_prefix_source = vformat(R"(
#version 450

#define RADIX_SIZE %d

layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer BinCountBuffer {
    uint bin_counts[];
} bin_counts_buf;

layout(set = 0, binding = 1, std430) restrict buffer BinPrefixBuffer {
    uint bin_prefix[];
} bin_prefix_buf;

layout(push_constant) uniform PushConstants {
    uint bin_offset;
    uint pad0;
    uint pad1;
    uint pad2;
} params;

void main() {
    uint prefix = 0;
    for (uint bin = 0; bin < RADIX_SIZE; ++bin) {
        uint idx = params.bin_offset + bin;
        uint count = bin_counts_buf.bin_counts[idx];
        bin_prefix_buf.bin_prefix[idx] = prefix;
        prefix += count;
    }
}
        )",
            variant.radix_size);

    RID bin_prefix_shader_file = create_compute_shader_from_spirv(device, bin_prefix_source);
    if (!bin_prefix_shader_file.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }
    variant.bin_prefix_shader = bin_prefix_shader_file;

    // Pass 4: stable scatter using precomputed prefixes.
    // Uses per-bin bitmasks to compute local ranks in parallel (no serialized lane loop).
    // GPU-driven: reads num_keys from indirect buffer at binding 6.
    uint32_t mask_words = (workgroup_size + 31u) / 32u;
    String scatter_source = vformat(R"(
#version 450

#define RADIX_BITS %d
#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d
#define MASK_WORDS %d

%s

layout(local_size_x = WORKGROUP_SIZE) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    %s keys[];
} keys_in;

layout(set = 0, binding = 1, std430) restrict buffer KeysOut {
    %s keys[];
} keys_out;

layout(set = 0, binding = 2, std430) restrict readonly buffer ValuesIn {
    uint values[];
} values_in;

layout(set = 0, binding = 3, std430) restrict buffer ValuesOut {
    uint values[];
} values_out;

layout(set = 0, binding = 4, std430) restrict readonly buffer WorkgroupPrefixBuffer {
    uint wg_prefix[];
} wg_prefix_buf;

layout(set = 0, binding = 5, std430) restrict readonly buffer BinPrefixBuffer {
    uint bin_prefix[];
} bin_prefix_buf;

// GPU-driven indirect buffer layout matches IndirectDispatchLayout (element_count after dispatch_xyz).
layout(set = 0, binding = 6, std430) restrict readonly buffer IndirectCount {
    uint dispatch_xyz[3];
    uint element_count;
} indirect;

layout(push_constant) uniform PushConstants {
    uint bit_shift;
    uint histogram_offset;
    uint workgroup_stride;
    uint bin_offset;
    uint wg_prefix_offset;
    uint workgroup_count;
    uint _pad0;
    uint _pad1;
} params;

%s

shared uint local_bases[RADIX_SIZE];
shared uint local_offsets[RADIX_SIZE];
shared uint bin_masks[RADIX_SIZE * MASK_WORDS];

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint num_keys = indirect.element_count;
    uint threads = gl_NumWorkGroups.x * WORKGROUP_SIZE;
    uint keys_per_thread = (num_keys + threads - 1) / threads;

    if (lid < RADIX_SIZE) {
        local_offsets[lid] = 0u;
        uint base_idx = params.wg_prefix_offset + gl_WorkGroupID.x * params.workgroup_stride + lid;
        uint bin_idx = params.bin_offset + lid;
        local_bases[lid] = wg_prefix_buf.wg_prefix[base_idx] + bin_prefix_buf.bin_prefix[bin_idx];
    }
    barrier();

    for (uint iter = 0; iter < keys_per_thread; ++iter) {
        for (uint i = lid; i < RADIX_SIZE * MASK_WORDS; i += WORKGROUP_SIZE) {
            bin_masks[i] = 0u;
        }
        barrier();

        uint idx = gid + iter * threads;
        bool valid = idx < num_keys;

%s

%s
        barrier();

        if (valid) {
            uint base = radix * MASK_WORDS;
            uint word_index = lid >> 5u;
            uint bit_mask = 1u << (lid & 31u);
            uint rank = 0u;
            for (uint w = 0u; w < word_index; ++w) {
                rank += bitCount(bin_masks[base + w]);
            }
            uint word = bin_masks[base + word_index];
            rank += bitCount(word & (bit_mask - 1u));
            uint pos = local_bases[radix] + local_offsets[radix] + rank;
            keys_out.keys[pos] = key;
            values_out.values[pos] = value;
        }
        barrier();

        if (lid < RADIX_SIZE) {
            uint base = lid * MASK_WORDS;
            uint count = 0u;
            for (uint w = 0u; w < MASK_WORDS; ++w) {
                count += bitCount(bin_masks[base + w]);
            }
            local_offsets[lid] += count;
        }
        barrier();
    }
}
        )",
            variant.radix_bits,
            variant.radix_size,
            workgroup_size,
            mask_words,
            subgroup_preamble,
            key_type,
            key_type,
            key_helper,
            (use_64bit_keys
                    ? String("        uvec2 key = uvec2(0u);\n        uint value = 0u;\n        uint radix = 0u;\n        if (valid) {\n            key = keys_in.keys[idx];\n            value = values_in.values[idx];\n            radix = get_radix(key, params.bit_shift);\n        }")
                    : String("        uint key = 0u;\n        uint value = 0u;\n        uint radix = 0u;\n        if (valid) {\n            key = keys_in.keys[idx];\n            value = values_in.values[idx];\n            radix = get_radix(key, params.bit_shift);\n        }")),
            scatter_bin_update);

    RID scatter_shader_file = create_compute_shader_from_spirv(device, scatter_source);
    if (!scatter_shader_file.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }
    variant.scatter_shader = scatter_shader_file;

    variant.histogram_pipeline = device->compute_pipeline_create(variant.histogram_shader);
    if (!variant.histogram_pipeline.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }

    variant.wg_prefix_pipeline = device->compute_pipeline_create(variant.wg_prefix_shader);
    if (!variant.wg_prefix_pipeline.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }

    variant.bin_prefix_pipeline = device->compute_pipeline_create(variant.bin_prefix_shader);
    if (!variant.bin_prefix_pipeline.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }

    variant.scatter_pipeline = device->compute_pipeline_create(variant.scatter_shader);
    if (!variant.scatter_pipeline.is_valid()) {
        cleanup_variant(variant);
        return ERR_CANT_CREATE;
    }

    variants.push_back(variant);
    return OK;
}

const RadixSort::RadixVariant *RadixSort::get_variant(uint32_t radix_bits) const {
    for (const RadixVariant &variant : variants) {
        if (variant.radix_bits == radix_bits) {
            return &variant;
        }
    }
    return nullptr;
}

const RadixSort::RadixVariant *RadixSort::select_variant(uint32_t element_count) const {
    uint32_t desired_bits = primary_radix_bits;
    if (secondary_radix_bits != 0 && element_count >= secondary_threshold) {
        desired_bits = secondary_radix_bits;
    }

    const RadixVariant *variant = get_variant(desired_bits);
    if (!variant && !variants.is_empty()) {
        variant = &variants[0];
    }
    return variant;
}

uint32_t RadixSort::get_workgroup_count(uint32_t element_count) const {
    return MAX<uint32_t>(1, (element_count + workgroup_size - 1) / workgroup_size);
}

Error RadixSort::initialize(RenderingDevice *p_rd, uint32_t p_max_elements) {
    if (!p_rd || p_max_elements == 0) {
        return ERR_INVALID_PARAMETER;
    }

    // Clean up any existing resources before re-initializing
    shutdown();

    rd = p_rd;
    max_elements = p_max_elements;

    RenderingDevice *resource_rd = nullptr;
    Error init_err = _init_sorter_devices(rd, local_rd, resource_rd);
    if (init_err != OK) {
        shutdown();
        return init_err;
    }
    resource_device = resource_rd;
    resource_device_generation = resource_rd->get_device_instance_id();
    RenderingDevice *command_rd = local_rd;

    const GPUSortingConfig &config = g_gpu_sorting_config;
    workgroup_size = config.workgroup_size > 0 ? config.workgroup_size : GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    primary_radix_bits = config.radix_bits;
    secondary_radix_bits = 0;
    secondary_threshold = 0;

    if (!device_supports_workgroup(command_rd, workgroup_size)) {
        GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] RadixSort unavailable on devices without %d-thread workgroups", workgroup_size));
        shutdown();
        return ERR_UNAVAILABLE;
    }

    uint64_t subgroup_ops = command_rd->limit_get(RenderingDevice::LIMIT_SUBGROUP_OPERATIONS);
    uint64_t subgroup_stages = command_rd->limit_get(RenderingDevice::LIMIT_SUBGROUP_IN_SHADERS);
    bool has_basic = (subgroup_ops & RenderingDevice::SUBGROUP_BASIC_BIT) != 0;
    bool has_ballot = (subgroup_ops & RenderingDevice::SUBGROUP_BALLOT_BIT) != 0;
    bool has_compute = (subgroup_stages & RenderingDevice::SHADER_STAGE_COMPUTE_BIT) != 0;
    subgroups_available = !_subgroup_prefix_forced_off() && has_basic && has_ballot && has_compute;
    if (!subgroups_available && _subgroup_prefix_forced_off()) {
        GS_LOG_WARN_DEFAULT("[GPU Sort] Subgroup prefix kernels forced OFF by project setting.");
    }

    key_config.key_bits = (key_config.key_bits == 64) ? 64 : 32;
    if (key_config.tile_bits > key_config.key_bits) {
        key_config.tile_bits = key_config.key_bits;
    }
    if (key_config.depth_bits > key_config.key_bits) {
        key_config.depth_bits = key_config.key_bits;
    }
    if (key_config.tile_bits + key_config.depth_bits > key_config.key_bits) {
        if (key_config.tile_bits >= key_config.key_bits) {
            key_config.tile_bits = key_config.key_bits;
            key_config.depth_bits = 0;
        } else {
            key_config.depth_bits = key_config.key_bits - key_config.tile_bits;
        }
    }
    use_64bit_keys = key_config.key_bits > 32;
    key_stride_words = use_64bit_keys ? 2u : 1u;
    key_stride_bytes = key_stride_words * sizeof(uint32_t);

    variants.clear();
    Error err = create_variant(resource_rd, primary_radix_bits);
    if (err != OK) {
        _cleanup_partial_init(resource_rd);
        return err;
    }
    if (secondary_radix_bits != 0) {
        err = create_variant(resource_rd, secondary_radix_bits);
        if (err != OK) {
        _cleanup_partial_init(resource_rd);
            return err;
        }
    }

    if (variants.is_empty()) {
        _cleanup_partial_init(resource_rd);
        ERR_FAIL_V_MSG(ERR_CANT_CREATE, "RadixSort: No shader variants created");
    }

    max_radix_size = 0;
    max_num_passes = 0;
    for (const RadixVariant &variant : variants) {
        max_radix_size = MAX(max_radix_size, variant.radix_size);
        max_num_passes = MAX(max_num_passes, variant.num_passes);
    }

    max_workgroups = MAX<uint32_t>(1, (max_elements + workgroup_size - 1) / workgroup_size);
    workgroup_stride = max_radix_size;
    histogram_stride = max_workgroups * max_radix_size;

    // Phase 2: Macro to create buffer with cleanup on failure
#define RADIX_CREATE_BUFFER(var, size, name)     do {         var = resource_rd->storage_buffer_create(size);         if (!var.is_valid()) {             GS_LOG_ERROR_DEFAULT(vformat("RadixSort: Failed to create %s (size=%llu)", name, (uint64_t)(size)));             _cleanup_partial_init(resource_rd);             return ERR_CANT_CREATE;         }         resource_rd->set_resource_name(var, name);     } while (0)

    uint64_t histogram_bytes = uint64_t(histogram_stride) * uint64_t(max_num_passes) * sizeof(uint32_t);
    RADIX_CREATE_BUFFER(histogram_buffer, histogram_bytes, "GS_RadixSortHistogramBuffer");

    uint64_t wg_prefix_bytes = histogram_bytes;
    RADIX_CREATE_BUFFER(wg_prefix_buffer, wg_prefix_bytes, "GS_RadixSortWgPrefixBuffer");

    uint64_t bin_bytes = uint64_t(max_num_passes) * uint64_t(max_radix_size) * sizeof(uint32_t);
    RADIX_CREATE_BUFFER(bin_counts_buffer, bin_bytes, "GS_RadixSortBinCountsBuffer");
    RADIX_CREATE_BUFFER(bin_prefix_buffer, bin_bytes, "GS_RadixSortBinPrefixBuffer");

    RADIX_CREATE_BUFFER(temp_keys_buffer, uint64_t(max_elements) * uint64_t(key_stride_bytes), "GS_RadixSortTempKeysBuffer");
    RADIX_CREATE_BUFFER(temp_values_buffer, uint64_t(max_elements) * sizeof(uint32_t), "GS_RadixSortTempValuesBuffer");

    // GPU-driven count staging buffer (dispatch_xyz[3] + element_count).
    // Used when sort() is called with an explicit count instead of sort_indirect()
    Vector<uint8_t> count_init;
    count_init.resize(GaussianSplatting::kIndirectDispatchHeaderSize);
    count_init.fill(0);
    indirect_count_buffer = resource_rd->storage_buffer_create(GaussianSplatting::kIndirectDispatchHeaderSize, count_init,
            RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
    if (!indirect_count_buffer.is_valid()) {
        GS_LOG_ERROR_DEFAULT("RadixSort: Failed to create GS_RadixSortIndirectCountBuffer");
        _cleanup_partial_init(resource_rd);
        return ERR_CANT_CREATE;
    }
    resource_rd->set_resource_name(indirect_count_buffer, "GS_RadixSortIndirectCountBuffer");

    Vector<uint8_t> dispatch_args_init;
    dispatch_args_init.resize(GaussianSplatting::kIndirectDispatchHeaderSize);
    dispatch_args_init.fill(0);
    indirect_dispatch_args_buffer = resource_rd->storage_buffer_create(GaussianSplatting::kIndirectDispatchHeaderSize,
            dispatch_args_init, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
    if (!indirect_dispatch_args_buffer.is_valid()) {
        GS_LOG_ERROR_DEFAULT("RadixSort: Failed to create GS_RadixSortIndirectDispatchArgs");
        _cleanup_partial_init(resource_rd);
        return ERR_CANT_CREATE;
    }
    resource_rd->set_resource_name(indirect_dispatch_args_buffer, "GS_RadixSortIndirectDispatchArgs");

    String dispatch_source = R"(
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer IndirectCount {
    uint dispatch_xyz[3];
    uint element_count;
    uint overflow_flag;
    uint unclamped_total;
} indirect_in;

layout(set = 0, binding = 1, std430) buffer DispatchArgs {
    uint dispatch_xyz[3];
    uint pad0;
} indirect_out;

layout(push_constant) uniform PushConstants {
    uint workgroup_size;
    uint pad0;
    uint pad1;
    uint pad2;
} params;

void main() {
    uint count = indirect_in.element_count;
    uint groups = (count + params.workgroup_size - 1u) / params.workgroup_size;
    if (groups == 0u) {
        groups = 1u;
    }
    indirect_out.dispatch_xyz[0] = groups;
    indirect_out.dispatch_xyz[1] = 1u;
    indirect_out.dispatch_xyz[2] = 1u;
}
)";

    indirect_dispatch_shader = create_compute_shader_from_spirv(resource_rd, dispatch_source);
    if (!indirect_dispatch_shader.is_valid()) {
        GS_LOG_ERROR_DEFAULT("RadixSort: Failed to create indirect dispatch compute shader");
        _cleanup_partial_init(resource_rd);
        return ERR_CANT_CREATE;
    }
    indirect_dispatch_pipeline = resource_rd->compute_pipeline_create(indirect_dispatch_shader);
    if (!indirect_dispatch_pipeline.is_valid()) {
        GS_LOG_ERROR_DEFAULT("RadixSort: Failed to create indirect dispatch compute pipeline");
        _cleanup_partial_init(resource_rd);
        return ERR_CANT_CREATE;
    }

#undef RADIX_CREATE_BUFFER

    uniform_sets.clear();
    uniform_owner = nullptr;

    GS_LOG_INFO_DEFAULT(vformat("RadixSort initialized for up to %d elements (primary %d-bit radix)",
            max_elements, primary_radix_bits));
    return OK;
}



// Phase 2: Cleanup partially-created resources on init failure
void RadixSort::_cleanup_partial_init(RenderingDevice *p_rd) {
    if (!p_rd) {
        return;
    }

    _free_uniform_sets(p_rd, uniform_sets);
    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    // Free buffers that may have been created before failure
#define SAFE_FREE(buf) \
    do { \
        if (buf.is_valid()) { \
            p_rd->free(buf); \
            buf = RID(); \
        } \
    } while (0)

    SAFE_FREE(histogram_buffer);
    SAFE_FREE(wg_prefix_buffer);
    SAFE_FREE(bin_counts_buffer);
    SAFE_FREE(bin_prefix_buffer);
    SAFE_FREE(temp_keys_buffer);
    SAFE_FREE(temp_values_buffer);
    SAFE_FREE(indirect_count_buffer);
    SAFE_FREE(indirect_dispatch_args_buffer);

#undef SAFE_FREE

    if (indirect_dispatch_pipeline.is_valid()) {
        if (p_rd->compute_pipeline_is_valid(indirect_dispatch_pipeline)) {
            p_rd->free(indirect_dispatch_pipeline);
        }
        indirect_dispatch_pipeline = RID();
    }
    if (indirect_dispatch_shader.is_valid()) {
        p_rd->free(indirect_dispatch_shader);
        indirect_dispatch_shader = RID();
    }

    // Clean up shader variants
    for (RadixVariant &variant : variants) {
        if (variant.scatter_pipeline.is_valid()) {
            if (p_rd->compute_pipeline_is_valid(variant.scatter_pipeline)) {
                p_rd->free(variant.scatter_pipeline);
            }
            variant.scatter_pipeline = RID();
        }
        if (variant.wg_prefix_pipeline.is_valid()) {
            if (p_rd->compute_pipeline_is_valid(variant.wg_prefix_pipeline)) {
                p_rd->free(variant.wg_prefix_pipeline);
            }
            variant.wg_prefix_pipeline = RID();
        }
        if (variant.bin_prefix_pipeline.is_valid()) {
            if (p_rd->compute_pipeline_is_valid(variant.bin_prefix_pipeline)) {
                p_rd->free(variant.bin_prefix_pipeline);
            }
            variant.bin_prefix_pipeline = RID();
        }
        if (variant.histogram_pipeline.is_valid()) {
            if (p_rd->compute_pipeline_is_valid(variant.histogram_pipeline)) {
                p_rd->free(variant.histogram_pipeline);
            }
            variant.histogram_pipeline = RID();
        }
        if (variant.scatter_shader.is_valid()) {
            p_rd->free(variant.scatter_shader);
            variant.scatter_shader = RID();
        }
        if (variant.wg_prefix_shader.is_valid()) {
            p_rd->free(variant.wg_prefix_shader);
            variant.wg_prefix_shader = RID();
        }
        if (variant.bin_prefix_shader.is_valid()) {
            p_rd->free(variant.bin_prefix_shader);
            variant.bin_prefix_shader = RID();
        }
        if (variant.histogram_shader.is_valid()) {
            p_rd->free(variant.histogram_shader);
            variant.histogram_shader = RID();
        }
    }
    variants.clear();

    // Clear device pointers and generations
    uniform_sets.clear();
    uniform_owner = nullptr;
    uniform_owner_generation = 0;
    resource_device = nullptr;
    resource_device_generation = 0;
    local_rd = nullptr;
    rd = nullptr;
}

void RadixSort::shutdown() {
    wait_for_completion();

    // Phase 3: Try to properly free resources if we still have a valid device
    // Check if resource_device is still valid by comparing with the current shared device
    RenderingDevice *current_shared = get_submission_device();
    // During shutdown the manager singleton may be gone; trust resource_device.
    bool device_still_valid = (resource_device != nullptr) &&
            (current_shared == nullptr || resource_device == current_shared);

    // ISSUE-010: Also validate device generation to detect recycled/stale pointers.
    if (device_still_valid && resource_device_generation != 0) {
        device_still_valid = ResourceOwnerMismatchContract::is_device_generation_valid(
                resource_device, resource_device_generation);
    }

    if (device_still_valid) {
        // Device is still valid - use cleanup helper to properly free resources
        _cleanup_partial_init(resource_device);
    } else {
        // Device may be gone or different - just invalidate RIDs without freeing
        // Resources will be cleaned up when the owning device is destroyed
        for (RadixVariant &variant : variants) {
            variant.scatter_pipeline = RID();
            variant.wg_prefix_pipeline = RID();
            variant.bin_prefix_pipeline = RID();
            variant.histogram_pipeline = RID();
            variant.scatter_shader = RID();
            variant.wg_prefix_shader = RID();
            variant.bin_prefix_shader = RID();
            variant.histogram_shader = RID();
        }
        variants.clear();
        uniform_sets.clear();
        uniform_owner = nullptr;
        uniform_owner_generation = 0;
        histogram_buffer = RID();
        wg_prefix_buffer = RID();
        bin_counts_buffer = RID();
        bin_prefix_buffer = RID();
        temp_keys_buffer = RID();
        temp_values_buffer = RID();
        indirect_count_buffer = RID();
        indirect_dispatch_args_buffer = RID();
        indirect_dispatch_shader = RID();
        indirect_dispatch_pipeline = RID();
        resource_device = nullptr;
        resource_device_generation = 0;
        local_rd = nullptr;
        rd = nullptr;
    }

    // Reset state values
    histogram_stride = 0;
    workgroup_stride = 0;
    max_workgroups = 0;
    max_radix_size = 0;
    max_num_passes = 0;
    primary_radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS;
    secondary_radix_bits = 0;
    secondary_threshold = 0;
    key_stride_words = 1;
    key_stride_bytes = sizeof(uint32_t);
    use_64bit_keys = false;
}

Error RadixSort::sort(RID keys_buffer, RID values_buffer, uint32_t count) {
    if (count == 0) {
        WARN_PRINT_ONCE("RadixSort::sort called with zero elements; nothing to sort.");
        return OK;
    }
    uint64_t sig = sort_async(keys_buffer, values_buffer, count);
    ERR_FAIL_COND_V(sig == 0, ERR_CANT_CREATE);
    wait_for_completion();
    return OK;
}

Error RadixSort::sort_indirect(RID keys_buffer, RID values_buffer, RID count_buffer) {
    SortPreflightResult preflight = _validate_sort_preflight(keys_buffer, values_buffer, 0, count_buffer, true);
    if (!preflight.is_ok()) {
        last_preflight_error = preflight.code;
        GS_LOG_ERROR_DEFAULT(vformat("RadixSort::sort_indirect preflight failed (%s): %s",
                _sort_preflight_error_name(preflight.code), preflight.message));
        return preflight.error;
    }
    last_preflight_error = SortPreflightError::NONE;

    // GPU-driven path: shaders read element_count directly from count_buffer
    uint64_t sig = _sort_indirect_internal(keys_buffer, values_buffer, count_buffer);
    if (sig == 0) {
        return ERR_CANT_CREATE;
    }
    wait_for_completion();
    return OK;
}

uint64_t RadixSort::sort_indirect_async(RID keys_buffer, RID values_buffer, RID count_buffer) {
    SortPreflightResult preflight = _validate_sort_preflight(keys_buffer, values_buffer, 0, count_buffer, true);
    if (!preflight.is_ok()) {
        last_preflight_error = preflight.code;
        GS_LOG_ERROR_DEFAULT(vformat("RadixSort::sort_indirect_async preflight failed (%s): %s",
                _sort_preflight_error_name(preflight.code), preflight.message));
        return 0;
    }
    last_preflight_error = SortPreflightError::NONE;

    // GPU-driven async path: dispatch sort without CPU blocking.
    // GPU command ordering ensures sort completes before subsequent rasterization.
    return _sort_indirect_internal(keys_buffer, values_buffer, count_buffer);
}

uint64_t RadixSort::sort_async(RID keys_buffer, RID values_buffer, uint32_t count) {
    if (count == 0) {
        WARN_PRINT_ONCE("RadixSort::sort_async called with zero elements; nothing to sort.");
        return ++timeline_value; // Return valid timeline value (no-op sort).
    }
    return _sort_async_internal(keys_buffer, values_buffer, count, RID(), 0, RID(), 0);
}

uint64_t RadixSort::sort_async_with_timeline(RID keys_buffer, RID values_buffer, uint32_t count, RID p_wait_semaphore,
        uint64_t p_wait_value, RID p_signal_semaphore, uint64_t p_signal_value) {
    if (count == 0) {
        WARN_PRINT_ONCE("RadixSort::sort_async_with_timeline called with zero elements; nothing to sort.");
        return ++timeline_value; // Return valid timeline value (no-op sort).
    }
    return _sort_async_internal(keys_buffer, values_buffer, count, p_wait_semaphore, p_wait_value, p_signal_semaphore, p_signal_value);
}

uint64_t RadixSort::_sort_indirect_internal(RID keys_buffer, RID values_buffer, RID count_buffer) {
    SortPreflightResult preflight = _validate_sort_preflight(keys_buffer, values_buffer, 0, count_buffer, true);
    if (!preflight.is_ok()) {
        last_preflight_error = preflight.code;
        GS_LOG_ERROR_DEFAULT(vformat("RadixSort::_sort_indirect_internal preflight failed (%s): %s",
                _sort_preflight_error_name(preflight.code), preflight.message));
        return 0;
    }
    last_preflight_error = SortPreflightError::NONE;

    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *compute_rd = _acquire_submission_device(local_rd, submission_lock);
    ERR_FAIL_NULL_V_MSG(compute_rd, 0, "Local rendering device unavailable for RadixSort::_sort_indirect_internal");

    RenderingDevice *resource_rd = resource_device;
    ERR_FAIL_NULL_V_MSG(resource_rd, 0, "Primary rendering device unavailable for RadixSort::_sort_indirect_internal");

    // Use primary variant for indirect sorting
    const RadixVariant *variant = get_variant(primary_radix_bits);
    ERR_FAIL_NULL_V_MSG(variant, 0, "Radix sort variant not initialized");

    if (is_sorting.load()) {
        GS_LOG_ERROR_DEFAULT("RadixSort::_sort_indirect_internal called while a sort is already in flight");
        return 0;
    }

    // ISSUE-010: Use generation-safe cleanup for prior uniform sets.
    _free_uniform_sets_safe(uniform_owner, uniform_owner_generation, resource_rd, uniform_sets);

    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    is_sorting = true;

    // GPU-driven: fall back to max_elements on direct dispatch; indirect dispatch uses GPU-prepared args.
    uint32_t num_passes = variant->num_passes;
    uint32_t workgroups = get_workgroup_count(max_elements);

    PassUniformSets pass_uniform_sets;
    if (_create_pass_uniform_sets(resource_rd, variant, keys_buffer, values_buffer, count_buffer, "GS_RadixSort64", pass_uniform_sets) != OK) {
        GS_LOG_ERROR_DEFAULT("RadixSort::_sort_indirect_internal failed to create reused pass uniform sets");
        is_sorting = false;
        return 0;
    }

    RID dispatch_uniform_set;
    RID dispatch_args_buffer = indirect_dispatch_args_buffer;
    bool use_indirect_dispatch = false;
    if (indirect_dispatch_shader.is_valid() && indirect_dispatch_pipeline.is_valid() && dispatch_args_buffer.is_valid()) {
        Vector<RD::Uniform> dispatch_uniforms;
        dispatch_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, count_buffer));
        dispatch_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, dispatch_args_buffer));
        dispatch_uniform_set = resource_rd->uniform_set_create(dispatch_uniforms, indirect_dispatch_shader, 0);
        if (dispatch_uniform_set.is_valid()) {
            resource_rd->set_resource_name(dispatch_uniform_set, "GS_RadixSort_IndirectDispatchSet");
            uniform_sets.push_back(dispatch_uniform_set);
            use_indirect_dispatch = true;
        }
    }


    if (use_indirect_dispatch) {
        ScopedGpuMarkerEx dispatch_marker(compute_rd, "GS_RadixSort_IndirectArgs", PassColors::SORTING);
        ComputeListID dispatch_list = compute_rd->compute_list_begin();
        if (dispatch_list != RD::INVALID_ID) {
            compute_rd->compute_list_bind_compute_pipeline(dispatch_list, indirect_dispatch_pipeline);
            compute_rd->compute_list_bind_uniform_set(dispatch_list, dispatch_uniform_set, 0);
            struct DispatchParams {
                uint32_t workgroup_size;
                uint32_t pad0;
                uint32_t pad1;
                uint32_t pad2;
            } dispatch_params = { workgroup_size, 0u, 0u, 0u };
            compute_rd->compute_list_set_push_constant(dispatch_list, &dispatch_params, sizeof(dispatch_params));
            compute_rd->compute_list_dispatch(dispatch_list, 1, 1, 1);
            compute_rd->compute_list_add_barrier(dispatch_list);
            compute_rd->compute_list_end();
            gs_device_utils::safe_submit(compute_rd);
        } else {
            use_indirect_dispatch = false;
        }
    }

    auto record_commands = [this, variant, pass_uniform_sets, num_passes, workgroups,
            use_indirect_dispatch, dispatch_args_buffer](RenderingDevice *command_rd, ComputeListID compute_list) {
        for (uint32_t pass = 0; pass < num_passes; ++pass) {
            bool use_primary = (pass & 1) == 0;
            RID histogram_set = use_primary ? pass_uniform_sets.histogram_even : pass_uniform_sets.histogram_odd;
            RID scatter_set = use_primary ? pass_uniform_sets.scatter_even : pass_uniform_sets.scatter_odd;
            uint32_t histogram_offset = pass * histogram_stride;
            uint32_t bin_offset = pass * variant->radix_size;

            // GPU Debug: Histogram pass (64-bit keys)
            {
                ScopedGpuMarkerEx histogram_marker(command_rd, "GS_RadixSort64_Histogram", PassColors::SORTING);
                command_rd->compute_list_bind_compute_pipeline(compute_list, variant->histogram_pipeline);
                command_rd->compute_list_bind_uniform_set(compute_list, histogram_set, 0);

                // Struct padded to 16 bytes for SPIR-V alignment
                struct HistogramParams {
                    uint32_t bit_shift;
                    uint32_t histogram_offset;
                    uint32_t workgroup_stride;
                    uint32_t _pad0;
                } hist_params = { pass * variant->radix_bits, histogram_offset, workgroup_stride, 0 };

                command_rd->compute_list_set_push_constant(compute_list, &hist_params, sizeof(hist_params));
                if (use_indirect_dispatch) {
                    command_rd->compute_list_dispatch_indirect(compute_list, dispatch_args_buffer, 0);
                } else {
                    command_rd->compute_list_dispatch(compute_list, workgroups, 1, 1);
                }
            }
            command_rd->compute_list_add_barrier(compute_list);

            // GPU Debug: Workgroup prefix sum pass (64-bit keys)
            {
                _record_wg_prefix_pass(command_rd, compute_list, variant, pass_uniform_sets.wg_prefix,
                        histogram_offset, workgroup_stride, bin_offset, "GS_RadixSort64_WGPrefix");
            }
            command_rd->compute_list_add_barrier(compute_list);

            // GPU Debug: Bin prefix sum pass (64-bit keys)
            {
                _record_bin_prefix_pass(command_rd, compute_list, variant, pass_uniform_sets.bin_prefix, bin_offset,
                        "GS_RadixSort64_BinPrefix");
            }
            command_rd->compute_list_add_barrier(compute_list);

            // GPU Debug: Scatter pass (64-bit keys)
            {
                _record_scatter_pass(command_rd, compute_list, variant, scatter_set,
                        pass * variant->radix_bits, histogram_offset, workgroup_stride, bin_offset, workgroups,
                        use_indirect_dispatch, dispatch_args_buffer, "GS_RadixSort64_Scatter");
            }

            if (pass + 1 < num_passes) {
                command_rd->compute_list_add_barrier(compute_list);
            }
        }

        ERR_FAIL_COND_MSG((num_passes & 1) != 0,
                "RadixSort assumes an even number of passes so results stay in the source buffer.");
    };

    uint64_t submit_start = OS::get_singleton()->get_ticks_usec();

    ScopedGpuMarkerEx sort_marker(compute_rd, "GS_RadixSort64", PassColors::SORTING);
    ComputeListID compute_list = compute_rd->compute_list_begin();
    record_commands(compute_rd, compute_list);
    compute_rd->compute_list_end();
    // PERF: Use safe_submit without sync - GPU barriers ensure correct ordering.
    // Blocking sync here was causing 15 FPS cap.
    gs_device_utils::safe_submit(compute_rd);

    current_sort_value = ++timeline_value;

    float cpu_submit_time_ms = (OS::get_singleton()->get_ticks_usec() - submit_start) / 1000.0f;
    metrics_collector.record_async_sort(0, cpu_submit_time_ms);

    // ISSUE-010: Use generation-safe cleanup.
    _free_uniform_sets_safe(uniform_owner, uniform_owner_generation, resource_rd, uniform_sets);
    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    is_sorting = false;
    return current_sort_value;
}

bool RadixSort::is_ready() const {
    return !is_sorting.load();
}

void RadixSort::wait_for_completion() {
    if (!is_sorting.load()) {
        // ISSUE-010: Use generation-safe cleanup to avoid dereferencing stale device pointers.
        _free_uniform_sets_safe(uniform_owner, uniform_owner_generation, resource_device, uniform_sets);
        uniform_owner = nullptr;
        uniform_owner_generation = 0;
        return;
    }

    if (local_rd) {
        GaussianSplatManager::ScopedSubmissionLock submission_lock;
        if (RenderingDevice *submission_rd = _acquire_submission_device(local_rd, submission_lock)) {
            // Flush the shared queue to honour the host timeline.
            gs_device_utils::safe_submit_and_sync(submission_rd);
        }
    }

    // ISSUE-010: Use generation-safe cleanup to avoid dereferencing stale device pointers.
    _free_uniform_sets_safe(uniform_owner, uniform_owner_generation, resource_device, uniform_sets);
    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    current_sort_value = 0;
    is_sorting = false;
}

// OneSweepSort Implementation

OneSweepSort::OneSweepSort() {
    current_sort_value = 0;
}

OneSweepSort::~OneSweepSort() {
    shutdown();
}

void OneSweepSort::_bind_methods() {
    // Bind methods for GDScript if needed
}

Error OneSweepSort::initialize(RenderingDevice *p_rd, uint32_t p_max_elements) {
    if (!p_rd || p_max_elements == 0) {
        return ERR_INVALID_PARAMETER;
    }

    // Clean up any existing resources before re-initializing
    shutdown();

    rd = p_rd;

    RenderingDevice *resource_rd = nullptr;
    Error init_err = _init_sorter_devices(rd, local_rd, resource_rd);
    if (init_err != OK) {
        shutdown();
        return init_err;
    }

    max_elements = p_max_elements;

    resource_device = resource_rd;
    resource_device_generation = resource_rd->get_device_instance_id();
    RenderingDevice *command_rd = local_rd;

    if (!device_supports_workgroup(command_rd, WORKGROUP_SIZE)) {
        GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] OneSweepSort disabled; device workgroup capacity below %d threads",
                WORKGROUP_SIZE));
        shutdown();
        return ERR_UNAVAILABLE;
    }

    auto fail = [&](Error err) {
        shutdown();
        return err;
    };

    // Create global histogram shader
    RID global_histogram_shader_file = create_compute_shader_from_spirv(resource_rd,
            vformat(R"(
#version 450

#define RADIX_BITS %d
#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d

layout(local_size_x = WORKGROUP_SIZE) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    float keys[];
} keys_in;

layout(set = 0, binding = 1, std430) restrict buffer GlobalHistogram {
    uint global_hist[];
} global_histogram;

layout(push_constant) uniform PushConstants {
    uint num_keys;
    uint bit_shift;
    uint pad0;
    uint pad1;
} params;

shared uint local_histogram[RADIX_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Initialize local histogram
    if (tid < RADIX_SIZE) {
        local_histogram[tid] = 0;
    }
    barrier();

    // Count keys
    uint elements_per_workgroup = (params.num_keys + gl_NumWorkGroups.x - 1) / gl_NumWorkGroups.x;
    uint start_idx = gl_WorkGroupID.x * elements_per_workgroup;
    uint end_idx = min(start_idx + elements_per_workgroup, params.num_keys);

    for (uint idx = start_idx + tid; idx < end_idx; idx += WORKGROUP_SIZE) {
        if (idx < params.num_keys) {
            float key = keys_in.keys[idx];
            uint ikey = floatBitsToUint(key);
            uint digit = (ikey >> params.bit_shift) & (RADIX_SIZE - 1);
            atomicAdd(local_histogram[digit], 1);
        }
    }
    barrier();

    // Accumulate to global histogram
    if (tid < RADIX_SIZE) {
        atomicAdd(global_histogram.global_hist[tid], local_histogram[tid]);
    }
}
            )",
                    RADIX_BITS,
                    RADIX_SIZE,
                    WORKGROUP_SIZE));
    if (!global_histogram_shader_file.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    global_histogram_shader = global_histogram_shader_file;

    // Create digit binning shader for chained scan
    RID digit_binning_shader_file = create_compute_shader_from_spirv(resource_rd,
            vformat(R"(
#version 450

#define RADIX_BITS %d
#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d
#define CHAINING_FACTOR %d

layout(local_size_x = WORKGROUP_SIZE) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    float keys[];
} keys_in;

layout(set = 0, binding = 1, std430) restrict buffer DigitHistogram {
    uint digit_hist[];
} digit_histogram;

layout(set = 0, binding = 2, std430) restrict buffer ChainedScan {
    uint chained_scan[];
} chained_scan_buf;

layout(push_constant) uniform PushConstants {
    uint num_keys;
    uint bit_shift;
    uint pass_id;
    uint pad0;
} params;

shared uint local_histogram[RADIX_SIZE];
shared uint local_scan[RADIX_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Initialize shared memory
    if (tid < RADIX_SIZE) {
        local_histogram[tid] = 0;
        local_scan[tid] = 0;
    }
    barrier();

    // Count digits in this workgroup
    uint elements_per_workgroup = (params.num_keys + gl_NumWorkGroups.x - 1) / gl_NumWorkGroups.x;
    uint start_idx = wid * elements_per_workgroup;
    uint end_idx = min(start_idx + elements_per_workgroup, params.num_keys);

    for (uint idx = start_idx + tid; idx < end_idx; idx += WORKGROUP_SIZE) {
        if (idx < params.num_keys) {
            float key = keys_in.keys[idx];
            uint ikey = floatBitsToUint(key);
            uint digit = (ikey >> params.bit_shift) & (RADIX_SIZE - 1);
            atomicAdd(local_histogram[digit], 1);
        }
    }
    barrier();

    // Store local histogram
    if (tid < RADIX_SIZE) {
        digit_histogram.digit_hist[wid * RADIX_SIZE + tid] = local_histogram[tid];
    }

    // Chained scan within workgroup
    if (tid < RADIX_SIZE) {
        local_scan[tid] = local_histogram[tid];
    }
    barrier();

    // Perform scan
    for (uint stride = 1; stride < RADIX_SIZE; stride *= 2) {
        if (tid >= stride && tid < RADIX_SIZE) {
            local_scan[tid] += local_scan[tid - stride];
        }
        barrier();
    }

    // Store scan results for chaining
    if (tid < RADIX_SIZE) {
        chained_scan_buf.chained_scan[wid * RADIX_SIZE + tid] =
            (tid > 0) ? local_scan[tid - 1] : 0;
    }
}
            )",
                    RADIX_BITS,
                    RADIX_SIZE,
                    WORKGROUP_SIZE,
                    CHAINING_FACTOR));
    if (!digit_binning_shader_file.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    digit_binning_shader = digit_binning_shader_file;

    // Create chained scan shader
    RID chained_scan_shader_file = create_compute_shader_from_spirv(resource_rd,
            vformat(R"(
#version 450

#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d

layout(local_size_x = WORKGROUP_SIZE) in;

layout(set = 0, binding = 0, std430) restrict buffer GlobalHistogram {
    uint global_hist[];
} global_histogram;

layout(set = 0, binding = 1, std430) restrict buffer ChainedScan {
    uint chained_scan[];
} chained_scan_buf;

layout(push_constant) uniform PushConstants {
    uint num_workgroups;
    uint pad0;
    uint pad1;
    uint pad2;
} params;

shared uint scan_scratch[RADIX_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Load global histogram
    if (tid < RADIX_SIZE) {
        scan_scratch[tid] = global_histogram.global_hist[tid];
    }
    barrier();

    // Exclusive scan
    for (uint stride = 1; stride < RADIX_SIZE; stride *= 2) {
        if (tid >= stride && tid < RADIX_SIZE) {
            scan_scratch[tid] += scan_scratch[tid - stride];
        }
        barrier();
    }

    // Store prefix sums
    if (tid < RADIX_SIZE) {
        global_histogram.global_hist[tid] = (tid > 0) ? scan_scratch[tid - 1] : 0;

        // Chain with local scans
        for (uint wg = 0; wg < params.num_workgroups; ++wg) {
            chained_scan_buf.chained_scan[wg * RADIX_SIZE + tid] +=
                global_histogram.global_hist[tid];
        }
    }
}
            )",
                    RADIX_SIZE,
                    WORKGROUP_SIZE));
    if (!chained_scan_shader_file.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    chained_scan_shader = chained_scan_shader_file;

    // Create scatter shader
    RID scatter_shader_file = create_compute_shader_from_spirv(resource_rd,
            vformat(R"(
#version 450

#define RADIX_BITS %d
#define RADIX_SIZE %d
#define WORKGROUP_SIZE %d

layout(local_size_x = WORKGROUP_SIZE) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer KeysIn {
    float keys[];
} keys_in;

layout(set = 0, binding = 1, std430) restrict writeonly buffer KeysOut {
    float keys[];
} keys_out;

layout(set = 0, binding = 2, std430) restrict readonly buffer ValuesIn {
    uint values[];
} values_in;

layout(set = 0, binding = 3, std430) restrict writeonly buffer ValuesOut {
    uint values[];
} values_out;

layout(set = 0, binding = 4, std430) restrict buffer ChainedScan {
    uint chained_scan[];
} chained_scan_buf;

layout(push_constant) uniform PushConstants {
    uint num_keys;
    uint bit_shift;
    uint pad0;
    uint pad1;
} params;

shared uint local_offsets[RADIX_SIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Load chained scan offsets
    if (tid < RADIX_SIZE) {
        local_offsets[tid] = chained_scan_buf.chained_scan[wid * RADIX_SIZE + tid];
    }
    barrier();

    // Scatter elements
    uint elements_per_workgroup = (params.num_keys + gl_NumWorkGroups.x - 1) / gl_NumWorkGroups.x;
    uint start_idx = wid * elements_per_workgroup;
    uint end_idx = min(start_idx + elements_per_workgroup, params.num_keys);

    for (uint idx = start_idx + tid; idx < end_idx; idx += WORKGROUP_SIZE) {
        if (idx < params.num_keys) {
            float key = keys_in.keys[idx];
            uint value = values_in.values[idx];
            uint ikey = floatBitsToUint(key);
            uint digit = (ikey >> params.bit_shift) & (RADIX_SIZE - 1);

            uint output_pos = atomicAdd(local_offsets[digit], 1);
            keys_out.keys[output_pos] = key;
            values_out.values[output_pos] = value;
        }
    }
}
            )",
                    RADIX_BITS,
                    RADIX_SIZE,
                    WORKGROUP_SIZE));
    if (!scatter_shader_file.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    scatter_shader = scatter_shader_file;

    // Create compute pipelines
    global_histogram_pipeline = resource_rd->compute_pipeline_create(global_histogram_shader);
    if (!global_histogram_pipeline.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }

    digit_binning_pipeline = resource_rd->compute_pipeline_create(digit_binning_shader);
    if (!digit_binning_pipeline.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }

    chained_scan_pipeline = resource_rd->compute_pipeline_create(chained_scan_shader);
    if (!chained_scan_pipeline.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }

    scatter_pipeline = resource_rd->compute_pipeline_create(scatter_shader);
    if (!scatter_pipeline.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }

    // Create buffers
    uint32_t max_workgroups = (max_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    global_histogram_buffer = resource_rd->storage_buffer_create(RADIX_SIZE * sizeof(uint32_t));
    if (!global_histogram_buffer.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    resource_rd->set_resource_name(global_histogram_buffer, "GS_OneSweepGlobalHistogramBuffer");

    digit_histogram_buffer = resource_rd->storage_buffer_create(max_workgroups * RADIX_SIZE * sizeof(uint32_t));
    if (!digit_histogram_buffer.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    resource_rd->set_resource_name(digit_histogram_buffer, "GS_OneSweepDigitHistogramBuffer");

    chained_scan_buffer = resource_rd->storage_buffer_create(max_workgroups * RADIX_SIZE * sizeof(uint32_t));
    if (!chained_scan_buffer.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    resource_rd->set_resource_name(chained_scan_buffer, "GS_OneSweepChainedScanBuffer");

    temp_keys_buffer = resource_rd->storage_buffer_create(max_elements * sizeof(float));
    if (!temp_keys_buffer.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    resource_rd->set_resource_name(temp_keys_buffer, "GS_OneSweepTempKeysBuffer");

    temp_values_buffer = resource_rd->storage_buffer_create(max_elements * sizeof(uint32_t));
    if (!temp_values_buffer.is_valid()) {
        return fail(ERR_CANT_CREATE);
    }
    resource_rd->set_resource_name(temp_values_buffer, "GS_OneSweepTempValuesBuffer");

    // Timeline synchronization is driven by timeline_value; no device-level semaphore setup required.

    GS_LOG_INFO_DEFAULT("OneSweepSort initialized for up to " + itos(max_elements) + " elements");
    return OK;
}

void OneSweepSort::shutdown() {
    wait_for_completion();

    RenderingDevice *device = resource_device;
    bool device_still_valid = _device_is_active(device);

    // ISSUE-010: Also validate device generation to detect recycled/stale pointers.
    if (device_still_valid && resource_device_generation != 0) {
        device_still_valid = ResourceOwnerMismatchContract::is_device_generation_valid(
                device, resource_device_generation);
    }

    if (device_still_valid && device) {
        if (global_histogram_pipeline.is_valid() && device->compute_pipeline_is_valid(global_histogram_pipeline)) {
            device->free(global_histogram_pipeline);
        }
        if (digit_binning_pipeline.is_valid() && device->compute_pipeline_is_valid(digit_binning_pipeline)) {
            device->free(digit_binning_pipeline);
        }
        if (chained_scan_pipeline.is_valid() && device->compute_pipeline_is_valid(chained_scan_pipeline)) {
            device->free(chained_scan_pipeline);
        }
        if (scatter_pipeline.is_valid() && device->compute_pipeline_is_valid(scatter_pipeline)) {
            device->free(scatter_pipeline);
        }
        if (global_histogram_shader.is_valid()) {
            device->free(global_histogram_shader);
        }
        if (digit_binning_shader.is_valid()) {
            device->free(digit_binning_shader);
        }
        if (chained_scan_shader.is_valid()) {
            device->free(chained_scan_shader);
        }
        if (scatter_shader.is_valid()) {
            device->free(scatter_shader);
        }
        if (global_histogram_buffer.is_valid()) {
            device->free(global_histogram_buffer);
        }
        if (digit_histogram_buffer.is_valid()) {
            device->free(digit_histogram_buffer);
        }
        if (chained_scan_buffer.is_valid()) {
            device->free(chained_scan_buffer);
        }
        if (temp_keys_buffer.is_valid()) {
            device->free(temp_keys_buffer);
        }
        if (temp_values_buffer.is_valid()) {
            device->free(temp_values_buffer);
        }
    }

    global_histogram_shader = RID();
    digit_binning_shader = RID();
    chained_scan_shader = RID();
    scatter_shader = RID();
    global_histogram_pipeline = RID();
    digit_binning_pipeline = RID();
    chained_scan_pipeline = RID();
    scatter_pipeline = RID();
    global_histogram_buffer = RID();
    digit_histogram_buffer = RID();
    chained_scan_buffer = RID();
    temp_keys_buffer = RID();
    temp_values_buffer = RID();

    uniform_sets.clear();
    uniform_owner = nullptr;
    uniform_owner_generation = 0;

    local_rd = nullptr;
    resource_device = nullptr;
    resource_device_generation = 0;
    rd = nullptr;
}

Error OneSweepSort::sort(RID keys_buffer, RID values_buffer, uint32_t count) {
    ERR_FAIL_COND_V(!keys_buffer.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(!values_buffer.is_valid(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(count == 0 || count > max_elements, ERR_INVALID_PARAMETER);

    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    RenderingDevice *compute_rd = _acquire_submission_device(local_rd, submission_lock);
    ERR_FAIL_NULL_V_MSG(compute_rd, ERR_CANT_CREATE, "Local rendering device unavailable for OneSweepSort::sort");

    RenderingDevice *resource_rd = resource_device;
    ERR_FAIL_NULL_V_MSG(resource_rd, ERR_CANT_CREATE, "Primary rendering device unavailable for OneSweepSort::sort");

    // Measures CPU-side command recording + synchronous submit time, NOT GPU execution time.
    // OneSweep uses safe_submit_and_sync per pass which includes GPU wait, so this is closer
    // to total wall-clock time but still not a pure GPU metric.
    // TODO: Replace with GPU timestamps when RenderingDevice exposes per-dispatch query API.
    start_cpu_record_timing();

    // OneSweep uses 8-bit radix, requiring 4 passes for 32-bit floats
    const uint32_t num_passes = 32 / RADIX_BITS;
    const uint32_t workgroups = (count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    RID current_keys = keys_buffer;
    RID current_values = values_buffer;
    RID next_keys = temp_keys_buffer;
    RID next_values = temp_values_buffer;

    for (uint32_t pass = 0; pass < num_passes; ++pass) {
        uint32_t bit_shift = pass * RADIX_BITS;

        // Use GPU buffer_clear() instead of CPU zero-init + buffer_update (BUF-1 optimization)
        uint32_t global_hist_size = RADIX_SIZE * sizeof(uint32_t);
        uint32_t per_wg_hist_size = workgroups * RADIX_SIZE * sizeof(uint32_t);
        resource_rd->buffer_clear(global_histogram_buffer, 0, global_hist_size);
        resource_rd->buffer_clear(digit_histogram_buffer, 0, per_wg_hist_size);
        resource_rd->buffer_clear(chained_scan_buffer, 0, per_wg_hist_size);
        gs_device_utils::safe_submit_and_sync(resource_rd);

        // Create uniform sets
        Vector<RD::Uniform> global_hist_uniforms;
        global_hist_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, current_keys));
        global_hist_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, global_histogram_buffer));
        RID global_hist_uniform_set = resource_rd->uniform_set_create(global_hist_uniforms, global_histogram_shader, 0);
        if (global_hist_uniform_set.is_valid()) {
            resource_rd->set_resource_name(global_hist_uniform_set, vformat("GS_OneSweep_GlobalHistSet_Pass%d", pass));
        }

        Vector<RD::Uniform> digit_binning_uniforms;
        digit_binning_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, current_keys));
        digit_binning_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, digit_histogram_buffer));
        digit_binning_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, chained_scan_buffer));
        RID digit_binning_uniform_set = resource_rd->uniform_set_create(digit_binning_uniforms, digit_binning_shader, 0);
        if (digit_binning_uniform_set.is_valid()) {
            resource_rd->set_resource_name(digit_binning_uniform_set, vformat("GS_OneSweep_DigitBinningSet_Pass%d", pass));
        }

        Vector<RD::Uniform> chained_scan_uniforms;
        chained_scan_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, global_histogram_buffer));
        chained_scan_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, chained_scan_buffer));
        RID chained_scan_uniform_set = resource_rd->uniform_set_create(chained_scan_uniforms, chained_scan_shader, 0);
        if (chained_scan_uniform_set.is_valid()) {
            resource_rd->set_resource_name(chained_scan_uniform_set, vformat("GS_OneSweep_ChainedScanSet_Pass%d", pass));
        }

        Vector<RD::Uniform> scatter_uniforms;
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, current_keys));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, next_keys));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, current_values));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, next_values));
        scatter_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, chained_scan_buffer));
        RID scatter_uniform_set = resource_rd->uniform_set_create(scatter_uniforms, scatter_shader, 0);
        if (scatter_uniform_set.is_valid()) {
            resource_rd->set_resource_name(scatter_uniform_set, vformat("GS_OneSweep_ScatterSet_Pass%d", pass));
        }

        ComputeListID compute_list = compute_rd->compute_list_begin();

        // Phase 1: Global histogram
        compute_rd->compute_list_bind_compute_pipeline(compute_list, global_histogram_pipeline);
        compute_rd->compute_list_bind_uniform_set(compute_list, global_hist_uniform_set, 0);

        struct GlobalHistParams {
            uint32_t num_keys;
            uint32_t bit_shift;
            uint32_t pad0;
            uint32_t pad1;
        } global_params = { count, bit_shift, 0, 0 };

        compute_rd->compute_list_set_push_constant(compute_list, &global_params, sizeof(global_params));
        compute_rd->compute_list_dispatch(compute_list, workgroups, 1, 1);
        compute_rd->compute_list_add_barrier(compute_list);

        // Phase 2: Digit binning
        compute_rd->compute_list_bind_compute_pipeline(compute_list, digit_binning_pipeline);
        compute_rd->compute_list_bind_uniform_set(compute_list, digit_binning_uniform_set, 0);

        struct DigitBinningParams {
            uint32_t num_keys;
            uint32_t bit_shift;
            uint32_t pass_id;
            uint32_t pad0;
        } binning_params = { count, bit_shift, pass, 0 };

        compute_rd->compute_list_set_push_constant(compute_list, &binning_params, sizeof(binning_params));
        compute_rd->compute_list_dispatch(compute_list, workgroups, 1, 1);
        compute_rd->compute_list_add_barrier(compute_list);

        // Phase 3: Chained scan
        compute_rd->compute_list_bind_compute_pipeline(compute_list, chained_scan_pipeline);
        compute_rd->compute_list_bind_uniform_set(compute_list, chained_scan_uniform_set, 0);

        struct ChainedScanParams {
            uint32_t num_workgroups;
            uint32_t pad0;
            uint32_t pad1;
            uint32_t pad2;
        } scan_params = { workgroups, 0, 0, 0 };

        compute_rd->compute_list_set_push_constant(compute_list, &scan_params, sizeof(scan_params));
        compute_rd->compute_list_dispatch(compute_list, 1, 1, 1);
        compute_rd->compute_list_add_barrier(compute_list);

        // Phase 4: Scatter
        compute_rd->compute_list_bind_compute_pipeline(compute_list, scatter_pipeline);
        compute_rd->compute_list_bind_uniform_set(compute_list, scatter_uniform_set, 0);
        compute_rd->compute_list_set_push_constant(compute_list, &global_params, sizeof(global_params));
        compute_rd->compute_list_dispatch(compute_list, workgroups, 1, 1);

        compute_rd->compute_list_end();
        gs_device_utils::safe_submit_and_sync(compute_rd);

        // Clean up uniform sets
        if (resource_rd->uniform_set_is_valid(global_hist_uniform_set)) {
            resource_rd->free(global_hist_uniform_set);
        }
        if (resource_rd->uniform_set_is_valid(digit_binning_uniform_set)) {
            resource_rd->free(digit_binning_uniform_set);
        }
        if (resource_rd->uniform_set_is_valid(chained_scan_uniform_set)) {
            resource_rd->free(chained_scan_uniform_set);
        }
        if (resource_rd->uniform_set_is_valid(scatter_uniform_set)) {
            resource_rd->free(scatter_uniform_set);
        }

        // Ping-pong buffers
        std::swap(current_keys, next_keys);
        std::swap(current_values, next_values);
    }

    // Copy final result back if needed
    if (current_keys != keys_buffer) {
        compute_rd->buffer_copy(current_keys, keys_buffer, 0, 0, count * sizeof(float));
        compute_rd->buffer_copy(current_values, values_buffer, 0, 0, count * sizeof(uint32_t));
    }

    // Update metrics (CPU recording + sync time, not pure GPU execution time).
    float sort_time_ms = end_cpu_record_timing();
    metrics_collector.record_sort(count, sort_time_ms, true);

    // Log performance data for Issue #126.
    log_sorting_performance(count, sort_time_ms, get_algorithm_name());
    return OK;
}

uint64_t OneSweepSort::sort_async(RID keys_buffer, RID values_buffer, uint32_t count) {
    Error err = sort(keys_buffer, values_buffer, count);
    if (err != OK) {
        return 0;
    }
    return ++timeline_value;
}

bool OneSweepSort::is_ready() const {
    return !is_sorting.load();
}

void OneSweepSort::wait_for_completion() {
    // Uniform sets auto-free when dependencies are freed (Godot PR 103113)

    uniform_sets.clear();
    uniform_owner = nullptr;

    current_sort_value = 0;
    is_sorting = false;
}


// ============================================================================
// Static Capability Probes
// ============================================================================

// BitonicSort capability probes
bool BitonicSort::is_supported(RenderingDevice *p_rd) {
    const ComputeCapabilityProbe probe = _probe_compute_capabilities(p_rd);
    return _supports_compute_profile(probe, WORKGROUP_SIZE, 2u, WORKGROUP_SIZE * sizeof(uint32_t));
}

SorterCapabilities BitonicSort::get_capabilities() {
    SorterCapabilities caps;
    caps.required_workgroup_size = WORKGROUP_SIZE;
    caps.max_supported_key_bits = 32;  // Bitonic uses float keys
    caps.supports_indirect = false;
    caps.supports_64bit_keys = false;
    caps.requires_power_of_two = false;  // We pad internally
    return caps;
}

// RadixSort capability probes
bool RadixSort::is_supported(RenderingDevice *p_rd) {
    const GPUSortingConfig &config = g_gpu_sorting_config;
    uint32_t required_workgroup_size = config.workgroup_size > 0 ? config.workgroup_size : GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    const uint32_t required_shared_memory = GPUSortingConstants::RADIX_SIZE * sizeof(uint32_t);
    const ComputeCapabilityProbe probe = _probe_compute_capabilities(p_rd);
    return _supports_compute_profile(probe, required_workgroup_size, MIN_STORAGE_BUFFERS_PER_SET, required_shared_memory);
}

SorterCapabilities RadixSort::get_capabilities() {
    const GPUSortingConfig &config = g_gpu_sorting_config;
    uint32_t required_workgroup_size = config.workgroup_size > 0 ? config.workgroup_size : GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    SorterCapabilities caps;
    caps.required_workgroup_size = required_workgroup_size;
    caps.max_supported_key_bits = 64;
    caps.supports_indirect = true;
    caps.supports_64bit_keys = true;
    caps.requires_power_of_two = false;
    return caps;
}

// OneSweepSort capability probes
bool OneSweepSort::is_supported(RenderingDevice *p_rd) {
    const ComputeCapabilityProbe probe = _probe_compute_capabilities(p_rd);
    if (!_supports_compute_profile(probe, WORKGROUP_SIZE, MIN_STORAGE_BUFFERS_PER_SET, WORKGROUP_SIZE * sizeof(uint32_t))) {
        return false;
    }
    return _supports_required_subgroups(probe);
}

SorterCapabilities OneSweepSort::get_capabilities() {
    SorterCapabilities caps;
    caps.required_workgroup_size = WORKGROUP_SIZE;
    caps.max_supported_key_bits = 32;
    caps.supports_indirect = false;
    caps.supports_64bit_keys = false;
    caps.requires_power_of_two = false;
    return caps;
}

static AlgorithmProbe _probe_algorithm(GPUSorterFactory::SortingAlgorithm algorithm, RenderingDevice *rd) {
    AlgorithmProbe probe;
    switch (algorithm) {
        case GPUSorterFactory::ALGORITHM_BITONIC:
            probe.supported = rd ? BitonicSort::is_supported(rd) : true;
            probe.supports_indirect = false;
            probe.capabilities = BitonicSort::get_capabilities();
            break;
        case GPUSorterFactory::ALGORITHM_RADIX:
            probe.supported = rd ? RadixSort::is_supported(rd) : true;
            probe.capabilities = RadixSort::get_capabilities();
            probe.supports_indirect = probe.capabilities.supports_indirect && probe.supported;
            break;
        case GPUSorterFactory::ALGORITHM_ONESWEEP:
            probe.supported = rd ? OneSweepSort::is_supported(rd) : true;
            probe.supports_indirect = false;
            probe.capabilities = OneSweepSort::get_capabilities();
            break;
        case GPUSorterFactory::ALGORITHM_AUTO:
            probe.supported = rd ? (RadixSort::is_supported(rd) || BitonicSort::is_supported(rd) || OneSweepSort::is_supported(rd)) : true;
            probe.supports_indirect = rd ? RadixSort::is_supported(rd) : true;
            probe.capabilities = RadixSort::get_capabilities();
            break;
        default:
            break;
    }
    return probe;
}

// Factory probe implementations
bool GPUSorterFactory::probe_is_supported(SortingAlgorithm algorithm, RenderingDevice *rd) {
    return _probe_algorithm(algorithm, rd).supported;
}

bool GPUSorterFactory::probe_supports_indirect(SortingAlgorithm algorithm) {
    return _probe_algorithm(algorithm, nullptr).supports_indirect;
}

bool GPUSorterFactory::probe_supports_indirect(SortingAlgorithm algorithm, RenderingDevice *rd) {
    return _probe_algorithm(algorithm, rd).supports_indirect;
}

SorterCapabilities GPUSorterFactory::probe_capabilities(SortingAlgorithm algorithm) {
    return _probe_algorithm(algorithm, nullptr).capabilities;
}

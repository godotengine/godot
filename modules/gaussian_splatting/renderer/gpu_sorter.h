#ifndef GPU_SORTER_H
#define GPU_SORTER_H

// =============================================================================
// GPU Sorting Algorithms
// =============================================================================
// Three sorting backends are provided.  Each targets a different sweet-spot on
// the latency/throughput/capability curve:
//
// 1. BitonicSort - O(n log^2 n) parallel comparison sort.
//    - Simple, reliable, no auxiliary memory beyond the input buffers.
//    - Best for small datasets (< ~64 K elements) where launch overhead
//      dominates and the extra passes are negligible.
//    - Does NOT support indirect dispatch (element count must be known on CPU).
//
// 2. RadixSort - O(n) linear-time 4/8-bit radix sort.  ** RECOMMENDED DEFAULT **
//    - Supports indirect dispatch (GPU-driven element count).
//    - Supports 64-bit sort keys.
//    - Best general-purpose choice: high throughput, pipelined, indirect-capable.
//    - GPUSorterFactory::ALGORITHM_AUTO selects this for most workloads.
//
// 3. OneSweepSort - O(n) single-pass radix sort with chained decoupled look-back.
//    - Lowest theoretical constant factor for very large datasets (> 1 M).
//    - Does NOT support indirect dispatch (lacks GPU-driven element count path).
//    - More complex; requires chained-scan buffer and careful tuning.
//    - Prefer RadixSort unless benchmarks prove OneSweep is faster for the
//      target workload and indirect dispatch is not required.
//
// Use GPUSorterFactory to create instances.  ALGORITHM_AUTO will probe device
// capabilities and fall back through RadixSort -> BitonicSort as needed.
// =============================================================================

#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"
#include "gpu_sorting_constants.h"
#include <atomic>
#include <chrono>

// Forward declarations
class RenderingDevice;

struct SortKeyConfig {
    uint32_t key_bits = 64;     // Total bits in sort key (32 or 64)
    uint32_t tile_bits = 32;    // Bits reserved for tile_id
    uint32_t depth_bits = 32;   // Bits reserved for depth key
    bool enable_tie_breaker = false; // Reserve low bits for splat_id tie-break
    bool require_stable = false; // Force stable sorting (Radix only)

    static SortKeyConfig from_settings();
};

// Capability metadata for sorter algorithms - enables filtering without instantiation
struct SorterCapabilities {
    uint32_t required_workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;  // Minimum workgroup size required
    uint32_t max_supported_key_bits = 64;    // Maximum key bits supported
    bool supports_indirect = false;          // Can use GPU-driven element count
    bool supports_64bit_keys = false;        // Supports 64-bit sort keys
    bool requires_power_of_two = false;      // Requires power-of-two element count
};

// Sorting performance metrics
// NOTE: Timing values measure CPU-side command recording/submission time, NOT actual
// GPU execution time. Godot's RenderingDevice does not currently expose per-dispatch
// GPU timestamp queries. BitonicSort attempts to use capture_timestamp() for GPU timing
// when available, but OneSweep and RadixSort async paths report CPU recording time only.
// TODO: Replace with GPU timestamps when RenderingDevice exposes per-dispatch query API.
struct SortingMetrics {
    // CPU-side time to record/submit sort commands (not GPU execution time).
    // For BitonicSort, this may reflect actual GPU time when capture_timestamp() succeeds.
    float last_sort_time_ms = 0.0f;
    float avg_sort_time_ms = 0.0f;
    float peak_sort_time_ms = 0.0f;
    uint32_t total_sorts = 0;
    uint32_t async_sorts = 0;
    float async_speedup = 1.0f;
    uint64_t total_elements_sorted = 0;
    float bandwidth_utilization = 0.0f; // Percentage of theoretical max
    uint32_t fallback_events = 0;
    String last_fallback_reason;
    Dictionary fallback_reason_counts;
};

enum class SortPreflightError : uint8_t {
    NONE = 0,
    INVALID_KEYS_BUFFER,
    INVALID_VALUES_BUFFER,
    INVALID_COUNT_BUFFER,
    INVALID_ELEMENT_COUNT,
    ELEMENT_COUNT_EXCEEDS_CAPACITY,
    UNSUPPORTED_KEY_FORMAT,
    RESOURCE_DEVICE_UNAVAILABLE,
    SUBMISSION_DEVICE_UNAVAILABLE,
};

struct SortPreflightResult {
    SortPreflightError code = SortPreflightError::NONE;
    Error error = OK;
    String message;

    bool is_ok() const { return code == SortPreflightError::NONE; }
};

// Helper for consistent metrics tracking.
class SortingMetricsCollector {
public:
    void record_sort(uint32_t element_count, float time_ms, bool used_gpu);
    void record_async_sort(uint32_t element_count, float time_ms);
    void record_fallback(const String &reason);
    SortingMetrics get_metrics() const { return metrics; }
    float last_sort_time_ms() const { return metrics.last_sort_time_ms; }
    uint32_t total_sorts() const { return metrics.total_sorts; }
    void set_bandwidth_utilization(float utilization) { metrics.bandwidth_utilization = utilization; }
    void set_async_speedup(float speedup) { metrics.async_speedup = speedup; }
    void reset() { metrics = SortingMetrics(); }

private:
    SortingMetrics metrics;
};

// Abstract interface for GPU sorting algorithms
class IGPUSorter : public RefCounted {
    GDCLASS(IGPUSorter, RefCounted);

protected:
    RenderingDevice *rd = nullptr;
    RenderingDevice *local_rd = nullptr;
    SortingMetricsCollector metrics_collector;
    std::atomic<bool> is_sorting{false};
    uint32_t max_elements = 0;

    // CPU-side timing utilities for measuring command recording/submission time.
    // WARNING: These measure wall-clock time on the CPU, NOT actual GPU execution time.
    // GPU sort dispatches run asynchronously on the GPU after submission; the real GPU
    // execution time can only be measured via GPU timestamp queries.
    // TODO: Replace with RenderingDevice GPU timestamps when per-dispatch queries are available.
    std::chrono::high_resolution_clock::time_point sort_start_time;

    void start_cpu_record_timing() {
        sort_start_time = std::chrono::high_resolution_clock::now();
    }

    float end_cpu_record_timing() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - sort_start_time);
        return duration.count() / 1000.0f; // Convert to milliseconds
    }

    static void _bind_methods() {}

public:
    virtual ~IGPUSorter() = default;

    // Core sorting interface
    virtual Error initialize(RenderingDevice *p_rd, uint32_t p_max_elements) = 0;
    virtual void shutdown() = 0;

    // Synchronous sorting
    virtual Error sort(RID keys_buffer, RID values_buffer, uint32_t count) = 0;
    // GPU-driven element count (reads count from IndirectDispatchLayout::element_count).
    // Count buffer layout: dispatch_xyz[3] (12 bytes) + element_count (4 bytes).
    virtual Error sort_indirect(RID keys_buffer, RID values_buffer, RID count_buffer) { return ERR_UNAVAILABLE; }
    // Async variant - returns timeline value without CPU blocking. Use for pipelined rendering.
    virtual uint64_t sort_indirect_async(RID keys_buffer, RID values_buffer, RID count_buffer) { return 0; }

    // Asynchronous sorting with compute queue
    virtual uint64_t sort_async(RID keys_buffer, RID values_buffer, uint32_t count) = 0;
    virtual bool is_ready() const = 0;
    virtual void wait_for_completion() = 0;

    // Performance queries
    virtual float get_last_sort_time_ms() const { return metrics_collector.last_sort_time_ms(); }
    virtual SortingMetrics get_metrics() const { return metrics_collector.get_metrics(); }
    virtual void reset_metrics() { metrics_collector.reset(); }
    virtual void record_fallback_reason(const String &reason) { metrics_collector.record_fallback(reason); }

    // Algorithm-specific info
    virtual String get_algorithm_name() const = 0;
    virtual uint32_t get_max_elements() const { return max_elements; }
    virtual bool supports_non_power_of_two() const = 0;
    // DEPRECATED: prefer algorithm-specific benchmarks via get_metrics().
    // The returned float is a unitless relative indicator (lower = fewer passes):
    //   0.5 = O(n log^2 n) parallel (BitonicSort)
    //   1.0 = O(n) linear (RadixSort / OneSweepSort)
    // These values do NOT reflect real-world throughput.  Use SortingMetrics
    // (last_sort_time_ms, bandwidth_utilization) for actual performance data.
    virtual float get_theoretical_complexity() const = 0;

    // Capability queries - override in subclasses for algorithm-specific behavior
    virtual bool supports_indirect() const { return false; }

    // Key configuration (default no-op for algorithms that don't need it)
    virtual void set_key_config(const SortKeyConfig &p_cfg) {}
    virtual SortPreflightError get_last_preflight_error() const { return SortPreflightError::NONE; }
};

// Bitonic Sort implementation - Simple and reliable
class BitonicSort : public IGPUSorter {
    GDCLASS(BitonicSort, IGPUSorter);

private:
    struct BitonicParams {
        uint32_t stage = 0;
        uint32_t pass_in_stage = 0;
        uint32_t num_elements = 0;
        uint32_t block_size = 0;
    };

    // Shader resources
    RID bitonic_shader;
    RID bitonic_pipeline;
    RID uniform_set;
    RenderingDevice *uniform_owner = nullptr;
    uint64_t uniform_owner_generation = 0; // ISSUE-010: device instance ID at time of assignment
    RenderingDevice *pipeline_device = nullptr;
    RenderingDevice *resource_device = nullptr;
    uint64_t resource_device_generation = 0; // ISSUE-010: device instance ID at time of assignment

    // Async compute resources
    std::atomic<uint64_t> timeline_value{0};
    uint64_t current_sort_value = 0;

    // Local size configuration
    static constexpr uint32_t WORKGROUP_SIZE = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    static constexpr uint32_t MAX_SHARED_ELEMENTS = 2048; // Limited by shared memory

    // Helper methods
    uint32_t next_power_of_two(uint32_t n) const;
    void dispatch_bitonic_pass(RenderingDevice *p_rd, RenderingDevice::ComputeListID p_command_list, uint32_t stage, uint32_t pass, uint32_t num_elements);

    // Work batching optimization (Issue #108)
    bool _requires_synchronization(uint32_t stage, uint32_t pass, uint32_t num_elements) const;

protected:
    static void _bind_methods();

public:
    BitonicSort();
    ~BitonicSort() override;

    // IGPUSorter implementation
    Error initialize(RenderingDevice *p_rd, uint32_t p_max_elements) override;
    void shutdown() override;

    Error sort(RID keys_buffer, RID values_buffer, uint32_t count) override;
    uint64_t sort_async(RID keys_buffer, RID values_buffer, uint32_t count) override;

    bool is_ready() const override;
    void wait_for_completion() override;

    String get_algorithm_name() const override { return "Bitonic Sort"; }
    bool supports_non_power_of_two() const override { return true; } // We pad internally
    float get_theoretical_complexity() const override { return 0.5f; } // O(log^2 n) parallel
    bool supports_indirect() const override { return false; } // Bitonic does not support indirect

    // Static capability probe - check if algorithm is supported without instantiation
    static bool is_supported(RenderingDevice *p_rd);
    static SorterCapabilities get_capabilities();
};

// Radix Sort implementation - Future upgrade
class RadixSort : public IGPUSorter {
    GDCLASS(RadixSort, IGPUSorter);

private:
    SortKeyConfig key_config;
    // Radix sort shader resources
    struct RadixVariant {
        uint32_t radix_bits = 4;
        uint32_t radix_size = 16;
        uint32_t num_passes = 8;
        RID histogram_shader;
        RID scatter_shader;
        RID histogram_pipeline;
        RID scatter_pipeline;
        RID wg_prefix_shader;
        RID wg_prefix_pipeline;
        RID bin_prefix_shader;
        RID bin_prefix_pipeline;
    };

    struct PassUniformSets {
        RID histogram_even;
        RID histogram_odd;
        RID wg_prefix;
        RID bin_prefix;
        RID scatter_even;
        RID scatter_odd;
    };

    LocalVector<RadixVariant> variants;

    // Intermediate buffers
    RID histogram_buffer;
    RID wg_prefix_buffer;
    RID bin_counts_buffer;
    RID bin_prefix_buffer;
    RID temp_keys_buffer;
    RID temp_values_buffer;

    // Async compute resources
    std::atomic<uint64_t> timeline_value{0};
    uint64_t current_sort_value = 0;

    // Uniform sets for GPU binding
    LocalVector<RID> uniform_sets;
    RenderingDevice *uniform_owner = nullptr;
    uint64_t uniform_owner_generation = 0; // ISSUE-010: device instance ID at time of assignment
    RenderingDevice *resource_device = nullptr;
    uint64_t resource_device_generation = 0; // ISSUE-010: device instance ID at time of assignment

    uint32_t workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    uint32_t histogram_stride = 0;
    uint32_t workgroup_stride = 0;
    uint32_t max_workgroups = 0;
    uint32_t max_radix_size = 16;
    uint32_t max_num_passes = 8;
    uint32_t primary_radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS;
    uint32_t secondary_radix_bits = 0;
    uint32_t secondary_threshold = 0;
    uint32_t key_stride_words = 1;
    uint32_t key_stride_bytes = sizeof(uint32_t);
    bool use_64bit_keys = false;
    bool subgroups_available = false;
    SortPreflightError last_preflight_error = SortPreflightError::NONE;

    // GPU-driven indirect buffer for sort_indirect (stores element count per IndirectDispatchLayout)
    RID indirect_count_buffer;
    RID indirect_dispatch_args_buffer;
    RID indirect_dispatch_shader;
    RID indirect_dispatch_pipeline;

protected:
    static void _bind_methods();

    Error create_variant(RenderingDevice *device, uint32_t radix_bits);
    Error _create_pass_uniform_sets(RenderingDevice *resource_rd, const RadixVariant *variant, RID keys_buffer, RID values_buffer,
            RID count_buffer, const String &label_prefix, PassUniformSets &r_sets);
    void _reset_pass_uniform_sets(PassUniformSets &p_sets);
    const RadixVariant *get_variant(uint32_t radix_bits) const;
    const RadixVariant *select_variant(uint32_t element_count) const;
    uint32_t get_workgroup_count(uint32_t element_count) const;
    uint64_t _sort_async_internal(RID keys_buffer, RID values_buffer, uint32_t count, RID p_wait_semaphore,
            uint64_t p_wait_value, RID p_signal_semaphore, uint64_t p_signal_value_override);
    uint64_t _sort_indirect_internal(RID keys_buffer, RID values_buffer, RID count_buffer);
    SortPreflightResult _validate_sort_preflight(RID keys_buffer, RID values_buffer, uint32_t count, RID count_buffer, bool p_indirect) const;
    void _cleanup_partial_init(RenderingDevice *p_rd);  // Phase 2: Cleanup on init failure

public:
    RadixSort();
    ~RadixSort() override;

    Error initialize(RenderingDevice *p_rd, uint32_t p_max_elements) override;
    void shutdown() override;

    Error sort(RID keys_buffer, RID values_buffer, uint32_t count) override;
    Error sort_indirect(RID keys_buffer, RID values_buffer, RID count_buffer) override;
    uint64_t sort_indirect_async(RID keys_buffer, RID values_buffer, RID count_buffer) override;
    uint64_t sort_async(RID keys_buffer, RID values_buffer, uint32_t count) override;
    uint64_t sort_async_with_timeline(RID keys_buffer, RID values_buffer, uint32_t count, RID p_wait_semaphore,
            uint64_t p_wait_value, RID p_signal_semaphore, uint64_t p_signal_value);

    bool is_ready() const override;
    void wait_for_completion() override;

    String get_algorithm_name() const override { return "Radix Sort"; }
    bool supports_non_power_of_two() const override { return true; }
    float get_theoretical_complexity() const override { return 1.0f; } // O(n) parallel
    bool supports_indirect() const override { return true; } // RadixSort supports indirect
    SortPreflightError get_last_preflight_error() const override { return last_preflight_error; }

    void set_key_config(const SortKeyConfig &p_cfg) { key_config = p_cfg; }

    // Static capability probe - check if algorithm is supported without instantiation
    static bool is_supported(RenderingDevice *p_rd);
    static SorterCapabilities get_capabilities();
};

// OneSweep Sort - Ultimate performance for large datasets
class OneSweepSort : public IGPUSorter {
    GDCLASS(OneSweepSort, IGPUSorter);

private:
    // OneSweep shader resources
    RID global_histogram_shader;
    RID digit_binning_shader;
    RID chained_scan_shader;
    RID scatter_shader;
    RID global_histogram_pipeline;
    RID digit_binning_pipeline;
    RID chained_scan_pipeline;
    RID scatter_pipeline;

    // Intermediate buffers
    RID global_histogram_buffer;
    RID digit_histogram_buffer;
    RID chained_scan_buffer;
    RID temp_keys_buffer;
    RID temp_values_buffer;

    // Async compute resources
    std::atomic<uint64_t> timeline_value{0};
    uint64_t current_sort_value = 0;

    // Uniform sets for GPU binding
    LocalVector<RID> uniform_sets;
    RenderingDevice *uniform_owner = nullptr;
    uint64_t uniform_owner_generation = 0; // ISSUE-010: device instance ID at time of assignment
    RenderingDevice *resource_device = nullptr;
    uint64_t resource_device_generation = 0; // ISSUE-010: device instance ID at time of assignment

    static constexpr uint32_t RADIX_BITS = GPUSortingConstants::RADIX_BITS;
    static constexpr uint32_t RADIX_SIZE = GPUSortingConstants::RADIX_SIZE; // 2^8
    static constexpr uint32_t WORKGROUP_SIZE = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    static constexpr uint32_t CHAINING_FACTOR = 4;

protected:
    static void _bind_methods();

public:
    OneSweepSort();
    ~OneSweepSort() override;

    Error initialize(RenderingDevice *p_rd, uint32_t p_max_elements) override;
    void shutdown() override;

    Error sort(RID keys_buffer, RID values_buffer, uint32_t count) override;
    uint64_t sort_async(RID keys_buffer, RID values_buffer, uint32_t count) override;

    bool is_ready() const override;
    void wait_for_completion() override;

    String get_algorithm_name() const override { return "OneSweep Sort"; }
    bool supports_non_power_of_two() const override { return true; }
    float get_theoretical_complexity() const override { return 1.0f; } // O(n) with lower constant
    bool supports_indirect() const override { return false; } // OneSweep does not support indirect yet

    // Static capability probe - check if algorithm is supported without instantiation
    static bool is_supported(RenderingDevice *p_rd);
    static SorterCapabilities get_capabilities();
};

// Factory for creating sorters
class GPUSorterFactory {
public:
    enum SortingAlgorithm {
        ALGORITHM_RADIX = 0,
        ALGORITHM_AUTO = 1,
        ALGORITHM_BITONIC = 2,
        ALGORITHM_ONESWEEP = 3
    };

    struct PolicyProbe {
        bool supported = false;
        bool supports_indirect = false;
        bool supports_64bit_keys = false;
    };

    struct PolicyDecision {
        SortingAlgorithm preferred_algorithm = ALGORITHM_RADIX;
        SortingAlgorithm selected_algorithm = ALGORITHM_RADIX;
        String fallback_reason;
    };

    static Ref<IGPUSorter> create_sorter(SortingAlgorithm algorithm, RenderingDevice *rd, uint32_t max_elements,
            const SortKeyConfig &p_key_config = SortKeyConfig::from_settings());
    static SortingAlgorithm get_best_algorithm_for_size(uint32_t element_count, const SortKeyConfig &key_config);
    static SortingAlgorithm get_best_algorithm_for_size(uint32_t element_count, const SortKeyConfig &key_config, RenderingDevice *rd);
    static PolicyDecision evaluate_auto_policy(uint32_t element_count, const SortKeyConfig &key_config,
            const PolicyProbe &radix_probe, const PolicyProbe &bitonic_probe, const PolicyProbe &onesweep_probe,
            bool require_indirect, bool require_64bit_keys);

    // Capability probes - query algorithm capabilities without instantiation
    static bool probe_is_supported(SortingAlgorithm algorithm, RenderingDevice *rd);
    static bool probe_supports_indirect(SortingAlgorithm algorithm);
    static bool probe_supports_indirect(SortingAlgorithm algorithm, RenderingDevice *rd);
    static SorterCapabilities probe_capabilities(SortingAlgorithm algorithm);
};

VARIANT_ENUM_CAST(GPUSorterFactory::SortingAlgorithm);

#endif // GPU_SORTER_H

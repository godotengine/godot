#ifndef GPU_SORTING_CONFIG_H
#define GPU_SORTING_CONFIG_H

#include "core/string/ustring.h"
#include "core/config/project_settings.h"
#include "gpu_sorting_constants.h"

// GPU Sorting Configuration for radix-only pipeline
struct GPUSortingConfig {
    enum SubgroupPrefixMode : uint8_t {
        SUBGROUP_PREFIX_AUTO = 0,
        SUBGROUP_PREFIX_FORCE_OFF = 1,
    };

    // Performance targets
    float target_sort_time_ms = 2.0f;        // Target sorting time per frame
    // Maximum supported elements for instance/depth sort buffers.
    // Overlap-record buffers are controlled by max_overlap_records.
    uint32_t max_sort_elements = 50000000;

    // Overlap record budget for global composite sort pipeline.
    // Each Gaussian that overlaps a tile generates one record. A scene with N splats
    // where each covers ~M tiles on average produces N*M records. Close camera views
    // or large splats can push this multiplier to 50-100+.
    // Default: 100 million records (~1.2 GB VRAM for key+value buffers).
    // Minimum: 100,000 (for small test scenes).
    // Maximum: ~200 million (practical VRAM limit; 2.4 GB for buffers alone).
    uint32_t max_overlap_records = 100000000;
    // Optional lower bound used by adaptive overlap budgeting.
    // Kept <= max_overlap_records via accessor clamp.
    uint32_t max_overlap_records_adaptive_min = 100000;
    // Enables adaptive overlap budget feedback loop in TileRenderer.
    bool adaptive_overlap_budget_enabled = false;

    // Soft cap for per-tile raster iterations in fragment/compute rasterizers.
    // Alpha saturation provides natural early termination (typically ~100-500
    // splats); this cap only guards pathological edge cases.  Raised from 8192
    // to avoid silent splat dropping and tile overflow artifacts at close range.
    uint32_t max_raster_splats_per_tile = 65536;

    // Radix parameters
    uint32_t radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS; // Primary radix sort bits (4 or 8)
    uint32_t workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE; // GPU workgroup size

    // Key layout
    uint32_t key_bits = 32;                  // Sort key width (32 or 64)
    uint32_t tile_bits = 16;                 // Bits reserved for tile_id in the key
    uint32_t depth_bits = 16;                // Bits reserved for depth in the key
    bool enable_tie_breaker = false;         // Reserve low bits for splat_id tie-breaks

    // Performance monitoring and logging
    bool enable_performance_logging = false; // Log sorting performance metrics
    uint32_t performance_log_interval = 100; // Log every N frames
    bool enable_bandwidth_monitoring = true; // Monitor memory bandwidth usage
    bool debug_validate_prefix = false;      // Cross-check GPU prefix totals against CPU reference (debug only; stalls)
    bool enable_prefix_readback = false;     // Enable sync readback of overlap count (debug/validation only; causes GPU stall)
    bool profiling_preserve_gpu_timestamps = false; // Skip synchronous prefix readback to keep timestamp buffers intact (profiling-only; estimates overlap count)
    bool enable_compute_raster = true;       // Compute rasterizer with multi-pass batched shared memory loading
    bool strict_global_sort = true;          // Enforce exact global depth-sort semantics (disables unsafe fallback shortcuts)
    bool validate_sorted_output = false;     // Debug validation of sorted key monotonicity after GPU sort
    bool enable_stage_timestamps = true;     // Capture per-stage GPU timestamps in tile renderer
    uint8_t subgroup_prefix_mode = SUBGROUP_PREFIX_AUTO; // Runtime subgroup policy for radix prefix kernels

    // Project settings integration
    void load_from_project_settings();
    void save_to_project_settings() const;
    void reset_to_defaults();

    // Configuration validation
    bool validate() const;
    String get_validation_errors() const;

    // Accessors used by tile renderer overlap-budget policy hooks.
    uint32_t get_overlap_records_hard_cap() const { return max_overlap_records; }
    uint32_t get_overlap_records_adaptive_min() const {
        return max_overlap_records_adaptive_min < max_overlap_records
                ? max_overlap_records_adaptive_min
                : max_overlap_records;
    }

    // GPU Performance Tier Presets
    // These presets configure sorting parameters optimized for different GPU capabilities.
    //
    // LOW TIER: Integrated GPUs (Intel UHD, AMD Vega), older discrete GPUs
    //   - Minimizes memory usage and compute load
    //   - Smaller workgroups for better occupancy on limited hardware
    //   - 32-bit keys to reduce bandwidth
    //
    // MEDIUM TIER: Mid-range discrete GPUs (GTX 1060, RTX 3050, RX 5600)
    //   - Balanced settings for good performance and quality
    //   - Standard workgroup sizes
    //   - 64-bit keys for better precision
    //
    // HIGH TIER: High-end GPUs (RTX 3070+, RX 6800+)
    //   - Maximum throughput settings
    //   - Larger element counts and workgroups
    //   - Full precision and monitoring enabled
    //
    // ULTRA TIER: Enthusiast GPUs (RTX 4090, RX 7900 XTX)
    //   - No compromises, maximum quality
    //   - Highest element counts
    //   - All optional features enabled

    static GPUSortingConfig preset_low();
    static GPUSortingConfig preset_medium();
    static GPUSortingConfig preset_high();
    static GPUSortingConfig preset_ultra();

    // Apply a preset by name (case-insensitive: "low", "medium", "high", "ultra")
    // Returns true if preset was recognized and applied.
    bool apply_preset(const String &p_preset_name);

    // Returns the preset name that best matches current settings, or "custom"
    String get_current_preset_name() const;

    // Performance analytics
    void log_performance_data(uint32_t element_count, float sort_time_ms, const String &algorithm) const;
    void print_config_summary() const;

    // Constants for project settings paths
    static const String SECTION_PATH;
    static const String TARGET_TIME_PATH;
    static const String MAX_ELEMENTS_PATH;
    static const String MAX_OVERLAP_RECORDS_PATH;
    static const String MAX_RASTER_SPLATS_PER_TILE_PATH;
    static const String RADIX_BITS_PATH;
    static const String WORKGROUP_SIZE_PATH;
    static const String KEY_BITS_PATH;
    static const String TILE_BITS_PATH;
    static const String DEPTH_BITS_PATH;
    static const String ENABLE_TIE_BREAKER_PATH;
    static const String PERFORMANCE_LOGGING_PATH;
    static const String LOG_INTERVAL_PATH;
    static const String BANDWIDTH_MONITORING_PATH;
    static const String DEBUG_VALIDATE_PREFIX_PATH;
    static const String ENABLE_PREFIX_READBACK_PATH;
    static const String PROFILING_PRESERVE_TIMESTAMPS_PATH;
    static const String ENABLE_COMPUTE_RASTER_PATH;
    static const String STRICT_GLOBAL_SORT_PATH;
    static const String VALIDATE_SORTED_OUTPUT_PATH;
    static const String ENABLE_STAGE_TIMESTAMPS_PATH;
    static const String SUBGROUP_PREFIX_MODE_PATH;
    static const String GPU_PRESET_PATH;
};

// Global configuration instance
extern GPUSortingConfig g_gpu_sorting_config;

// Configuration management functions
void initialize_gpu_sorting_config();
void log_sorting_performance(uint32_t element_count, float sort_time_ms, const String &algorithm);

#endif // GPU_SORTING_CONFIG_H

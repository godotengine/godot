#include "gpu_sorting_config.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "../logger/gs_logger.h"

namespace {

static void _apply_instance_pipeline_overrides(GPUSortingConfig &config) {
    bool needs_override = config.key_bits != 64;
    if (needs_override) {
        config.key_bits = 64;
        if (config.tile_bits > config.key_bits) {
            config.tile_bits = config.key_bits;
        }
        if (config.depth_bits > config.key_bits) {
            config.depth_bits = config.key_bits;
        }
        if (config.tile_bits + config.depth_bits > config.key_bits) {
            if (config.tile_bits >= config.key_bits) {
                config.tile_bits = config.key_bits;
                config.depth_bits = 0;
            } else {
                config.depth_bits = config.key_bits - config.tile_bits;
            }
        }
        GS_LOG_GPU_SORT_INFO("[GPUSortingConfig] Instance pipeline requires 64-bit sort keys; overriding settings.");
    }
}

} // namespace

// Project settings paths
const String GPUSortingConfig::SECTION_PATH = "rendering/gaussian_splatting/gpu_sorting/";
const String GPUSortingConfig::TARGET_TIME_PATH = SECTION_PATH + "target_sort_time_ms";
const String GPUSortingConfig::MAX_ELEMENTS_PATH = SECTION_PATH + "max_sort_elements";
const String GPUSortingConfig::MAX_OVERLAP_RECORDS_PATH = SECTION_PATH + "max_overlap_records";
const String GPUSortingConfig::MAX_RASTER_SPLATS_PER_TILE_PATH = SECTION_PATH + "max_raster_splats_per_tile";
const String GPUSortingConfig::RADIX_BITS_PATH = SECTION_PATH + "radix_bits";
const String GPUSortingConfig::WORKGROUP_SIZE_PATH = SECTION_PATH + "workgroup_size";
const String GPUSortingConfig::KEY_BITS_PATH = SECTION_PATH + "key_bits";
const String GPUSortingConfig::TILE_BITS_PATH = SECTION_PATH + "tile_bits";
const String GPUSortingConfig::DEPTH_BITS_PATH = SECTION_PATH + "depth_bits";
const String GPUSortingConfig::ENABLE_TIE_BREAKER_PATH = SECTION_PATH + "enable_tie_breaker";
const String GPUSortingConfig::PERFORMANCE_LOGGING_PATH = SECTION_PATH + "enable_performance_logging";
const String GPUSortingConfig::LOG_INTERVAL_PATH = SECTION_PATH + "performance_log_interval";
const String GPUSortingConfig::BANDWIDTH_MONITORING_PATH = SECTION_PATH + "enable_bandwidth_monitoring";
const String GPUSortingConfig::DEBUG_VALIDATE_PREFIX_PATH = SECTION_PATH + "debug_validate_prefix";
const String GPUSortingConfig::ENABLE_PREFIX_READBACK_PATH = SECTION_PATH + "enable_prefix_readback";
const String GPUSortingConfig::PROFILING_PRESERVE_TIMESTAMPS_PATH = SECTION_PATH + "profiling_preserve_gpu_timestamps";
const String GPUSortingConfig::ENABLE_COMPUTE_RASTER_PATH = SECTION_PATH + "enable_compute_raster";
const String GPUSortingConfig::STRICT_GLOBAL_SORT_PATH = "rendering/gaussian_splatting/sorting/strict_global_sort";
const String GPUSortingConfig::VALIDATE_SORTED_OUTPUT_PATH = "rendering/gaussian_splatting/sorting/validate_sorted_output";
const String GPUSortingConfig::ENABLE_STAGE_TIMESTAMPS_PATH = SECTION_PATH + "enable_stage_timestamps";
const String GPUSortingConfig::SUBGROUP_PREFIX_MODE_PATH = SECTION_PATH + "subgroup_prefix_mode";
const String GPUSortingConfig::GPU_PRESET_PATH = SECTION_PATH + "gpu_preset";

// Global instance
GPUSortingConfig g_gpu_sorting_config;

void GPUSortingConfig::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    // Check if a preset is specified first
    String preset_name = ps->get_setting(GPU_PRESET_PATH, "");
    if (!preset_name.is_empty() && preset_name != "custom") {
        if (apply_preset(preset_name)) {
            // Preset applied successfully - still load debug/logging overrides
            enable_performance_logging = ps->get_setting(PERFORMANCE_LOGGING_PATH, enable_performance_logging);
            performance_log_interval = ps->get_setting(LOG_INTERVAL_PATH, performance_log_interval);
            enable_bandwidth_monitoring = ps->get_setting(BANDWIDTH_MONITORING_PATH, enable_bandwidth_monitoring);
            debug_validate_prefix = ps->get_setting(DEBUG_VALIDATE_PREFIX_PATH, debug_validate_prefix);
            enable_prefix_readback = ps->get_setting(ENABLE_PREFIX_READBACK_PATH, enable_prefix_readback);
            profiling_preserve_gpu_timestamps = ps->get_setting(PROFILING_PRESERVE_TIMESTAMPS_PATH, profiling_preserve_gpu_timestamps);
            enable_compute_raster = ps->get_setting(ENABLE_COMPUTE_RASTER_PATH, enable_compute_raster);
            max_raster_splats_per_tile = ps->get_setting(MAX_RASTER_SPLATS_PER_TILE_PATH, max_raster_splats_per_tile);
            strict_global_sort = ps->get_setting(STRICT_GLOBAL_SORT_PATH, strict_global_sort);
            validate_sorted_output = ps->get_setting(VALIDATE_SORTED_OUTPUT_PATH, validate_sorted_output);
            enable_stage_timestamps = ps->get_setting(ENABLE_STAGE_TIMESTAMPS_PATH, enable_stage_timestamps);
            subgroup_prefix_mode = static_cast<uint8_t>(ps->get_setting(SUBGROUP_PREFIX_MODE_PATH, int(subgroup_prefix_mode)));
            _apply_instance_pipeline_overrides(*this);

            if (enable_performance_logging) {
                print_config_summary();
            }
            return;
        }
        // Fall through to manual config if preset not recognized
    }

    // Load individual settings (custom configuration)
    target_sort_time_ms = ps->get_setting(TARGET_TIME_PATH, 2.0f);
    bool has_elements = ps->has_setting(MAX_ELEMENTS_PATH);
    bool has_overlap = ps->has_setting(MAX_OVERLAP_RECORDS_PATH);
    max_sort_elements = ps->get_setting(MAX_ELEMENTS_PATH, 50000000);
    max_overlap_records = ps->get_setting(MAX_OVERLAP_RECORDS_PATH, 100000000);
    max_raster_splats_per_tile = ps->get_setting(MAX_RASTER_SPLATS_PER_TILE_PATH, 8192);
    GS_LOG_GPU_SORT_INFO(vformat("[GPUSortingConfig] LOADED: max_sort_elements=%d max_overlap_records=%d (has_elements=%d has_overlap=%d)",
            max_sort_elements, max_overlap_records, int(has_elements), int(has_overlap)));

    radix_bits = ps->get_setting(RADIX_BITS_PATH, GPUSortingConstants::DEFAULT_RADIX_BITS);
    workgroup_size = ps->get_setting(WORKGROUP_SIZE_PATH, GPUSortingConstants::DEFAULT_WORKGROUP_SIZE);
    key_bits = ps->get_setting(KEY_BITS_PATH, 32);
    tile_bits = ps->get_setting(TILE_BITS_PATH, 16);
    depth_bits = ps->get_setting(DEPTH_BITS_PATH, 16);
    enable_tie_breaker = ps->get_setting(ENABLE_TIE_BREAKER_PATH, false);

    enable_performance_logging = ps->get_setting(PERFORMANCE_LOGGING_PATH, false);
    performance_log_interval = ps->get_setting(LOG_INTERVAL_PATH, 100);
    enable_bandwidth_monitoring = ps->get_setting(BANDWIDTH_MONITORING_PATH, true);
    debug_validate_prefix = ps->get_setting(DEBUG_VALIDATE_PREFIX_PATH, false);
    enable_prefix_readback = ps->get_setting(ENABLE_PREFIX_READBACK_PATH, false);
    profiling_preserve_gpu_timestamps = ps->get_setting(PROFILING_PRESERVE_TIMESTAMPS_PATH, false);
    enable_compute_raster = ps->get_setting(ENABLE_COMPUTE_RASTER_PATH, false);
    strict_global_sort = ps->get_setting(STRICT_GLOBAL_SORT_PATH, true);
    validate_sorted_output = ps->get_setting(VALIDATE_SORTED_OUTPUT_PATH, false);
    enable_stage_timestamps = ps->get_setting(ENABLE_STAGE_TIMESTAMPS_PATH, true);
    subgroup_prefix_mode = static_cast<uint8_t>(ps->get_setting(SUBGROUP_PREFIX_MODE_PATH, int(SUBGROUP_PREFIX_AUTO)));

    _apply_instance_pipeline_overrides(*this);

    if (enable_performance_logging) {
        print_config_summary();
    }
}

void GPUSortingConfig::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    ps->set_setting(TARGET_TIME_PATH, target_sort_time_ms);
    ps->set_setting(MAX_ELEMENTS_PATH, max_sort_elements);
    ps->set_setting(MAX_OVERLAP_RECORDS_PATH, max_overlap_records);
    ps->set_setting(MAX_RASTER_SPLATS_PER_TILE_PATH, max_raster_splats_per_tile);

    ps->set_setting(RADIX_BITS_PATH, radix_bits);
    ps->set_setting(WORKGROUP_SIZE_PATH, workgroup_size);
    ps->set_setting(KEY_BITS_PATH, key_bits);
    ps->set_setting(TILE_BITS_PATH, tile_bits);
    ps->set_setting(DEPTH_BITS_PATH, depth_bits);
    ps->set_setting(ENABLE_TIE_BREAKER_PATH, enable_tie_breaker);

    ps->set_setting(PERFORMANCE_LOGGING_PATH, enable_performance_logging);
    ps->set_setting(LOG_INTERVAL_PATH, performance_log_interval);
    ps->set_setting(BANDWIDTH_MONITORING_PATH, enable_bandwidth_monitoring);
    ps->set_setting(DEBUG_VALIDATE_PREFIX_PATH, debug_validate_prefix);
    ps->set_setting(ENABLE_PREFIX_READBACK_PATH, enable_prefix_readback);
    ps->set_setting(PROFILING_PRESERVE_TIMESTAMPS_PATH, profiling_preserve_gpu_timestamps);
    ps->set_setting(ENABLE_COMPUTE_RASTER_PATH, enable_compute_raster);
    ps->set_setting(STRICT_GLOBAL_SORT_PATH, strict_global_sort);
    ps->set_setting(VALIDATE_SORTED_OUTPUT_PATH, validate_sorted_output);
    ps->set_setting(ENABLE_STAGE_TIMESTAMPS_PATH, enable_stage_timestamps);
    ps->set_setting(SUBGROUP_PREFIX_MODE_PATH, int(subgroup_prefix_mode));

    ps->save();

    GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Configuration saved to project settings");
}

void GPUSortingConfig::reset_to_defaults() {
    target_sort_time_ms = 2.0f;
    max_sort_elements = 50000000;
    max_overlap_records = 100000000;
    max_raster_splats_per_tile = 8192;
    radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS;
    workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE;
    key_bits = 32;  // Default 32-bit for 8 passes instead of 16
    tile_bits = 16;
    depth_bits = 16;
    enable_tie_breaker = false;
    enable_performance_logging = false;
    performance_log_interval = 100;
    enable_bandwidth_monitoring = true;
    debug_validate_prefix = false;
    enable_prefix_readback = false;  // Disabled by default; GPU-driven pipeline uses indirect dispatch
    profiling_preserve_gpu_timestamps = false;
    enable_compute_raster = false;
    strict_global_sort = true;
    validate_sorted_output = false;
    enable_stage_timestamps = true;
    subgroup_prefix_mode = SUBGROUP_PREFIX_AUTO;

    GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Reset to default configuration");
}

bool GPUSortingConfig::validate() const {
    // Maximum overlap records budget: 200M records uses ~2.4 GB VRAM (keys + values).
    // Minimum: 100K records to ensure small scenes work.
    const uint32_t MIN_OVERLAP_RECORDS = 100000;
    const uint32_t MAX_OVERLAP_RECORDS_LIMIT = 200000000;
    const uint32_t MIN_RASTER_SPLATS_PER_TILE = 256;
    const uint32_t MAX_RASTER_SPLATS_PER_TILE = 65536;

    return target_sort_time_ms > 0.1f &&
           max_sort_elements > 1000 &&
           max_overlap_records >= MIN_OVERLAP_RECORDS &&
           max_overlap_records <= MAX_OVERLAP_RECORDS_LIMIT &&
           max_raster_splats_per_tile >= MIN_RASTER_SPLATS_PER_TILE &&
           max_raster_splats_per_tile <= MAX_RASTER_SPLATS_PER_TILE &&
           (radix_bits == GPUSortingConstants::DEFAULT_RADIX_BITS || radix_bits == GPUSortingConstants::RADIX_BITS) &&
           (workgroup_size == 64 || workgroup_size == 128 || workgroup_size == 256 || workgroup_size == 512) &&
           (key_bits == 32 || key_bits == 64) &&
           (tile_bits + depth_bits > 0) &&
           (tile_bits + depth_bits <= key_bits) &&
           performance_log_interval > 0 &&
           subgroup_prefix_mode <= SUBGROUP_PREFIX_FORCE_OFF;
}

String GPUSortingConfig::get_validation_errors() const {
    String errors;

    if (target_sort_time_ms <= 0.1f) errors += "Target sort time must be > 0.1ms\n";
    if (max_sort_elements <= 1000) errors += "Max sort elements must be > 1000\n";
    if (max_overlap_records < 100000) {
        errors += "Max overlap records must be >= 100,000 (too low may cause render cutoff)\n";
    }
    if (max_overlap_records > 200000000) {
        errors += "Max overlap records must be <= 200,000,000 (exceeds practical VRAM limits)\n";
    }
    if (max_raster_splats_per_tile < 256) {
        errors += "Max raster splats per tile must be >= 256\n";
    }
    if (max_raster_splats_per_tile > 65536) {
        errors += "Max raster splats per tile must be <= 65,536\n";
    }
    if (radix_bits != 4 && radix_bits != 8) errors += "Radix bits must be 4 or 8\n";
    if (workgroup_size != 64 && workgroup_size != 128 && workgroup_size != 256 && workgroup_size != 512) {
        errors += "Workgroup size must be 64, 128, 256, or 512\n";
    }
    if (key_bits != 32 && key_bits != 64) {
        errors += "Key bits must be 32 or 64\n";
    }
    if (tile_bits + depth_bits == 0) {
        errors += "Tile/depth bit split must allocate at least one bit\n";
    } else if (tile_bits + depth_bits > key_bits) {
        errors += "Tile/depth bit split must fit within key_bits\n";
    }
    if (performance_log_interval <= 0) errors += "Performance log interval must be > 0\n";
    if (subgroup_prefix_mode > SUBGROUP_PREFIX_FORCE_OFF) {
        errors += "Subgroup prefix mode must be 0 (auto) or 1 (force_off)\n";
    }

    return errors;
}

void GPUSortingConfig::log_performance_data(uint32_t element_count, float sort_time_ms, const String &algorithm) const {
    if (!enable_performance_logging) {
        return;
    }

    static uint32_t log_counter = 0;
    log_counter++;

    if (log_counter % performance_log_interval == 0) {
        float throughput_msps = element_count / sort_time_ms / 1000.0f;
        bool meets_target = sort_time_ms <= target_sort_time_ms;
        String status = meets_target ? "✓" : "⚠";

        GS_LOG_GPU_SORT_INFO(vformat("[GPU Sort Performance] %s %s: %d elements, %.2f ms, %.1f M splats/s (target: %.1f ms)",
                status, algorithm, element_count, sort_time_ms, throughput_msps, target_sort_time_ms));

    if (enable_bandwidth_monitoring) {
        const float key_bytes = float(key_bits) / 8.0f;
        const float value_bytes = float(sizeof(uint32_t));
        float data_size_mb = (element_count * (key_bytes + value_bytes) * 2.0f) / (1024.0f * 1024.0f);
        float bandwidth_gbps = data_size_mb / sort_time_ms;
        GS_LOG_GPU_SORT_INFO(vformat("[GPU Sort Performance] Bandwidth: %.1f GB/s (%.1f MB transferred, key_bits=%d)",
                bandwidth_gbps, data_size_mb, key_bits));
    }
    if (profiling_preserve_gpu_timestamps) {
        GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Profiling mode: prefix readback skipped to preserve GPU timestamps (overlap count estimated)");
    }
}
}

void GPUSortingConfig::print_config_summary() const {
    GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] ========== Configuration Summary ==========");
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Active Preset: %s", get_current_preset_name()));
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Performance Target: %.1f ms per sort (instance/depth max %d elements)",
            target_sort_time_ms, max_sort_elements));
    // Calculate VRAM usage for overlap record buffers: keys (8 bytes each for 64-bit) + values (4 bytes each)
    const float overlap_vram_mb = float(max_overlap_records) * 12.0f / (1024.0f * 1024.0f);
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Overlap Budget: %d records (~%.0f MB VRAM for key+value buffers)",
            max_overlap_records, overlap_vram_mb));
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Raster Tile Cap: %d splats/tile", max_raster_splats_per_tile));
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Radix Configuration: %d-bit radix, workgroup size %d",
            radix_bits, workgroup_size));
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Key Layout: key_bits=%d, tile_bits=%d, depth_bits=%d, tie_breaker=%s",
            key_bits, tile_bits, depth_bits, enable_tie_breaker ? "enabled" : "disabled"));
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Performance Logging: %s (interval: %d frames, bandwidth: %s)",
            enable_performance_logging ? "enabled" : "disabled",
            performance_log_interval,
            enable_bandwidth_monitoring ? "on" : "off"));
    if (debug_validate_prefix) {
        GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Debug: prefix validation is ENABLED (will stall for buffer readback)");
    }
    if (enable_prefix_readback) {
        GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Prefix readback enabled (debug/validation mode; causes GPU stall)");
    } else {
        GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] GPU-driven indirect dispatch enabled (no CPU readback stall)");
    }
    if (enable_compute_raster) {
        GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Compute rasterizer enabled (experimental)");
    }
    GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Strict global sort: %s | sorted-output validation: %s | stage timestamps: %s | subgroup prefix mode: %d",
            strict_global_sort ? "enabled" : "disabled",
            validate_sorted_output ? "enabled" : "disabled",
            enable_stage_timestamps ? "enabled" : "disabled",
            int(subgroup_prefix_mode)));
    GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] ================================================");
}

// ============================================================================
// GPU Performance Tier Presets
// ============================================================================

GPUSortingConfig GPUSortingConfig::preset_low() {
    GPUSortingConfig config;

    // Performance targets - conservative for integrated/low-end GPUs
    config.target_sort_time_ms = 4.0f;       // Relaxed timing target
    config.max_sort_elements = 1000000;      // 1M splats max
    config.max_overlap_records = 10000000;   // 10M overlap records (~120 MB VRAM)
    config.max_raster_splats_per_tile = 4096;

    // Radix parameters - smaller workgroups for better occupancy
    config.radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS; // Standard 4-bit radix
    config.workgroup_size = 128;             // Smaller workgroups for limited compute units

    // Key layout - 32-bit keys to reduce memory bandwidth
    config.key_bits = 32;                    // 32-bit keys reduce bandwidth by 50%
    config.tile_bits = 16;                   // 16 bits for tile_id (65K tiles max)
    config.depth_bits = 16;                  // 16 bits for depth (sufficient precision)
    config.enable_tie_breaker = false;       // Disable to save bits
    // Performance monitoring - minimal overhead
    config.enable_performance_logging = false;
    config.performance_log_interval = 200;
    config.enable_bandwidth_monitoring = false;
    config.debug_validate_prefix = false;
    config.enable_prefix_readback = false;
    config.profiling_preserve_gpu_timestamps = false;
    config.enable_compute_raster = false;
    config.strict_global_sort = true;
    config.validate_sorted_output = false;
    config.enable_stage_timestamps = true;
    config.subgroup_prefix_mode = SUBGROUP_PREFIX_AUTO;

    return config;
}

GPUSortingConfig GPUSortingConfig::preset_medium() {
    GPUSortingConfig config;

    // Performance targets - balanced
    config.target_sort_time_ms = 2.5f;       // Reasonable timing target
    config.max_sort_elements = 5000000;      // 5M splats max
    config.max_overlap_records = 30000000;   // 30M overlap records (~360 MB VRAM)
    config.max_raster_splats_per_tile = 8192;

    // Radix parameters - standard configuration
    config.radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS; // Standard 4-bit radix
    config.workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE; // Standard workgroup size

    // Key layout - 64-bit for precision
    config.key_bits = 64;
    config.tile_bits = 32;                   // 32 bits for tile_id
    config.depth_bits = 32;                  // 32 bits for depth
    config.enable_tie_breaker = false;       // Usually not needed
    // Performance monitoring - basic
    config.enable_performance_logging = false;
    config.performance_log_interval = 100;
    config.enable_bandwidth_monitoring = true;
    config.debug_validate_prefix = false;
    config.enable_prefix_readback = false;
    config.profiling_preserve_gpu_timestamps = false;
    config.enable_compute_raster = false;
    config.strict_global_sort = true;
    config.validate_sorted_output = false;
    config.enable_stage_timestamps = true;
    config.subgroup_prefix_mode = SUBGROUP_PREFIX_AUTO;

    return config;
}

GPUSortingConfig GPUSortingConfig::preset_high() {
    GPUSortingConfig config;

    // Performance targets - aggressive
    config.target_sort_time_ms = 2.0f;       // Tight timing target
    config.max_sort_elements = 30000000;     // 30M splats max
    config.max_overlap_records = 100000000;  // 100M overlap records (~1.2 GB VRAM)
    config.max_raster_splats_per_tile = 12288;

    // Radix parameters - optimized for throughput
    config.radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS; // 4-bit radix (8-bit has higher memory pressure)
    config.workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE; // Standard workgroup size

    // Key layout - full precision
    config.key_bits = 64;
    config.tile_bits = 32;                   // 32 bits for tile_id
    config.depth_bits = 32;                  // 32 bits for depth
    config.enable_tie_breaker = false;       // Disable unless depth collisions are an issue
    // Performance monitoring - enabled for optimization
    config.enable_performance_logging = false;
    config.performance_log_interval = 100;
    config.enable_bandwidth_monitoring = true;
    config.debug_validate_prefix = false;
    config.enable_prefix_readback = false;
    config.profiling_preserve_gpu_timestamps = false;
    config.enable_compute_raster = false;    // Enable when stable
    config.strict_global_sort = true;
    config.validate_sorted_output = false;
    config.enable_stage_timestamps = true;
    config.subgroup_prefix_mode = SUBGROUP_PREFIX_AUTO;

    return config;
}

GPUSortingConfig GPUSortingConfig::preset_ultra() {
    GPUSortingConfig config;

    // Performance targets - maximum capacity
    config.target_sort_time_ms = 1.5f;       // Aggressive timing target
    config.max_sort_elements = 50000000;     // 50M splats max
    config.max_overlap_records = 150000000;  // 150M overlap records (~1.8 GB VRAM)
    config.max_raster_splats_per_tile = 16384;

    // Radix parameters - maximum throughput
    config.radix_bits = GPUSortingConstants::DEFAULT_RADIX_BITS; // 4-bit radix for consistent performance
    config.workgroup_size = GPUSortingConstants::DEFAULT_WORKGROUP_SIZE; // Can try 512 on very high-end GPUs

    // Key layout - maximum precision with tie-breaker
    config.key_bits = 64;
    config.tile_bits = 32;                   // 32 bits for tile_id
    config.depth_bits = 32;                  // 32 bits for depth
    config.enable_tie_breaker = true;        // Enable for maximum sort stability
    // Performance monitoring - full diagnostics available
    config.enable_performance_logging = false;  // User can enable
    config.performance_log_interval = 50;
    config.enable_bandwidth_monitoring = true;
    config.debug_validate_prefix = false;
    config.enable_prefix_readback = false;
    config.profiling_preserve_gpu_timestamps = false;
    config.enable_compute_raster = false;       // Enable when stable
    config.strict_global_sort = true;
    config.validate_sorted_output = false;
    config.enable_stage_timestamps = true;
    config.subgroup_prefix_mode = SUBGROUP_PREFIX_AUTO;

    return config;
}

bool GPUSortingConfig::apply_preset(const String &p_preset_name) {
    String preset_lower = p_preset_name.to_lower().strip_edges();

    GPUSortingConfig new_config;
    bool recognized = false;

    if (preset_lower == "low" || preset_lower == "performance") {
        new_config = preset_low();
        recognized = true;
    } else if (preset_lower == "medium" || preset_lower == "balanced") {
        new_config = preset_medium();
        recognized = true;
    } else if (preset_lower == "high" || preset_lower == "quality") {
        new_config = preset_high();
        recognized = true;
    } else if (preset_lower == "ultra" || preset_lower == "maximum") {
        new_config = preset_ultra();
        recognized = true;
    }

    if (recognized) {
        // Apply all settings from preset
        target_sort_time_ms = new_config.target_sort_time_ms;
        max_sort_elements = new_config.max_sort_elements;
        max_overlap_records = new_config.max_overlap_records;
        max_raster_splats_per_tile = new_config.max_raster_splats_per_tile;
        radix_bits = new_config.radix_bits;
        workgroup_size = new_config.workgroup_size;
        key_bits = new_config.key_bits;
        tile_bits = new_config.tile_bits;
        depth_bits = new_config.depth_bits;
        enable_tie_breaker = new_config.enable_tie_breaker;
        enable_performance_logging = new_config.enable_performance_logging;
        performance_log_interval = new_config.performance_log_interval;
        enable_bandwidth_monitoring = new_config.enable_bandwidth_monitoring;
        debug_validate_prefix = new_config.debug_validate_prefix;
        enable_prefix_readback = new_config.enable_prefix_readback;
        profiling_preserve_gpu_timestamps = new_config.profiling_preserve_gpu_timestamps;
        enable_compute_raster = new_config.enable_compute_raster;
        strict_global_sort = new_config.strict_global_sort;
        validate_sorted_output = new_config.validate_sorted_output;
        enable_stage_timestamps = new_config.enable_stage_timestamps;
        subgroup_prefix_mode = new_config.subgroup_prefix_mode;

        GS_LOG_GPU_SORT_INFO(vformat("[GPU Sorting Config] Applied '%s' preset (max %d elements, %d-bit keys, workgroup %d)",
                preset_lower, max_sort_elements, key_bits, workgroup_size));
        return true;
    }

    GS_LOG_GPU_SORT_ERROR(vformat("[GPU Sorting Config] Unknown preset '%s'. Valid presets: low, medium, high, ultra", p_preset_name));
    return false;
}

String GPUSortingConfig::get_current_preset_name() const {
    // Check against known presets
    GPUSortingConfig low = preset_low();
    GPUSortingConfig medium = preset_medium();
    GPUSortingConfig high = preset_high();
    GPUSortingConfig ultra = preset_ultra();

    // Compare key parameters to identify preset
    auto matches = [this](const GPUSortingConfig &p_preset) -> bool {
        return max_sort_elements == p_preset.max_sort_elements &&
               key_bits == p_preset.key_bits &&
               workgroup_size == p_preset.workgroup_size &&
               tile_bits == p_preset.tile_bits &&
               depth_bits == p_preset.depth_bits;
    };

    if (matches(low)) {
        return "low";
    } else if (matches(medium)) {
        return "medium";
    } else if (matches(high)) {
        return "high";
    } else if (matches(ultra)) {
        return "ultra";
    }

    return "custom";
}

void initialize_gpu_sorting_config() {
    // Register settings with GLOBAL_DEF so they can be read from project.godot
    GLOBAL_DEF(GPUSortingConfig::GPU_PRESET_PATH, "high");
    GLOBAL_DEF(GPUSortingConfig::TARGET_TIME_PATH, 2.0f);
    GLOBAL_DEF(GPUSortingConfig::MAX_ELEMENTS_PATH, 50000000);
    GLOBAL_DEF(GPUSortingConfig::MAX_OVERLAP_RECORDS_PATH, 100000000);
    GLOBAL_DEF(GPUSortingConfig::MAX_RASTER_SPLATS_PER_TILE_PATH, 8192);
    GLOBAL_DEF(GPUSortingConfig::RADIX_BITS_PATH, GPUSortingConstants::DEFAULT_RADIX_BITS);
    GLOBAL_DEF(GPUSortingConfig::WORKGROUP_SIZE_PATH, GPUSortingConstants::DEFAULT_WORKGROUP_SIZE);
    GLOBAL_DEF(GPUSortingConfig::KEY_BITS_PATH, 64);
    GLOBAL_DEF(GPUSortingConfig::TILE_BITS_PATH, 32);
    GLOBAL_DEF(GPUSortingConfig::DEPTH_BITS_PATH, 32);
    GLOBAL_DEF(GPUSortingConfig::ENABLE_TIE_BREAKER_PATH, false);
    GLOBAL_DEF(GPUSortingConfig::PERFORMANCE_LOGGING_PATH, g_gpu_sorting_config.enable_performance_logging);
    GLOBAL_DEF(GPUSortingConfig::LOG_INTERVAL_PATH, g_gpu_sorting_config.performance_log_interval);
    GLOBAL_DEF(GPUSortingConfig::BANDWIDTH_MONITORING_PATH, g_gpu_sorting_config.enable_bandwidth_monitoring);
    GLOBAL_DEF(GPUSortingConfig::ENABLE_COMPUTE_RASTER_PATH, g_gpu_sorting_config.enable_compute_raster);
    GLOBAL_DEF(GPUSortingConfig::STRICT_GLOBAL_SORT_PATH, g_gpu_sorting_config.strict_global_sort);
    GLOBAL_DEF(GPUSortingConfig::VALIDATE_SORTED_OUTPUT_PATH, g_gpu_sorting_config.validate_sorted_output);
    GLOBAL_DEF(GPUSortingConfig::ENABLE_STAGE_TIMESTAMPS_PATH, g_gpu_sorting_config.enable_stage_timestamps);
    GLOBAL_DEF(GPUSortingConfig::SUBGROUP_PREFIX_MODE_PATH, int(g_gpu_sorting_config.subgroup_prefix_mode));
#ifdef DEBUG_ENABLED
    GLOBAL_DEF(GPUSortingConfig::ENABLE_PREFIX_READBACK_PATH, g_gpu_sorting_config.enable_prefix_readback);
    GLOBAL_DEF(GPUSortingConfig::DEBUG_VALIDATE_PREFIX_PATH, g_gpu_sorting_config.debug_validate_prefix);
    GLOBAL_DEF(GPUSortingConfig::PROFILING_PRESERVE_TIMESTAMPS_PATH, g_gpu_sorting_config.profiling_preserve_gpu_timestamps);
#endif
    
    g_gpu_sorting_config.load_from_project_settings();

    if (!g_gpu_sorting_config.validate()) {
        GS_LOG_GPU_SORT_ERROR("[GPU Sorting Config] Invalid configuration detected:");
        GS_LOG_GPU_SORT_ERROR(g_gpu_sorting_config.get_validation_errors());
        GS_LOG_GPU_SORT_INFO("[GPU Sorting Config] Resetting to defaults...");
        g_gpu_sorting_config.reset_to_defaults();
        g_gpu_sorting_config.save_to_project_settings();
    }
}

void log_sorting_performance(uint32_t element_count, float sort_time_ms, const String &algorithm) {
    g_gpu_sorting_config.log_performance_data(element_count, sort_time_ms, algorithm);
}

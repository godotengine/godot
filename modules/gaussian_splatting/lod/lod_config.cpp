#include "lod_config.h"
#include "../core/gs_project_settings.h"
#include "../core/quality_tier_config.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "../logger/gs_logger.h"
#include <cfloat>  // For FLT_MAX

namespace {
static bool _is_data_log_enabled() { return gs::settings::is_data_log_enabled(); }
} // namespace

// Static path constants
const String LODConfig::ENABLED_PATH = LOD_CONFIG_ENABLED_PATH;
const String LODConfig::NUM_LEVELS_PATH = LOD_CONFIG_NUM_LEVELS_PATH;
const String LODConfig::MAX_DISTANCE_PATH = LOD_CONFIG_MAX_DISTANCE_PATH;
const String LODConfig::BASE_THRESHOLD_PATH = LOD_CONFIG_BASE_THRESHOLD_PATH;

// Global configuration instance
LODConfig g_lod_config;

void LODConfig::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    auto get_bool = [ps](const StringName &name, bool fallback) -> bool {
        if (!ps->has_setting(name)) return fallback;
        Variant value = ps->get_setting_with_override(name);
        if (value.get_type() == Variant::BOOL) {
            return value.operator bool();
        }
        if (value.get_type() == Variant::INT) {
            return value.operator int64_t() != 0;
        }
        return fallback;
    };

    auto get_int = [ps](const StringName &name, int fallback) -> int {
        if (!ps->has_setting(name)) return fallback;
        Variant value = ps->get_setting_with_override(name);
        if (value.get_type() == Variant::INT) {
            return static_cast<int>(value.operator int64_t());
        }
        if (value.get_type() == Variant::FLOAT) {
            return static_cast<int>(Math::round(value.operator double()));
        }
        return fallback;
    };

    auto get_float = [ps](const StringName &name, float fallback) -> float {
        if (!ps->has_setting(name)) return fallback;
        Variant value = ps->get_setting_with_override(name);
        if (value.get_type() == Variant::FLOAT) {
            return static_cast<float>(value.operator double());
        }
        if (value.get_type() == Variant::INT) {
            return static_cast<float>(value.operator int64_t());
        }
        return fallback;
    };

    enabled = get_bool(LOD_CONFIG_ENABLED_PATH, enabled);
    num_levels = get_int(LOD_CONFIG_NUM_LEVELS_PATH, num_levels);

    // Sentinel-based tier seeding for max_distance and base_threshold.
    // -1.0f means "not explicitly set by user" -- check active tier.
    float raw_max_distance = get_float(LOD_CONFIG_MAX_DISTANCE_PATH, -1.0f);
    float raw_base_threshold = get_float(LOD_CONFIG_BASE_THRESHOLD_PATH, -1.0f);

    // Resolve sentinels via tier config.
    const String tier_preset = ps->get_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
    QualityTierConfig tier_config;
    bool has_tier = get_quality_tier_config(tier_preset, tier_config);

    if (raw_max_distance < 0.0f) {
        max_distance = (has_tier && tier_config.lod_max_distance > 0.0f) ? tier_config.lod_max_distance : 100.0f;
    } else {
        max_distance = raw_max_distance;
    }

    if (raw_base_threshold < 0.0f) {
        base_threshold = (has_tier && tier_config.lod_base_threshold > 0.0f) ? tier_config.lod_base_threshold : 10.0f;
    } else {
        base_threshold = raw_base_threshold;
    }

    // Write resolved values back so consumers that read ProjectSettings
    // directly (e.g. GPUCuller::update_culling_settings) never see the
    // sentinel -1.0f.
    if (raw_max_distance < 0.0f) {
        ps->set_setting(LOD_CONFIG_MAX_DISTANCE_PATH, max_distance);
    }
    if (raw_base_threshold < 0.0f) {
        ps->set_setting(LOD_CONFIG_BASE_THRESHOLD_PATH, base_threshold);
    }

    splat_skip_enabled = get_bool(LOD_CONFIG_SPLAT_SKIP_ENABLED_PATH, splat_skip_enabled);
    sh_reduction_enabled = get_bool(LOD_CONFIG_SH_REDUCTION_ENABLED_PATH, sh_reduction_enabled);
    opacity_fade_enabled = get_bool(LOD_CONFIG_OPACITY_FADE_ENABLED_PATH, opacity_fade_enabled);
    debug_visualization = get_bool(LOD_CONFIG_DEBUG_VISUALIZATION_PATH, debug_visualization);

    // Validate and clamp values
    num_levels = CLAMP(num_levels, 2, 8);
    max_distance = MAX(base_threshold * 2.0f, max_distance);
    base_threshold = MAX(1.0f, base_threshold);
}

void LODConfig::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    ps->set_setting(LOD_CONFIG_ENABLED_PATH, enabled);
    ps->set_setting(LOD_CONFIG_NUM_LEVELS_PATH, num_levels);
    ps->set_setting(LOD_CONFIG_MAX_DISTANCE_PATH, max_distance);
    ps->set_setting(LOD_CONFIG_BASE_THRESHOLD_PATH, base_threshold);
    ps->set_setting(LOD_CONFIG_SPLAT_SKIP_ENABLED_PATH, splat_skip_enabled);
    ps->set_setting(LOD_CONFIG_SH_REDUCTION_ENABLED_PATH, sh_reduction_enabled);
    ps->set_setting(LOD_CONFIG_OPACITY_FADE_ENABLED_PATH, opacity_fade_enabled);
    ps->set_setting(LOD_CONFIG_DEBUG_VISUALIZATION_PATH, debug_visualization);
}

void LODConfig::reset_to_defaults() {
    enabled = true;
    num_levels = 4;
    max_distance = 100.0f;
    base_threshold = 10.0f;
    splat_skip_enabled = true;
    sh_reduction_enabled = true;
    opacity_fade_enabled = true;
    debug_visualization = false;
}

bool LODConfig::validate() const {
    if (num_levels < 2 || num_levels > 8) {
        return false;
    }
    if (max_distance <= 0.0f) {
        return false;
    }
    if (base_threshold <= 0.0f || base_threshold >= max_distance) {
        return false;
    }
    return true;
}

int LODConfig::calculate_lod_level(float distance) const {
    if (!enabled || distance <= 0.0f) {
        return 0;  // Highest detail
    }

    // Octree-GS formula: L = floor(min(max(log2(d_max/d), 0), K-1))
    // d = distance, d_max = max_distance, K = num_levels
    // PERF: Precomputed 1/log(2) to avoid computing log(2.0f) every call
    // log2(x) = log(x) * (1/log(2)) = log(x) * LOG2_E where LOG2_E ≈ 1.4426950408889634
    static constexpr float LOG2_E = 1.4426950408889634f;  // 1.0 / log(2.0)

    float ratio = max_distance / MAX(distance, 0.001f);
    float log_ratio = Math::log(ratio) * LOG2_E;  // log2(ratio) using single log() call

    // Clamp to valid LOD range [0, num_levels - 1]
    // Note: Higher log_ratio (closer objects) = lower LOD level (more detail)
    // We invert: closer = LOD 0, farther = LOD K-1
    float inverted = static_cast<float>(num_levels - 1) - log_ratio;
    int lod_level = static_cast<int>(Math::floor(CLAMP(inverted, 0.0f, static_cast<float>(num_levels - 1))));

    return lod_level;
}

int LODConfig::get_splat_skip_factor(int lod_level) const {
    if (!splat_skip_enabled || lod_level <= 0) {
        return 1;  // Render all splats
    }

    // Skip factor doubles per LOD level: 1, 2, 4, 8, 16...
    return 1 << CLAMP(lod_level, 0, 7);
}

int LODConfig::get_sh_band_for_lod(int lod_level) const {
    if (!sh_reduction_enabled) {
        return 3;  // Full SH3 quality
    }

    // Map LOD level to SH band: LOD 0 -> SH3, LOD 1 -> SH2, LOD 2 -> SH1, LOD 3+ -> SH0
    return CLAMP(3 - lod_level, 0, 3);
}

float LODConfig::get_opacity_multiplier(float distance) const {
    if (!opacity_fade_enabled) {
        return 1.0f;
    }

    if (distance <= base_threshold) {
        return 1.0f;  // Full opacity for close objects
    }

    if (distance >= max_distance) {
        return 0.0f;  // Fully faded at max distance
    }

    // Linear fade between base_threshold and max_distance
    float fade_range = max_distance - base_threshold;
    float fade_distance = distance - base_threshold;
    return 1.0f - (fade_distance / fade_range);
}

float LODConfig::get_distance_threshold(int lod_level) const {
    if (lod_level <= 0) {
        return base_threshold;
    }

    // Calculate distance threshold for each LOD level
    // Uses exponential spacing: base * 2^lod_level
    float multiplier = static_cast<float>(1 << lod_level);
    return MIN(base_threshold * multiplier, max_distance);
}

void LODConfig::print_config_summary() const {
    if (!_is_data_log_enabled()) {
        return;
    }
    GS_LOG_STREAMING_INFO(vformat("[LOD Config] enabled=%s, levels=%d, max_dist=%.1f, base=%.1f",
            enabled ? "true" : "false", num_levels, max_distance, base_threshold));
    GS_LOG_STREAMING_INFO(vformat("[LOD Config] splat_skip=%s, sh_reduce=%s, opacity_fade=%s, debug=%s",
            splat_skip_enabled ? "true" : "false",
            sh_reduction_enabled ? "true" : "false",
            opacity_fade_enabled ? "true" : "false",
            debug_visualization ? "true" : "false"));

    // Print distance thresholds for each LOD level
    for (int i = 0; i < num_levels; i++) {
        GS_LOG_STREAMING_DEBUG(vformat("[LOD Config] Level %d: distance <= %.1f, skip=%d, SH=%d",
                i, get_distance_threshold(i), get_splat_skip_factor(i), get_sh_band_for_lod(i)));
    }
}

// ChunkLODMetadata implementation
void ChunkLODMetadata::update_from_distance(float p_distance, const LODConfig& config) {
    float old_lod_level = lod_level;
    distance = p_distance;

    lod_level = config.calculate_lod_level(distance);
    sh_band_level = config.get_sh_band_for_lod(lod_level);
    splat_skip_factor = config.get_splat_skip_factor(lod_level);
    opacity_multiplier = config.get_opacity_multiplier(distance);

    // Mark for update if LOD level changed
    needs_update = (lod_level != old_lod_level);
}

// LODDebugStats implementation
void LODDebugStats::reset() {
    for (int i = 0; i < 8; i++) {
        lod_level_counts[i] = 0;
    }
    for (int i = 0; i < 4; i++) {
        sh_band_counts[i] = 0;
    }
    total_chunks = 0;
    total_splats_original = 0;
    total_splats_after_skip = 0;
    splat_reduction_ratio = 0.0f;
    min_distance = FLT_MAX;
    max_distance = 0.0f;
    avg_distance = 0.0f;
}

void LODDebugStats::update_from_chunks(const ChunkLODMetadata* chunks, uint32_t count) {
    reset();
    if (!chunks || count == 0) {
        return;
    }

    total_chunks = count;
    float total_distance = 0.0f;

    for (uint32_t i = 0; i < count; i++) {
        const ChunkLODMetadata& chunk = chunks[i];

        // LOD level distribution
        if (chunk.lod_level >= 0 && chunk.lod_level < 8) {
            lod_level_counts[chunk.lod_level]++;
        }

        // SH band distribution
        if (chunk.sh_band_level >= 0 && chunk.sh_band_level < 4) {
            sh_band_counts[chunk.sh_band_level]++;
        }

        // Distance statistics
        if (chunk.distance < min_distance) {
            min_distance = chunk.distance;
        }
        if (chunk.distance > max_distance) {
            max_distance = chunk.distance;
        }
        total_distance += chunk.distance;
    }

    avg_distance = total_distance / static_cast<float>(count);

    if (min_distance == FLT_MAX) {
        min_distance = 0.0f;
    }
}

String LODDebugStats::to_string() const {
    String result = vformat("[LOD Stats] %d chunks, dist=[%.1f, %.1f], avg=%.1f\n",
            total_chunks, min_distance, max_distance, avg_distance);

    result += "[LOD Levels] ";
    for (int i = 0; i < 8; i++) {
        if (lod_level_counts[i] > 0) {
            result += vformat("L%d:%d ", i, lod_level_counts[i]);
        }
    }
    result += "\n";

    result += "[SH Bands] ";
    for (int i = 0; i < 4; i++) {
        if (sh_band_counts[i] > 0) {
            result += vformat("SH%d:%d ", i, sh_band_counts[i]);
        }
    }

    if (splat_reduction_ratio > 0.0f) {
        result += vformat("\n[Reduction] %.1f%% (%d -> %d splats)",
                splat_reduction_ratio * 100.0f, total_splats_original, total_splats_after_skip);
    }

    return result;
}

// Global initialization
void initialize_lod_config() {
    register_lod_project_settings();
    g_lod_config.load_from_project_settings();
    GS_LOG_STREAMING_INFO("[LOD] Configuration loaded from project settings");
    if (g_lod_config.debug_visualization) {
        g_lod_config.print_config_summary();
    }
}

void register_lod_project_settings() {
    // This function is called during module initialization to register
    // LOD-related project settings with default values
    GLOBAL_DEF(LOD_CONFIG_ENABLED_PATH, true);
    GLOBAL_DEF(LOD_CONFIG_NUM_LEVELS_PATH, 4);
    GLOBAL_DEF(LOD_CONFIG_MAX_DISTANCE_PATH, -1.0f);
    GLOBAL_DEF(LOD_CONFIG_BASE_THRESHOLD_PATH, -1.0f);
    GLOBAL_DEF(LOD_CONFIG_SPLAT_SKIP_ENABLED_PATH, true);
    GLOBAL_DEF(LOD_CONFIG_SH_REDUCTION_ENABLED_PATH, true);
    GLOBAL_DEF(LOD_CONFIG_OPACITY_FADE_ENABLED_PATH, true);
    GLOBAL_DEF(LOD_CONFIG_DEBUG_VISUALIZATION_PATH, false);

    // LOD blending settings (LODGE technique) - eliminates popping during LOD transitions
    GLOBAL_DEF("rendering/gaussian_splatting/lod/blend_enabled", true);
    GLOBAL_DEF("rendering/gaussian_splatting/lod/blend_distance", 5.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/lod/hysteresis_zone", 0.5f);

    // Legacy LOD settings (kept for compatibility)
    GLOBAL_DEF(LOD_CONFIG_MIN_SCREEN_SIZE_PIXELS_PATH, 1.5f);
    GLOBAL_DEF(LOD_CONFIG_BIAS_PATH, 1.0f);
}

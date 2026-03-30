#include "quantization_config.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "../core/gs_project_settings.h"
#include "../core/quality_tier_config.h"
#include "../logger/gs_logger.h"

// Project settings paths
const String QuantizationConfig::SECTION_PATH = "rendering/gaussian_splatting/compression/";
const String QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH = SECTION_PATH + "per_chunk_quantization";
const String QuantizationConfig::POSITION_BITS_PATH = SECTION_PATH + "position_bits";
const String QuantizationConfig::SCALE_BITS_PATH = SECTION_PATH + "scale_bits";
const String QuantizationConfig::QUANTIZE_SCALES_PATH = SECTION_PATH + "quantize_scales";
const String QuantizationConfig::MIN_CHUNK_SIZE_PATH = SECTION_PATH + "min_chunk_size";
const String QuantizationConfig::MAX_CHUNK_SIZE_PATH = SECTION_PATH + "max_chunk_size";
const String QuantizationConfig::ADAPTIVE_CHUNK_SIZE_PATH = SECTION_PATH + "adaptive_chunk_size";

// Global instance
QuantizationConfig g_quantization_config;

void QuantizationConfig::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    // Sentinel-based tier seeding for per_chunk_quantization.
    // -1 means "not explicitly set by user" -- check active tier.
    int raw_quantization = gs::settings::get_int(ps, PER_CHUNK_QUANTIZATION_PATH, -1);
    if (raw_quantization < 0) {
        // Sentinel: user never set this.
        per_chunk_quantization = false; // Code default.
        const String tier_preset = ps->get_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
        QualityTierConfig tier_config;
        if (get_quality_tier_config(tier_preset, tier_config) && tier_config.quantization_enabled >= 0) {
            per_chunk_quantization = (tier_config.quantization_enabled != 0);
        }
    } else {
        per_chunk_quantization = (raw_quantization != 0);
    }

    position_bits = ps->get_setting(POSITION_BITS_PATH, 16);
    scale_bits = ps->get_setting(SCALE_BITS_PATH, 12);
    quantize_scales = ps->get_setting(QUANTIZE_SCALES_PATH, false);
    min_chunk_size = ps->get_setting(MIN_CHUNK_SIZE_PATH, 256);
    max_chunk_size = ps->get_setting(MAX_CHUNK_SIZE_PATH, 8192);
    adaptive_chunk_size = ps->get_setting(ADAPTIVE_CHUNK_SIZE_PATH, true);

    // Clamp values to valid ranges
    position_bits = CLAMP(position_bits, 8u, 24u);
    scale_bits = CLAMP(scale_bits, 8u, 16u);
    min_chunk_size = MAX(64u, min_chunk_size);
    max_chunk_size = MAX(min_chunk_size, max_chunk_size);

    if (per_chunk_quantization) {
        print_config_summary();
    }
}

void QuantizationConfig::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    ps->set_setting(PER_CHUNK_QUANTIZATION_PATH, per_chunk_quantization ? 1 : 0); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    ps->set_setting(POSITION_BITS_PATH, (int)position_bits); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    ps->set_setting(SCALE_BITS_PATH, (int)scale_bits); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    ps->set_setting(QUANTIZE_SCALES_PATH, quantize_scales); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    ps->set_setting(MIN_CHUNK_SIZE_PATH, (int)min_chunk_size); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    ps->set_setting(MAX_CHUNK_SIZE_PATH, (int)max_chunk_size); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    ps->set_setting(ADAPTIVE_CHUNK_SIZE_PATH, adaptive_chunk_size); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION

    ps->save();

    GS_LOG_STREAMING_INFO("[Quantization Config] Configuration saved to project settings");
}

void QuantizationConfig::reset_to_defaults() {
    per_chunk_quantization = false;
    position_bits = 16;
    scale_bits = 12;
    quantize_scales = false;
    min_chunk_size = 256;
    max_chunk_size = 8192;
    adaptive_chunk_size = true;

    GS_LOG_STREAMING_INFO("[Quantization Config] Reset to default configuration");
}

bool QuantizationConfig::validate() const {
    // Position bits must be in valid range
    if (position_bits < 8 || position_bits > 24) {
        return false;
    }

    // Scale bits must be in valid range
    if (scale_bits < 8 || scale_bits > 16) {
        return false;
    }

    // Chunk size constraints
    if (min_chunk_size < 64) {
        return false;
    }
    if (max_chunk_size < min_chunk_size) {
        return false;
    }

    return true;
}

String QuantizationConfig::get_validation_errors() const {
    String errors;

    if (position_bits < 8) {
        errors += "Position bits must be >= 8\n";
    }
    if (position_bits > 24) {
        errors += "Position bits must be <= 24\n";
    }
    if (scale_bits < 8) {
        errors += "Scale bits must be >= 8\n";
    }
    if (scale_bits > 16) {
        errors += "Scale bits must be <= 16\n";
    }
    if (min_chunk_size < 64) {
        errors += "Minimum chunk size must be >= 64\n";
    }
    if (max_chunk_size < min_chunk_size) {
        errors += "Maximum chunk size must be >= minimum chunk size\n";
    }

    return errors;
}

float QuantizationConfig::get_position_compression_ratio() const {
    if (!per_chunk_quantization) {
        return 1.0f;
    }

    // Original: 3 floats (12 bytes) per position
    // Quantized: 3 * position_bits / 8 bytes + small overhead for chunk bounds
    // Assuming chunk overhead is amortized over min_chunk_size Gaussians
    float original_bytes = 12.0f;
    float quantized_bytes = (3.0f * float(position_bits)) / 8.0f;

    // Add amortized chunk bounds overhead (6 floats = 24 bytes for min/max)
    float chunk_overhead = 24.0f / float(min_chunk_size);
    quantized_bytes += chunk_overhead;

    return original_bytes / quantized_bytes;
}

float QuantizationConfig::get_scale_compression_ratio() const {
    if (!per_chunk_quantization || !quantize_scales) {
        return 1.0f;
    }

    // Original: 3 floats (12 bytes) per scale
    // Quantized: 3 * scale_bits / 8 bytes + small overhead for chunk bounds
    float original_bytes = 12.0f;
    float quantized_bytes = (3.0f * float(scale_bits)) / 8.0f;

    // Add amortized chunk bounds overhead (6 floats = 24 bytes for min/max)
    float chunk_overhead = 24.0f / float(min_chunk_size);
    quantized_bytes += chunk_overhead;

    return original_bytes / quantized_bytes;
}

float QuantizationConfig::get_total_compression_ratio() const {
    if (!per_chunk_quantization) {
        return 1.0f;
    }

    // PackedGaussian is 144 bytes total
    // Position: 12 bytes, Scale: 12 bytes
    const float total_bytes = 144.0f;
    const float position_bytes = 12.0f;
    const float scale_bytes = 12.0f;

    float saved_position = position_bytes * (1.0f - 1.0f / get_position_compression_ratio());
    float saved_scale = quantize_scales ? scale_bytes * (1.0f - 1.0f / get_scale_compression_ratio()) : 0.0f;

    float new_size = total_bytes - saved_position - saved_scale;
    return total_bytes / new_size;
}

void QuantizationConfig::print_config_summary() const {
    GS_LOG_STREAMING_INFO("[Quantization Config] ========== Configuration Summary ==========");
    GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Per-Chunk Quantization: %s",
            per_chunk_quantization ? "ENABLED" : "disabled"));

    if (per_chunk_quantization) {
        GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Position Bits: %d (%d levels)",
                position_bits, get_position_levels()));
        GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Scale Quantization: %s",
                quantize_scales ? "enabled" : "disabled"));
        if (quantize_scales) {
            GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Scale Bits: %d (%d levels)",
                    scale_bits, get_scale_levels()));
        }
        GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Chunk Size Range: %d - %d",
                min_chunk_size, max_chunk_size));
        GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Adaptive Chunk Size: %s",
                adaptive_chunk_size ? "enabled" : "disabled"));

        float pos_ratio = get_position_compression_ratio();
        float total_ratio = get_total_compression_ratio();

        GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Position Compression: %.2fx",
                pos_ratio));
        if (quantize_scales) {
            GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Scale Compression: %.2fx",
                    get_scale_compression_ratio()));
        }
        GS_LOG_STREAMING_INFO(vformat("[Quantization Config] Total Compression: %.2fx (%.1f%% savings)",
                total_ratio, (1.0f - 1.0f / total_ratio) * 100.0f));
    }

    GS_LOG_STREAMING_INFO("[Quantization Config] ================================================");
}

void register_quantization_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    // Per-chunk quantization enable (sentinel -1 = auto from tier, 0 = off, 1 = on)
    if (!ps->has_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH)) {
        ps->set_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, -1); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, -1);
    ps->set_custom_property_info(PropertyInfo(
        Variant::INT,
        QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH,
        PROPERTY_HINT_ENUM,
        "Auto (Tier Default):-1,Disabled:0,Enabled:1"
    ));

    // Position bits
    if (!ps->has_setting(QuantizationConfig::POSITION_BITS_PATH)) {
        ps->set_setting(QuantizationConfig::POSITION_BITS_PATH, 16);
    }
    ps->set_initial_value(QuantizationConfig::POSITION_BITS_PATH, 16);
    ps->set_custom_property_info(PropertyInfo(
        Variant::INT,
        QuantizationConfig::POSITION_BITS_PATH,
        PROPERTY_HINT_RANGE,
        "8,24,1"
    ));

    // Scale bits
    if (!ps->has_setting(QuantizationConfig::SCALE_BITS_PATH)) {
        ps->set_setting(QuantizationConfig::SCALE_BITS_PATH, 12);
    }
    ps->set_initial_value(QuantizationConfig::SCALE_BITS_PATH, 12);
    ps->set_custom_property_info(PropertyInfo(
        Variant::INT,
        QuantizationConfig::SCALE_BITS_PATH,
        PROPERTY_HINT_RANGE,
        "8,16,1"
    ));

    // Quantize scales
    if (!ps->has_setting(QuantizationConfig::QUANTIZE_SCALES_PATH)) {
        ps->set_setting(QuantizationConfig::QUANTIZE_SCALES_PATH, false);
    }
    ps->set_initial_value(QuantizationConfig::QUANTIZE_SCALES_PATH, false);
    ps->set_custom_property_info(PropertyInfo(
        Variant::BOOL,
        QuantizationConfig::QUANTIZE_SCALES_PATH,
        PROPERTY_HINT_NONE,
        ""
    ));

    // Minimum chunk size
    if (!ps->has_setting(QuantizationConfig::MIN_CHUNK_SIZE_PATH)) {
        ps->set_setting(QuantizationConfig::MIN_CHUNK_SIZE_PATH, 256);
    }
    ps->set_initial_value(QuantizationConfig::MIN_CHUNK_SIZE_PATH, 256);
    ps->set_custom_property_info(PropertyInfo(
        Variant::INT,
        QuantizationConfig::MIN_CHUNK_SIZE_PATH,
        PROPERTY_HINT_RANGE,
        "64,4096,64"
    ));

    // Maximum chunk size
    if (!ps->has_setting(QuantizationConfig::MAX_CHUNK_SIZE_PATH)) {
        ps->set_setting(QuantizationConfig::MAX_CHUNK_SIZE_PATH, 8192);
    }
    ps->set_initial_value(QuantizationConfig::MAX_CHUNK_SIZE_PATH, 8192);
    ps->set_custom_property_info(PropertyInfo(
        Variant::INT,
        QuantizationConfig::MAX_CHUNK_SIZE_PATH,
        PROPERTY_HINT_RANGE,
        "256,65536,256"
    ));

    // Adaptive chunk size
    if (!ps->has_setting(QuantizationConfig::ADAPTIVE_CHUNK_SIZE_PATH)) {
        ps->set_setting(QuantizationConfig::ADAPTIVE_CHUNK_SIZE_PATH, true);
    }
    ps->set_initial_value(QuantizationConfig::ADAPTIVE_CHUNK_SIZE_PATH, true);
    ps->set_custom_property_info(PropertyInfo(
        Variant::BOOL,
        QuantizationConfig::ADAPTIVE_CHUNK_SIZE_PATH,
        PROPERTY_HINT_NONE,
        ""
    ));
}

void initialize_quantization_config() {
    // Register project settings first
    register_quantization_project_settings();

    // Load configuration
    g_quantization_config.load_from_project_settings();

    if (!g_quantization_config.validate()) {
        GS_LOG_STREAMING_ERROR("[Quantization Config] Invalid configuration detected:");
        GS_LOG_STREAMING_ERROR(g_quantization_config.get_validation_errors());
        GS_LOG_STREAMING_INFO("[Quantization Config] Resetting to defaults...");
        g_quantization_config.reset_to_defaults();
        g_quantization_config.save_to_project_settings();
    }
}

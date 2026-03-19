#include "float16_config.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "../logger/gs_logger.h"

// Project settings paths
const String Float16Config::SECTION_PATH = "rendering/gaussian_splatting/data/";
const String Float16Config::USE_FLOAT16_STORAGE_PATH = SECTION_PATH + "use_float16_storage";
const String Float16Config::FLOAT16_POSITIONS_PATH = SECTION_PATH + "float16_positions";
const String Float16Config::FLOAT16_SH_PATH = SECTION_PATH + "float16_sh_coefficients";
const String Float16Config::FLOAT16_ROTATIONS_PATH = SECTION_PATH + "float16_rotations";
const String Float16Config::ENABLE_QUANTIZATION_PATH = SECTION_PATH + "enable_position_quantization";
const String Float16Config::QUANTIZATION_CHUNK_SIZE_PATH = SECTION_PATH + "quantization_chunk_size";

// Global instance
Float16Config g_float16_config;

void Float16Config::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    use_float16_storage = ps->get_setting(USE_FLOAT16_STORAGE_PATH, false);
    float16_positions = ps->get_setting(FLOAT16_POSITIONS_PATH, true);
    float16_sh_coefficients = ps->get_setting(FLOAT16_SH_PATH, true);
    float16_rotations = ps->get_setting(FLOAT16_ROTATIONS_PATH, true);
    enable_position_quantization = ps->get_setting(ENABLE_QUANTIZATION_PATH, true);
    quantization_chunk_size = ps->get_setting(QUANTIZATION_CHUNK_SIZE_PATH, 4096);

    // Compute packed size based on settings
    packed_gaussian_size_fp16 = compute_packed_size();

    if (use_float16_storage) {
        print_config_summary();
    }
}

void Float16Config::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    ps->set_setting(USE_FLOAT16_STORAGE_PATH, use_float16_storage);
    ps->set_setting(FLOAT16_POSITIONS_PATH, float16_positions);
    ps->set_setting(FLOAT16_SH_PATH, float16_sh_coefficients);
    ps->set_setting(FLOAT16_ROTATIONS_PATH, float16_rotations);
    ps->set_setting(ENABLE_QUANTIZATION_PATH, enable_position_quantization);
    ps->set_setting(QUANTIZATION_CHUNK_SIZE_PATH, quantization_chunk_size);

    ps->save();

    GS_LOG_STREAMING_INFO("[Float16 Config] Configuration saved to project settings");
}

void Float16Config::reset_to_defaults() {
    use_float16_storage = false;
    float16_positions = true;
    float16_sh_coefficients = true;
    float16_rotations = true;
    enable_position_quantization = true;
    quantization_chunk_size = 4096;
    packed_gaussian_size_fp16 = compute_packed_size();

    GS_LOG_STREAMING_INFO("[Float16 Config] Reset to default configuration");
}

bool Float16Config::validate() const {
    // Quantization chunk size must be reasonable
    if (quantization_chunk_size < 64 || quantization_chunk_size > 65536) {
        return false;
    }

    // At least one field should use float16 if storage is enabled
    if (use_float16_storage && !float16_positions && !float16_sh_coefficients && !float16_rotations) {
        return false;
    }

    return true;
}

String Float16Config::get_validation_errors() const {
    String errors;

    if (quantization_chunk_size < 64) {
        errors += "Quantization chunk size must be >= 64\n";
    }
    if (quantization_chunk_size > 65536) {
        errors += "Quantization chunk size must be <= 65536\n";
    }
    if (use_float16_storage && !float16_positions && !float16_sh_coefficients && !float16_rotations) {
        errors += "At least one field must use Float16 when storage is enabled\n";
    }

    return errors;
}

float Float16Config::get_compression_ratio() const {
    if (!use_float16_storage) {
        return 1.0f;
    }

    uint32_t new_size = compute_packed_size();
    return float(packed_gaussian_size_fp32) / float(new_size);
}

uint32_t Float16Config::compute_packed_size() const {
    if (!use_float16_storage) {
        return packed_gaussian_size_fp32;
    }

    // PackedGaussian current layout (144 bytes):
    //   position[3]: 12 bytes
    //   opacity: 4 bytes
    //   scale[3]: 12 bytes
    //   area: 4 bytes
    //   rotation[4]: 16 bytes
    //   sh: 64 bytes (dc[4] + encoded[12])
    //   normal[3]: 12 bytes
    //   stroke_age: 4 bytes
    //   brush_axes[2]: 8 bytes
    //   painterly_meta: 4 bytes
    //   sh_metadata: 4 bytes
    // Total: 144 bytes

    uint32_t size = 0;

    // Position: 12 bytes FP32 -> 6 bytes FP16 (+ chunk offset overhead)
    if (float16_positions) {
        size += 6;  // 3 x half
    } else {
        size += 12; // 3 x float
    }

    // Opacity: keep as 4 bytes (or use packed 8-bit in existing layout)
    size += 4;

    // Scale: always FP32 (precision-sensitive) - 12 bytes
    size += 12;

    // Area: keep as FP32 - 4 bytes
    size += 4;

    // Rotation: 16 bytes FP32 -> 8 bytes FP16
    if (float16_rotations) {
        size += 8;  // 4 x half
    } else {
        size += 16; // 4 x float
    }

    // SH: DC[4] + encoded[12] = 64 bytes FP32 -> 32 bytes FP16
    // Note: SH is already somewhat compressed with RGB9E5 encoding
    // For FP16 we can keep DC as FP32 for base color quality
    if (float16_sh_coefficients) {
        size += 16; // DC as FP32 (4 floats)
        size += 24; // encoded[12] as FP16 (12 halfs = 24 bytes)
    } else {
        size += 64; // Full FP32
    }

    // Normal, stroke_age, brush_axes, painterly_meta, sh_metadata: 32 bytes
    size += 12; // normal[3]
    size += 4;  // stroke_age
    size += 8;  // brush_axes[2]
    size += 4;  // painterly_meta
    size += 4;  // sh_metadata

    // Align to 16 bytes for GPU compatibility
    size = ((size + 15) / 16) * 16;

    return size;
}

void Float16Config::print_config_summary() const {
    GS_LOG_STREAMING_INFO("[Float16 Config] ========== Configuration Summary ==========");
    GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Float16 Storage: %s", use_float16_storage ? "ENABLED" : "disabled"));

    if (use_float16_storage) {
        GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Float16 Positions: %s", float16_positions ? "yes" : "no"));
        GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Float16 SH Coefficients: %s", float16_sh_coefficients ? "yes" : "no"));
        GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Float16 Rotations: %s", float16_rotations ? "yes" : "no"));
        GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Position Quantization: %s (chunk size: %d)",
                enable_position_quantization ? "enabled" : "disabled", quantization_chunk_size));

        uint32_t new_size = compute_packed_size();
        float ratio = get_compression_ratio();
        float savings = (1.0f - 1.0f / ratio) * 100.0f;

        GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Packed Size: %d bytes (was %d bytes)", new_size, packed_gaussian_size_fp32));
        GS_LOG_STREAMING_INFO(vformat("[Float16 Config] Compression Ratio: %.2fx (%.1f%% memory savings)", ratio, savings));
    }

    GS_LOG_STREAMING_INFO("[Float16 Config] ================================================");
}

void initialize_float16_config() {
    g_float16_config.load_from_project_settings();

    if (!g_float16_config.validate()) {
        GS_LOG_STREAMING_ERROR("[Float16 Config] Invalid configuration detected:");
        GS_LOG_STREAMING_ERROR(g_float16_config.get_validation_errors());
        GS_LOG_STREAMING_INFO("[Float16 Config] Resetting to defaults...");
        g_float16_config.reset_to_defaults();
        g_float16_config.save_to_project_settings();
    }
}

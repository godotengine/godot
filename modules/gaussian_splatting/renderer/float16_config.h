#ifndef FLOAT16_CONFIG_H
#define FLOAT16_CONFIG_H

#include "core/string/ustring.h"
#include "core/config/project_settings.h"

/**
 * @file float16_config.h
 * @brief Configuration for Float16 (half-precision) storage of Gaussian Splatting data.
 *
 * Float16 storage provides approximately 2x compression compared to Float32 with
 * minimal quality impact for most fields. Key benefits:
 * - 50% memory reduction for eligible fields
 * - 20-30% streaming bandwidth improvement
 * - Faster GPU memory transfers
 *
 * Fields eligible for Float16 storage:
 * - Position (xyz) - with per-chunk offset for improved precision
 * - SH coefficients (already low dynamic range)
 * - Rotation quaternion (normalized values in [-1, 1])
 *
 * Fields that remain Float32 (precision-sensitive):
 * - Scale (affects Gaussian shape critically)
 * - Opacity (already stored as 8-bit unorm in packed format)
 */

struct Float16Config {
    // Master enable flag for Float16 storage
    bool use_float16_storage = false;

    // Per-field Float16 options (only apply when use_float16_storage is true)
    bool float16_positions = true;      // Position xyz (requires quantization offset)
    bool float16_sh_coefficients = true; // Spherical harmonics (already low range)
    bool float16_rotations = true;       // Quaternion components (normalized)

    // Per-chunk quantization for positions (improves FP16 precision)
    // Each chunk stores a 3D offset (center) and data is relative to that offset
    bool enable_position_quantization = true;
    uint32_t quantization_chunk_size = 4096;  // Gaussians per quantization chunk

    // Runtime computed values
    uint32_t packed_gaussian_size_fp32 = 144;  // Current PackedGaussian size
    uint32_t packed_gaussian_size_fp16 = 0;    // Computed when enabled

    // Project settings paths
    static const String SECTION_PATH;
    static const String USE_FLOAT16_STORAGE_PATH;
    static const String FLOAT16_POSITIONS_PATH;
    static const String FLOAT16_SH_PATH;
    static const String FLOAT16_ROTATIONS_PATH;
    static const String ENABLE_QUANTIZATION_PATH;
    static const String QUANTIZATION_CHUNK_SIZE_PATH;

    // Configuration management
    void load_from_project_settings();
    void save_to_project_settings() const;
    void reset_to_defaults();

    // Validation
    bool validate() const;
    String get_validation_errors() const;

    // Compute expected memory savings
    float get_compression_ratio() const;
    uint32_t compute_packed_size() const;

    // Print configuration summary
    void print_config_summary() const;
};

// Global configuration instance
extern Float16Config g_float16_config;

// Initialization function (called during module init)
void initialize_float16_config();

#endif // FLOAT16_CONFIG_H

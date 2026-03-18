#ifndef QUANTIZATION_CONFIG_H
#define QUANTIZATION_CONFIG_H

#include "core/string/ustring.h"
#include "core/config/project_settings.h"

/**
 * @file quantization_config.h
 * @brief Configuration for per-chunk quantization of Gaussian Splatting data.
 *
 * Per-chunk quantization provides approximately 4x compression for position data
 * with minimal quality loss. Based on Unity's Gaussian Splatting implementation:
 * - Each chunk stores min/max bounds for position normalization
 * - Positions stored as normalized values relative to chunk bounds
 * - Can combine with Float16 storage for additional compression
 *
 * Key benefits:
 * - 4x compression ratio for position data
 * - Minimal visual quality loss (imperceptible in most cases)
 * - Reduced GPU memory usage and streaming bandwidth
 * - Better cache coherency for spatially-local Gaussians
 */

struct QuantizationConfig {
    // Master enable flag for per-chunk quantization
    bool per_chunk_quantization = false;

    // Bit depth for position quantization (8-24 bits, default 16)
    // 8 bits:  256 levels, ~0.4% precision, 4x compression
    // 16 bits: 65536 levels, ~0.0015% precision, 2x compression (vs FP32)
    // 24 bits: 16M levels, full precision retained, 1.33x compression
    uint32_t position_bits = 16;

    // Bit depth for scale quantization (8-16 bits, default 12)
    uint32_t scale_bits = 12;

    // Enable scale quantization (in addition to position)
    bool quantize_scales = false;

    // Minimum chunk size for quantization (smaller chunks = better precision)
    uint32_t min_chunk_size = 256;

    // Maximum chunk size for quantization (larger chunks = more memory efficient)
    uint32_t max_chunk_size = 8192;

    // Auto-adjust chunk size based on spatial extent
    bool adaptive_chunk_size = true;

    // Project settings paths
    static const String SECTION_PATH;
    static const String PER_CHUNK_QUANTIZATION_PATH;
    static const String POSITION_BITS_PATH;
    static const String SCALE_BITS_PATH;
    static const String QUANTIZE_SCALES_PATH;
    static const String MIN_CHUNK_SIZE_PATH;
    static const String MAX_CHUNK_SIZE_PATH;
    static const String ADAPTIVE_CHUNK_SIZE_PATH;

    // Configuration management
    void load_from_project_settings();
    void save_to_project_settings() const;
    void reset_to_defaults();

    // Validation
    bool validate() const;
    String get_validation_errors() const;

    // Compute expected memory savings
    float get_position_compression_ratio() const;
    float get_scale_compression_ratio() const;
    float get_total_compression_ratio() const;

    // Get number of quantization levels for position
    uint32_t get_position_levels() const { return (1u << position_bits) - 1; }

    // Get number of quantization levels for scale
    uint32_t get_scale_levels() const { return (1u << scale_bits) - 1; }

    // Print configuration summary
    void print_config_summary() const;
};

// Global configuration instance
extern QuantizationConfig g_quantization_config;

// Initialization function (called during module init)
void initialize_quantization_config();

// Register project settings (called once during module registration)
void register_quantization_project_settings();

#endif // QUANTIZATION_CONFIG_H

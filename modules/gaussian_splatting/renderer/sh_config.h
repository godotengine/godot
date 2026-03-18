#ifndef SH_CONFIG_H
#define SH_CONFIG_H

#include "core/string/ustring.h"
#include "core/config/project_settings.h"

/**
 * @file sh_config.h
 * @brief Spherical Harmonics configuration for memory/quality tradeoff.
 *
 * Provides runtime configuration of SH band count to balance memory usage
 * against view-dependent color quality:
 *
 * - SH0 (DC only):   Base color, 3 values per splat (~4x memory reduction)
 * - SH1 (1st order): + basic view dependence, 12 values total
 * - SH2 (2nd order): + enhanced view dependence, 27 values total
 * - SH3 (3rd order): Full quality, 48 values total (default)
 *
 * Progressive loading allows instant display with SH0, loading higher
 * bands as bandwidth allows.
 */

// Project settings paths
#define SH_CONFIG_SECTION "rendering/gaussian_splatting/rendering/"
#define SH_CONFIG_BANDS_PATH SH_CONFIG_SECTION "sh_bands"
#define SH_CONFIG_DC_LOGIT_PATH SH_CONFIG_SECTION "dc_is_logit"
#define SH_CONFIG_PROGRESSIVE_PATH "rendering/gaussian_splatting/streaming/sh_progressive_load"

/**
 * @enum SHBandLevel
 * @brief Spherical Harmonics band configuration levels.
 */
enum SHBandLevel {
    SH_BAND_0 = 0,  ///< DC only (base color, 1 coefficient = 3 values)
    SH_BAND_1 = 1,  ///< DC + 1st order (4 coefficients = 12 values)
    SH_BAND_2 = 2,  ///< DC + 2nd order (9 coefficients = 27 values)
    SH_BAND_3 = 3,  ///< DC + 3rd order (16 coefficients = 48 values) - default
    SH_BAND_MAX = SH_BAND_3
};

/**
 * @struct SHConfig
 * @brief Runtime configuration for Spherical Harmonics rendering.
 */
struct SHConfig {
    // Current SH band level (0-3)
    SHBandLevel sh_bands = SH_BAND_3;

    // Enable progressive SH loading (SH0 first, then higher bands)
    bool progressive_load = false;

    // Treat DC as logit and apply sigmoid (legacy compatibility)
    bool dc_is_logit = false;

    // Project settings integration
    void load_from_project_settings();
    void save_to_project_settings() const;
    void reset_to_defaults();

    // Validation
    bool validate() const;

    // Utility functions
    static int get_coefficient_count(SHBandLevel level);
    static int get_float_count(SHBandLevel level);
    static const char* get_band_name(SHBandLevel level);
    static float get_memory_multiplier(SHBandLevel level);

    // Debug output
    void print_config_summary() const;

    // Constants for project settings paths
    static const String BANDS_PATH;
    static const String DC_LOGIT_PATH;
    static const String PROGRESSIVE_PATH;
};

// Global configuration instance
extern SHConfig g_sh_config;

// Configuration management functions
void initialize_sh_config();

#endif // SH_CONFIG_H

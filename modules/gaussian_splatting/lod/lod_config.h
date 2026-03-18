#ifndef LOD_CONFIG_H
#define LOD_CONFIG_H

#include "core/string/ustring.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include <cfloat>  // For FLT_MAX

/**
 * @file lod_config.h
 * @brief Distance-based LOD configuration for Gaussian Splatting streaming.
 *
 * Implements the Octree-GS LOD selection formula:
 *   L = floor(min(max(log2(d_max/d), 0), K-1))
 *
 * Where:
 *   d = distance from camera to chunk/splat
 *   d_max = maximum render distance
 *   K = number of LOD levels
 *   L = selected LOD level (0 = highest detail, K-1 = lowest)
 *
 * LOD reduction strategies:
 *   - Splat skipping: Skip every Nth splat based on LOD level
 *   - SH band reduction: Use fewer spherical harmonics bands at distance
 *   - Opacity fade: Reduce opacity for distant splats
 */

// Project settings paths for LOD configuration
#define LOD_CONFIG_SECTION "rendering/gaussian_splatting/lod/"
#define LOD_CONFIG_ENABLED_PATH LOD_CONFIG_SECTION "enabled"
#define LOD_CONFIG_NUM_LEVELS_PATH LOD_CONFIG_SECTION "num_levels"
#define LOD_CONFIG_MAX_DISTANCE_PATH LOD_CONFIG_SECTION "max_distance"
#define LOD_CONFIG_BASE_THRESHOLD_PATH LOD_CONFIG_SECTION "base_threshold"
#define LOD_CONFIG_SPLAT_SKIP_ENABLED_PATH LOD_CONFIG_SECTION "splat_skip_enabled"
#define LOD_CONFIG_SH_REDUCTION_ENABLED_PATH LOD_CONFIG_SECTION "sh_reduction_enabled"
#define LOD_CONFIG_OPACITY_FADE_ENABLED_PATH LOD_CONFIG_SECTION "opacity_fade_enabled"
#define LOD_CONFIG_DEBUG_VISUALIZATION_PATH LOD_CONFIG_SECTION "debug_visualization"
#define LOD_CONFIG_MIN_SCREEN_SIZE_PIXELS_PATH LOD_CONFIG_SECTION "min_screen_size_pixels"
#define LOD_CONFIG_BIAS_PATH LOD_CONFIG_SECTION "bias"

/**
 * @struct LODConfig
 * @brief Runtime configuration for distance-based LOD selection.
 */
struct LODConfig {
    // Core LOD settings
    bool enabled = true;              // Enable distance-based LOD
    int num_levels = 4;               // Number of LOD levels (2-8)
    float max_distance = 100.0f;      // Maximum render distance
    float base_threshold = 10.0f;     // Distance for LOD 0 (highest detail)

    // LOD reduction strategies
    bool splat_skip_enabled = true;   // Skip splats based on LOD level
    bool sh_reduction_enabled = true; // Reduce SH bands at distance
    bool opacity_fade_enabled = true; // Fade opacity at distance

    // Debug
    bool debug_visualization = false; // Show LOD level colors

    // Project settings integration
    void load_from_project_settings();
    void save_to_project_settings() const;
    void reset_to_defaults();

    // Validation
    bool validate() const;

    /**
     * @brief Calculate LOD level using Octree-GS formula.
     *
     * L = floor(min(max(log2(d_max/d), 0), K-1))
     *
     * @param distance Distance from camera to splat/chunk
     * @return LOD level (0 = highest detail, num_levels-1 = lowest)
     */
    int calculate_lod_level(float distance) const;

    /**
     * @brief Get splat skip factor for a given LOD level.
     *
     * LOD 0: render all splats (skip 1)
     * LOD 1: render every 2nd splat (skip 2)
     * LOD 2: render every 4th splat (skip 4)
     * LOD N: render every 2^N splat
     *
     * @param lod_level The LOD level
     * @return Skip factor (1 = render all, 2 = skip half, etc.)
     */
    int get_splat_skip_factor(int lod_level) const;

    /**
     * @brief Get SH band level for a given LOD level.
     *
     * Maps LOD level to SH band level (0-3):
     *   LOD 0: SH3 (full quality)
     *   LOD 1: SH2
     *   LOD 2: SH1
     *   LOD 3+: SH0 (DC only)
     *
     * @param lod_level The LOD level
     * @return SH band level (0-3)
     */
    int get_sh_band_for_lod(int lod_level) const;

    /**
     * @brief Get opacity multiplier for a given distance.
     *
     * Returns 1.0 for close objects, fades to 0.0 at max_distance.
     *
     * @param distance Distance from camera
     * @return Opacity multiplier (0.0 - 1.0)
     */
    float get_opacity_multiplier(float distance) const;

    /**
     * @brief Get distance threshold for a specific LOD level.
     *
     * @param lod_level The LOD level
     * @return Distance threshold for that LOD level
     */
    float get_distance_threshold(int lod_level) const;

    // Debug output
    void print_config_summary() const;

    // Constants for project settings paths
    static const String ENABLED_PATH;
    static const String NUM_LEVELS_PATH;
    static const String MAX_DISTANCE_PATH;
    static const String BASE_THRESHOLD_PATH;
};

/**
 * @struct ChunkLODMetadata
 * @brief LOD metadata stored per streaming chunk.
 */
struct ChunkLODMetadata {
    int lod_level = 0;               // Current LOD level for this chunk
    float distance = 0.0f;           // Distance from camera
    int sh_band_level = 3;           // Current SH band level
    int splat_skip_factor = 1;       // Current splat skip factor
    float opacity_multiplier = 1.0f; // Current opacity multiplier
    bool needs_update = false;       // LOD changed, needs re-upload

    void update_from_distance(float p_distance, const LODConfig& config);
};

/**
 * @struct LODDebugStats
 * @brief Debug statistics for LOD system monitoring.
 */
struct LODDebugStats {
    // Per-LOD level counts
    uint32_t lod_level_counts[8] = {};  // Chunks per LOD level (up to 8 levels)
    uint32_t total_chunks = 0;

    // Reduction statistics
    uint32_t total_splats_original = 0;
    uint32_t total_splats_after_skip = 0;
    float splat_reduction_ratio = 0.0f;

    // SH band distribution
    uint32_t sh_band_counts[4] = {};    // Chunks per SH band (SH0-SH3)

    // Distance statistics
    float min_distance = 0.0f;
    float max_distance = 0.0f;
    float avg_distance = 0.0f;

    void reset();
    void update_from_chunks(const ChunkLODMetadata* chunks, uint32_t count);
    String to_string() const;
};

// Global configuration instance
extern LODConfig g_lod_config;

// Configuration management functions
void initialize_lod_config();
void register_lod_project_settings();

#endif // LOD_CONFIG_H

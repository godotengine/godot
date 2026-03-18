/**
 * @file render_config_types.h
 * @brief Configuration type definitions for GaussianSplatRenderer.
 *
 * This header contains configuration structs that control rendering behavior.
 * Extracted from inline .inc files for better code organization and IDE support.
 */

#ifndef GAUSSIAN_RENDER_CONFIG_TYPES_H
#define GAUSSIAN_RENDER_CONFIG_TYPES_H

#include "core/object/ref_counted.h"
#include "core/math/color.h"
#include "core/templates/hash_map.h"
#include "core/templates/rid.h"

// Forward declarations
class ColorGradingResource;

/**
 * @namespace GaussianRenderConfig
 * @brief Contains standalone configuration types that can be used independently.
 *
 * Note: Some config types (RenderConfig, CullingConfig, PainterlyConfig) remain
 * defined inside GaussianSplatRenderer class because they reference class enums.
 */
namespace GaussianRenderConfig {

/**
 * @struct StateUniformData
 * @brief GPU uniform data for interactive state rendering effects.
 *
 * This struct is used for shader uploads and must maintain 16-byte alignment
 * for std140 uniform buffer layout compatibility.
 */
struct StateUniformData {
    float highlight_strength = 0.0f;
    float outline_width = 0.0f;
    float state = 0.0f;
    float reserved = 0.0f;
    Color highlight_color = Color(1.2, 1.2, 0.8, 1.0);
    Color outline_color = Color(1.0, 0.5, 0.0, 1.0);
};
static_assert(sizeof(StateUniformData) % 16 == 0, "StateUniformData must stay 16-byte aligned for std140 uploads");

} // namespace GaussianRenderConfig

#endif // GAUSSIAN_RENDER_CONFIG_TYPES_H

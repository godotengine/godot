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
 * Renderer-scoped config types live here as standalone types or small templates.
 * GaussianSplatRenderer re-exports them with class-local aliases.
 */
namespace GaussianRenderConfig {

template <typename RenderModeT>
struct RenderConfig {
	RenderModeT render_mode = {};
	float opacity_multiplier = 1.0f;
	Ref<class ColorGradingResource> color_grading;
};

struct CullingConfig {
	bool solid_coverage_enabled = false;
	float solid_coverage_alpha_floor = 0.95f;
};

struct PainterlyConfig {
	bool enabled = false;
	bool enable_strokes = true;
	bool low_end_mode = false;
	float internal_scale = 1.0f;
	float edge_threshold = 0.25f;
	float edge_intensity = 1.5f;
	float stroke_length = 32.0f;
	float stroke_opacity = 0.8f;
	float gamma = 2.2f;
};

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

template <typename InteractiveStateT>
struct InteractiveStateConfig {
	StateUniformData uniform_data;
	InteractiveStateT current_state = {};
	HashMap<InteractiveStateT, RID> state_shaders;
	bool state_dirty = false;
	Color highlight_color = Color(1.2, 1.2, 0.8, 1.0);
	Color outline_color = Color(1.0, 0.5, 0.0, 1.0);
	float outline_width = 2.0f;
	bool highlight_enabled = false;
	bool outline_enabled = false;
};

} // namespace GaussianRenderConfig

#endif // GAUSSIAN_RENDER_CONFIG_TYPES_H

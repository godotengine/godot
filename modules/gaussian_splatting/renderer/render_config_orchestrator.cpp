#include "render_config_orchestrator.h"

#include "render_data_orchestrator.h"
#include "painterly_pass_graph.h"
#include "../interfaces/interactive_state_manager.h"
#include "../interfaces/painterly_renderer.h"
#include "../logger/gs_logger.h"
#include "../resources/color_grading_resource.h"

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

RenderConfigOrchestrator::RenderConfigOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		interactive_state_manager(p_dependencies.interactive_state_manager),
		painterly_renderer(p_dependencies.painterly_renderer) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(interactive_state_manager);
	ERR_FAIL_NULL(painterly_renderer);
}

void RenderConfigOrchestrator::set_render_mode(GaussianSplatRenderer::RenderMode p_mode) {
	render_config.render_mode = p_mode;
}

void RenderConfigOrchestrator::set_opacity_multiplier(float p_opacity) {
	float clamped = CLAMP(p_opacity, 0.0f, 1.0f);
	if (Math::is_equal_approx(render_config.opacity_multiplier, clamped)) {
		return;
	}
	render_config.opacity_multiplier = clamped;
	if (renderer) {
		renderer->invalidate_cached_render();
	}
}

void RenderConfigOrchestrator::set_color_grading(const Ref<ColorGradingResource> &p_grading) {
	if (render_config.color_grading == p_grading) {
		return;
	}
	render_config.color_grading = p_grading;
	if (renderer) {
		renderer->invalidate_cached_render();
	}
}

void RenderConfigOrchestrator::set_interactive_state(GaussianSplatRenderer::InteractiveState p_state) {
	if (interactive_state.current_state == p_state) {
		return;
	}

	if (interactive_state_manager->is_valid()) {
		if (!interactive_state_manager->ptr()->apply_renderer_state(renderer, p_state)) {
			return;
		}
		return;
	}

	GaussianSplatRenderer::InteractiveState old_state = interactive_state.current_state;
	if (old_state == GaussianSplatRenderer::STATE_DISABLED &&
			(p_state == GaussianSplatRenderer::STATE_HOVERED || p_state == GaussianSplatRenderer::STATE_SELECTED)) {
		GS_LOG_WARN_DEFAULT("Invalid state transition: Cannot hover or select disabled splats");
		return;
	}

	interactive_state.current_state = p_state;
}

void RenderConfigOrchestrator::set_solid_coverage_enabled(bool p_enabled) {
	culling_config.solid_coverage_enabled = p_enabled;
}

void RenderConfigOrchestrator::set_solid_coverage_alpha_floor(float p_alpha) {
	culling_config.solid_coverage_alpha_floor = CLAMP(p_alpha, 0.0f, 1.0f);
}

void RenderConfigOrchestrator::set_painterly_enabled(bool p_enabled) {
	if (painterly_config.enabled == p_enabled) {
		return;
	}
	painterly_config.enabled = p_enabled;
	if (!painterly_config.enabled) {
		if (painterly_renderer->is_valid()) {
			PainterlyPassGraph *pass_graph = painterly_renderer->ptr()->get_pass_graph();
			if (pass_graph) {
				pass_graph->reset();
			}
		}
	}
}

void RenderConfigOrchestrator::set_painterly_low_end_mode(bool p_enabled) {
	painterly_config.low_end_mode = p_enabled;
}

void RenderConfigOrchestrator::set_painterly_enable_strokes(bool p_enabled) {
	painterly_config.enable_strokes = p_enabled;
}

void RenderConfigOrchestrator::set_painterly_internal_scale(float p_scale) {
	painterly_config.internal_scale = CLAMP(p_scale, 0.25f, 1.0f);
}

void RenderConfigOrchestrator::set_painterly_edge_threshold(float p_threshold) {
	painterly_config.edge_threshold = CLAMP(p_threshold, 0.0f, 1.0f);
}

void RenderConfigOrchestrator::set_painterly_edge_intensity(float p_intensity) {
	painterly_config.edge_intensity = CLAMP(p_intensity, 0.0f, 8.0f);
}

void RenderConfigOrchestrator::set_painterly_stroke_length(float p_length) {
	painterly_config.stroke_length = CLAMP(p_length, 1.0f, 256.0f);
}

void RenderConfigOrchestrator::set_painterly_stroke_opacity(float p_opacity) {
	painterly_config.stroke_opacity = CLAMP(p_opacity, 0.0f, 1.0f);
}

void RenderConfigOrchestrator::set_painterly_gamma(float p_gamma) {
	painterly_config.gamma = CLAMP(p_gamma, 0.5f, 4.0f);
}

// Getter definitions moved to header (inline)

void GaussianSplatRenderer::set_render_mode(RenderMode p_mode) {
	config_orchestrator->set_render_mode(p_mode);
}

void GaussianSplatRenderer::set_opacity_multiplier(float p_opacity) {
	config_orchestrator->set_opacity_multiplier(p_opacity);
}

void GaussianSplatRenderer::set_color_grading(const Ref<ColorGradingResource> &p_grading) {
	config_orchestrator->set_color_grading(p_grading);
}

void GaussianSplatRenderer::set_interactive_state(InteractiveState p_state) {
	config_orchestrator->set_interactive_state(p_state);
}

void GaussianSplatRenderer::set_solid_coverage_enabled(bool p_enabled) {
	config_orchestrator->set_solid_coverage_enabled(p_enabled);
}

void GaussianSplatRenderer::set_solid_coverage_alpha_floor(float p_alpha) {
	config_orchestrator->set_solid_coverage_alpha_floor(p_alpha);
}

void GaussianSplatRenderer::set_painterly_enabled(bool p_enabled) {
	config_orchestrator->set_painterly_enabled(p_enabled);
}

void GaussianSplatRenderer::set_painterly_low_end_mode(bool p_enabled) {
	config_orchestrator->set_painterly_low_end_mode(p_enabled);
}

void GaussianSplatRenderer::set_painterly_enable_strokes(bool p_enabled) {
	config_orchestrator->set_painterly_enable_strokes(p_enabled);
}

void GaussianSplatRenderer::set_painterly_internal_scale(float p_scale) {
	config_orchestrator->set_painterly_internal_scale(p_scale);
}

void GaussianSplatRenderer::set_painterly_edge_threshold(float p_threshold) {
	config_orchestrator->set_painterly_edge_threshold(p_threshold);
}

void GaussianSplatRenderer::set_painterly_edge_intensity(float p_intensity) {
	config_orchestrator->set_painterly_edge_intensity(p_intensity);
}

void GaussianSplatRenderer::set_painterly_stroke_length(float p_length) {
	config_orchestrator->set_painterly_stroke_length(p_length);
}

void GaussianSplatRenderer::set_painterly_stroke_opacity(float p_opacity) {
	config_orchestrator->set_painterly_stroke_opacity(p_opacity);
}

void GaussianSplatRenderer::set_painterly_gamma(float p_gamma) {
	config_orchestrator->set_painterly_gamma(p_gamma);
}

void GaussianSplatRenderer::set_gaussian_asset(const Ref<GaussianSplatAsset> &p_asset) {
	data_orchestrator->set_gaussian_asset(p_asset);
}

#ifndef GAUSSIAN_RENDER_CONFIG_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_CONFIG_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

class InteractiveStateManager;
class PainterlyRenderer;

class RenderConfigOrchestrator {
public:
	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		Ref<InteractiveStateManager> *interactive_state_manager = nullptr;
		Ref<PainterlyRenderer> *painterly_renderer = nullptr;
	};

	explicit RenderConfigOrchestrator(const Dependencies &p_dependencies);

	void set_render_mode(GaussianSplatRenderer::RenderMode p_mode);
	void set_opacity_multiplier(float p_opacity);
	void set_color_grading(const Ref<class ColorGradingResource> &p_grading);

	void set_interactive_state(GaussianSplatRenderer::InteractiveState p_state);
	void set_solid_coverage_enabled(bool p_enabled);
	void set_solid_coverage_alpha_floor(float p_alpha);

	void set_painterly_enabled(bool p_enabled);
	void set_painterly_low_end_mode(bool p_enabled);
	void set_painterly_enable_strokes(bool p_enabled);
	void set_painterly_internal_scale(float p_scale);
	void set_painterly_edge_threshold(float p_threshold);
	void set_painterly_edge_intensity(float p_intensity);
	void set_painterly_stroke_length(float p_length);
	void set_painterly_stroke_opacity(float p_opacity);
	void set_painterly_gamma(float p_gamma);

	GaussianSplatRenderer::RenderConfig &get_render_config() { return render_config; }
	const GaussianSplatRenderer::RenderConfig &get_render_config() const { return render_config; }
	GaussianSplatRenderer::InteractiveStateConfig &get_interactive_state() { return interactive_state; }
	const GaussianSplatRenderer::InteractiveStateConfig &get_interactive_state() const { return interactive_state; }
	GaussianSplatRenderer::CullingConfig &get_culling_config() { return culling_config; }
	const GaussianSplatRenderer::CullingConfig &get_culling_config() const { return culling_config; }
	GaussianSplatRenderer::PainterlyConfig &get_painterly_config() { return painterly_config; }
	const GaussianSplatRenderer::PainterlyConfig &get_painterly_config() const { return painterly_config; }

private:
	GaussianSplatRenderer *renderer = nullptr;
	Ref<InteractiveStateManager> *interactive_state_manager = nullptr;
	Ref<PainterlyRenderer> *painterly_renderer = nullptr;
	GaussianSplatRenderer::RenderConfig render_config;
	GaussianSplatRenderer::InteractiveStateConfig interactive_state;
	GaussianSplatRenderer::CullingConfig culling_config;
	GaussianSplatRenderer::PainterlyConfig painterly_config;
};

#endif

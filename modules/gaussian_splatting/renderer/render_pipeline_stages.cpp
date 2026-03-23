#include "render_pipeline_stages.h"

#include "../core/gs_project_settings.h"
#include "render_debug_state_orchestrator.h"
#include "render_diagnostics_orchestrator.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/color.h"
#include "core/math/math_defs.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "core/templates/hashfuncs.h"
#include "gpu_debug_utils.h"
#include "../interfaces/output_compositor.h"
#include "../interfaces/gpu_culler.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../interfaces/interactive_state_manager.h"
#include "../interfaces/debug_overlay_system.h"
#include "../interfaces/overflow_auto_tuner.h"
#include "../interfaces/tile_rasterizer.h"
#include "../interfaces/painterly_renderer.h"
#include "../logger/gs_debug_trace.h"
#include "../logger/gs_logger.h"
#include "../resources/color_grading_resource.h"
#include "gpu_sorting_config.h"
#include "instance_pipeline_contract.h"
#include "pipeline_feature_set.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include <cstring>

namespace {
static GaussianSplatRenderer::StageResult _make_stage_result(
		GaussianSplatRenderer::StageResult::StageStatus p_status,
		const String &p_reason = String(),
		bool p_is_error = false,
		GaussianSplatRenderer::RenderFallbackReason p_fallback_reason = GaussianSplatRenderer::RenderFallbackReason::NONE) {
	GaussianSplatRenderer::StageResult result;
	result.status = p_status;
	result.is_error = p_is_error;
	result.reason = p_reason;
	result.fallback_reason = p_fallback_reason;
	return result;
}

static String _stage_status_label(GaussianSplatRenderer::StageResult::StageStatus p_status) {
	switch (p_status) {
		case GaussianSplatRenderer::StageResult::StageStatus::SUCCESS:
			return "success";
		case GaussianSplatRenderer::StageResult::StageStatus::SKIPPED:
			return "skipped";
		case GaussianSplatRenderer::StageResult::StageStatus::FALLBACK:
			return "fallback";
		case GaussianSplatRenderer::StageResult::StageStatus::FAILED:
			return "failed";
	}
	return "unknown";
}

static String _fallback_reason_label(GaussianSplatRenderer::RenderFallbackReason p_reason) {
	switch (p_reason) {
		case GaussianSplatRenderer::RenderFallbackReason::NONE:
			return "none";
		case GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE:
			return "data_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::STREAMING_DATA_UNAVAILABLE:
			return "streaming_data_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::GPU_CULLER_UNAVAILABLE:
			return "gpu_culler_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS:
			return "no_visible_splats";
		case GaussianSplatRenderer::RenderFallbackReason::RENDERING_DEVICE_UNAVAILABLE:
			return "rendering_device_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::RASTER_REUSED_CACHED_RENDER:
			return "raster_reused_cached_render";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_UNAVAILABLE:
			return "painterly_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_PASS_GRAPH_UNAVAILABLE:
			return "painterly_pass_graph_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_MATERIAL_UNAVAILABLE:
			return "painterly_material_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_RENDER_FAILED:
			return "painterly_render_failed";
		case GaussianSplatRenderer::RenderFallbackReason::TILE_FALLBACK_FAILED:
			return "tile_fallback_failed";
		case GaussianSplatRenderer::RenderFallbackReason::OUTPUT_COMPOSITOR_UNAVAILABLE:
			return "output_compositor_unavailable";
	}
	return "unknown";
}

static bool _pipeline_trace_enabled(GaussianSplatRenderer *p_renderer) {
	return p_renderer && p_renderer->get_debug_config().enable_pipeline_trace;
}

constexpr char GS_SCENE_COMPOSITE_DEPTH_TEST_SETTING[] = "rendering/gaussian_splatting/composite/depth_test";
constexpr bool GS_SCENE_COMPOSITE_DEPTH_TEST_DEFAULT = true;

static bool _is_scene_depth_composite_expected(RenderDataRD *p_render_data) {
	if (!p_render_data || !p_render_data->render_buffers.is_valid()) {
		return false;
	}

	bool depth_test_enabled = GS_SCENE_COMPOSITE_DEPTH_TEST_DEFAULT;
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings && project_settings->has_setting(GS_SCENE_COMPOSITE_DEPTH_TEST_SETTING)) {
		depth_test_enabled = (bool)project_settings->get_setting_with_override(GS_SCENE_COMPOSITE_DEPTH_TEST_SETTING);
	}
	if (!depth_test_enabled) {
		return false;
	}

	RenderSceneBuffersRD *render_buffers_rd = Object::cast_to<RenderSceneBuffersRD>(p_render_data->render_buffers.ptr());
	return render_buffers_rd && render_buffers_rd->get_depth_texture().is_valid();
}

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static bool _get_bool_setting(ProjectSettings *p_settings, const StringName &p_name, bool p_fallback) {
	return gs::settings::get_bool(p_settings, p_name, p_fallback);
}

static float _get_float_setting(ProjectSettings *p_settings, const StringName &p_name, float p_fallback) {
	return gs::settings::get_float(p_settings, p_name, p_fallback);
}

static int _get_int_setting(ProjectSettings *p_settings, const StringName &p_name, int p_fallback) {
	return static_cast<int>(gs::settings::get_uint(p_settings, p_name, static_cast<uint32_t>(p_fallback)));
}

static RD::DataFormat _resolve_compute_friendly_raster_format(RD::DataFormat p_format) {
	switch (p_format) {
		case RD::DATA_FORMAT_R8G8B8A8_UNORM:
			return p_format;
		case RD::DATA_FORMAT_R8G8B8A8_SRGB:
		case RD::DATA_FORMAT_B8G8R8A8_UNORM:
		case RD::DATA_FORMAT_B8G8R8A8_SRGB:
		case RD::DATA_FORMAT_A8B8G8R8_UNORM_PACK32:
		case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
			return RD::DATA_FORMAT_R8G8B8A8_UNORM;
		default:
			return RD::DATA_FORMAT_R8G8B8A8_UNORM;
	}
}

static void _begin_pipeline_trace(GaussianSplatRenderer *p_renderer) {
	if (!p_renderer) {
		return;
	}
	GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	const uint64_t frame_id = state_view.get_frame_state_view().frame_counter;
	uint64_t trace_frame_id = frame_id;
	if (Engine::get_singleton()) {
		trace_frame_id = Engine::get_singleton()->get_process_frames();
	}
	GaussianSplatting::debug_trace_begin_frame(trace_frame_id);
	auto &debug_state = p_renderer->get_debug_state();
	debug_state.pipeline_trace_generation++;
	if (!_pipeline_trace_enabled(p_renderer)) {
		debug_state.pipeline_events.clear();
		debug_state.pipeline_events_frame = frame_id;
		debug_state.pipeline_events_valid = false;
		return;
	}
	debug_state.pipeline_events.clear();
	debug_state.pipeline_events_frame = frame_id;
	debug_state.pipeline_events_valid = true;
}

static void _record_pipeline_event(GaussianSplatRenderer *p_renderer, const char *p_stage, const String &p_message,
		uint32_t p_input_count, uint32_t p_output_count, bool p_is_error,
		GaussianSplatRenderer::RenderFallbackReason p_fallback_reason = GaussianSplatRenderer::RenderFallbackReason::NONE,
		const String &p_route_uid = String()) {
	if (!_pipeline_trace_enabled(p_renderer)) {
		return;
	}
	auto &debug_state = p_renderer->get_debug_state();
	if (!debug_state.pipeline_events_valid) {
		GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
		const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
		debug_state.pipeline_events_frame = state_view.get_frame_state_view().frame_counter;
		debug_state.pipeline_events_valid = true;
	}
	GaussianSplatRenderer::PipelineEvent event;
	event.stage = p_stage;
	event.message = p_message;
	event.route_uid = p_route_uid;
	event.input_count = p_input_count;
	event.output_count = p_output_count;
	event.is_error = p_is_error;
	event.fallback_reason = p_fallback_reason;
	debug_state.pipeline_events.push_back(event);
}

static RID _get_sort_indices_buffer(const GaussianSplatRenderer::IFrameStateView &p_state_view) {
	GPUSortingPipeline *sorting_pipeline = p_state_view.get_sorting_pipeline();
	return sorting_pipeline ? sorting_pipeline->get_sort_indices_buffer() : RID();
}

static _FORCE_INLINE_ uint64_t _hash_u64(uint64_t p_value, uint64_t p_seed) {
	return hash64_murmur3_64(p_value, p_seed);
}

static _FORCE_INLINE_ uint64_t _hash_float_bits(float p_value, uint64_t p_seed) {
	union {
		float f;
		uint32_t u;
	} bits = { p_value };
	return hash64_murmur3_64(static_cast<uint64_t>(bits.u), p_seed);
}

static _FORCE_INLINE_ uint64_t _hash_bool(bool p_value, uint64_t p_seed) {
	return hash64_murmur3_64(p_value ? 1ull : 0ull, p_seed);
}

static uint64_t _hash_vector3(const Vector3 &p_vec, uint64_t p_seed) {
	p_seed = _hash_float_bits(p_vec.x, p_seed);
	p_seed = _hash_float_bits(p_vec.y, p_seed);
	p_seed = _hash_float_bits(p_vec.z, p_seed);
	return p_seed;
}

static uint64_t _hash_transform3d(const Transform3D &p_transform, uint64_t p_seed) {
	p_seed = _hash_vector3(p_transform.basis.get_column(0), p_seed);
	p_seed = _hash_vector3(p_transform.basis.get_column(1), p_seed);
	p_seed = _hash_vector3(p_transform.basis.get_column(2), p_seed);
	p_seed = _hash_vector3(p_transform.origin, p_seed);
	return p_seed;
}

static uint64_t _hash_color(const Color &p_color, uint64_t p_seed) {
	p_seed = _hash_float_bits(p_color.r, p_seed);
	p_seed = _hash_float_bits(p_color.g, p_seed);
	p_seed = _hash_float_bits(p_color.b, p_seed);
	p_seed = _hash_float_bits(p_color.a, p_seed);
	return p_seed;
}

static uint64_t _compute_color_grading_signature(const GaussianSplatRenderer::RenderConfig &p_render_config) {
	uint64_t seed = HASH_MURMUR3_SEED;
	seed = _hash_u64(static_cast<uint64_t>(p_render_config.render_mode), seed);
	seed = _hash_float_bits(p_render_config.opacity_multiplier, seed);

	const Ref<ColorGradingResource> &grading = p_render_config.color_grading;
	if (!grading.is_valid()) {
		return _hash_u64(0ull, seed);
	}

	seed = _hash_u64(1ull, seed);
	seed = _hash_u64(reinterpret_cast<uint64_t>(grading.ptr()), seed);
	seed = _hash_bool(grading->get_enabled(), seed);
	seed = _hash_float_bits(grading->get_exposure(), seed);
	seed = _hash_float_bits(grading->get_contrast(), seed);
	seed = _hash_float_bits(grading->get_saturation(), seed);
	seed = _hash_float_bits(grading->get_temperature(), seed);
	seed = _hash_float_bits(grading->get_tint(), seed);
	seed = _hash_float_bits(grading->get_hue_shift(), seed);
	return seed;
}

static uint64_t _compute_lighting_signature(const RenderDataRD *p_render_data, uint64_t p_frame_id) {
	uint64_t seed = HASH_MURMUR3_SEED;
	if (!p_render_data) {
		// Without RenderDataRD we cannot enumerate scene lights; avoid stale cache reuse.
		return _hash_u64(p_frame_id, seed);
	}

	seed = _hash_u64(1ull, seed);
	seed = _hash_u64(p_render_data->shadow_atlas.get_id(), seed);
	seed = _hash_u64(p_render_data->cluster_buffer.get_id(), seed);
	seed = _hash_u64(static_cast<uint64_t>(p_render_data->cluster_size), seed);
	seed = _hash_u64(static_cast<uint64_t>(p_render_data->cluster_max_elements), seed);
	seed = _hash_u64(static_cast<uint64_t>(p_render_data->directional_light_count), seed);

	if (p_render_data->scene_data) {
		seed = _hash_u64(static_cast<uint64_t>(p_render_data->scene_data->camera_visible_layers), seed);
		seed = _hash_u64(static_cast<uint64_t>(p_render_data->scene_data->directional_light_count), seed);
	}

	float direct_light_scale = 0.5f;
	float indirect_sh_scale = 1.0f;
	float shadow_strength = 1.0f;
	bool sh_dc_logit = false;
	float shadow_receiver_bias_scale = 0.2f;
	float shadow_receiver_bias_min = 0.0f;
	float shadow_receiver_bias_max = 0.0f;
	bool wind_enabled = false;
	float wind_strength = 0.0f;
	float wind_frequency = 1.0f;
	float wind_spatial_frequency = 0.1f;
	float wind_time_scale = 1.0f;
	Vector3 wind_direction = Vector3(1.0f, 0.0f, 0.0f);
	int max_effectors = 1;
	bool sphere_effector_enabled = false;
	Vector3 sphere_effector_center = Vector3();
	float sphere_effector_radius = 0.0f;
	float sphere_effector_strength = 0.0f;
	float sphere_effector_falloff = 2.0f;
	float sphere_effector_frequency = 2.0f;
	if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
		static const StringName direct_path("rendering/gaussian_splatting/lighting/direct_light_scale");
		static const StringName indirect_path("rendering/gaussian_splatting/lighting/indirect_sh_scale");
		static const StringName shadow_path("rendering/gaussian_splatting/lighting/shadow_strength");
		static const StringName sh_dc_logit_path("rendering/gaussian_splatting/lighting/dc_logit");
		static const StringName shadow_bias_scale_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_scale");
		static const StringName shadow_bias_min_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_min");
		static const StringName shadow_bias_max_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_max");
		static const StringName wind_enabled_path("rendering/gaussian_splatting/animation/wind_enabled");
		static const StringName wind_direction_x_path("rendering/gaussian_splatting/animation/wind_direction_x");
		static const StringName wind_direction_y_path("rendering/gaussian_splatting/animation/wind_direction_y");
		static const StringName wind_direction_z_path("rendering/gaussian_splatting/animation/wind_direction_z");
		static const StringName wind_strength_path("rendering/gaussian_splatting/animation/wind_strength");
		static const StringName wind_frequency_path("rendering/gaussian_splatting/animation/wind_frequency");
		static const StringName wind_spatial_frequency_path("rendering/gaussian_splatting/animation/wind_spatial_frequency");
		static const StringName wind_time_scale_path("rendering/gaussian_splatting/animation/wind_time_scale");
		static const StringName max_effectors_path("rendering/gaussian_splatting/effects/max_effectors");
		static const StringName sphere_effector_enabled_path("rendering/gaussian_splatting/effects/sphere_effector_enabled");
		static const StringName sphere_effector_center_x_path("rendering/gaussian_splatting/effects/sphere_effector_center_x");
		static const StringName sphere_effector_center_y_path("rendering/gaussian_splatting/effects/sphere_effector_center_y");
		static const StringName sphere_effector_center_z_path("rendering/gaussian_splatting/effects/sphere_effector_center_z");
		static const StringName sphere_effector_radius_path("rendering/gaussian_splatting/effects/sphere_effector_radius");
		static const StringName sphere_effector_strength_path("rendering/gaussian_splatting/effects/sphere_effector_strength");
		static const StringName sphere_effector_falloff_path("rendering/gaussian_splatting/effects/sphere_effector_falloff");
		static const StringName sphere_effector_frequency_path("rendering/gaussian_splatting/effects/sphere_effector_frequency");

		direct_light_scale = _get_float_setting(ps, direct_path, direct_light_scale);
		indirect_sh_scale = _get_float_setting(ps, indirect_path, indirect_sh_scale);
		shadow_strength = _get_float_setting(ps, shadow_path, shadow_strength);
		sh_dc_logit = _get_bool_setting(ps, sh_dc_logit_path, sh_dc_logit);
		shadow_receiver_bias_scale = _get_float_setting(ps, shadow_bias_scale_path, shadow_receiver_bias_scale);
		shadow_receiver_bias_min = _get_float_setting(ps, shadow_bias_min_path, shadow_receiver_bias_min);
		shadow_receiver_bias_max = _get_float_setting(ps, shadow_bias_max_path, shadow_receiver_bias_max);

		wind_enabled = _get_bool_setting(ps, wind_enabled_path, wind_enabled);
		wind_direction.x = _get_float_setting(ps, wind_direction_x_path, wind_direction.x);
		wind_direction.y = _get_float_setting(ps, wind_direction_y_path, wind_direction.y);
		wind_direction.z = _get_float_setting(ps, wind_direction_z_path, wind_direction.z);
		wind_strength = _get_float_setting(ps, wind_strength_path, wind_strength);
		wind_frequency = _get_float_setting(ps, wind_frequency_path, wind_frequency);
		wind_spatial_frequency = _get_float_setting(ps, wind_spatial_frequency_path, wind_spatial_frequency);
		wind_time_scale = _get_float_setting(ps, wind_time_scale_path, wind_time_scale);

		max_effectors = _get_int_setting(ps, max_effectors_path, max_effectors);
		sphere_effector_enabled = _get_bool_setting(ps, sphere_effector_enabled_path, sphere_effector_enabled);
		sphere_effector_center.x = _get_float_setting(ps, sphere_effector_center_x_path, sphere_effector_center.x);
		sphere_effector_center.y = _get_float_setting(ps, sphere_effector_center_y_path, sphere_effector_center.y);
		sphere_effector_center.z = _get_float_setting(ps, sphere_effector_center_z_path, sphere_effector_center.z);
		sphere_effector_radius = _get_float_setting(ps, sphere_effector_radius_path, sphere_effector_radius);
		sphere_effector_strength = _get_float_setting(ps, sphere_effector_strength_path, sphere_effector_strength);
		sphere_effector_falloff = _get_float_setting(ps, sphere_effector_falloff_path, sphere_effector_falloff);
		sphere_effector_frequency = _get_float_setting(ps, sphere_effector_frequency_path, sphere_effector_frequency);
	}
	seed = _hash_float_bits(direct_light_scale, seed);
	seed = _hash_float_bits(indirect_sh_scale, seed);
	seed = _hash_float_bits(shadow_strength, seed);
	seed = _hash_bool(sh_dc_logit, seed);
	seed = _hash_float_bits(shadow_receiver_bias_scale, seed);
	seed = _hash_float_bits(shadow_receiver_bias_min, seed);
	seed = _hash_float_bits(shadow_receiver_bias_max, seed);
	seed = _hash_bool(wind_enabled, seed);
	seed = _hash_vector3(wind_direction, seed);
	seed = _hash_float_bits(wind_strength, seed);
	seed = _hash_float_bits(wind_frequency, seed);
	seed = _hash_float_bits(wind_spatial_frequency, seed);
	seed = _hash_float_bits(wind_time_scale, seed);
	const int capped_effectors = CLAMP(max_effectors, 0, 1);
	seed = _hash_u64(static_cast<uint64_t>(capped_effectors), seed);
	const bool sphere_effective_enabled = capped_effectors > 0 && sphere_effector_enabled;
	seed = _hash_bool(sphere_effective_enabled, seed);
	seed = _hash_vector3(sphere_effector_center, seed);
	seed = _hash_float_bits(MAX(0.0f, sphere_effector_radius), seed);
	seed = _hash_float_bits(sphere_effector_strength, seed);
	seed = _hash_float_bits(MAX(0.001f, sphere_effector_falloff), seed);
	if (wind_enabled && wind_strength > 0.0f) {
		const float wind_time_seconds = float(double(p_frame_id) * (1.0 / 60.0) * double(MAX(wind_time_scale, 0.0f)));
		seed = _hash_float_bits(wind_time_seconds, seed);
	}

	if (RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton()) {
		seed = _hash_u64(light_storage->get_directional_light_buffer().get_id(), seed);
		seed = _hash_u64(light_storage->get_omni_light_buffer().get_id(), seed);
		seed = _hash_u64(light_storage->get_spot_light_buffer().get_id(), seed);
		seed = _hash_u64(static_cast<uint64_t>(light_storage->get_omni_light_count()), seed);
		seed = _hash_u64(static_cast<uint64_t>(light_storage->get_spot_light_count()), seed);

		const PagedArray<RID> *lights = p_render_data->lights;
		if (!lights) {
			// If the visible light list is unavailable, inspector edits cannot be tracked reliably.
			return _hash_u64(p_frame_id, seed);
		}

		const uint64_t light_count = lights->size();
		seed = _hash_u64(light_count, seed);
		for (uint64_t i = 0; i < light_count; i++) {
			const RID light_instance = (*lights)[i];
			seed = _hash_u64(light_instance.get_id(), seed);
			if (!light_storage->owns_light_instance(light_instance)) {
				continue;
			}

			const RID base_light = light_storage->light_instance_get_base_light(light_instance);
			seed = _hash_u64(base_light.get_id(), seed);
			if (base_light.is_valid()) {
				seed = _hash_u64(light_storage->light_get_version(base_light), seed);
				// Important: LightStorage::light_set_param() does not bump version for all inspector params
				// (for example LIGHT_PARAM_ENERGY), so hash explicit light properties for correctness.
				const RS::LightType light_type = light_storage->light_get_type(base_light);
				seed = _hash_u64(static_cast<uint64_t>(light_type), seed);
				seed = _hash_color(light_storage->light_get_color(base_light), seed);
				seed = _hash_bool(light_storage->light_has_shadow(base_light), seed);
				seed = _hash_bool(light_storage->light_is_negative(base_light), seed);
				seed = _hash_bool(light_storage->light_get_reverse_cull_face_mode(base_light), seed);
				seed = _hash_u64(static_cast<uint64_t>(light_storage->light_get_cull_mask(base_light)), seed);
				seed = _hash_u64(static_cast<uint64_t>(light_storage->light_get_shadow_caster_mask(base_light)), seed);
				seed = _hash_u64(light_storage->light_get_projector(base_light).get_id(), seed);
				seed = _hash_bool(light_storage->light_is_distance_fade_enabled(base_light), seed);
				seed = _hash_float_bits(light_storage->light_get_distance_fade_begin(base_light), seed);
				seed = _hash_float_bits(light_storage->light_get_distance_fade_shadow(base_light), seed);
				seed = _hash_float_bits(light_storage->light_get_distance_fade_length(base_light), seed);
				seed = _hash_u64(static_cast<uint64_t>(light_storage->light_get_bake_mode(base_light)), seed);
				seed = _hash_u64(static_cast<uint64_t>(light_storage->light_get_max_sdfgi_cascade(base_light)), seed);
				for (int param = 0; param < RS::LIGHT_PARAM_MAX; param++) {
					seed = _hash_float_bits(light_storage->light_get_param(base_light, static_cast<RS::LightParam>(param)), seed);
				}

				if (light_type == RS::LIGHT_DIRECTIONAL) {
					seed = _hash_u64(static_cast<uint64_t>(light_storage->light_directional_get_shadow_mode(base_light)), seed);
					seed = _hash_u64(static_cast<uint64_t>(light_storage->light_directional_get_sky_mode(base_light)), seed);
					seed = _hash_bool(light_storage->light_directional_get_blend_splits(base_light), seed);
				} else if (light_type == RS::LIGHT_OMNI) {
					seed = _hash_u64(static_cast<uint64_t>(light_storage->light_omni_get_shadow_mode(base_light)), seed);
				}
			}
			seed = _hash_transform3d(light_storage->light_instance_get_base_transform(light_instance), seed);
			seed = _hash_u64(static_cast<uint64_t>(light_storage->light_instance_get_cull_mask(light_instance)), seed);
		}
	}

	return seed;
}

static uint64_t _compute_cull_config_signature(const GaussianSplatRenderer &p_renderer,
		const GaussianSplatRenderer::IFrameStateView &p_state_view) {
	uint64_t seed = HASH_MURMUR3_SEED;
	const GPUCuller *gpu_culler = p_state_view.get_gpu_culler();
	if (!gpu_culler) {
		return _hash_u64(0ull, seed);
	}

	const GPUCuller::CullingConfig &config = gpu_culler->get_config();
	const GPUCuller::CullingState &state = gpu_culler->get_state();
	seed = _hash_bool(config.lod_enabled, seed);
	seed = _hash_float_bits(config.lod_bias, seed);
	seed = _hash_bool(config.frustum_culling, seed);
	seed = _hash_bool(config.gpu_culling_enabled, seed);
	seed = _hash_bool(config.temporal_coherence, seed);
	seed = _hash_bool(config.cluster_culling_enabled, seed);
	seed = _hash_u64(config.cluster_target_size, seed);
	seed = _hash_float_bits(config.cluster_frustum_slack, seed);
	seed = _hash_bool(config.cluster_use_morton_order, seed);
	seed = _hash_bool(config.cluster_use_indirect_dispatch, seed);
	seed = _hash_float_bits(config.lod_min_screen_size, seed);
	seed = _hash_float_bits(config.lod_max_distance, seed);
	seed = _hash_float_bits(config.importance_cull_threshold, seed);
	seed = _hash_float_bits(config.cull_radius_multiplier, seed);
	seed = _hash_float_bits(config.cull_frustum_plane_slack, seed);
	seed = _hash_float_bits(config.cull_near_tolerance, seed);
	seed = _hash_float_bits(config.cull_far_tolerance, seed);
	seed = _hash_bool(config.opacity_aware_culling, seed);
	seed = _hash_float_bits(config.visibility_threshold, seed);
	seed = _hash_bool(config.distance_cull_enabled, seed);
	seed = _hash_float_bits(config.distance_cull_start, seed);
	seed = _hash_float_bits(config.distance_cull_max_rate, seed);
	seed = _hash_float_bits(state.tiny_splat_screen_radius_px, seed);
	seed = _hash_u64(static_cast<uint64_t>(MAX(0, p_renderer.get_performance_settings().max_splats)), seed);
	return seed;
}


static void _record_validation_event(GaussianSplatRenderer *p_renderer, const char *p_stage,
		const GaussianSplatRenderer::StageIO &p_io) {
	if (!p_renderer || !p_io.validation_failed) {
		return;
	}
	String message = p_io.validation_error;
	if (message.is_empty()) {
		message = "validation_failed";
	}
	_record_pipeline_event(p_renderer, p_stage, message, p_io.input_count, p_io.output_count, true);
}

struct StageIOValidationConfig {
	bool failed = false;
	bool count_invalid = false;
	bool input_missing = false;
	bool output_missing = false;
	bool record_event = true;
	String failed_error;
	String count_error;
	String input_error;
	String output_error;
};

static void _init_stage_io(GaussianSplatRenderer::StageIO &p_io, uint64_t p_frame_id,
		uint32_t p_input_count, uint32_t p_output_count, const RID &p_input_buffer,
		const RID &p_output_buffer, bool p_validated) {
	p_io.frame_id = p_frame_id;
	p_io.input_count = p_input_count;
	p_io.output_count = p_output_count;
	p_io.input_buffer = p_input_buffer;
	p_io.output_buffer = p_output_buffer;
	p_io.validated = p_validated;
}

static void _finalize_stage_io(GaussianSplatRenderer *p_renderer, const char *p_stage,
		GaussianSplatRenderer::StageIO &p_io, const StageIOValidationConfig &p_config) {
	p_io.validation_failed = p_io.validated &&
			(p_config.failed || p_config.count_invalid || p_config.input_missing || p_config.output_missing);
	if (!p_io.validation_failed) {
		p_io.validation_error = String();
	} else if (p_config.failed) {
		p_io.validation_error = p_config.failed_error;
	} else if (p_config.count_invalid) {
		p_io.validation_error = p_config.count_error;
	} else if (p_config.input_missing) {
		p_io.validation_error = p_config.input_error;
	} else if (p_config.output_missing) {
		p_io.validation_error = p_config.output_error;
	} else {
		p_io.validation_error = String();
	}
	if (p_config.record_event) {
		_record_validation_event(p_renderer, p_stage, p_io);
	}
}
} // namespace

// Frame planning static helpers (moved from GaussianSplatRenderer)

RenderPipelineStages::DataSourcePlan RenderPipelineStages::build_data_source_plan(
		const SceneState &p_scene_state,
		const StreamingState &p_streaming_state,
		const SortingState &p_sorting_state,
		const ResourceState &p_resource_state,
		const SubsystemState &p_subsystem_state) {
	DataSourcePlan plan;
	String error;
	Error status = GaussianSplatRenderer::get_active_data_source(p_scene_state, p_streaming_state, p_sorting_state,
			p_resource_state, p_subsystem_state, plan.source, error);
	if (status != OK && error.is_empty()) {
		error = "Active data source unavailable";
	}
	plan.error = error;
	plan.using_real_data = (status == OK);
	return plan;
}

void RenderPipelineStages::apply_data_source_plan(const DataSourcePlan &p_plan, PerformanceMetrics &p_metrics,
		const ResourceState &p_resource_state) {
	p_metrics.data_source = p_plan.source.source_name;
	p_metrics.using_real_data = p_plan.using_real_data;
	p_metrics.data_source_error = p_plan.using_real_data ? String() : p_plan.error;
	(void)p_resource_state;
}

RenderPipelineStages::RenderFramePlan RenderPipelineStages::build_frame_plan(
		const SceneState &p_scene_state,
		const StreamingState &p_streaming_state,
		const SortingState &p_sorting_state,
		const ResourceState &p_resource_state,
		const SubsystemState &p_subsystem_state,
		const PipelineFeatureSet *p_pipeline_features,
		bool p_has_render_data,
		const String &p_cull_skip_reason,
		const String &p_sort_skip_reason,
		RenderFallbackReason p_cull_skip_reason_code,
		RenderFallbackReason p_sort_skip_reason_code,
		bool p_set_skip_metrics,
		bool p_clear_cull_state_on_skip) {
	RenderFramePlan plan;
	plan.has_render_data = p_has_render_data;
	plan.set_skip_metrics = p_set_skip_metrics;
	plan.clear_cull_state_on_skip = p_clear_cull_state_on_skip;
	plan.compute_raster_policy = (p_pipeline_features && p_pipeline_features->enable_fast_raster)
			? GaussianSplatting::ComputeRasterPolicy::ForceOn
			: GaussianSplatting::ComputeRasterPolicy::Default;
	plan.cull_skip_reason = p_cull_skip_reason;
	plan.sort_skip_reason = p_sort_skip_reason;
	plan.cull_skip_reason_code = p_cull_skip_reason_code;
	plan.sort_skip_reason_code = p_sort_skip_reason_code;
	plan.data_source = build_data_source_plan(p_scene_state, p_streaming_state, p_sorting_state,
			p_resource_state, p_subsystem_state);
	return plan;
}

// Frame context preparation (moved from GaussianSplatRenderer::_prepare_render_frame_context)

void RenderPipelineStages::prepare_frame_context(RenderDataRD *p_render_data,
		const Transform3D &p_world_to_camera_transform,
		const Projection &p_projection, const Projection &p_render_projection,
		bool p_defer_render_buffers_commit,
		RenderFrameContext &r_context) {
	ERR_FAIL_NULL(renderer);

	RenderSceneBuffersRD *render_buffers_rd = nullptr;
	if (p_render_data && p_render_data->render_buffers.is_valid()) {
		render_buffers_rd = Object::cast_to<RenderSceneBuffersRD>(p_render_data->render_buffers.ptr());
	}

	RID render_target;
	if (render_buffers_rd) {
		RD::DataFormat override_format = renderer->get_view_state().active_viewport_color_format;
		if (render_buffers_rd->has_internal_texture()) {
			render_target = render_buffers_rd->get_internal_texture();
			if (override_format == RD::DATA_FORMAT_MAX && render_target.is_valid()) {
				RenderingDevice *viewport_device = renderer->get_main_rendering_device();
				RenderingDevice *target_device = renderer->get_resource_owner(render_target, viewport_device);
				RD::TextureFormat target_format = renderer->get_texture_format(target_device, render_target);
				if (target_format.format != RD::DATA_FORMAT_MAX) {
					override_format = target_format.format;
				}
			}
		}
		renderer->set_manual_viewport_format(override_format, "render_sorted_set_override");
		if (override_format != RD::DATA_FORMAT_MAX) {
			renderer->set_active_viewport_format(override_format, "render_sorted_override_active");
		}
	}

	Size2i viewport_size = renderer->get_view_state().manual_viewport_override;
	if (render_buffers_rd) {
		viewport_size = render_buffers_rd->get_internal_size();
	}

	if (viewport_size.x <= 0 || viewport_size.y <= 0) {
		viewport_size = Size2i(1280, 720);
	}

	r_context.render_data = p_render_data;
	r_context.render_buffers = render_buffers_rd;
	r_context.render_target = render_target;
	r_context.world_to_camera_transform = p_world_to_camera_transform;
	r_context.projection = p_projection;
	r_context.cull_projection = renderer->build_cull_projection(p_render_data, p_projection);
	r_context.render_projection = p_render_projection;
	r_context.viewport_size = viewport_size;
	r_context.viewport_format = renderer->get_view_state().manual_viewport_format_override;
	r_context.defer_commit = p_defer_render_buffers_commit && render_buffers_rd != nullptr;
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	r_context.frame_id = state_view.get_frame_state_view().frame_counter;
	r_context.painterly_enabled = renderer->get_painterly_config().enabled;
	r_context.state_provider = nullptr;
	renderer->validate_cull_projection_contract(p_render_data, p_projection, r_context.cull_projection,
			"render_pipeline_stages::prepare_render_frame_context");
	renderer->update_pipeline_features(state_view.get_rendering_device());
	r_context.deps.output_compositor = state_view.get_output_compositor();
	r_context.deps.gpu_culler = state_view.get_gpu_culler();
	r_context.deps.painterly_renderer = state_view.get_painterly_renderer();
	r_context.deps.sorting_pipeline = state_view.get_sorting_pipeline();
	r_context.deps.rendering_device = state_view.get_rendering_device();
	r_context.deps.scene_state = &renderer->get_scene_state();
	r_context.deps.streaming_state = &renderer->get_streaming_state();
	r_context.deps.sorting_state = &frame_provider.get_sorting_state();
	r_context.deps.render_config = &frame_provider.get_render_config();
	r_context.deps.jacobian_debug = &frame_provider.get_jacobian_debug();
	r_context.deps.resource_state = &frame_provider.get_resource_state();
	r_context.deps.frame_state = &frame_provider.get_frame_state();
	r_context.deps.performance_state = &frame_provider.get_performance_state();
	r_context.deps.subsystem_state = &frame_provider.get_subsystem_state();
	r_context.deps.pipeline_features = state_view.get_pipeline_features();
	DEV_ASSERT(r_context.deps.validate());
}

// Frame entry execution (moved from GaussianSplatRenderer::_run_pipeline_entry)

void RenderPipelineStages::execute_frame_entry(const RenderFrameContext &p_frame_context,
		bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
		RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
		bool p_set_skip_metrics, bool p_clear_cull_state_on_skip) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_COND(!p_frame_context.deps.validate());

	// Copy frame context first, then build frame_plan and update deps.
	// The provider must be constructed AFTER this so it sees the updated deps.
	RenderFrameContext frame_context = p_frame_context;
	GaussianSplatRenderer::FrameStateProvider preplan_provider(renderer, &frame_context.deps);
	const GaussianSplatRenderer::IFrameStateView &preplan_view = preplan_provider;

	// Build initial state references from deps (before provider construction).
	const SceneState &scene_state = frame_context.deps.scene_state
			? *frame_context.deps.scene_state
			: preplan_view.get_scene_state();
	const StreamingState &streaming_state = frame_context.deps.streaming_state
			? *frame_context.deps.streaming_state
			: preplan_view.get_streaming_state();
	SortingState &sorting_state = frame_context.deps.sorting_state
			? *frame_context.deps.sorting_state
			: preplan_provider.get_sorting_state();
	ResourceState &resource_state = frame_context.deps.resource_state
			? *frame_context.deps.resource_state
			: preplan_provider.get_resource_state();
	SubsystemState &subsystem_state_ref = frame_context.deps.subsystem_state
			? *frame_context.deps.subsystem_state
			: preplan_provider.get_subsystem_state();
	const PipelineFeatureSet *pipeline_features = frame_context.deps.pipeline_features
			? frame_context.deps.pipeline_features
			: preplan_view.get_pipeline_features();

	RenderFramePlan frame_plan = build_frame_plan(scene_state, streaming_state, sorting_state, resource_state,
			subsystem_state_ref, pipeline_features, p_has_render_data, p_cull_skip_reason,
			p_sort_skip_reason, p_cull_skip_reason_code, p_sort_skip_reason_code, p_set_skip_metrics,
			p_clear_cull_state_on_skip);

	// Update deps with the frame_plan BEFORE constructing provider.
	frame_context.deps.frame_plan = &frame_plan;

	// Now construct the provider from the updated frame_context.deps.
	const GaussianSplatRenderer::IFrameStateProvider *context_provider = frame_context.state_provider;
	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer, &frame_context.deps);
	const GaussianSplatRenderer::IFrameStateProvider &state_provider =
			context_provider ? *context_provider : fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	frame_context.state_provider = &state_provider;
	DEV_ASSERT(frame_context.deps.frame_plan);
	frame_context.snapshot.valid = true;
	frame_context.snapshot.visible_splats = 0;
	frame_context.snapshot.sorted_splats = 0;
	frame_context.snapshot.cull_visible_domain = GaussianSplatRenderer::IndexDomain::UNKNOWN;
	frame_context.snapshot.sorted_index_domain = GaussianSplatRenderer::IndexDomain::UNKNOWN;
	GaussianSplatRenderer::FrameState &frame_state_ref = state_provider.get_frame_state();
	GaussianSplatRenderer::SortingState &sorting_state_ref = sorting_state;
	auto update_counts_from_snapshot = [&]() {
		frame_state_ref.visible_splat_count.store(frame_context.snapshot.visible_splats, std::memory_order_release);
		sorting_state_ref.sorted_splat_count = frame_context.snapshot.sorted_splats;
	};

	if (!frame_plan.has_render_data) {
		if (debug_state_orchestrator) {
			renderer->get_debug_state().route_uid = RenderRouteUID::COMMON_SKIP_NO_DATA;
		}
		GPUCuller *gpu_culler = state_view.get_gpu_culler();
		if (frame_plan.clear_cull_state_on_skip && gpu_culler) {
			auto &cull_state = gpu_culler->get_state();
			cull_state.culled_indices.clear();
			cull_state.culled_distances_sq.clear();
			cull_state.culled_importance_weights.clear();
		}
		update_counts_from_snapshot();
		if (frame_plan.set_skip_metrics && frame_context.metrics) {
			frame_context.metrics->cull = GaussianSplatRenderer::CullStageOutput();
			frame_context.metrics->sort = GaussianSplatRenderer::SortStageOutput();
			frame_context.metrics->cull_result = _make_stage_result(
					StageResult::StageStatus::SKIPPED,
					frame_plan.cull_skip_reason,
					false,
					frame_plan.cull_skip_reason_code);
			frame_context.metrics->sort_result = _make_stage_result(
					StageResult::StageStatus::SKIPPED,
					frame_plan.sort_skip_reason,
					false,
					frame_plan.sort_skip_reason_code);
		}
		render_sorted_splats_with_context(frame_context);
		return;
	}

	Size2i cull_viewport_size = frame_context.viewport_size;
	if (frame_context.render_buffers) {
		Size2i target_size = frame_context.render_buffers->get_target_size();
		if (target_size.x > 0 && target_size.y > 0) {
			cull_viewport_size = target_size;
		}
	}
	GaussianSplatRenderer::CullStageInput cull_input{
		frame_context.frame_id,
		frame_context.world_to_camera_transform,
		frame_context.cull_projection,
		cull_viewport_size,
		frame_context.metrics,
		&state_provider
	};
	GaussianSplatRenderer::CullStageOutput cull_output;
	execute_cull_stage(cull_input, cull_output);
	GaussianSplatRenderer::SortStageInput sort_input{
		frame_context.frame_id,
		frame_context.world_to_camera_transform,
		cull_output.visible_count,
		frame_context.metrics,
		&state_provider,
		cull_output.visible_domain
	};
	GaussianSplatRenderer::SortStageOutput sort_output;
	execute_sort_stage(sort_input, sort_output);
	frame_context.snapshot.cull_visible_domain = cull_output.visible_domain;
	frame_context.snapshot.sorted_index_domain = sort_output.output_domain;
	if (sort_output.output_domain == GaussianSplatRenderer::IndexDomain::SPLAT_REF) {
		frame_context.snapshot.visible_splats = sort_output.sorted_count;
	} else if (sort_output.output_domain == GaussianSplatRenderer::IndexDomain::GAUSSIAN_GLOBAL) {
		frame_context.snapshot.visible_splats =
				(sort_output.sorted_count > 0 ? sort_output.sorted_count : cull_output.visible_count);
	} else if (cull_output.visible_domain == GaussianSplatRenderer::IndexDomain::CHUNK_REF) {
		frame_context.snapshot.visible_splats = sort_output.sorted_count;
	} else {
		frame_context.snapshot.visible_splats = cull_output.visible_count;
	}
	frame_context.snapshot.sorted_splats = sort_output.sorted_count;
	update_counts_from_snapshot();
	frame_plan.data_source = build_data_source_plan(scene_state, streaming_state, sorting_state,
			resource_state, subsystem_state_ref);
	render_sorted_splats_with_context(frame_context);
}

struct RenderPipelineStages::CullStage {
	using StageResult = GaussianSplatRenderer::StageResult;

	RenderPipelineStages *pipeline = nullptr;
	GaussianSplatRenderer *renderer = nullptr;

	CullStage(RenderPipelineStages *p_pipeline, GaussianSplatRenderer *p_renderer) :
			pipeline(p_pipeline),
			renderer(p_renderer) {
	}

	StageResult execute(const GaussianSplatRenderer::CullStageInput &p_input,
			GaussianSplatRenderer::CullStageOutput &r_output) {
		StageResult result;
		r_output = GaussianSplatRenderer::CullStageOutput();
		GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
		const GaussianSplatRenderer::IFrameStateProvider &state_provider =
				p_input.state_provider ? *p_input.state_provider : fallback_provider;
		const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
		GPUCuller *gpu_culler = state_view.get_gpu_culler();
		if (!gpu_culler) {
			result = _make_stage_result(StageResult::StageStatus::FAILED, "Culling failed: GPU culler unavailable", true,
					GaussianSplatRenderer::RenderFallbackReason::GPU_CULLER_UNAVAILABLE);
			_record_pipeline_event(renderer, "cull",
					"fail: gpu_culler unavailable",
					0, 0, true,
					GaussianSplatRenderer::RenderFallbackReason::GPU_CULLER_UNAVAILABLE,
					RenderRouteUID::COMMON_FAIL_NO_DEVICE);
			if (p_input.metrics) {
				p_input.metrics->cull = r_output;
				p_input.metrics->cull_result = result;
				auto &io = p_input.metrics->cull_io;
				_init_stage_io(io, p_input.frame_id, r_output.candidate_count, r_output.visible_count,
						RID(), RID(), r_output.candidate_count > 0);
				StageIOValidationConfig validation;
				validation.failed = true;
				validation.failed_error = "GPU culler unavailable";
				_finalize_stage_io(renderer, "cull", io, validation);
			}
			return result;
		}

		const auto &buffers = renderer->get_instance_pipeline_buffers();
		const bool instance_cull_path_requested = renderer->has_instance_pipeline_buffers() &&
				GaussianSplatting::InstancePipelineContract::has_cull_buffers(buffers);
		if (instance_cull_path_requested) {
			_record_pipeline_event(renderer, "cull",
					vformat("instance_path inst_count=%d chunk_count=%d max_vis_chunks=%d",
							buffers.instance_count, buffers.dispatch_chunk_count, buffers.max_visible_chunks),
					buffers.instance_count, buffers.max_visible_chunks, false,
					GaussianSplatRenderer::RenderFallbackReason::NONE,
					RenderRouteUID::INSTANCE_CULL_GPU);
			GPUCuller::InstancePipelineInputs instance_inputs;
			instance_inputs.instance_buffer = buffers.instance_buffer;
			instance_inputs.asset_meta_buffer = buffers.asset_meta_buffer;
			instance_inputs.asset_chunk_index_buffer = buffers.asset_chunk_index_buffer;
			instance_inputs.chunk_meta_buffer = buffers.chunk_meta_buffer;
			instance_inputs.visible_chunk_buffer = buffers.visible_chunk_buffer;
			instance_inputs.counter_buffer = buffers.counter_buffer;
			instance_inputs.instance_count = buffers.instance_count;
			instance_inputs.dispatch_chunk_count = buffers.dispatch_chunk_count;
			instance_inputs.max_visible_chunks = buffers.max_visible_chunks;
			instance_inputs.device = state_view.get_rendering_device();
			gpu_culler->set_instance_pipeline_inputs(instance_inputs);
		} else {
			result = _make_stage_result(StageResult::StageStatus::SKIPPED, "Culling skipped: instance buffers missing", false,
					GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE);
			_record_pipeline_event(renderer, "cull",
					"skip: instance buffers missing",
					0, 0, false,
					GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
					RenderRouteUID::COMMON_SKIP_NO_DATA);
			gpu_culler->clear_instance_pipeline_inputs();
			if (p_input.metrics) {
				p_input.metrics->cull = r_output;
				p_input.metrics->cull_result = result;
				auto &io = p_input.metrics->cull_io;
				_init_stage_io(io, p_input.frame_id, 0, 0, RID(), RID(), false);
				StageIOValidationConfig validation;
				validation.failed = true;
				validation.failed_error = "Instance cull buffers missing";
				_finalize_stage_io(renderer, "cull", io, validation);
			}
			return result;
		}

		gpu_culler->get_config().last_cull_viewport_size = p_input.viewport_size;
		r_output = pipeline->cull_for_view(p_input.world_to_camera_transform, p_input.projection, p_input.viewport_size);
		if (r_output.visible_domain == GaussianSplatRenderer::IndexDomain::UNKNOWN) {
			r_output.visible_domain = instance_cull_path_requested
					? GaussianSplatRenderer::IndexDomain::CHUNK_REF
					: GaussianSplatRenderer::IndexDomain::GAUSSIAN_GLOBAL;
		}
		if (r_output.visible_count > 0 && r_output.visible_domain == GaussianSplatRenderer::IndexDomain::UNKNOWN) {
			result = _make_stage_result(StageResult::StageStatus::FAILED,
					"Culling failed: output domain unresolved", true,
					GaussianSplatRenderer::RenderFallbackReason::NONE);
		}
		_record_pipeline_event(renderer, "cull",
				vformat("output candidates=%d visible=%d domain=%s",
						r_output.candidate_count, r_output.visible_count,
						GaussianRenderState::index_domain_to_string(r_output.visible_domain)),
				r_output.candidate_count, r_output.visible_count, false,
				GaussianSplatRenderer::RenderFallbackReason::NONE,
				RenderRouteUID::INSTANCE_CULL_GPU);

		r_output.has_visible = r_output.visible_count > 0;
		if (!r_output.has_visible) {
			r_output.visible_count = 0;
		}

		if (p_input.metrics) {
			p_input.metrics->cull = r_output;
			p_input.metrics->cull_result = result;
			const auto &cull_state = gpu_culler->get_state();
			auto &io = p_input.metrics->cull_io;
			_init_stage_io(io, p_input.frame_id, r_output.candidate_count, r_output.visible_count,
					RID(), cull_state.gpu_visible_indices_buffer, r_output.candidate_count > 0);
			const bool count_invalid = io.output_count > io.input_count;
			const bool buffer_missing = cull_state.gpu_visible_indices_count > 0 && !io.output_buffer.is_valid();
			StageIOValidationConfig validation;
			validation.count_invalid = count_invalid;
			validation.output_missing = buffer_missing;
			validation.count_error = "Culling output exceeds candidate count";
			validation.output_error = "Culling output buffer invalid";
			_finalize_stage_io(renderer, "cull", io, validation);
		}
		return result;
	}
};

struct RenderPipelineStages::SortStage {
	using StageResult = GaussianSplatRenderer::StageResult;

	RenderPipelineStages *pipeline = nullptr;
	GaussianSplatRenderer *renderer = nullptr;

	SortStage(RenderPipelineStages *p_pipeline, GaussianSplatRenderer *p_renderer) :
			pipeline(p_pipeline),
			renderer(p_renderer) {
	}

		StageResult execute(const GaussianSplatRenderer::SortStageInput &p_input,
				GaussianSplatRenderer::SortStageOutput &r_output) {
			StageResult result;
			r_output = GaussianSplatRenderer::SortStageOutput();
			r_output.input_count = p_input.input_count;
			r_output.input_domain = p_input.input_domain;

			GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
			const GaussianSplatRenderer::IFrameStateProvider &state_provider =
					p_input.state_provider ? *p_input.state_provider : fallback_provider;
			const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
			auto finalize_sort_metrics = [&](const StageResult &p_result) {
				if (!p_input.metrics) {
					return;
				}
				p_input.metrics->sort = r_output;
				p_input.metrics->sort_result = p_result;
				auto &io = p_input.metrics->sort_io;
				_init_stage_io(io, p_input.frame_id, r_output.input_count, r_output.sorted_count,
						RID(), _get_sort_indices_buffer(state_view), r_output.input_count > 0);
				const bool count_invalid = p_input.input_domain == GaussianSplatRenderer::IndexDomain::GAUSSIAN_GLOBAL &&
						io.output_count > io.input_count;
				StageIOValidationConfig validation;
				validation.count_invalid = count_invalid;
				validation.count_error = "Sort output exceeds input count";
				_finalize_stage_io(renderer, "sort", io, validation);
			};

			const String sort_gpu_uid = RenderRouteUID::INSTANCE_SORT_GPU;
			GPUCuller *gpu_culler = state_view.get_gpu_culler();
			if (!gpu_culler) {
				result = _make_stage_result(StageResult::StageStatus::FAILED, "Sort failed: GPU culler unavailable", true,
						GaussianSplatRenderer::RenderFallbackReason::GPU_CULLER_UNAVAILABLE);
				_record_pipeline_event(renderer, "sort", "fail: gpu_culler unavailable",
						r_output.input_count, 0, true,
						GaussianSplatRenderer::RenderFallbackReason::GPU_CULLER_UNAVAILABLE,
						RenderRouteUID::COMMON_FAIL_NO_DEVICE);
				finalize_sort_metrics(result);
				return result;
			}

			GPUSortingPipeline *sorting_pipeline = state_view.get_sorting_pipeline();
			const auto &buffers = renderer->get_instance_pipeline_buffers();
			const bool instance_buffers_ready = renderer->has_instance_pipeline_buffers() &&
					GaussianSplatting::InstancePipelineContract::has_sort_buffers(buffers);
			const bool input_domain_chunk = p_input.input_domain == GaussianSplatRenderer::IndexDomain::CHUNK_REF;
			const bool input_domain_global = p_input.input_domain == GaussianSplatRenderer::IndexDomain::GAUSSIAN_GLOBAL;
			if (!input_domain_chunk && !input_domain_global) {
				if (r_output.input_count == 0) {
					result = _make_stage_result(StageResult::StageStatus::SKIPPED, "Sort skipped: unresolved input domain", false,
							GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE);
					_record_pipeline_event(renderer, "sort", "skip: unresolved input domain",
							r_output.input_count, 0, false,
							GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
							RenderRouteUID::COMMON_SKIP_NO_DATA);
					finalize_sort_metrics(result);
					return result;
				}
				result = _make_stage_result(StageResult::StageStatus::FAILED,
						vformat("Sort failed: unsupported input index domain '%s'",
								GaussianRenderState::index_domain_to_string(p_input.input_domain)),
						true,
						GaussianSplatRenderer::RenderFallbackReason::NONE);
				_record_pipeline_event(renderer, "sort", "fail: unsupported input domain",
						r_output.input_count, 0, true,
						GaussianSplatRenderer::RenderFallbackReason::NONE,
						RenderRouteUID::COMMON_FAIL_SORT_FAILED);
				finalize_sort_metrics(result);
				return result;
			}
			if (input_domain_chunk && !instance_buffers_ready) {
				if (sorting_pipeline) {
					sorting_pipeline->clear_instance_pipeline_inputs();
				}
				if (r_output.input_count == 0) {
					result = _make_stage_result(StageResult::StageStatus::SKIPPED, "Sort skipped: instance buffers missing", false,
							GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE);
					_record_pipeline_event(renderer, "sort", "skip: instance buffers missing",
							r_output.input_count, 0, false,
							GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
							RenderRouteUID::COMMON_SKIP_NO_DATA);
					finalize_sort_metrics(result);
					return result;
				}
				result = _make_stage_result(StageResult::StageStatus::FAILED,
						"Sort failed: chunk-domain input requires instance sort buffers", true,
						GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE);
				_record_pipeline_event(renderer, "sort", "fail: chunk-domain input without instance buffers",
						r_output.input_count, 0, true,
						GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
						RenderRouteUID::COMMON_FAIL_SORT_FAILED);
				finalize_sort_metrics(result);
				return result;
			}
			if (input_domain_global && instance_buffers_ready) {
				if (sorting_pipeline) {
					sorting_pipeline->clear_instance_pipeline_inputs();
				}
				if (r_output.input_count == 0) {
					result = _make_stage_result(StageResult::StageStatus::SKIPPED, "Sort skipped: incompatible input domain", false,
							GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE);
					_record_pipeline_event(renderer, "sort", "skip: global-domain input with instance buffers",
							r_output.input_count, 0, false,
							GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
							RenderRouteUID::COMMON_SKIP_NO_DATA);
					finalize_sort_metrics(result);
					return result;
				}
				result = _make_stage_result(StageResult::StageStatus::FAILED,
						"Sort failed: global-domain input is incompatible with instance sort buffers", true,
						GaussianSplatRenderer::RenderFallbackReason::NONE);
				_record_pipeline_event(renderer, "sort", "fail: global-domain input with instance buffers",
						r_output.input_count, 0, true,
						GaussianSplatRenderer::RenderFallbackReason::NONE,
						RenderRouteUID::COMMON_FAIL_SORT_FAILED);
				finalize_sort_metrics(result);
				return result;
			}

			if (sorting_pipeline) {
				if (input_domain_chunk) {
					GPUSortingPipeline::InstancePipelineInputs instance_inputs;
					instance_inputs.atlas_gaussian_buffer = buffers.atlas_gaussian_buffer;
					instance_inputs.quantization_buffer = buffers.quantization_required ? buffers.quantization_buffer : RID();
					instance_inputs.instance_buffer = buffers.instance_buffer;
					instance_inputs.chunk_meta_buffer = buffers.chunk_meta_buffer;
					instance_inputs.visible_chunk_buffer = buffers.visible_chunk_buffer;
					instance_inputs.splat_ref_buffer = buffers.splat_ref_buffer;
					instance_inputs.sort_key_buffer = buffers.sort_key_buffer;
					instance_inputs.sort_value_buffer = buffers.sort_value_buffer;
					instance_inputs.counter_buffer = buffers.counter_buffer;
					instance_inputs.chunk_dispatch_buffer = buffers.chunk_dispatch_buffer;
					instance_inputs.indirect_count_buffer = buffers.indirect_count_buffer;
					instance_inputs.instance_count_buffer = buffers.instance_count_buffer;
					// FIX: Use buffer capacity instead of stale async readback value.
					// The GPU-side counter in instance_chunk_dispatch.glsl drives actual
					// dispatch count; this value only guards visible_chunk_buffer reads
					// in depth_compute.glsl. Using the stale readback caused missing
					// chunks when visibility increased during fast camera rotation.
					uint32_t visible_chunk_count = buffers.max_visible_chunks;
					instance_inputs.visible_chunk_count = visible_chunk_count;
					instance_inputs.max_visible_chunks = buffers.max_visible_chunks;
					instance_inputs.max_visible_splats = buffers.max_visible_splats;
					instance_inputs.max_chunk_splats = buffers.max_chunk_splats;
					instance_inputs.device = state_view.get_rendering_device();
					sorting_pipeline->set_instance_pipeline_inputs(instance_inputs);
				} else {
					sorting_pipeline->clear_instance_pipeline_inputs();
				}
			}

			if (r_output.input_count == 0 && input_domain_global) {
				result = _make_stage_result(StageResult::StageStatus::SKIPPED, "Sort skipped: no visible splats", false,
						GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS);
				_record_pipeline_event(renderer, "sort", "skip: no visible splats",
						0, 0, false,
						GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS,
						RenderRouteUID::COMMON_SKIP_NO_VISIBLE);
				finalize_sort_metrics(result);
				return result;
			}

			GaussianSplatRenderer::SortStageSummary summary =
					pipeline->sort_for_view(p_input.world_to_camera_transform, p_input.input_domain);
			r_output.did_sort = true;
			r_output.sorted_count = summary.sorted_count;
			r_output.sort_time_ms = summary.sort_time_ms;
			r_output.input_domain = summary.input_domain;
			r_output.output_domain = summary.output_domain;
			const GaussianSplatRenderer::IndexDomain expected_output_domain = input_domain_chunk
					? GaussianSplatRenderer::IndexDomain::SPLAT_REF
					: GaussianSplatRenderer::IndexDomain::GAUSSIAN_GLOBAL;
			const bool input_domain_mismatch = r_output.input_domain != p_input.input_domain;
			const bool output_domain_mismatch = r_output.output_domain != expected_output_domain;
			RID sort_indices_buffer = _get_sort_indices_buffer(state_view);
			const bool output_missing = r_output.sorted_count > 0 && !sort_indices_buffer.is_valid();
			const bool count_missing = r_output.input_count > 0 &&
					(input_domain_global && r_output.sorted_count == 0);
			const bool sort_failed = input_domain_mismatch || output_domain_mismatch || count_missing || output_missing;
			if (sort_failed) {
				String reason;
				if (input_domain_mismatch) {
					reason = vformat("Sort failed: input domain mismatch (expected=%s actual=%s)",
							GaussianRenderState::index_domain_to_string(p_input.input_domain),
							GaussianRenderState::index_domain_to_string(r_output.input_domain));
				} else if (output_domain_mismatch) {
					reason = vformat("Sort failed: output domain mismatch (expected=%s actual=%s)",
							GaussianRenderState::index_domain_to_string(expected_output_domain),
							GaussianRenderState::index_domain_to_string(r_output.output_domain));
				} else if (output_missing) {
					reason = "Sort failed: output buffer invalid";
				} else {
					reason = "Sort failed: no sorted output for non-zero input";
				}
				result = _make_stage_result(StageResult::StageStatus::FAILED, reason, true,
						GaussianSplatRenderer::RenderFallbackReason::NONE);
				r_output.sorted_count = 0;
			}

			if (p_input.metrics) {
				p_input.metrics->sort = r_output;
				p_input.metrics->sort_result = result;
				const auto &cull_state = gpu_culler->get_state();
				auto &io = p_input.metrics->sort_io;
				_init_stage_io(io, p_input.frame_id, r_output.input_count, r_output.sorted_count,
						cull_state.gpu_visible_indices_buffer, sort_indices_buffer,
						r_output.input_count > 0);
				const bool count_invalid = input_domain_global && io.output_count > io.input_count;
				const bool buffer_missing = io.output_count > 0 && !io.output_buffer.is_valid();
				StageIOValidationConfig validation;
				validation.count_invalid = count_invalid;
				validation.output_missing = buffer_missing;
				validation.count_error = "Sort output exceeds input count";
				validation.output_error = "Sort output buffer invalid";
				_finalize_stage_io(renderer, "sort", io, validation);
			}

			String sort_message = result.reason;
			if (sort_message.is_empty()) {
				sort_message = "status=" + _stage_status_label(result.status);
			}
			String sort_route_uid = renderer->get_debug_state().sort_route_uid;
			if (RenderRouteUID::is_sort_route_uid_missing(sort_route_uid)) {
				sort_route_uid = result.status == StageResult::StageStatus::FAILED
						? String(RenderRouteUID::COMMON_FAIL_SORT_FAILED)
						: sort_gpu_uid;
			}
			_record_pipeline_event(renderer, "sort", sort_message,
					r_output.input_count, r_output.sorted_count,
					result.is_error || result.status == StageResult::StageStatus::FAILED,
					result.fallback_reason,
					sort_route_uid);
			return result;
		}
};

struct RenderPipelineStages::RasterStage {
	using StageResult = GaussianSplatRenderer::StageResult;

	RenderPipelineStages *pipeline = nullptr;
	GaussianSplatRenderer *renderer = nullptr;

	RasterStage(RenderPipelineStages *p_pipeline, GaussianSplatRenderer *p_renderer) :
			pipeline(p_pipeline),
			renderer(p_renderer) {
	}

		Error render_tile_fallback(const Size2i &p_viewport_size, RD::DataFormat p_target_format,
				const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
				RenderDataRD *p_render_data, uint32_t p_sorted_splat_count, GaussianSplatRenderer::IndexDomain p_sorted_index_domain,
				const GaussianSplatRenderer::IFrameStateProvider &p_state_provider,
				RID &r_color_output, RID &r_depth_output);
	StageResult resolve_painterly_output(const GaussianSplatRenderer::RasterStageInput &p_input,
			GaussianSplatRenderer::RasterStageOutput &r_output);
	bool try_reuse_cached_render(const GaussianSplatRenderer::RasterStageInput &p_input,
			GaussianSplatRenderer::RasterStageOutput &r_output);
	StageResult render_baseline_stage(const GaussianSplatRenderer::RasterStageInput &p_input,
			GaussianSplatRenderer::RasterStageOutput &r_output, const StageResult &p_fallback_context,
			uint64_t p_frame_start_usec);
	StageResult render_painterly_or_baseline_stage(const GaussianSplatRenderer::RasterStageInput &p_input,
			GaussianSplatRenderer::RasterStageOutput &r_output, const StageResult &p_painterly_status,
			uint64_t p_frame_start_usec);
};

struct RenderPipelineStages::CompositeStage {
	using StageResult = GaussianSplatRenderer::StageResult;

	GaussianSplatRenderer *renderer = nullptr;

	explicit CompositeStage(GaussianSplatRenderer *p_renderer) :
			renderer(p_renderer) {
	}

	StageResult execute(const GaussianSplatRenderer::CompositeStageInput &p_input,
			bool &r_did_composite);
};

struct RenderPipelineStages::RasterCompositeStage {
	using StageResult = GaussianSplatRenderer::StageResult;

	GaussianSplatRenderer *renderer = nullptr;
	RasterStage *raster_stage = nullptr;
	CompositeStage *composite_stage = nullptr;

	RasterCompositeStage(GaussianSplatRenderer *p_renderer, RasterStage *p_raster_stage,
			CompositeStage *p_composite_stage) :
			renderer(p_renderer),
			raster_stage(p_raster_stage),
			composite_stage(p_composite_stage) {
	}

	bool execute(const GaussianSplatRenderer::RenderFrameContext &p_context, uint64_t p_frame_start_usec,
			GaussianSplatRenderer::RasterStageOutput &r_raster_output, StageResult &r_raster_result,
			StageResult &r_composite_result, float &r_composite_time_ms, bool &r_composite_executed) {
		const GaussianSplatRenderer::IFrameStateProvider *context_provider = p_context.state_provider;
		GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer, &p_context.deps);
		const GaussianSplatRenderer::IFrameStateProvider *active_provider =
				context_provider ? context_provider : &fallback_provider;
		const GaussianSplatRenderer::IFrameStateView &state_view = *active_provider;
		OutputCompositor *output_compositor = state_view.get_output_compositor();
		ERR_FAIL_NULL_V(raster_stage, false);
		ERR_FAIL_NULL_V(composite_stage, false);
		GaussianSplatRenderer::RasterStageInput raster_input;
		raster_input.frame_id = p_context.frame_id;
		raster_input.render_data = p_context.render_data;
		raster_input.world_to_camera_transform = p_context.world_to_camera_transform;
		raster_input.projection = p_context.projection;
		raster_input.render_projection = p_context.render_projection;
		raster_input.viewport_size = p_context.viewport_size;
		raster_input.viewport_format = p_context.viewport_format;
		if (p_context.metrics) {
			raster_input.sorted_splat_count = p_context.metrics->sort.sorted_count;
			raster_input.sort_time_ms = p_context.metrics->sort.sort_time_ms;
			raster_input.sorted_index_domain = p_context.metrics->sort.output_domain;
		} else if (p_context.snapshot.valid) {
			raster_input.sorted_splat_count = p_context.snapshot.sorted_splats;
			raster_input.sort_time_ms = 0.0f;
			raster_input.sorted_index_domain = p_context.snapshot.sorted_index_domain;
		} else {
			raster_input.sorted_splat_count = 0;
			raster_input.sort_time_ms = 0.0f;
			raster_input.sorted_index_domain = GaussianSplatRenderer::IndexDomain::UNKNOWN;
		}
		raster_input.content_generation = renderer->get_instance_pipeline_content_generation();
		raster_input.cull_config_signature = _compute_cull_config_signature(*renderer, state_view);
		raster_input.color_grading_signature = _compute_color_grading_signature(state_view.get_render_config_view());
		raster_input.lighting_signature = _compute_lighting_signature(p_context.render_data, p_context.frame_id);
		raster_input.metrics = p_context.metrics;
		raster_input.state_provider = active_provider;
		raster_input.painterly_requested = p_context.painterly_enabled;

		r_raster_output.internal_size = raster_input.viewport_size;
		r_raster_output.reused_cached_render = false;
		r_raster_output.render_time_ms = 0.0f;
		r_raster_output.raster_path = "unknown";
		r_raster_output.sorted_splat_count = raster_input.sorted_splat_count;
		r_raster_output.content_generation = raster_input.content_generation;
		{
			Ref<TileRenderer> tile_renderer = renderer->get_tile_renderer();
			r_raster_output.shader_defines_hash = tile_renderer.is_valid() ? tile_renderer->get_shader_defines_hash() : 0;
		}

		StageResult painterly_status = raster_stage->resolve_painterly_output(raster_input, r_raster_output);
		const bool reused_cached_render = raster_stage->try_reuse_cached_render(raster_input, r_raster_output);

		if (_pipeline_trace_enabled(renderer)) {
			String raster_probe_uid;
			if (reused_cached_render) {
				raster_probe_uid = RenderRouteUID::INSTANCE_RASTER_CACHED;
			} else if (r_raster_output.painterly_active) {
				raster_probe_uid = RenderRouteUID::INSTANCE_RASTER_PAINTERLY;
			}
			_record_pipeline_event(renderer, "raster",
					vformat("reused_cache=%s painterly=%s color_valid=%s",
							reused_cached_render ? "YES" : "no",
							r_raster_output.painterly_active ? "YES" : "no",
							r_raster_output.color.is_valid() ? "YES" : "no"),
					raster_input.sorted_splat_count, r_raster_output.color.is_valid() ? 1u : 0u, false,
					GaussianSplatRenderer::RenderFallbackReason::NONE,
					raster_probe_uid);
		}

		if (reused_cached_render) {
			if (painterly_status.status == StageResult::StageStatus::FALLBACK) {
				r_raster_result = painterly_status;
			} else {
				r_raster_result = _make_stage_result(StageResult::StageStatus::SKIPPED, "Raster skipped: reused cached render", false,
						GaussianSplatRenderer::RenderFallbackReason::RASTER_REUSED_CACHED_RENDER);
			}
		} else {
			r_raster_result = raster_stage->render_painterly_or_baseline_stage(raster_input, r_raster_output, painterly_status,
					p_frame_start_usec);
		}

		if (p_context.metrics && (r_raster_output.reused_cached_render || r_raster_output.painterly_active)) {
			auto &io = p_context.metrics->raster_io;
			const uint32_t input_count = raster_input.sorted_splat_count;
			_init_stage_io(io, p_context.frame_id, input_count,
					r_raster_output.color.is_valid() ? 1u : 0u,
					_get_sort_indices_buffer(state_view), r_raster_output.color, input_count > 0);
			StageIOValidationConfig validation;
			validation.failed = r_raster_result.status == StageResult::StageStatus::FAILED;
			validation.failed_error = "Raster stage failed";
			validation.record_event = false;
			_finalize_stage_io(renderer, "raster", io, validation);
		}

		if (r_raster_result.status == StageResult::StageStatus::FAILED) {
			r_composite_result = _make_stage_result(StageResult::StageStatus::SKIPPED,
					"Composite skipped: raster stage failed", false,
					r_raster_result.fallback_reason);
			if (p_context.metrics) {
				p_context.metrics->raster = r_raster_output;
				p_context.metrics->raster_result = r_raster_result;
				p_context.metrics->composite_result = r_composite_result;
				auto &io = p_context.metrics->composite_io;
				const uint32_t input_count = r_raster_output.color.is_valid() ? 1u : 0u;
				_init_stage_io(io, p_context.frame_id, input_count, 0, r_raster_output.color, RID(), input_count > 0);
				StageIOValidationConfig validation;
				validation.failed = true;
				validation.failed_error = "Composite skipped: raster stage failed";
				validation.record_event = false;
				_finalize_stage_io(renderer, "composite", io, validation);
			}
			return false;
		}

		if (!r_raster_output.reused_cached_render && output_compositor) {
			const bool require_scene_depth = _is_scene_depth_composite_expected(raster_input.render_data);
			if (require_scene_depth && !r_raster_output.depth.is_valid()) {
				output_compositor->invalidate_cached_render();
			} else {
				output_compositor->update_render_cache_signature(raster_input.world_to_camera_transform,
						raster_input.projection, raster_input.viewport_size, r_raster_output.painterly_active,
						r_raster_output.depth, r_raster_output.internal_size, r_raster_output.color,
						raster_input.content_generation, raster_input.cull_config_signature,
						raster_input.color_grading_signature, raster_input.lighting_signature, require_scene_depth);
			}
		}

		GaussianSplatRenderer::CompositeStageInput composite_input;
		composite_input.frame_id = p_context.frame_id;
		composite_input.render_data = p_context.render_data;
		composite_input.render_buffers = p_context.render_buffers;
		composite_input.render_target = p_context.render_target;
		composite_input.viewport_size = p_context.viewport_size;
		composite_input.defer_commit = p_context.defer_commit;
		composite_input.raster_output = r_raster_output;
		composite_input.metrics = p_context.metrics;
		composite_input.state_provider = active_provider;
		uint64_t composite_start_usec = OS::get_singleton()->get_ticks_usec();
		r_composite_result = composite_stage->execute(composite_input, r_composite_executed);
		uint64_t composite_end_usec = OS::get_singleton()->get_ticks_usec();
		r_composite_time_ms = r_composite_executed ? (composite_end_usec - composite_start_usec) / 1000.0f : 0.0f;
		if (p_context.metrics) {
			p_context.metrics->raster = r_raster_output;
			p_context.metrics->raster_result = r_raster_result;
			p_context.metrics->composite_time_ms = r_composite_time_ms;
			p_context.metrics->composite_executed = r_composite_executed;
			p_context.metrics->composite_result = r_composite_result;
		}
		return r_composite_result.status != StageResult::StageStatus::FAILED;
	}
};

RenderPipelineStages::RenderPipelineStages(GaussianSplatRenderer *p_renderer) :
		renderer(p_renderer) {
	ERR_FAIL_NULL(renderer);

	cull_stage = std::make_unique<CullStage>(this, renderer);
	sort_stage = std::make_unique<SortStage>(this, renderer);
	raster_stage = std::make_unique<RasterStage>(this, renderer);
	composite_stage = std::make_unique<CompositeStage>(renderer);
	raster_composite_stage = std::make_unique<RasterCompositeStage>(renderer, raster_stage.get(), composite_stage.get());
}

RenderPipelineStages::~RenderPipelineStages() = default;

void RenderPipelineStages::set_debug_state_orchestrator(RenderDebugStateOrchestrator *p_debug_state_orchestrator) {
	debug_state_orchestrator = p_debug_state_orchestrator;
}

void RenderPipelineStages::set_diagnostics_orchestrator(RenderDiagnosticsOrchestrator *p_diagnostics_orchestrator) {
	diagnostics_orchestrator = p_diagnostics_orchestrator;
}

RenderPipelineStages::StageResult RenderPipelineStages::execute_cull_stage(
		const GaussianSplatRenderer::CullStageInput &p_input,
		GaussianSplatRenderer::CullStageOutput &r_output) {
	ERR_FAIL_NULL_V(cull_stage.get(), StageResult());
	return cull_stage->execute(p_input, r_output);
}

RenderPipelineStages::StageResult RenderPipelineStages::execute_sort_stage(
		const GaussianSplatRenderer::SortStageInput &p_input,
		GaussianSplatRenderer::SortStageOutput &r_output) {
	ERR_FAIL_NULL_V(sort_stage.get(), StageResult());
	return sort_stage->execute(p_input, r_output);
}

GaussianSplatRenderer::CullStageOutput RenderPipelineStages::cull_for_view(const Transform3D &p_world_to_camera_transform,
		const Projection &p_projection, const Size2i &p_viewport_size) {
	return renderer->cull_for_view(p_world_to_camera_transform, p_projection, p_viewport_size);
}

GaussianSplatRenderer::SortStageSummary RenderPipelineStages::sort_for_view(
		const Transform3D &p_world_to_camera_transform, GaussianSplatRenderer::IndexDomain p_input_domain) {
	return renderer->sort_gaussians_for_view(p_world_to_camera_transform, p_input_domain);
}

void RenderPipelineStages::reset_debug_overlay_metrics(float p_sort_ms) {
	ERR_FAIL_NULL(debug_state_orchestrator);
	debug_state_orchestrator->reset_debug_overlay_metrics(p_sort_ms);
}

void RenderPipelineStages::store_stage_metrics(const GaussianSplatRenderer::StageMetrics &p_metrics) {
	ERR_FAIL_NULL(debug_state_orchestrator);
	debug_state_orchestrator->store_stage_metrics(p_metrics);
}

void RenderPipelineStages::clear_stage_metrics() {
	ERR_FAIL_NULL(debug_state_orchestrator);
	debug_state_orchestrator->clear_stage_metrics();
}

void RenderPipelineStages::increment_frame_counter() {
	ERR_FAIL_NULL(diagnostics_orchestrator);
	diagnostics_orchestrator->increment_frame_counter();
}

void RenderPipelineStages::finalize_frame_metrics(uint64_t p_frame_start_usec) {
	ERR_FAIL_NULL(diagnostics_orchestrator);
	diagnostics_orchestrator->finalize_frame_metrics(p_frame_start_usec);
}

void RenderPipelineStages::reset_render_state_for_frame(const GaussianSplatRenderer::IFrameStateProvider *p_state_provider) {
	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
	const GaussianSplatRenderer::IFrameStateProvider &state_provider =
			p_state_provider ? *p_state_provider : fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	OutputCompositor *output_compositor = state_view.get_output_compositor();
	if (!output_compositor) {
		return;
	}

	auto &output_cache = output_compositor->get_cache_state();
	output_compositor->set_final_render_texture(RID());
	output_cache.has_valid_render = false;
	output_compositor->invalidate_cached_render();
	output_cache.last_viewport_copy_success = false;
	output_cache.last_viewport_copy_source_size = Size2i();
	output_cache.last_viewport_copy_dest_size = Size2i();
	GaussianSplatRenderer::PerformanceState &performance_state = state_provider.get_performance_state();
	auto &metrics = performance_state.metrics;
	metrics.gpu_frame_time_ms = 0.0f;
	metrics.gpu_tile_binning_time_ms = 0.0f;
	metrics.gpu_tile_raster_time_ms = 0.0f;
	metrics.gpu_tile_prefix_time_ms = 0.0f;
	metrics.gpu_tile_resolve_time_ms = 0.0f;
	metrics.gpu_timeline_inflight_frames = 0;
	metrics.gpu_timeline_completed_frames = 0;
	metrics.gpu_timeline_stall_count = 0;
	metrics.gpu_timeline_stall_ms = 0.0f;
	metrics.gpu_timeline_last_value = 0;
}

Error RenderPipelineStages::RasterStage::render_tile_fallback(const Size2i &p_viewport_size, RD::DataFormat p_target_format,
		const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
		RenderDataRD *p_render_data, uint32_t p_sorted_splat_count, GaussianSplatRenderer::IndexDomain p_sorted_index_domain,
		const GaussianSplatRenderer::IFrameStateProvider &p_state_provider,
		RID &r_color_output, RID &r_depth_output) {
	const GaussianSplatRenderer::IFrameStateView &state_view = p_state_provider;
	GaussianSplatRenderer::SubsystemState &subsystem_state = p_state_provider.get_subsystem_state();
	const GaussianSplatRenderer::SubsystemState &subsystem_state_view = state_view.get_subsystem_state_view();
	const GaussianSplatRenderer::RenderConfig &render_config = p_state_provider.get_render_config_view();
	const GaussianSplatRenderer::JacobianDebugConfig &jacobian_debug = p_state_provider.get_jacobian_debug_view();
	GaussianSplatRenderer::PerformanceState &performance_state = p_state_provider.get_performance_state();
	// Simplified tile fallback - assumes tile_renderer/rasterizer initialized in _create_gpu_resources_safe()
	if (!renderer->ensure_rendering_device("_render_tile_fallback")) {
		return ERR_UNCONFIGURED;
	}

	// Ensure tile renderer is initialized
	if (!renderer->get_tile_renderer_state().renderer.is_valid() || !subsystem_state.rasterizer.is_valid()) {
		if (!renderer->get_tile_renderer_state().init_failed) {
			GS_LOG_WARN_DEFAULT("[TileRenderer] Not initialized; call _create_gpu_resources_safe() first");
		}
		return ERR_UNCONFIGURED;
	}

	const int width = MAX(1, p_viewport_size.x);
	const int height = MAX(1, p_viewport_size.y);
	if (p_sorted_splat_count > 0 && p_sorted_index_domain != GaussianSplatRenderer::IndexDomain::SPLAT_REF) {
		GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Index-domain contract violation: raster requires splat_ref sort output (got %s)",
				GaussianRenderState::index_domain_to_string(p_sorted_index_domain)));
		return ERR_INVALID_DATA;
	}

	// Determine target format
	RD::DataFormat target_format = p_target_format;
	if (target_format == RD::DATA_FORMAT_MAX) {
		target_format = renderer->get_view_state().active_viewport_color_format;
	}
	if (target_format == RD::DATA_FORMAT_MAX) {
		target_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	}
	const RD::DataFormat raster_output_format = _resolve_compute_friendly_raster_format(target_format);
	if (raster_output_format != target_format) {
		WARN_PRINT_ONCE(vformat("[TileRenderer] Tile fallback format override: requested=%d resolved=%d",
				int(target_format), int(raster_output_format)));
	}

	// Reconfigure tile renderer for current viewport if needed
	subsystem_state.rasterizer->set_output_format(raster_output_format);

	// Collect data sources from the frame plan (mandatory).
	const GaussianSplatRenderer::RenderFramePlan *frame_plan = state_view.get_frame_plan();
	ERR_FAIL_NULL_V_MSG(frame_plan, ERR_UNCONFIGURED, "RenderFramePlan missing in raster stage.");
	const auto &instance_buffers = renderer->get_instance_pipeline_buffers();
	const bool instance_buffers_ready = renderer->has_instance_pipeline_buffers() &&
			GaussianSplatting::InstancePipelineContract::has_raster_buffers(instance_buffers);
	if (!instance_buffers_ready) {
		GS_LOG_ERROR_DEFAULT("[TileRenderer] Instance pipeline enabled but raster buffers are missing.");
		return ERR_UNCONFIGURED;
	}

	RID gaussian_buffer = instance_buffers.atlas_gaussian_buffer;
	RID sorted_indices = subsystem_state.sorting_pipeline.is_valid() ?
			subsystem_state.sorting_pipeline->get_sort_indices_buffer() :
			RID();
	uint32_t splat_count = p_sorted_splat_count > 0 ? p_sorted_splat_count : 0;
	uint32_t total_gaussians = instance_buffers.atlas_gaussian_count;
	if (total_gaussians == 0) {
		const auto &streaming_state = state_view.get_streaming_state();
		if (streaming_state.shared_dynamic_asset_handle.is_valid()) {
			total_gaussians = streaming_state.shared_dynamic_asset_handle.gaussian_count;
		}
		if (total_gaussians == 0 && instance_buffers.max_visible_splats > 0) {
			total_gaussians = instance_buffers.max_visible_splats;
			WARN_PRINT_ONCE("[TileRenderer] atlas_gaussian_count missing; using max_visible_splats for atlas bounds.");
		}
	}
	if (!sorted_indices.is_valid()) {
		GS_LOG_ERROR_DEFAULT("[TileRenderer] Instance pipeline enabled but sorted indices buffer is invalid.");
		return ERR_UNCONFIGURED;
	}
	if (splat_count > 0 && total_gaussians == 0) {
		GS_LOG_ERROR_DEFAULT("[TileRenderer] atlas_gaussian_count must be set when splat_count > 0");
		return ERR_INVALID_DATA;
	}

    // DEBUG: Log which data source path is used
    static int path_debug_count = 0;
    if (GaussianSplatting::is_debug_frame_logging_enabled() && ++path_debug_count <= 5) {
		const char *source_name = "instance_pipeline";
        GS_LOG_RENDERER_DEBUG(vformat("[DATA-PATH] #%d source=%s sorted_indices=%s gaussian_buffer=%s splat_count=%d",
            path_debug_count,
            source_name,
			sorted_indices.is_valid() ? "valid" : "INVALID",
			gaussian_buffer.is_valid() ? "valid" : "INVALID",
			splat_count));
	}

	// Prepare render params
	TileRenderer::RenderParams render_params;
	render_params.gaussian_buffer = gaussian_buffer;
	render_params.sorted_indices = sorted_indices;
	render_params.splat_count = p_sorted_splat_count > 0 ? MIN(splat_count, p_sorted_splat_count) : splat_count;
	render_params.overlap_record_count = render_params.splat_count;
	render_params.total_gaussians = total_gaussians;
	render_params.viewport_size = Vector2i(width, height);
	render_params.world_to_camera_transform = p_world_to_camera_transform;
	render_params.projection = p_projection;
	render_params.render_projection = p_render_projection;
	render_params.tile_size = TileRenderer::DEFAULT_TILE_SIZE;
	render_params.interactive_state_uniform = RID();
	render_params.output_is_premultiplied = true;
	render_params.opacity_multiplier = render_config.opacity_multiplier;
	render_params.color_grading = render_config.color_grading;
	render_params.scene_uniform_buffer = RID();
	render_params.directional_light_buffer = RID();
	render_params.cluster_buffer = RID();
	render_params.shadow_atlas = RID();
	render_params.omni_light_count = 0;
	render_params.spot_light_count = 0;
	render_params.cluster_size = 0;
	render_params.cluster_max_elements = 0;
	render_params.light_mask = 0xFFFFFFFFu;
	float direct_light_scale = 0.5f;
	float indirect_sh_scale = 1.0f;
	float shadow_strength = 1.0f;
	bool sh_dc_logit = false;
	float shadow_receiver_bias_scale = 0.2f;
	float shadow_receiver_bias_min = 0.0f;
	float shadow_receiver_bias_max = 0.0f;
	bool wind_enabled = false;
	Vector3 wind_direction(1.0f, 0.0f, 0.0f);
	float wind_strength = 0.0f;
	float wind_frequency = 1.0f;
	float wind_spatial_frequency = 0.1f;
	float wind_time_scale = 1.0f;
	int max_effectors = 1;
	bool sphere_effector_enabled = false;
	Vector3 sphere_effector_center = Vector3();
	float sphere_effector_radius = 0.0f;
	float sphere_effector_strength = 0.0f;
	float sphere_effector_falloff = 2.0f;
	float sphere_effector_frequency = 2.0f;
	if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
		static const StringName direct_path("rendering/gaussian_splatting/lighting/direct_light_scale");
		static const StringName indirect_path("rendering/gaussian_splatting/lighting/indirect_sh_scale");
		static const StringName shadow_path("rendering/gaussian_splatting/lighting/shadow_strength");
		static const StringName sh_dc_logit_path("rendering/gaussian_splatting/lighting/dc_logit");
		static const StringName shadow_bias_scale_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_scale");
		static const StringName shadow_bias_min_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_min");
		static const StringName shadow_bias_max_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_max");
		static const StringName wind_enabled_path("rendering/gaussian_splatting/animation/wind_enabled");
		static const StringName wind_direction_x_path("rendering/gaussian_splatting/animation/wind_direction_x");
		static const StringName wind_direction_y_path("rendering/gaussian_splatting/animation/wind_direction_y");
		static const StringName wind_direction_z_path("rendering/gaussian_splatting/animation/wind_direction_z");
		static const StringName wind_strength_path("rendering/gaussian_splatting/animation/wind_strength");
		static const StringName wind_frequency_path("rendering/gaussian_splatting/animation/wind_frequency");
		static const StringName wind_spatial_frequency_path("rendering/gaussian_splatting/animation/wind_spatial_frequency");
		static const StringName wind_time_scale_path("rendering/gaussian_splatting/animation/wind_time_scale");
		static const StringName max_effectors_path("rendering/gaussian_splatting/effects/max_effectors");
		static const StringName sphere_effector_enabled_path("rendering/gaussian_splatting/effects/sphere_effector_enabled");
		static const StringName sphere_effector_center_x_path("rendering/gaussian_splatting/effects/sphere_effector_center_x");
		static const StringName sphere_effector_center_y_path("rendering/gaussian_splatting/effects/sphere_effector_center_y");
		static const StringName sphere_effector_center_z_path("rendering/gaussian_splatting/effects/sphere_effector_center_z");
		static const StringName sphere_effector_radius_path("rendering/gaussian_splatting/effects/sphere_effector_radius");
		static const StringName sphere_effector_strength_path("rendering/gaussian_splatting/effects/sphere_effector_strength");
		static const StringName sphere_effector_falloff_path("rendering/gaussian_splatting/effects/sphere_effector_falloff");
		static const StringName sphere_effector_frequency_path("rendering/gaussian_splatting/effects/sphere_effector_frequency");

		direct_light_scale = _get_float_setting(ps, direct_path, direct_light_scale);
		indirect_sh_scale = _get_float_setting(ps, indirect_path, indirect_sh_scale);
		shadow_strength = _get_float_setting(ps, shadow_path, shadow_strength);
		sh_dc_logit = _get_bool_setting(ps, sh_dc_logit_path, sh_dc_logit);
		shadow_receiver_bias_scale = _get_float_setting(ps, shadow_bias_scale_path, shadow_receiver_bias_scale);
		shadow_receiver_bias_min = _get_float_setting(ps, shadow_bias_min_path, shadow_receiver_bias_min);
		shadow_receiver_bias_max = _get_float_setting(ps, shadow_bias_max_path, shadow_receiver_bias_max);
		wind_enabled = _get_bool_setting(ps, wind_enabled_path, wind_enabled);
		wind_direction.x = _get_float_setting(ps, wind_direction_x_path, wind_direction.x);
		wind_direction.y = _get_float_setting(ps, wind_direction_y_path, wind_direction.y);
		wind_direction.z = _get_float_setting(ps, wind_direction_z_path, wind_direction.z);
		wind_strength = _get_float_setting(ps, wind_strength_path, wind_strength);
		wind_frequency = _get_float_setting(ps, wind_frequency_path, wind_frequency);
		wind_spatial_frequency = _get_float_setting(ps, wind_spatial_frequency_path, wind_spatial_frequency);
		wind_time_scale = _get_float_setting(ps, wind_time_scale_path, wind_time_scale);
		max_effectors = _get_int_setting(ps, max_effectors_path, max_effectors);
		sphere_effector_enabled = _get_bool_setting(ps, sphere_effector_enabled_path, sphere_effector_enabled);
		sphere_effector_center.x = _get_float_setting(ps, sphere_effector_center_x_path, sphere_effector_center.x);
		sphere_effector_center.y = _get_float_setting(ps, sphere_effector_center_y_path, sphere_effector_center.y);
		sphere_effector_center.z = _get_float_setting(ps, sphere_effector_center_z_path, sphere_effector_center.z);
		sphere_effector_radius = _get_float_setting(ps, sphere_effector_radius_path, sphere_effector_radius);
		sphere_effector_strength = _get_float_setting(ps, sphere_effector_strength_path, sphere_effector_strength);
		sphere_effector_falloff = _get_float_setting(ps, sphere_effector_falloff_path, sphere_effector_falloff);
		sphere_effector_frequency = _get_float_setting(ps, sphere_effector_frequency_path, sphere_effector_frequency);
	}
	render_params.direct_light_scale = CLAMP(direct_light_scale, 0.0f, 4.0f);
	render_params.indirect_sh_scale = CLAMP(indirect_sh_scale, 0.0f, 4.0f);
	render_params.shadow_strength = CLAMP(shadow_strength, 0.0f, 1.0f);
	render_params.sh_dc_logit = sh_dc_logit;
	render_params.shadow_receiver_bias_scale = MAX(0.0f, shadow_receiver_bias_scale);
	render_params.shadow_receiver_bias_min = MAX(0.0f, shadow_receiver_bias_min);
	render_params.shadow_receiver_bias_max = MAX(0.0f, shadow_receiver_bias_max);
	render_params.enable_direct_lighting = true;
	render_params.normal_mode = 0;
	render_params.direct_lighting_mode = 1;
	render_params.wind_enabled = wind_enabled;
	render_params.wind_direction = wind_direction;
	render_params.wind_strength = MAX(0.0f, wind_strength);
	render_params.wind_frequency = MAX(0.0f, wind_frequency);
	render_params.wind_spatial_frequency = wind_spatial_frequency;
	render_params.wind_time_seconds = float(double(state_view.get_frame_state_view().frame_counter) * (1.0 / 60.0) *
			double(MAX(wind_time_scale, 0.0f)));
	const int capped_effectors = CLAMP(max_effectors, 0, 1);
	const bool sphere_effective_enabled = capped_effectors > 0 && sphere_effector_enabled;
	render_params.sphere_effector_enabled = sphere_effective_enabled;
	render_params.sphere_effector_center = sphere_effector_center;
	render_params.sphere_effector_radius = MAX(0.0f, sphere_effector_radius);
	render_params.sphere_effector_strength = sphere_effector_strength;
	render_params.sphere_effector_falloff = MAX(0.001f, sphere_effector_falloff);
	render_params.sphere_effector_frequency = MAX(0.1f, sphere_effector_frequency);
	if (instance_buffers_ready) {
		render_params.instance_buffer = instance_buffers.instance_buffer;
		render_params.splat_ref_buffer = instance_buffers.splat_ref_buffer;
		render_params.quantization_buffer = instance_buffers.quantization_required ? instance_buffers.quantization_buffer : RID();
		render_params.instance_indirect_count_buffer = instance_buffers.instance_count_buffer;
		render_params.instance_indirect_dispatch_buffer = instance_buffers.indirect_count_buffer;
		render_params.max_visible_splats = instance_buffers.max_visible_splats;
	} else {
		// Keep instance-path RIDs explicitly cleared when buffers are unavailable.
		// This prevents transient stale/garbage descriptors from being interpreted
		// as valid indirect dispatch/count buffers by tile stages.
		render_params.instance_buffer = RID();
		render_params.splat_ref_buffer = RID();
		render_params.quantization_buffer = RID();
		render_params.instance_indirect_count_buffer = RID();
		render_params.instance_indirect_dispatch_buffer = RID();
	}
	if (p_render_data && p_render_data->scene_data) {
		render_params.scene_uniform_buffer = p_render_data->scene_data->get_uniform_buffer();
		render_params.light_mask = p_render_data->scene_data->camera_visible_layers;
		bool want_projection_log = GaussianSplatting::is_debug_frame_logging_enabled();
		if (!want_projection_log && subsystem_state_view.debug_overlay_system.is_valid()) {
			want_projection_log = subsystem_state_view.debug_overlay_system->get_show_projection_issues();
		}
		if (want_projection_log) {
			const Projection scene_projection = p_render_data->scene_data->get_cam_projection();
			if (!scene_projection.is_same(p_render_projection)) {
				static int projection_mismatch_log_counter = 0;
				if (++projection_mismatch_log_counter % 60 == 1) {
					GS_LOG_RENDERER_DEBUG("[LIGHTING] GS render projection differs from scene_data projection (TAA jitter/flip_y or viewport mismatch suspected).");
				}
			}
		}
	}
	if (p_render_data) {
		render_params.shadow_atlas = p_render_data->shadow_atlas;
	}
	if (p_render_data && p_render_data->cluster_buffer.is_valid()) {
		render_params.cluster_buffer = p_render_data->cluster_buffer;
		render_params.cluster_size = p_render_data->cluster_size;
		render_params.cluster_max_elements = p_render_data->cluster_max_elements;
	}
	if (GaussianSplatting::is_debug_force_unclustered_lights_enabled()) {
		render_params.cluster_buffer = RID();
		render_params.cluster_size = 0;
		render_params.cluster_max_elements = 0;
	}
	if (RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton()) {
		render_params.directional_light_buffer = light_storage->get_directional_light_buffer();
		uint32_t omni_count = light_storage->get_omni_light_count();
		uint32_t spot_count = light_storage->get_spot_light_count();
		const bool use_clustered = render_params.cluster_buffer.is_valid() &&
				render_params.cluster_size > 0u &&
				render_params.cluster_max_elements > 0u;
		if (!use_clustered) {
			omni_count = MIN(omni_count, uint32_t(TileRenderer::MAX_OMNI_LIGHTS));
			spot_count = MIN(spot_count, uint32_t(TileRenderer::MAX_SPOT_LIGHTS));
		}
		render_params.omni_light_count = omni_count;
		render_params.spot_light_count = spot_count;
	}

	if (GaussianSplatting::is_debug_frame_logging_enabled()) {
		static int render_params_log_count = 0;
		if (render_params_log_count < 10 || (render_params_log_count % 300) == 0) {
			bool has_render_data = (p_render_data != nullptr);
			bool has_scene_data = has_render_data && (p_render_data->scene_data != nullptr);
			RID ubo = has_scene_data ? p_render_data->scene_data->get_uniform_buffer() : RID();
			uint32_t dir_count = has_scene_data ? p_render_data->scene_data->directional_light_count : 0;
			GS_LOG_RENDERER_INFO(vformat("[LIGHTING-BIND] Frame %d: render_data=%s scene_data=%s ubo=%s dir_light_count=%d omni=%d spot=%d",
					render_params_log_count,
					has_render_data ? "YES" : "NO",
					has_scene_data ? "YES" : "NO",
					ubo.is_valid() ? "VALID" : "INVALID",
					dir_count,
					render_params.omni_light_count,
					render_params.spot_light_count));
		}
		render_params_log_count++;
	}
	const PipelineFeatureSet *pipeline_features = state_view.get_pipeline_features();
	render_params.compute_raster_policy = frame_plan->compute_raster_policy;
	if (pipeline_features) {
		render_params.request_packed_stage_data = pipeline_features->enable_packed_stage_data;
		render_params.request_tighter_bounds = pipeline_features->enable_tighter_bounds;
		render_params.request_sh_amortization = pipeline_features->enable_sh_amortization;
		render_params.sh_amortization_divisor = pipeline_features->sh_amortization_divisor;
	}

	// Apply debug options
	if (pipeline && pipeline->debug_state_orchestrator) {
		pipeline->debug_state_orchestrator->apply_debug_options_to_render_params(render_params);
	}

	// Culling configuration (tile binning)
	if (subsystem_state.gpu_culler.is_valid()) {
		const GPUCuller::CullingConfig &cull_config = subsystem_state.gpu_culler->get_config();
		const GPUCuller::CullingState &cull_state = subsystem_state.gpu_culler->get_state();
		render_params.distance_cull_enabled = cull_config.distance_cull_enabled;
		render_params.distance_cull_start = cull_config.distance_cull_start;
		render_params.distance_cull_max_rate = cull_config.distance_cull_max_rate;
		// Subpixel splat culling - filter splats smaller than this radius in pixels
		render_params.tiny_splat_screen_radius = cull_state.tiny_splat_screen_radius_px;
	}

	// Apply Jacobian diagnostic toggles
	render_params.jacobian_bypass_radius_depth_floor = jacobian_debug.bypass_radius_depth_floor;
	render_params.jacobian_bypass_j_col2_clamp = jacobian_debug.bypass_j_col2_clamp;
	render_params.jacobian_invert_j_col2_sign = jacobian_debug.invert_j_col2_sign;
	render_params.max_conic_aspect = jacobian_debug.max_conic_aspect;

	// Instance pipeline supplies per-instance inverse rotations; keep global rotation identity.
	render_params.instance_rotation_inverse = Basis();
	render_params.instance_rotation_valid = false;

	// Update interactive state GPU buffer
	if (subsystem_state.interactive_state_manager.is_valid() &&
			subsystem_state.interactive_state_manager->is_initialized()) {
		subsystem_state.interactive_state_manager->update_gpu_state();
	}

	// Handle zero splat count
	if (render_params.splat_count == 0) {
		performance_state.metrics.raster_path = "none";
		RID color_output = subsystem_state.rasterizer->get_output_texture();
		RID depth_output = subsystem_state.rasterizer->has_depth_output() ?
			subsystem_state.rasterizer->get_depth_texture() : RID();
		r_color_output = color_output;
		r_depth_output = depth_output;
		return OK;
	}

	// Render through TileRasterizer interface
	RenderingDevice *tile_device = state_view.get_rendering_device();
	if (subsystem_state.interactive_state_manager.is_valid()) {
		render_params.interactive_state_uniform =
			subsystem_state.interactive_state_manager->ensure_state_uniform_buffer(renderer, tile_device);
	}
	RasterResult raster_result = subsystem_state.rasterizer->render_direct(tile_device, render_params);

	r_color_output = raster_result.output_texture;
	r_depth_output = raster_result.has_depth ? raster_result.depth_texture : RID();

	if (!r_color_output.is_valid()) {
		return FAILED;
	}

	// Track outputs
	renderer->update_tile_renderer_output_tracking(r_color_output, raster_result.output_owner,
			r_depth_output, raster_result.has_depth ? raster_result.depth_owner : nullptr);

	// Update performance metrics
	RasterPerformance raster_perf = subsystem_state.rasterizer->get_performance();

	RasterStats raster_stats = subsystem_state.rasterizer->get_render_stats();
	performance_state.metrics.raster_path = raster_stats.last_raster_used_compute ? "compute" : "fragment";
	if (pipeline && pipeline->debug_state_orchestrator) {
		pipeline->debug_state_orchestrator->update_raster_metrics(raster_perf, raster_stats);
	}

	// Apply overflow feedback if enabled
	if (subsystem_state.gpu_culler.is_valid() &&
			subsystem_state.gpu_culler->get_state().overflow_autotune_enabled &&
			subsystem_state.rasterizer.is_valid()) {
		RasterOverflowStats overflow_stats = subsystem_state.rasterizer->get_overflow_stats();
		subsystem_state.gpu_culler->apply_overflow_feedback(overflow_stats, render_params.splat_count,
				subsystem_state.rasterizer->get_tile_count(), subsystem_state.overflow_auto_tuner.ptr(),
				raster_stats.average_splats_per_tile);
	}

	if (pipeline && pipeline->debug_state_orchestrator) {
		pipeline->debug_state_orchestrator->clear_overlay_dirty_flags();
	}

	return OK;
}

RenderPipelineStages::StageResult RenderPipelineStages::RasterStage::resolve_painterly_output(
		const GaussianSplatRenderer::RasterStageInput &p_input,
		GaussianSplatRenderer::RasterStageOutput &r_output) {
	StageResult result;
	r_output.painterly_active = p_input.painterly_requested;
	if (!r_output.painterly_active) {
		return result;
	}

	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
	const GaussianSplatRenderer::IFrameStateProvider *active_provider =
			p_input.state_provider ? p_input.state_provider : &fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = *active_provider;
	PainterlyRenderer *painterly_renderer = state_view.get_painterly_renderer();
	if (!painterly_renderer) {
		result = _make_stage_result(StageResult::StageStatus::FALLBACK,
				"[Painterly] Painterly renderer not available; falling back to baseline pipeline", true,
				GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_UNAVAILABLE);
		r_output.painterly_active = false;
		return result;
	}

	if (!painterly_renderer->get_pass_graph()) {
		result = _make_stage_result(StageResult::StageStatus::FALLBACK,
				"[Painterly] Painterly pass graph unavailable; falling back to baseline pipeline", true,
				GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_PASS_GRAPH_UNAVAILABLE);
		r_output.painterly_active = false;
		return result;
	}

	Ref<PainterlyMaterial> painterly_material = painterly_renderer->get_material();
	if (!painterly_material.is_valid()) {
		result = _make_stage_result(StageResult::StageStatus::FALLBACK,
				"[Painterly] Painterly renderer requires a PainterlyMaterial; falling back to baseline pipeline", true,
				GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_MATERIAL_UNAVAILABLE);
		r_output.painterly_active = false;
	}
	return result;
}

bool RenderPipelineStages::RasterStage::try_reuse_cached_render(const GaussianSplatRenderer::RasterStageInput &p_input,
		GaussianSplatRenderer::RasterStageOutput &r_output) {
	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
	const GaussianSplatRenderer::IFrameStateProvider &state_provider =
			p_input.state_provider ? *p_input.state_provider : fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	OutputCompositor *output_compositor = state_view.get_output_compositor();
	GaussianSplatRenderer::PerformanceState &performance_state = state_provider.get_performance_state();
	if (!output_compositor) {
		return false;
	}

	const bool require_scene_depth = _is_scene_depth_composite_expected(p_input.render_data);
	RID cached_render = output_compositor->get_final_render_texture();
	if (!output_compositor->can_reuse_cached_render(p_input.world_to_camera_transform, p_input.projection,
				p_input.viewport_size, r_output.painterly_active, cached_render,
				p_input.content_generation, p_input.cull_config_signature,
				p_input.color_grading_signature, p_input.lighting_signature, require_scene_depth)) {
		return false;
	}

	RID cached_depth = output_compositor->get_cached_render_depth();
	if (require_scene_depth && !cached_depth.is_valid()) {
		output_compositor->invalidate_cached_render();
		return false;
	}

	auto &output_cache = output_compositor->get_cache_state();
	r_output.color = cached_render;
	r_output.depth = cached_depth;
	r_output.internal_size = output_cache.cached_render_internal_size;
	r_output.reused_cached_render = r_output.color.is_valid();
	if (r_output.reused_cached_render) {
		r_output.raster_path = "cached";
		performance_state.metrics.raster_path = "cached";
	}
	return r_output.reused_cached_render;
}

RenderPipelineStages::StageResult RenderPipelineStages::RasterStage::render_baseline_stage(
		const GaussianSplatRenderer::RasterStageInput &p_input,
		GaussianSplatRenderer::RasterStageOutput &r_output, const StageResult &p_fallback_context,
		uint64_t p_frame_start_usec) {
	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
	const GaussianSplatRenderer::IFrameStateProvider &state_provider =
			p_input.state_provider ? *p_input.state_provider : fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	GaussianSplatRenderer::FrameState &frame_state = state_provider.get_frame_state();
	GaussianSplatRenderer::PerformanceState &performance_state = state_provider.get_performance_state();
	GaussianSplatRenderer::ResourceState &resource_state = state_provider.get_resource_state();
	OutputCompositor *output_compositor = state_view.get_output_compositor();
	uint64_t fallback_start = OS::get_singleton()->get_ticks_usec();
	Error fallback_err = render_tile_fallback(p_input.viewport_size, p_input.viewport_format,
			p_input.world_to_camera_transform, p_input.projection, p_input.render_projection,
			p_input.render_data, p_input.sorted_splat_count, p_input.sorted_index_domain,
			state_provider, r_output.color, r_output.depth);
	r_output.raster_path = performance_state.metrics.raster_path;
	uint64_t fallback_end = OS::get_singleton()->get_ticks_usec();
	const float render_time_ms = (fallback_end - fallback_start) / 1000.0f;
	frame_state.render_time_ms = render_time_ms;
	r_output.render_time_ms = render_time_ms;
	auto fill_raster_io = [&](bool p_failed) {
		if (!p_input.metrics) {
			return;
		}
		auto &io = p_input.metrics->raster_io;
		const uint32_t input_count = p_input.sorted_splat_count;
		_init_stage_io(io, p_input.frame_id, input_count,
				r_output.color.is_valid() ? 1u : 0u,
				_get_sort_indices_buffer(state_view), r_output.color, input_count > 0);
		const bool input_missing = io.input_count > 0 && !io.input_buffer.is_valid();
		const bool output_missing = io.output_count > 0 && !io.output_buffer.is_valid();
		StageIOValidationConfig validation;
		validation.failed = p_failed;
		validation.input_missing = input_missing;
		validation.output_missing = output_missing;
		validation.failed_error = "Raster output texture invalid";
		validation.input_error = "Raster input buffer invalid";
		validation.output_error = "Raster output buffer invalid";
		_finalize_stage_io(renderer, "raster", io, validation);
	};

	if (fallback_err != OK || !r_output.color.is_valid()) {
		performance_state.metrics.raster_path = "failed";
		r_output.raster_path = "failed";
		fill_raster_io(true);
		StageResult result = _make_stage_result(StageResult::StageStatus::FAILED,
				vformat("[GaussianSplatRenderer] Tile fallback failed: %d", fallback_err), true,
				GaussianSplatRenderer::RenderFallbackReason::TILE_FALLBACK_FAILED);

		pipeline->reset_render_state_for_frame(&state_provider);
		frame_state.render_time_ms = 0.0f;
		r_output.render_time_ms = 0.0f;
		pipeline->reset_debug_overlay_metrics(p_input.sort_time_ms);

		uint64_t frame_end = OS::get_singleton()->get_ticks_usec();
		float frame_time_ms = (frame_end - p_frame_start_usec) / 1000.0f;

		auto &metrics = performance_state.metrics;
		metrics.total_frames_rendered++;
		metrics.rendered_splat_count = 0;

		float alpha = 0.05f;
		if (metrics.total_frames_rendered == 1) {
			metrics.avg_frame_time_ms = frame_time_ms;
		} else {
			metrics.avg_frame_time_ms =
					metrics.avg_frame_time_ms * (1.0f - alpha) + frame_time_ms * alpha;
		}

		metrics.peak_frame_time_ms = MAX(metrics.peak_frame_time_ms, frame_time_ms);

		const GaussianSplatRenderer::RenderFramePlan *frame_plan = state_view.get_frame_plan();
		DEV_ASSERT(frame_plan);
		if (frame_plan) {
			GaussianSplatRenderer::apply_data_source_plan(frame_plan->data_source, metrics, resource_state);
		} else {
			GS_LOG_ERROR_DEFAULT("[GaussianSplatRenderer] RenderFramePlan missing; data source metrics unavailable");
		}

		if (output_compositor) {
			auto &output_cache = output_compositor->get_cache_state();
			output_cache.pending_render_buffers_size = Size2i();
			output_cache.pending_painterly_commit = false;
		}

		pipeline->increment_frame_counter();
		return result;
	}

	fill_raster_io(false);
	renderer->update_gpu_pass_metrics_from_tile_renderer();
	if (p_fallback_context.status == StageResult::StageStatus::FALLBACK) {
		return p_fallback_context;
	}
	return StageResult();
}

RenderPipelineStages::StageResult RenderPipelineStages::RasterStage::render_painterly_or_baseline_stage(
		const GaussianSplatRenderer::RasterStageInput &p_input,
		GaussianSplatRenderer::RasterStageOutput &r_output, const StageResult &p_painterly_status,
		uint64_t p_frame_start_usec) {
	if (!r_output.painterly_active) {
		return render_baseline_stage(p_input, r_output, p_painterly_status, p_frame_start_usec);
	}

	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
	const GaussianSplatRenderer::IFrameStateProvider &state_provider =
			p_input.state_provider ? *p_input.state_provider : fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	GaussianSplatRenderer::PerformanceState &performance_state = state_provider.get_performance_state();
	float painterly_render_time_ms = 0.0f;
	GaussianSplatRenderer::FrameState &frame_state = state_provider.get_frame_state();
	PainterlyRenderer *painterly_renderer = state_view.get_painterly_renderer();
	if (!painterly_renderer) {
		StageResult painterly_failure = _make_stage_result(StageResult::StageStatus::FALLBACK,
				"[Painterly] Painterly renderer not available; falling back to baseline pipeline", true,
				GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_UNAVAILABLE);
		r_output.painterly_active = false;
		r_output.internal_size = p_input.viewport_size;
		return render_baseline_stage(p_input, r_output, painterly_failure, p_frame_start_usec);
	}
	Error painterly_err = painterly_renderer->render_painterly_frame(renderer,
			p_input.viewport_size, p_input.viewport_format, p_input.world_to_camera_transform,
			p_input.projection, p_input.render_projection, r_output.color, r_output.internal_size,
			painterly_render_time_ms);
	if (painterly_err != OK) {
		StageResult painterly_failure;
		r_output.painterly_active = false;
		if (painterly_err == ERR_UNAVAILABLE) {
			painterly_failure = _make_stage_result(StageResult::StageStatus::FALLBACK,
					"[Painterly] ERROR: Pass graph failed to initialize properly", true,
					GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_PASS_GRAPH_UNAVAILABLE);
		} else {
			painterly_failure = _make_stage_result(StageResult::StageStatus::FALLBACK,
					vformat("[Painterly] Failed to populate painterly G-buffer: %d", painterly_err), true,
					GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_RENDER_FAILED);
		}
		r_output.internal_size = p_input.viewport_size;
		return render_baseline_stage(p_input, r_output, painterly_failure, p_frame_start_usec);
	}

	r_output.raster_path = "painterly";
	performance_state.metrics.raster_path = "painterly";
	frame_state.render_time_ms = painterly_render_time_ms;
	r_output.render_time_ms = painterly_render_time_ms;
	return StageResult();
}

RenderPipelineStages::StageResult RenderPipelineStages::CompositeStage::execute(
		const GaussianSplatRenderer::CompositeStageInput &p_input,
		bool &r_did_composite) {
	r_did_composite = false;
	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer);
	const GaussianSplatRenderer::IFrameStateProvider *active_provider =
			p_input.state_provider ? p_input.state_provider : &fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = *active_provider;
	OutputCompositor *output_compositor = state_view.get_output_compositor();
	if (!output_compositor) {
		if (p_input.metrics) {
			auto &io = p_input.metrics->composite_io;
			const uint32_t input_count = p_input.raster_output.color.is_valid() ? 1u : 0u;
			_init_stage_io(io, p_input.frame_id, input_count, 0, p_input.raster_output.color, RID(),
					input_count > 0);
			StageIOValidationConfig validation;
			validation.failed = true;
			validation.failed_error = "Composite failed: OutputCompositor unavailable";
			_finalize_stage_io(renderer, "composite", io, validation);
		}
		return _make_stage_result(StageResult::StageStatus::FAILED, "Composite failed: OutputCompositor unavailable", true,
				GaussianSplatRenderer::RenderFallbackReason::OUTPUT_COMPOSITOR_UNAVAILABLE);
	}

	auto &output_cache = output_compositor->get_cache_state();
	auto fill_composite_io = [&](bool p_failed, const String &p_error) {
		if (!p_input.metrics) {
			return;
		}
		auto &io = p_input.metrics->composite_io;
		const uint32_t input_count = p_input.raster_output.color.is_valid() ? 1u : 0u;
		const uint32_t output_count = (r_did_composite || output_cache.has_valid_render) ? 1u : 0u;
		_init_stage_io(io, p_input.frame_id, input_count, output_count, p_input.raster_output.color,
				output_compositor->get_final_render_texture(), input_count > 0);
		const bool input_missing = io.input_count > 0 && !io.input_buffer.is_valid();
		const bool output_missing = io.output_count > 0 && !io.output_buffer.is_valid();
		StageIOValidationConfig validation;
		validation.failed = p_failed;
		validation.input_missing = input_missing;
		validation.output_missing = output_missing;
		validation.failed_error = p_error;
		validation.input_error = "Composite input texture invalid";
		validation.output_error = "Composite output texture invalid";
		_finalize_stage_io(renderer, "composite", io, validation);
	};

	const uint32_t trace_input_count = p_input.raster_output.color.is_valid() ? 1u : 0u;
	const uint32_t trace_output_count = output_cache.has_valid_render ? 1u : 0u;
	if (_pipeline_trace_enabled(renderer)) {
		_record_pipeline_event(renderer, "composite",
				vformat("reused=%s rt_valid=%s last_rt_valid=%s rt_match=%s color_valid=%s",
						p_input.raster_output.reused_cached_render ? "YES" : "no",
						p_input.render_target.is_valid() ? "YES" : "no",
						output_cache.last_render_target.is_valid() ? "YES" : "no",
						(p_input.render_target == output_cache.last_render_target) ? "YES" : "no",
						p_input.raster_output.color.is_valid() ? "YES" : "no"),
				trace_input_count, trace_output_count, false);
	}

	if (p_input.raster_output.reused_cached_render) {
		// FIX: NEVER skip the composite just because the render target hasn't changed.
		// Godot clears the viewport each frame, so we MUST re-composite the cached render.
		// Fall through to actually composite the cached render to the viewport
	}

	RenderingDevice *rendering_device = state_view.get_rendering_device();
	if (!output_compositor->is_initialized() && rendering_device) {
		output_compositor->initialize(rendering_device);
	}
	if (p_input.raster_output.internal_size.x > 0 && p_input.raster_output.internal_size.y > 0) {
		output_compositor->set_internal_render_size(p_input.raster_output.internal_size);
	}
	// Local copy for mutable r_render_target parameter (function may update it during compositing).
	RID render_target_copy = p_input.render_target;
	output_compositor->integrate_final_output(renderer, p_input.render_data,
			p_input.render_buffers, p_input.raster_output.color, render_target_copy, p_input.viewport_size,
			p_input.defer_commit, p_input.raster_output.painterly_active, p_input.raster_output.depth);
	r_did_composite = true;
	if (_pipeline_trace_enabled(renderer)) {
		_record_pipeline_event(renderer, "composite",
				vformat("executed copy_success=%s",
						output_cache.last_viewport_copy_success ? "YES" : "no"),
				trace_input_count, 1u, false);
	}
	fill_composite_io(false, String());
	return StageResult();
}

bool RenderPipelineStages::execute_raster_composite_pipeline(const GaussianSplatRenderer::RenderFrameContext &p_context,
		uint64_t p_frame_start_usec, GaussianSplatRenderer::RasterStageOutput &r_raster_output,
		StageResult &r_raster_result, StageResult &r_composite_result, float &r_composite_time_ms,
		bool &r_composite_executed) {
	ERR_FAIL_NULL_V(raster_composite_stage.get(), false);
	return raster_composite_stage->execute(p_context, p_frame_start_usec, r_raster_output, r_raster_result,
			r_composite_result, r_composite_time_ms, r_composite_executed);
}

void RenderPipelineStages::render_sorted_splats_with_context(const GaussianSplatRenderer::RenderFrameContext &p_context) {
	ERR_FAIL_COND(!p_context.deps.validate());
	const GaussianSplatRenderer::IFrameStateProvider *context_provider = p_context.state_provider;
	GaussianSplatRenderer::FrameStateProvider fallback_provider(renderer, &p_context.deps);
	const GaussianSplatRenderer::IFrameStateProvider &state_provider =
			context_provider ? *context_provider : fallback_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	OutputCompositor *output_compositor = state_view.get_output_compositor();
	ERR_FAIL_COND_MSG(!output_compositor, "OutputCompositor not initialized");
	auto &output_cache = output_compositor->get_cache_state();
	output_cache.last_viewport_copy_success = false;
	output_cache.last_viewport_copy_source_size = Size2i();
	output_cache.last_viewport_copy_dest_size = Size2i();
	GaussianSplatRenderer::FrameState &frame_state = state_provider.get_frame_state();
	GaussianSplatRenderer::PerformanceState &performance_state = state_provider.get_performance_state();
	GaussianSplatRenderer::ResourceState &resource_state = state_provider.get_resource_state();
	frame_state.render_time_ms = 0.0f;
	auto &metrics = performance_state.metrics;
	metrics.raster_path = "unknown";
	metrics.gpu_frame_time_ms = 0.0f;
	metrics.gpu_tile_binning_time_ms = 0.0f;
	metrics.gpu_tile_raster_time_ms = 0.0f;
	const GaussianSplatRenderer::RenderFramePlan *frame_plan = state_view.get_frame_plan();
	DEV_ASSERT(frame_plan);
	ERR_FAIL_COND_MSG(!frame_plan, "RenderFramePlan missing in render_sorted_splats_with_context.");
	_begin_pipeline_trace(renderer);
	GaussianSplatRenderer::apply_data_source_plan(frame_plan->data_source, metrics, resource_state);
	const uint64_t frame_start_usec = OS::get_singleton()->get_ticks_usec();

	auto emit_stage_metrics = [&]() {
		if (p_context.metrics) {
			store_stage_metrics(*p_context.metrics);
		} else {
			clear_stage_metrics();
		}
	};
	auto finalize_frame = [&]() {
		emit_stage_metrics();
		increment_frame_counter();
		finalize_frame_metrics(frame_start_usec);
	};

	if (!p_context.metrics && !p_context.snapshot.valid) {
		performance_state.metrics.raster_path = "skipped";
		_record_pipeline_event(renderer, "raster", "fail: missing frame snapshot/metrics", 0, 0, true,
				GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
				RenderRouteUID::COMMON_SKIP_NO_DATA);
		_record_pipeline_event(renderer, "composite", "skip: missing frame snapshot/metrics", 0, 0, false,
				GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE,
				RenderRouteUID::COMMON_SKIP_NO_DATA);
		reset_render_state_for_frame(&state_provider);
		finalize_frame();
		return;
	}

	if (p_context.metrics) {
		log_stage_result("Cull", p_context.metrics->cull_result);
		log_stage_result("Sort", p_context.metrics->sort_result);
	}
	if (p_context.metrics && p_context.metrics->sort_result.status == StageResult::StageStatus::FAILED) {
		performance_state.metrics.raster_path = "skipped";
		const String reason = p_context.metrics->sort_result.reason.is_empty() ?
				"Sort failed; skipping raster/composite" :
				p_context.metrics->sort_result.reason;
		_record_pipeline_event(renderer, "raster", "skip: sort failed",
				p_context.metrics->sort.input_count, 0, true, p_context.metrics->sort_result.fallback_reason,
				RenderRouteUID::COMMON_FAIL_SORT_FAILED);
		_record_pipeline_event(renderer, "composite", "skip: sort failed",
				0, 0, false, p_context.metrics->sort_result.fallback_reason,
				RenderRouteUID::COMMON_FAIL_SORT_FAILED);
		StageResult raster_result = _make_stage_result(StageResult::StageStatus::FAILED,
				"Raster skipped: sort failed", true, p_context.metrics->sort_result.fallback_reason);
		StageResult composite_result = _make_stage_result(StageResult::StageStatus::SKIPPED,
				"Composite skipped: sort failed", false, p_context.metrics->sort_result.fallback_reason);
		p_context.metrics->raster_result = raster_result;
		p_context.metrics->composite_result = composite_result;
		auto &raster_io = p_context.metrics->raster_io;
		_init_stage_io(raster_io, p_context.frame_id, p_context.metrics->sort.input_count, 0,
				_get_sort_indices_buffer(state_view), RID(),
				p_context.metrics->sort.input_count > 0);
		StageIOValidationConfig raster_validation;
		raster_validation.failed = true;
		raster_validation.failed_error = reason;
		_finalize_stage_io(renderer, "raster", raster_io, raster_validation);
		auto &composite_io = p_context.metrics->composite_io;
		_init_stage_io(composite_io, p_context.frame_id, 0, 0, RID(), RID(), false);
		StageIOValidationConfig composite_validation;
		composite_validation.record_event = false;
		_finalize_stage_io(renderer, "composite", composite_io, composite_validation);
		reset_render_state_for_frame(&state_provider);
		finalize_frame();
		return;
	}
	uint32_t current_visible = p_context.snapshot.valid ? p_context.snapshot.visible_splats : 0;
	if (!p_context.snapshot.valid && p_context.metrics) {
		current_visible = p_context.metrics->cull.visible_domain == GaussianSplatRenderer::IndexDomain::CHUNK_REF
				? p_context.metrics->sort.sorted_count
				: p_context.metrics->cull.visible_count;
	}
	auto finalize_composite_io = [&]() {
		if (!p_context.metrics) {
			return;
		}
		auto &composite_io = p_context.metrics->composite_io;
		_init_stage_io(composite_io, p_context.frame_id, 0, 0, RID(), RID(), false);
		StageIOValidationConfig composite_validation;
		composite_validation.record_event = false;
		_finalize_stage_io(renderer, "composite", composite_io, composite_validation);
	};
	if (current_visible == 0) {
		performance_state.metrics.raster_path = "none";
		_record_pipeline_event(renderer, "raster", "skip: no visible splats", current_visible, 0, false,
				GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS,
				RenderRouteUID::COMMON_SKIP_NO_VISIBLE);
		_record_pipeline_event(renderer, "composite", "skip: no visible splats", 0, 0, false,
				GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS,
				RenderRouteUID::COMMON_SKIP_NO_VISIBLE);
		if (p_context.metrics) {
			p_context.metrics->raster_result = _make_stage_result(StageResult::StageStatus::SKIPPED,
					"Raster skipped: no visible splats", false,
					GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS);
			p_context.metrics->composite_result = _make_stage_result(StageResult::StageStatus::SKIPPED,
					"Composite skipped: no visible splats", false,
					GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS);
			auto &raster_io = p_context.metrics->raster_io;
			_init_stage_io(raster_io, p_context.frame_id, 0, 0, RID(), RID(), false);
			StageIOValidationConfig raster_validation;
			raster_validation.record_event = false;
			_finalize_stage_io(renderer, "raster", raster_io, raster_validation);
			finalize_composite_io();
		}
		reset_render_state_for_frame(&state_provider);
		finalize_frame();
		return;
	}

	if (!renderer->ensure_rendering_device("render_sorted_splats")) {
		performance_state.metrics.raster_path = "failed";
		_record_pipeline_event(renderer, "raster", "fail: RenderingDevice unavailable", current_visible, 0, true,
				GaussianSplatRenderer::RenderFallbackReason::RENDERING_DEVICE_UNAVAILABLE,
				RenderRouteUID::COMMON_FAIL_NO_DEVICE);
		_record_pipeline_event(renderer, "composite", "skip: raster failed", 0, 0, false,
				GaussianSplatRenderer::RenderFallbackReason::RENDERING_DEVICE_UNAVAILABLE,
				RenderRouteUID::COMMON_FAIL_NO_DEVICE);
		if (p_context.metrics) {
			p_context.metrics->raster_result = _make_stage_result(StageResult::StageStatus::FAILED,
					"Raster failed: RenderingDevice unavailable", true,
					GaussianSplatRenderer::RenderFallbackReason::RENDERING_DEVICE_UNAVAILABLE);
			p_context.metrics->composite_result = _make_stage_result(StageResult::StageStatus::FAILED,
					"Composite failed: RenderingDevice unavailable", true,
					GaussianSplatRenderer::RenderFallbackReason::RENDERING_DEVICE_UNAVAILABLE);
			auto &raster_io = p_context.metrics->raster_io;
			_init_stage_io(raster_io, p_context.frame_id, current_visible, 0,
					_get_sort_indices_buffer(state_view), RID(), current_visible > 0);
			StageIOValidationConfig raster_validation;
			raster_validation.failed = true;
			raster_validation.failed_error = "Raster failed: RenderingDevice unavailable";
			_finalize_stage_io(renderer, "raster", raster_io, raster_validation);
			finalize_composite_io();
		}
		reset_render_state_for_frame(&state_provider);
		finalize_frame();
		return;
	}

	output_cache.pending_painterly_commit = false;
	output_cache.pending_render_buffers_size = p_context.viewport_size;
	output_cache.render_buffers_commit_pending = false;

	GaussianSplatRenderer::RasterStageOutput raster_output;
	StageResult raster_result;
	StageResult composite_result;
	float composite_time_ms = 0.0f;
	bool composite_executed = false;
	auto record_raster_composite_events = [&]() {
		String raster_message = raster_result.reason;
		if (raster_message.is_empty()) {
			raster_message = "status=" + _stage_status_label(raster_result.status);
		}
		String raster_route_uid;
		if (raster_result.fallback_reason == GaussianSplatRenderer::RenderFallbackReason::RASTER_REUSED_CACHED_RENDER ||
				raster_output.reused_cached_render) {
			raster_route_uid = RenderRouteUID::INSTANCE_RASTER_CACHED;
		} else if (raster_output.painterly_active || raster_output.raster_path == "painterly") {
			raster_route_uid = RenderRouteUID::INSTANCE_RASTER_PAINTERLY;
		} else if (raster_output.raster_path == "compute") {
			raster_route_uid = RenderRouteUID::INSTANCE_RASTER_COMPUTE;
		} else if (raster_output.raster_path == "fragment") {
			raster_route_uid = RenderRouteUID::INSTANCE_RASTER_FRAGMENT;
		} else if (raster_result.status == StageResult::StageStatus::FAILED) {
			raster_route_uid = RenderRouteUID::COMMON_FAIL_NO_OUTPUT;
		}
		_record_pipeline_event(renderer, "raster", raster_message, current_visible, 0,
				raster_result.is_error || raster_result.status == StageResult::StageStatus::FAILED,
				raster_result.fallback_reason,
				raster_route_uid);
		if (!raster_route_uid.is_empty()) {
			renderer->get_debug_state().route_uid = raster_route_uid;
		}
		String composite_message = composite_result.reason;
		if (composite_message.is_empty()) {
			composite_message = "status=" + _stage_status_label(composite_result.status);
		}
		String composite_route_uid;
		if (composite_result.fallback_reason == GaussianSplatRenderer::RenderFallbackReason::OUTPUT_COMPOSITOR_UNAVAILABLE ||
				composite_result.status == StageResult::StageStatus::FAILED) {
			composite_route_uid = RenderRouteUID::COMMON_FAIL_NO_OUTPUT;
		}
		_record_pipeline_event(renderer, "composite", composite_message, 0, 0,
				composite_result.is_error || composite_result.status == StageResult::StageStatus::FAILED,
				composite_result.fallback_reason,
				composite_route_uid);
	};
	if (!execute_raster_composite_pipeline(p_context, frame_start_usec, raster_output, raster_result, composite_result,
				composite_time_ms, composite_executed)) {
		record_raster_composite_events();
		log_stage_result("Raster", raster_result);
		log_stage_result("Composite", composite_result);
		finalize_frame();
		return;
	}

	record_raster_composite_events();

	emit_stage_metrics();
	log_stage_result("Raster", raster_result);
	log_stage_result("Composite", composite_result);
	increment_frame_counter();
	finalize_frame_metrics(frame_start_usec);
}

void RenderPipelineStages::log_stage_result(const char *p_stage_label, const StageResult &p_result) const {
	String message = p_result.reason;
	if (message.is_empty()) {
		if (p_result.fallback_reason == GaussianSplatRenderer::RenderFallbackReason::NONE) {
			return;
		}
		message = "fallback=" + _fallback_reason_label(p_result.fallback_reason);
	} else if (p_result.fallback_reason != GaussianSplatRenderer::RenderFallbackReason::NONE) {
		message += " (fallback=" + _fallback_reason_label(p_result.fallback_reason) + ")";
	}
	const bool should_log = (p_result.status == StageResult::StageStatus::FAILED) ||
			(p_result.status == StageResult::StageStatus::FALLBACK && p_result.is_error);
	if (!should_log) {
		return;
	}
	GaussianSplatRenderer::FrameStateProvider frame_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
	if (state_view.get_frame_state_view().frame_counter % 60 != 0) {
		return;
	}
	const String log_message = vformat("[%s] %s", p_stage_label, message);
	if (p_result.status == StageResult::StageStatus::FAILED) {
		GS_LOG_ERROR_DEFAULT(log_message);
	} else {
		GS_LOG_WARN_DEFAULT(log_message);
	}
}

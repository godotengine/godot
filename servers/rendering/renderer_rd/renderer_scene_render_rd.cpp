/*************************************************************************/
/*  renderer_scene_render_rd.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "renderer_scene_render_rd.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "renderer_compositor_rd.h"
#include "servers/rendering/rendering_server_default.h"

void get_vogel_disk(float *r_kernel, int p_sample_count) {
	const float golden_angle = 2.4;

	for (int i = 0; i < p_sample_count; i++) {
		float r = Math::sqrt(float(i) + 0.5) / Math::sqrt(float(p_sample_count));
		float theta = float(i) * golden_angle;

		r_kernel[i * 4] = Math::cos(theta) * r;
		r_kernel[i * 4 + 1] = Math::sin(theta) * r;
	}
}

void RendererSceneRenderRD::sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_environment);
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	bool needs_sdfgi = env && env->sdfgi_enabled;

	if (!needs_sdfgi) {
		if (rb->sdfgi != nullptr) {
			//erase it
			rb->sdfgi->erase();
			memdelete(rb->sdfgi);
			rb->sdfgi = nullptr;
		}
		return;
	}

	static const uint32_t history_frames_to_converge[RS::ENV_SDFGI_CONVERGE_MAX] = { 5, 10, 15, 20, 25, 30 };
	uint32_t requested_history_size = history_frames_to_converge[gi.sdfgi_frames_to_converge];

	if (rb->sdfgi && (rb->sdfgi->num_cascades != env->sdfgi_cascades || rb->sdfgi->min_cell_size != env->sdfgi_min_cell_size || requested_history_size != rb->sdfgi->history_size || rb->sdfgi->uses_occlusion != env->sdfgi_use_occlusion || rb->sdfgi->y_scale_mode != env->sdfgi_y_scale)) {
		//configuration changed, erase
		rb->sdfgi->erase();
		memdelete(rb->sdfgi);
		rb->sdfgi = nullptr;
	}

	RendererSceneGIRD::SDFGI *sdfgi = rb->sdfgi;
	if (sdfgi == nullptr) {
		// re-create
		rb->sdfgi = gi.create_sdfgi(env, p_world_position, requested_history_size);
	} else {
		//check for updates
		rb->sdfgi->update(env, p_world_position);
	}
}

int RendererSceneRenderRD::sdfgi_get_pending_region_count(RID p_render_buffers) const {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);

	ERR_FAIL_COND_V(rb == nullptr, 0);

	if (rb->sdfgi == nullptr) {
		return 0;
	}

	int dirty_count = 0;
	for (uint32_t i = 0; i < rb->sdfgi->cascades.size(); i++) {
		const RendererSceneGIRD::SDFGI::Cascade &c = rb->sdfgi->cascades[i];

		if (c.dirty_regions == RendererSceneGIRD::SDFGI::Cascade::DIRTY_ALL) {
			dirty_count++;
		} else {
			for (int j = 0; j < 3; j++) {
				if (c.dirty_regions[j] != 0) {
					dirty_count++;
				}
			}
		}
	}

	return dirty_count;
}

AABB RendererSceneRenderRD::sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const {
	AABB bounds;
	Vector3i from;
	Vector3i size;
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(rb == nullptr, AABB());
	ERR_FAIL_COND_V(rb->sdfgi == nullptr, AABB());

	int c = rb->sdfgi->get_pending_region_data(p_region, from, size, bounds);
	ERR_FAIL_COND_V(c == -1, AABB());
	return bounds;
}

uint32_t RendererSceneRenderRD::sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const {
	AABB bounds;
	Vector3i from;
	Vector3i size;
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(rb == nullptr, -1);
	ERR_FAIL_COND_V(rb->sdfgi == nullptr, -1);

	return rb->sdfgi->get_pending_region_data(p_region, from, size, bounds);
}

RID RendererSceneRenderRD::sky_allocate() {
	return sky.allocate_sky_rid();
}
void RendererSceneRenderRD::sky_initialize(RID p_rid) {
	sky.initialize_sky_rid(p_rid);
}

void RendererSceneRenderRD::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
	sky.sky_set_radiance_size(p_sky, p_radiance_size);
}

void RendererSceneRenderRD::sky_set_mode(RID p_sky, RS::SkyMode p_mode) {
	sky.sky_set_mode(p_sky, p_mode);
}

void RendererSceneRenderRD::sky_set_material(RID p_sky, RID p_material) {
	sky.sky_set_material(p_sky, p_material);
}

Ref<Image> RendererSceneRenderRD::sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) {
	return sky.sky_bake_panorama(p_sky, p_energy, p_bake_irradiance, p_size);
}

RID RendererSceneRenderRD::environment_allocate() {
	return environment_owner.allocate_rid();
}
void RendererSceneRenderRD::environment_initialize(RID p_rid) {
	environment_owner.initialize_rid(p_rid, RendererSceneEnvironmentRD());
}

void RendererSceneRenderRD::environment_set_background(RID p_env, RS::EnvironmentBG p_bg) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->background = p_bg;
}

void RendererSceneRenderRD::environment_set_sky(RID p_env, RID p_sky) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky = p_sky;
}

void RendererSceneRenderRD::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky_custom_fov = p_scale;
}

void RendererSceneRenderRD::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky_orientation = p_orientation;
}

void RendererSceneRenderRD::environment_set_bg_color(RID p_env, const Color &p_color) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->bg_color = p_color;
}

void RendererSceneRenderRD::environment_set_bg_energy(RID p_env, float p_energy) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->bg_energy = p_energy;
}

void RendererSceneRenderRD::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->canvas_max_layer = p_max_layer;
}

void RendererSceneRenderRD::environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->set_ambient_light(p_color, p_ambient, p_energy, p_sky_contribution, p_reflection_source);
}

RS::EnvironmentBG RendererSceneRenderRD::environment_get_background(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_BG_MAX);
	return env->background;
}

RID RendererSceneRenderRD::environment_get_sky(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, RID());
	return env->sky;
}

float RendererSceneRenderRD::environment_get_sky_custom_fov(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->sky_custom_fov;
}

Basis RendererSceneRenderRD::environment_get_sky_orientation(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Basis());
	return env->sky_orientation;
}

Color RendererSceneRenderRD::environment_get_bg_color(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->bg_color;
}

float RendererSceneRenderRD::environment_get_bg_energy(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->bg_energy;
}

int RendererSceneRenderRD::environment_get_canvas_max_layer(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->canvas_max_layer;
}

Color RendererSceneRenderRD::environment_get_ambient_light_color(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->ambient_light;
}

RS::EnvironmentAmbientSource RendererSceneRenderRD::environment_get_ambient_source(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_AMBIENT_SOURCE_BG);
	return env->ambient_source;
}

float RendererSceneRenderRD::environment_get_ambient_light_energy(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->ambient_light_energy;
}

float RendererSceneRenderRD::environment_get_ambient_sky_contribution(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->ambient_sky_contribution;
}

RS::EnvironmentReflectionSource RendererSceneRenderRD::environment_get_reflection_source(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_REFLECTION_SOURCE_DISABLED);
	return env->reflection_source;
}

void RendererSceneRenderRD::environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->set_tonemap(p_tone_mapper, p_exposure, p_white, p_auto_exposure, p_min_luminance, p_max_luminance, p_auto_exp_speed, p_auto_exp_scale);
}

void RendererSceneRenderRD::environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->set_glow(p_enable, p_levels, p_intensity, p_strength, p_mix, p_bloom_threshold, p_blend_mode, p_hdr_bleed_threshold, p_hdr_bleed_scale, p_hdr_luminance_cap);
}

void RendererSceneRenderRD::environment_glow_set_use_bicubic_upscale(bool p_enable) {
	glow_bicubic_upscale = p_enable;
}

void RendererSceneRenderRD::environment_glow_set_use_high_quality(bool p_enable) {
	glow_high_quality = p_enable;
}

void RendererSceneRenderRD::environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	if (!is_dynamic_gi_supported()) {
		return;
	}

	env->set_sdfgi(p_enable, p_cascades, p_min_cell_size, p_y_scale, p_use_occlusion, p_bounce_feedback, p_read_sky, p_energy, p_normal_bias, p_probe_bias);
}

void RendererSceneRenderRD::environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_fog_aerial_perspective) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	env->set_fog(p_enable, p_light_color, p_light_energy, p_sun_scatter, p_density, p_height, p_height_density, p_fog_aerial_perspective);
}

bool RendererSceneRenderRD::environment_is_fog_enabled(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, false);

	return env->fog_enabled;
}
Color RendererSceneRenderRD::environment_get_fog_light_color(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->fog_light_color;
}
float RendererSceneRenderRD::environment_get_fog_light_energy(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_light_energy;
}
float RendererSceneRenderRD::environment_get_fog_sun_scatter(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_sun_scatter;
}
float RendererSceneRenderRD::environment_get_fog_density(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_density;
}
float RendererSceneRenderRD::environment_get_fog_height(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);

	return env->fog_height;
}
float RendererSceneRenderRD::environment_get_fog_height_density(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_height_density;
}

float RendererSceneRenderRD::environment_get_fog_aerial_perspective(RID p_env) const {
	const RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->fog_aerial_perspective;
}

void RendererSceneRenderRD::environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	if (!is_volumetric_supported()) {
		return;
	}

	env->set_volumetric_fog(p_enable, p_density, p_albedo, p_emission, p_emission_energy, p_anisotropy, p_length, p_detail_spread, p_gi_inject, p_temporal_reprojection, p_temporal_reprojection_amount, p_ambient_inject);
}

void RendererSceneRenderRD::environment_set_volumetric_fog_volume_size(int p_size, int p_depth) {
	volumetric_fog_size = p_size;
	volumetric_fog_depth = p_depth;
}

void RendererSceneRenderRD::environment_set_volumetric_fog_filter_active(bool p_enable) {
	volumetric_fog_filter_active = p_enable;
}

void RendererSceneRenderRD::environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) {
	gi.sdfgi_ray_count = p_ray_count;
}

void RendererSceneRenderRD::environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) {
	gi.sdfgi_frames_to_converge = p_frames;
}
void RendererSceneRenderRD::environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) {
	gi.sdfgi_frames_to_update_light = p_update;
}

void RendererSceneRenderRD::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	env->set_ssr(p_enable, p_max_steps, p_fade_int, p_fade_out, p_depth_tolerance);
}

void RendererSceneRenderRD::environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) {
	ssr_roughness_quality = p_quality;
}

RS::EnvironmentSSRRoughnessQuality RendererSceneRenderRD::environment_get_ssr_roughness_quality() const {
	return ssr_roughness_quality;
}

void RendererSceneRenderRD::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	env->set_ssao(p_enable, p_radius, p_intensity, p_power, p_detail, p_horizon, p_sharpness, p_light_affect, p_ao_channel_affect);
}

void RendererSceneRenderRD::environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
	ssao_quality = p_quality;
	ssao_half_size = p_half_size;
	ssao_adaptive_target = p_adaptive_target;
	ssao_blur_passes = p_blur_passes;
	ssao_fadeout_from = p_fadeout_from;
	ssao_fadeout_to = p_fadeout_to;
}

void RendererSceneRenderRD::environment_set_ssil(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_sharpness, float p_normal_rejection) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	env->ssil_enabled = p_enable;
	env->ssil_radius = p_radius;
	env->ssil_intensity = p_intensity;
	env->ssil_sharpness = p_sharpness;
	env->ssil_normal_rejection = p_normal_rejection;
}

void RendererSceneRenderRD::environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
	ssil_quality = p_quality;
	ssil_half_size = p_half_size;
	ssil_adaptive_target = p_adaptive_target;
	ssil_blur_passes = p_blur_passes;
	ssil_fadeout_from = p_fadeout_from;
	ssil_fadeout_to = p_fadeout_to;
}

bool RendererSceneRenderRD::environment_is_ssao_enabled(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_enabled;
}

float RendererSceneRenderRD::environment_get_ssao_ao_affect(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0.0);
	return env->ssao_ao_channel_affect;
}

float RendererSceneRenderRD::environment_get_ssao_light_affect(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0.0);
	return env->ssao_direct_light_affect;
}

bool RendererSceneRenderRD::environment_is_ssil_enabled(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssil_enabled;
}

bool RendererSceneRenderRD::environment_is_ssr_enabled(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssr_enabled;
}
bool RendererSceneRenderRD::environment_is_sdfgi_enabled(RID p_env) const {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->sdfgi_enabled;
}

bool RendererSceneRenderRD::is_environment(RID p_env) const {
	return environment_owner.owns(p_env);
}

Ref<Image> RendererSceneRenderRD::environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Ref<Image>());

	RS::EnvironmentBG environment_background = env->background;

	if (environment_background == RS::ENV_BG_CAMERA_FEED || environment_background == RS::ENV_BG_CANVAS || environment_background == RS::ENV_BG_KEEP) {
		return Ref<Image>(); //nothing to bake
	}

	RS::EnvironmentAmbientSource ambient_source = env->ambient_source;

	bool use_ambient_light = false;
	bool use_cube_map = false;
	if (ambient_source == RS::ENV_AMBIENT_SOURCE_BG && (environment_background == RS::ENV_BG_CLEAR_COLOR || environment_background == RS::ENV_BG_COLOR)) {
		use_ambient_light = true;
	} else {
		use_cube_map = (ambient_source == RS::ENV_AMBIENT_SOURCE_BG && environment_background == RS::ENV_BG_SKY) || ambient_source == RS::ENV_AMBIENT_SOURCE_SKY;
		use_ambient_light = use_cube_map || ambient_source == RS::ENV_AMBIENT_SOURCE_COLOR;
	}
	use_cube_map = use_cube_map || (environment_background == RS::ENV_BG_SKY && env->sky.is_valid());

	Color ambient_color;
	float ambient_color_sky_mix;
	if (use_ambient_light) {
		ambient_color_sky_mix = env->ambient_sky_contribution;
		const float ambient_energy = env->ambient_light_energy;
		ambient_color = env->ambient_light;
		ambient_color = ambient_color.to_linear();
		ambient_color.r *= ambient_energy;
		ambient_color.g *= ambient_energy;
		ambient_color.b *= ambient_energy;
	}

	if (use_cube_map) {
		Ref<Image> panorama = sky_bake_panorama(env->sky, env->bg_energy, p_bake_irradiance, p_size);
		if (use_ambient_light) {
			for (int x = 0; x < p_size.width; x++) {
				for (int y = 0; y < p_size.height; y++) {
					panorama->set_pixel(x, y, ambient_color.lerp(panorama->get_pixel(x, y), ambient_color_sky_mix));
				}
			}
		}
		return panorama;
	} else {
		const float bg_energy = env->bg_energy;
		Color panorama_color = ((environment_background == RS::ENV_BG_CLEAR_COLOR) ? storage->get_default_clear_color() : env->bg_color);
		panorama_color = panorama_color.to_linear();
		panorama_color.r *= bg_energy;
		panorama_color.g *= bg_energy;
		panorama_color.b *= bg_energy;

		if (use_ambient_light) {
			panorama_color = ambient_color.lerp(panorama_color, ambient_color_sky_mix);
		}

		Ref<Image> panorama;
		panorama.instantiate();
		panorama->create(p_size.width, p_size.height, false, Image::FORMAT_RGBAF);
		panorama->fill(panorama_color);
		return panorama;
	}

	return Ref<Image>();
}

////////////////////////////////////////////////////////////

RID RendererSceneRenderRD::fog_volume_instance_create(RID p_fog_volume) {
	FogVolumeInstance fvi;
	fvi.volume = p_fog_volume;
	return fog_volume_instance_owner.make_rid(fvi);
}
void RendererSceneRenderRD::fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) {
	FogVolumeInstance *fvi = fog_volume_instance_owner.get_or_null(p_fog_volume_instance);
	ERR_FAIL_COND(!fvi);
	fvi->transform = p_transform;
}
void RendererSceneRenderRD::fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) {
	FogVolumeInstance *fvi = fog_volume_instance_owner.get_or_null(p_fog_volume_instance);
	ERR_FAIL_COND(!fvi);
	fvi->active = p_active;
}

RID RendererSceneRenderRD::fog_volume_instance_get_volume(RID p_fog_volume_instance) const {
	FogVolumeInstance *fvi = fog_volume_instance_owner.get_or_null(p_fog_volume_instance);
	ERR_FAIL_COND_V(!fvi, RID());
	return fvi->volume;
}

Vector3 RendererSceneRenderRD::fog_volume_instance_get_position(RID p_fog_volume_instance) const {
	FogVolumeInstance *fvi = fog_volume_instance_owner.get_or_null(p_fog_volume_instance);
	ERR_FAIL_COND_V(!fvi, Vector3());

	return fvi->transform.get_origin();
}

////////////////////////////////////////////////////////////

RID RendererSceneRenderRD::reflection_atlas_create() {
	ReflectionAtlas ra;
	ra.count = GLOBAL_GET("rendering/reflections/reflection_atlas/reflection_count");
	ra.size = GLOBAL_GET("rendering/reflections/reflection_atlas/reflection_size");

	if (is_clustered_enabled()) {
		ra.cluster_builder = memnew(ClusterBuilderRD);
		ra.cluster_builder->set_shared(&cluster_builder_shared);
		ra.cluster_builder->setup(Size2i(ra.size, ra.size), max_cluster_elements, RID(), RID(), RID());
	} else {
		ra.cluster_builder = nullptr;
	}

	return reflection_atlas_owner.make_rid(ra);
}

void RendererSceneRenderRD::reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {
	ReflectionAtlas *ra = reflection_atlas_owner.get_or_null(p_ref_atlas);
	ERR_FAIL_COND(!ra);

	if (ra->size == p_reflection_size && ra->count == p_reflection_count) {
		return; //no changes
	}

	if (ra->cluster_builder) {
		// only if we're using our cluster
		ra->cluster_builder->setup(Size2i(ra->size, ra->size), max_cluster_elements, RID(), RID(), RID());
	}

	ra->size = p_reflection_size;
	ra->count = p_reflection_count;

	if (ra->reflection.is_valid()) {
		//clear and invalidate everything
		RD::get_singleton()->free(ra->reflection);
		ra->reflection = RID();
		RD::get_singleton()->free(ra->depth_buffer);
		ra->depth_buffer = RID();
		for (int i = 0; i < ra->reflections.size(); i++) {
			ra->reflections.write[i].data.clear_reflection_data();
			if (ra->reflections[i].owner.is_null()) {
				continue;
			}
			reflection_probe_release_atlas_index(ra->reflections[i].owner);
			//rp->atlasindex clear
		}

		ra->reflections.clear();
	}
}

int RendererSceneRenderRD::reflection_atlas_get_size(RID p_ref_atlas) const {
	ReflectionAtlas *ra = reflection_atlas_owner.get_or_null(p_ref_atlas);
	ERR_FAIL_COND_V(!ra, 0);

	return ra->size;
}

////////////////////////
RID RendererSceneRenderRD::reflection_probe_instance_create(RID p_probe) {
	ReflectionProbeInstance rpi;
	rpi.probe = p_probe;
	rpi.forward_id = _allocate_forward_id(FORWARD_ID_TYPE_REFLECTION_PROBE);

	return reflection_probe_instance_owner.make_rid(rpi);
}

void RendererSceneRenderRD::reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND(!rpi);

	rpi->transform = p_transform;
	rpi->dirty = true;
}

void RendererSceneRenderRD::reflection_probe_release_atlas_index(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND(!rpi);

	if (rpi->atlas.is_null()) {
		return; //nothing to release
	}
	ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(rpi->atlas);
	ERR_FAIL_COND(!atlas);
	ERR_FAIL_INDEX(rpi->atlas_index, atlas->reflections.size());
	atlas->reflections.write[rpi->atlas_index].owner = RID();
	rpi->atlas_index = -1;
	rpi->atlas = RID();
}

bool RendererSceneRenderRD::reflection_probe_instance_needs_redraw(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	if (rpi->rendering) {
		return false;
	}

	if (rpi->dirty) {
		return true;
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS) {
		return true;
	}

	return rpi->atlas_index == -1;
}

bool RendererSceneRenderRD::reflection_probe_instance_has_reflection(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	return rpi->atlas.is_valid();
}

bool RendererSceneRenderRD::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {
	ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(p_reflection_atlas);

	ERR_FAIL_COND_V(!atlas, false);

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	RD::get_singleton()->draw_command_begin_label("Reflection probe render");

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS && atlas->reflection.is_valid() && atlas->size != 256) {
		WARN_PRINT("ReflectionProbes set to UPDATE_ALWAYS must have an atlas size of 256. Please update the atlas size in the ProjectSettings.");
		reflection_atlas_set_size(p_reflection_atlas, 256, atlas->count);
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS && atlas->reflection.is_valid() && atlas->reflections[0].data.layers[0].mipmaps.size() != 8) {
		// Invalidate reflection atlas, need to regenerate
		RD::get_singleton()->free(atlas->reflection);
		atlas->reflection = RID();

		for (int i = 0; i < atlas->reflections.size(); i++) {
			if (atlas->reflections[i].owner.is_null()) {
				continue;
			}
			reflection_probe_release_atlas_index(atlas->reflections[i].owner);
		}

		atlas->reflections.clear();
	}

	if (atlas->reflection.is_null()) {
		int mipmaps = MIN(sky.roughness_layers, Image::get_image_required_mipmaps(atlas->size, atlas->size, Image::FORMAT_RGBAH) + 1);
		mipmaps = storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS ? 8 : mipmaps; // always use 8 mipmaps with real time filtering
		{
			//reflection atlas was unused, create:
			RD::TextureFormat tf;
			tf.array_layers = 6 * atlas->count;
			tf.format = _render_buffers_get_color_format();
			tf.texture_type = RD::TEXTURE_TYPE_CUBE_ARRAY;
			tf.mipmaps = mipmaps;
			tf.width = atlas->size;
			tf.height = atlas->size;
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | (_render_buffers_can_be_storage() ? RD::TEXTURE_USAGE_STORAGE_BIT : 0);

			atlas->reflection = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = atlas->size;
			tf.height = atlas->size;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			atlas->depth_buffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
		atlas->reflections.resize(atlas->count);
		for (int i = 0; i < atlas->count; i++) {
			atlas->reflections.write[i].data.update_reflection_data(storage, atlas->size, mipmaps, false, atlas->reflection, i * 6, storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS, sky.roughness_layers, _render_buffers_get_color_format());
			for (int j = 0; j < 6; j++) {
				atlas->reflections.write[i].fbs[j] = reflection_probe_create_framebuffer(atlas->reflections.write[i].data.layers[0].mipmaps[0].views[j], atlas->depth_buffer);
			}
		}

		Vector<RID> fb;
		fb.push_back(atlas->depth_buffer);
		atlas->depth_fb = RD::get_singleton()->framebuffer_create(fb);
	}

	if (rpi->atlas_index == -1) {
		for (int i = 0; i < atlas->reflections.size(); i++) {
			if (atlas->reflections[i].owner.is_null()) {
				rpi->atlas_index = i;
				break;
			}
		}
		//find the one used last
		if (rpi->atlas_index == -1) {
			//everything is in use, find the one least used via LRU
			uint64_t pass_min = 0;

			for (int i = 0; i < atlas->reflections.size(); i++) {
				ReflectionProbeInstance *rpi2 = reflection_probe_instance_owner.get_or_null(atlas->reflections[i].owner);
				if (rpi2->last_pass < pass_min) {
					pass_min = rpi2->last_pass;
					rpi->atlas_index = i;
				}
			}
		}
	}

	if (rpi->atlas_index != -1) { // should we fail if this is still -1 ?
		atlas->reflections.write[rpi->atlas_index].owner = p_instance;
	}

	rpi->atlas = p_reflection_atlas;
	rpi->rendering = true;
	rpi->dirty = false;
	rpi->processing_layer = 1;
	rpi->processing_side = 0;

	RD::get_singleton()->draw_command_end_label();

	return true;
}

RID RendererSceneRenderRD::reflection_probe_create_framebuffer(RID p_color, RID p_depth) {
	Vector<RID> fb;
	fb.push_back(p_color);
	fb.push_back(p_depth);
	return RD::get_singleton()->framebuffer_create(fb);
}

bool RendererSceneRenderRD::reflection_probe_instance_postprocess_step(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, false);
	ERR_FAIL_COND_V(!rpi->rendering, false);
	ERR_FAIL_COND_V(rpi->atlas.is_null(), false);

	ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(rpi->atlas);
	if (!atlas || rpi->atlas_index == -1) {
		//does not belong to an atlas anymore, cancel (was removed from atlas or atlas changed while rendering)
		rpi->rendering = false;
		return false;
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == RS::REFLECTION_PROBE_UPDATE_ALWAYS) {
		// Using real time reflections, all roughness is done in one step
		atlas->reflections.write[rpi->atlas_index].data.create_reflection_fast_filter(storage, false);
		rpi->rendering = false;
		rpi->processing_side = 0;
		rpi->processing_layer = 1;
		return true;
	}

	if (rpi->processing_layer > 1) {
		atlas->reflections.write[rpi->atlas_index].data.create_reflection_importance_sample(storage, false, 10, rpi->processing_layer, sky.sky_ggx_samples_quality);
		rpi->processing_layer++;
		if (rpi->processing_layer == atlas->reflections[rpi->atlas_index].data.layers[0].mipmaps.size()) {
			rpi->rendering = false;
			rpi->processing_side = 0;
			rpi->processing_layer = 1;
			return true;
		}
		return false;

	} else {
		atlas->reflections.write[rpi->atlas_index].data.create_reflection_importance_sample(storage, false, rpi->processing_side, rpi->processing_layer, sky.sky_ggx_samples_quality);
	}

	rpi->processing_side++;
	if (rpi->processing_side == 6) {
		rpi->processing_side = 0;
		rpi->processing_layer++;
	}

	return false;
}

uint32_t RendererSceneRenderRD::reflection_probe_instance_get_resolution(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, 0);

	ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, 0);
	return atlas->size;
}

RID RendererSceneRenderRD::reflection_probe_instance_get_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, RID());
	return atlas->reflections[rpi->atlas_index].fbs[p_index];
}

RID RendererSceneRenderRD::reflection_probe_instance_get_depth_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, RID());
	return atlas->depth_fb;
}

///////////////////////////////////////////////////////////

RID RendererSceneRenderRD::shadow_atlas_create() {
	return shadow_atlas_owner.make_rid(ShadowAtlas());
}

void RendererSceneRenderRD::_update_shadow_atlas(ShadowAtlas *shadow_atlas) {
	if (shadow_atlas->size > 0 && shadow_atlas->depth.is_null()) {
		RD::TextureFormat tf;
		tf.format = shadow_atlas->use_16_bits ? RD::DATA_FORMAT_D16_UNORM : RD::DATA_FORMAT_D32_SFLOAT;
		tf.width = shadow_atlas->size;
		tf.height = shadow_atlas->size;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		shadow_atlas->depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
		Vector<RID> fb_tex;
		fb_tex.push_back(shadow_atlas->depth);
		shadow_atlas->fb = RD::get_singleton()->framebuffer_create(fb_tex);
	}
}

void RendererSceneRenderRD::shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_COND(p_size < 0);
	p_size = next_power_of_2(p_size);

	if (p_size == shadow_atlas->size && p_16_bits == shadow_atlas->use_16_bits) {
		return;
	}

	// erasing atlas
	if (shadow_atlas->depth.is_valid()) {
		RD::get_singleton()->free(shadow_atlas->depth);
		shadow_atlas->depth = RID();
	}
	for (int i = 0; i < 4; i++) {
		//clear subdivisions
		shadow_atlas->quadrants[i].shadows.resize(0);
		shadow_atlas->quadrants[i].shadows.resize(1 << shadow_atlas->quadrants[i].subdivision);
	}

	//erase shadow atlas reference from lights
	for (const KeyValue<RID, uint32_t> &E : shadow_atlas->shadow_owners) {
		LightInstance *li = light_instance_owner.get_or_null(E.key);
		ERR_CONTINUE(!li);
		li->shadow_atlases.erase(p_atlas);
	}

	//clear owners
	shadow_atlas->shadow_owners.clear();

	shadow_atlas->size = p_size;
	shadow_atlas->use_16_bits = p_16_bits;
}

void RendererSceneRenderRD::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_INDEX(p_quadrant, 4);
	ERR_FAIL_INDEX(p_subdivision, 16384);

	uint32_t subdiv = next_power_of_2(p_subdivision);
	if (subdiv & 0xaaaaaaaa) { //sqrt(subdiv) must be integer
		subdiv <<= 1;
	}

	subdiv = int(Math::sqrt((float)subdiv));

	//obtain the number that will be x*x

	if (shadow_atlas->quadrants[p_quadrant].subdivision == subdiv) {
		return;
	}

	//erase all data from quadrant
	for (int i = 0; i < shadow_atlas->quadrants[p_quadrant].shadows.size(); i++) {
		if (shadow_atlas->quadrants[p_quadrant].shadows[i].owner.is_valid()) {
			shadow_atlas->shadow_owners.erase(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			LightInstance *li = light_instance_owner.get_or_null(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			ERR_CONTINUE(!li);
			li->shadow_atlases.erase(p_atlas);
		}
	}

	shadow_atlas->quadrants[p_quadrant].shadows.resize(0);
	shadow_atlas->quadrants[p_quadrant].shadows.resize(subdiv * subdiv);
	shadow_atlas->quadrants[p_quadrant].subdivision = subdiv;

	//cache the smallest subdiv (for faster allocation in light update)

	shadow_atlas->smallest_subdiv = 1 << 30;

	for (int i = 0; i < 4; i++) {
		if (shadow_atlas->quadrants[i].subdivision) {
			shadow_atlas->smallest_subdiv = MIN(shadow_atlas->smallest_subdiv, shadow_atlas->quadrants[i].subdivision);
		}
	}

	if (shadow_atlas->smallest_subdiv == 1 << 30) {
		shadow_atlas->smallest_subdiv = 0;
	}

	//resort the size orders, simple bublesort for 4 elements..

	int swaps = 0;
	do {
		swaps = 0;

		for (int i = 0; i < 3; i++) {
			if (shadow_atlas->quadrants[shadow_atlas->size_order[i]].subdivision < shadow_atlas->quadrants[shadow_atlas->size_order[i + 1]].subdivision) {
				SWAP(shadow_atlas->size_order[i], shadow_atlas->size_order[i + 1]);
				swaps++;
			}
		}
	} while (swaps > 0);
}

bool RendererSceneRenderRD::_shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow) {
	for (int i = p_quadrant_count - 1; i >= 0; i--) {
		int qidx = p_in_quadrants[i];

		if (shadow_atlas->quadrants[qidx].subdivision == (uint32_t)p_current_subdiv) {
			return false;
		}

		//look for an empty space
		int sc = shadow_atlas->quadrants[qidx].shadows.size();
		const ShadowAtlas::Quadrant::Shadow *sarr = shadow_atlas->quadrants[qidx].shadows.ptr();

		int found_free_idx = -1; //found a free one
		int found_used_idx = -1; //found existing one, must steal it
		uint64_t min_pass = 0; // pass of the existing one, try to use the least recently used one (LRU fashion)

		for (int j = 0; j < sc; j++) {
			if (!sarr[j].owner.is_valid()) {
				found_free_idx = j;
				break;
			}

			LightInstance *sli = light_instance_owner.get_or_null(sarr[j].owner);
			ERR_CONTINUE(!sli);

			if (sli->last_scene_pass != scene_pass) {
				//was just allocated, don't kill it so soon, wait a bit..
				if (p_tick - sarr[j].alloc_tick < shadow_atlas_realloc_tolerance_msec) {
					continue;
				}

				if (found_used_idx == -1 || sli->last_scene_pass < min_pass) {
					found_used_idx = j;
					min_pass = sli->last_scene_pass;
				}
			}
		}

		if (found_free_idx == -1 && found_used_idx == -1) {
			continue; //nothing found
		}

		if (found_free_idx == -1 && found_used_idx != -1) {
			found_free_idx = found_used_idx;
		}

		r_quadrant = qidx;
		r_shadow = found_free_idx;

		return true;
	}

	return false;
}

bool RendererSceneRenderRD::_shadow_atlas_find_omni_shadows(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow) {
	for (int i = p_quadrant_count - 1; i >= 0; i--) {
		int qidx = p_in_quadrants[i];

		if (shadow_atlas->quadrants[qidx].subdivision == (uint32_t)p_current_subdiv) {
			return false;
		}

		//look for an empty space
		int sc = shadow_atlas->quadrants[qidx].shadows.size();
		const ShadowAtlas::Quadrant::Shadow *sarr = shadow_atlas->quadrants[qidx].shadows.ptr();

		int found_idx = -1;
		uint64_t min_pass = 0; // sum of currently selected spots, try to get the least recently used pair

		for (int j = 0; j < sc - 1; j++) {
			uint64_t pass = 0;

			if (sarr[j].owner.is_valid()) {
				LightInstance *sli = light_instance_owner.get_or_null(sarr[j].owner);
				ERR_CONTINUE(!sli);

				if (sli->last_scene_pass == scene_pass) {
					continue;
				}

				//was just allocated, don't kill it so soon, wait a bit..
				if (p_tick - sarr[j].alloc_tick < shadow_atlas_realloc_tolerance_msec) {
					continue;
				}
				pass += sli->last_scene_pass;
			}

			if (sarr[j + 1].owner.is_valid()) {
				LightInstance *sli = light_instance_owner.get_or_null(sarr[j + 1].owner);
				ERR_CONTINUE(!sli);

				if (sli->last_scene_pass == scene_pass) {
					continue;
				}

				//was just allocated, don't kill it so soon, wait a bit..
				if (p_tick - sarr[j + 1].alloc_tick < shadow_atlas_realloc_tolerance_msec) {
					continue;
				}
				pass += sli->last_scene_pass;
			}

			if (found_idx == -1 || pass < min_pass) {
				found_idx = j;
				min_pass = pass;

				// we found two empty spots, no need to check the rest
				if (pass == 0) {
					break;
				}
			}
		}

		if (found_idx == -1) {
			continue; //nothing found
		}

		r_quadrant = qidx;
		r_shadow = found_idx;

		return true;
	}

	return false;
}

bool RendererSceneRenderRD::shadow_atlas_update_light(RID p_atlas, RID p_light_instance, float p_coverage, uint64_t p_light_version) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_atlas);
	ERR_FAIL_COND_V(!shadow_atlas, false);

	LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND_V(!li, false);

	if (shadow_atlas->size == 0 || shadow_atlas->smallest_subdiv == 0) {
		return false;
	}

	uint32_t quad_size = shadow_atlas->size >> 1;
	int desired_fit = MIN(quad_size / shadow_atlas->smallest_subdiv, next_power_of_2(quad_size * p_coverage));

	int valid_quadrants[4];
	int valid_quadrant_count = 0;
	int best_size = -1; //best size found
	int best_subdiv = -1; //subdiv for the best size

	//find the quadrants this fits into, and the best possible size it can fit into
	for (int i = 0; i < 4; i++) {
		int q = shadow_atlas->size_order[i];
		int sd = shadow_atlas->quadrants[q].subdivision;
		if (sd == 0) {
			continue; //unused
		}

		int max_fit = quad_size / sd;

		if (best_size != -1 && max_fit > best_size) {
			break; //too large
		}

		valid_quadrants[valid_quadrant_count++] = q;
		best_subdiv = sd;

		if (max_fit >= desired_fit) {
			best_size = max_fit;
		}
	}

	ERR_FAIL_COND_V(valid_quadrant_count == 0, false);

	uint64_t tick = OS::get_singleton()->get_ticks_msec();

	uint32_t old_key = ShadowAtlas::SHADOW_INVALID;
	uint32_t old_quadrant = ShadowAtlas::SHADOW_INVALID;
	uint32_t old_shadow = ShadowAtlas::SHADOW_INVALID;
	int old_subdivision = -1;

	bool should_realloc = false;
	bool should_redraw = false;

	if (shadow_atlas->shadow_owners.has(p_light_instance)) {
		old_key = shadow_atlas->shadow_owners[p_light_instance];
		old_quadrant = (old_key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		old_shadow = old_key & ShadowAtlas::SHADOW_INDEX_MASK;

		should_realloc = shadow_atlas->quadrants[old_quadrant].subdivision != (uint32_t)best_subdiv && (shadow_atlas->quadrants[old_quadrant].shadows[old_shadow].alloc_tick - tick > shadow_atlas_realloc_tolerance_msec);
		should_redraw = shadow_atlas->quadrants[old_quadrant].shadows[old_shadow].version != p_light_version;

		if (!should_realloc) {
			shadow_atlas->quadrants[old_quadrant].shadows.write[old_shadow].version = p_light_version;
			//already existing, see if it should redraw or it's just OK
			return should_redraw;
		}

		old_subdivision = shadow_atlas->quadrants[old_quadrant].subdivision;
	}

	bool is_omni = li->light_type == RS::LIGHT_OMNI;
	bool found_shadow = false;
	int new_quadrant = -1;
	int new_shadow = -1;

	if (is_omni) {
		found_shadow = _shadow_atlas_find_omni_shadows(shadow_atlas, valid_quadrants, valid_quadrant_count, old_subdivision, tick, new_quadrant, new_shadow);
	} else {
		found_shadow = _shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, old_subdivision, tick, new_quadrant, new_shadow);
	}

	if (found_shadow) {
		if (old_quadrant != ShadowAtlas::SHADOW_INVALID) {
			shadow_atlas->quadrants[old_quadrant].shadows.write[old_shadow].version = 0;
			shadow_atlas->quadrants[old_quadrant].shadows.write[old_shadow].owner = RID();

			if (old_key & ShadowAtlas::OMNI_LIGHT_FLAG) {
				shadow_atlas->quadrants[old_quadrant].shadows.write[old_shadow + 1].version = 0;
				shadow_atlas->quadrants[old_quadrant].shadows.write[old_shadow + 1].owner = RID();
			}
		}

		uint32_t new_key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
		new_key |= new_shadow;

		ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
		_shadow_atlas_invalidate_shadow(sh, p_atlas, shadow_atlas, new_quadrant, new_shadow);

		sh->owner = p_light_instance;
		sh->alloc_tick = tick;
		sh->version = p_light_version;

		if (is_omni) {
			new_key |= ShadowAtlas::OMNI_LIGHT_FLAG;

			int new_omni_shadow = new_shadow + 1;
			ShadowAtlas::Quadrant::Shadow *extra_sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_omni_shadow];
			_shadow_atlas_invalidate_shadow(extra_sh, p_atlas, shadow_atlas, new_quadrant, new_omni_shadow);

			extra_sh->owner = p_light_instance;
			extra_sh->alloc_tick = tick;
			extra_sh->version = p_light_version;
		}

		li->shadow_atlases.insert(p_atlas);

		//update it in map
		shadow_atlas->shadow_owners[p_light_instance] = new_key;
		//make it dirty, as it should redraw anyway
		return true;
	}

	return should_redraw;
}

void RendererSceneRenderRD::_shadow_atlas_invalidate_shadow(RendererSceneRenderRD::ShadowAtlas::Quadrant::Shadow *p_shadow, RID p_atlas, RendererSceneRenderRD::ShadowAtlas *p_shadow_atlas, uint32_t p_quadrant, uint32_t p_shadow_idx) {
	if (p_shadow->owner.is_valid()) {
		LightInstance *sli = light_instance_owner.get_or_null(p_shadow->owner);
		uint32_t old_key = p_shadow_atlas->shadow_owners[p_shadow->owner];

		if (old_key & ShadowAtlas::OMNI_LIGHT_FLAG) {
			uint32_t s = old_key & ShadowAtlas::SHADOW_INDEX_MASK;
			uint32_t omni_shadow_idx = p_shadow_idx + (s == (uint32_t)p_shadow_idx ? 1 : -1);
			RendererSceneRenderRD::ShadowAtlas::Quadrant::Shadow *omni_shadow = &p_shadow_atlas->quadrants[p_quadrant].shadows.write[omni_shadow_idx];
			omni_shadow->version = 0;
			omni_shadow->owner = RID();
		}

		p_shadow_atlas->shadow_owners.erase(p_shadow->owner);
		p_shadow->version = 0;
		p_shadow->owner = RID();
		sli->shadow_atlases.erase(p_atlas);
	}
}

void RendererSceneRenderRD::_update_directional_shadow_atlas() {
	if (directional_shadow.depth.is_null() && directional_shadow.size > 0) {
		RD::TextureFormat tf;
		tf.format = directional_shadow.use_16_bits ? RD::DATA_FORMAT_D16_UNORM : RD::DATA_FORMAT_D32_SFLOAT;
		tf.width = directional_shadow.size;
		tf.height = directional_shadow.size;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		directional_shadow.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
		Vector<RID> fb_tex;
		fb_tex.push_back(directional_shadow.depth);
		directional_shadow.fb = RD::get_singleton()->framebuffer_create(fb_tex);
	}
}
void RendererSceneRenderRD::directional_shadow_atlas_set_size(int p_size, bool p_16_bits) {
	p_size = nearest_power_of_2_templated(p_size);

	if (directional_shadow.size == p_size && directional_shadow.use_16_bits == p_16_bits) {
		return;
	}

	directional_shadow.size = p_size;
	directional_shadow.use_16_bits = p_16_bits;

	if (directional_shadow.depth.is_valid()) {
		RD::get_singleton()->free(directional_shadow.depth);
		directional_shadow.depth = RID();
		_base_uniforms_changed();
	}
}

void RendererSceneRenderRD::set_directional_shadow_count(int p_count) {
	directional_shadow.light_count = p_count;
	directional_shadow.current_light = 0;
}

static Rect2i _get_directional_shadow_rect(int p_size, int p_shadow_count, int p_shadow_index) {
	int split_h = 1;
	int split_v = 1;

	while (split_h * split_v < p_shadow_count) {
		if (split_h == split_v) {
			split_h <<= 1;
		} else {
			split_v <<= 1;
		}
	}

	Rect2i rect(0, 0, p_size, p_size);
	rect.size.width /= split_h;
	rect.size.height /= split_v;

	rect.position.x = rect.size.width * (p_shadow_index % split_h);
	rect.position.y = rect.size.height * (p_shadow_index / split_h);

	return rect;
}

int RendererSceneRenderRD::get_directional_light_shadow_size(RID p_light_intance) {
	ERR_FAIL_COND_V(directional_shadow.light_count == 0, 0);

	Rect2i r = _get_directional_shadow_rect(directional_shadow.size, directional_shadow.light_count, 0);

	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_intance);
	ERR_FAIL_COND_V(!light_instance, 0);

	switch (storage->light_directional_get_shadow_mode(light_instance->light)) {
		case RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL:
			break; //none
		case RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS:
			r.size.height /= 2;
			break;
		case RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS:
			r.size /= 2;
			break;
	}

	return MAX(r.size.width, r.size.height);
}

//////////////////////////////////////////////////

RID RendererSceneRenderRD::camera_effects_allocate() {
	return camera_effects_owner.allocate_rid();
}
void RendererSceneRenderRD::camera_effects_initialize(RID p_rid) {
	camera_effects_owner.initialize_rid(p_rid, CameraEffects());
}

void RendererSceneRenderRD::camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) {
	dof_blur_quality = p_quality;
	dof_blur_use_jitter = p_use_jitter;
}

void RendererSceneRenderRD::camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) {
	dof_blur_bokeh_shape = p_shape;
}

void RendererSceneRenderRD::camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {
	CameraEffects *camfx = camera_effects_owner.get_or_null(p_camera_effects);
	ERR_FAIL_COND(!camfx);

	camfx->dof_blur_far_enabled = p_far_enable;
	camfx->dof_blur_far_distance = p_far_distance;
	camfx->dof_blur_far_transition = p_far_transition;

	camfx->dof_blur_near_enabled = p_near_enable;
	camfx->dof_blur_near_distance = p_near_distance;
	camfx->dof_blur_near_transition = p_near_transition;

	camfx->dof_blur_amount = p_amount;
}

void RendererSceneRenderRD::camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) {
	CameraEffects *camfx = camera_effects_owner.get_or_null(p_camera_effects);
	ERR_FAIL_COND(!camfx);

	camfx->override_exposure_enabled = p_enable;
	camfx->override_exposure = p_exposure;
}

RID RendererSceneRenderRD::light_instance_create(RID p_light) {
	RID li = light_instance_owner.make_rid(LightInstance());

	LightInstance *light_instance = light_instance_owner.get_or_null(li);

	light_instance->self = li;
	light_instance->light = p_light;
	light_instance->light_type = storage->light_get_type(p_light);
	if (light_instance->light_type != RS::LIGHT_DIRECTIONAL) {
		light_instance->forward_id = _allocate_forward_id(light_instance->light_type == RS::LIGHT_OMNI ? FORWARD_ID_TYPE_OMNI_LIGHT : FORWARD_ID_TYPE_SPOT_LIGHT);
	}

	return li;
}

void RendererSceneRenderRD::light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->transform = p_transform;
}

void RendererSceneRenderRD::light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->aabb = p_aabb;
}

void RendererSceneRenderRD::light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale, float p_range_begin, const Vector2 &p_uv_scale) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	ERR_FAIL_INDEX(p_pass, 6);

	light_instance->shadow_transform[p_pass].camera = p_projection;
	light_instance->shadow_transform[p_pass].transform = p_transform;
	light_instance->shadow_transform[p_pass].farplane = p_far;
	light_instance->shadow_transform[p_pass].split = p_split;
	light_instance->shadow_transform[p_pass].bias_scale = p_bias_scale;
	light_instance->shadow_transform[p_pass].range_begin = p_range_begin;
	light_instance->shadow_transform[p_pass].shadow_texel_size = p_shadow_texel_size;
	light_instance->shadow_transform[p_pass].uv_scale = p_uv_scale;
}

void RendererSceneRenderRD::light_instance_mark_visible(RID p_light_instance) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->last_scene_pass = scene_pass;
}

RendererSceneRenderRD::ShadowCubemap *RendererSceneRenderRD::_get_shadow_cubemap(int p_size) {
	if (!shadow_cubemaps.has(p_size)) {
		ShadowCubemap sc;
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = p_size;
			tf.height = p_size;
			tf.texture_type = RD::TEXTURE_TYPE_CUBE;
			tf.array_layers = 6;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			sc.cubemap = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		for (int i = 0; i < 6; i++) {
			RID side_texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), sc.cubemap, i, 0);
			Vector<RID> fbtex;
			fbtex.push_back(side_texture);
			sc.side_fb[i] = RD::get_singleton()->framebuffer_create(fbtex);
		}

		shadow_cubemaps[p_size] = sc;
	}

	return &shadow_cubemaps[p_size];
}

//////////////////////////

RID RendererSceneRenderRD::decal_instance_create(RID p_decal) {
	DecalInstance di;
	di.decal = p_decal;
	di.forward_id = _allocate_forward_id(FORWARD_ID_TYPE_DECAL);
	return decal_instance_owner.make_rid(di);
}

void RendererSceneRenderRD::decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) {
	DecalInstance *di = decal_instance_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!di);
	di->transform = p_transform;
}

/////////////////////////////////

RID RendererSceneRenderRD::lightmap_instance_create(RID p_lightmap) {
	LightmapInstance li;
	li.lightmap = p_lightmap;
	return lightmap_instance_owner.make_rid(li);
}
void RendererSceneRenderRD::lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) {
	LightmapInstance *li = lightmap_instance_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!li);
	li->transform = p_transform;
}

/////////////////////////////////

RID RendererSceneRenderRD::voxel_gi_instance_create(RID p_base) {
	return gi.voxel_gi_instance_create(p_base);
}

void RendererSceneRenderRD::voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) {
	gi.voxel_gi_instance_set_transform_to_data(p_probe, p_xform);
}

bool RendererSceneRenderRD::voxel_gi_needs_update(RID p_probe) const {
	if (!is_dynamic_gi_supported()) {
		return false;
	}

	return gi.voxel_gi_needs_update(p_probe);
}

void RendererSceneRenderRD::voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<GeometryInstance *> &p_dynamic_objects) {
	if (!is_dynamic_gi_supported()) {
		return;
	}

	gi.voxel_gi_update(p_probe, p_update_light_instances, p_light_instances, p_dynamic_objects, this);
}

void RendererSceneRenderRD::_debug_sdfgi_probes(RID p_render_buffers, RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (!rb->sdfgi) {
		return; //nothing to debug
	}

	rb->sdfgi->debug_probes(p_draw_list, p_framebuffer, p_camera_with_transform);
}

////////////////////////////////
RID RendererSceneRenderRD::render_buffers_create() {
	RenderBuffers rb;
	rb.data = _create_render_buffer_data();
	return render_buffers_owner.make_rid(rb);
}

void RendererSceneRenderRD::_allocate_blur_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->blur[0].texture.is_null());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rb->width, rb->height, Image::FORMAT_RGBAH);

	// TODO make sure texture_create_shared_from_slice works for multiview

	RD::TextureFormat tf;
	tf.format = _render_buffers_get_color_format(); // RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	tf.width = rb->internal_width;
	tf.height = rb->internal_height;
	tf.texture_type = rb->view_count > 1 ? RD::TEXTURE_TYPE_2D_ARRAY : RD::TEXTURE_TYPE_2D;
	tf.array_layers = rb->view_count;
	tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	if (_render_buffers_can_be_storage()) {
		tf.usage_bits += RD::TEXTURE_USAGE_STORAGE_BIT;
	} else {
		tf.usage_bits += RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	}
	tf.mipmaps = mipmaps_required;

	rb->sss_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.width = rb->internal_width;
	tf.height = rb->internal_height;
	rb->blur[0].texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	//the second one is smaller (only used for separatable part of blur)
	tf.width >>= 1;
	tf.height >>= 1;
	tf.mipmaps--;
	rb->blur[1].texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

	int base_width = rb->internal_width;
	int base_height = rb->internal_height;

	for (uint32_t i = 0; i < mipmaps_required; i++) {
		RenderBuffers::Blur::Mipmap mm;
		mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->blur[0].texture, 0, i);

		mm.width = base_width;
		mm.height = base_height;

		if (!_render_buffers_can_be_storage()) {
			Vector<RID> fb;
			fb.push_back(mm.texture);

			mm.fb = RD::get_singleton()->framebuffer_create(fb);
		}

		if (!_render_buffers_can_be_storage()) {
			// and half texture, this is an intermediate result so just allocate a texture, is this good enough?
			tf.width = MAX(1, base_width >> 1);
			tf.height = base_height;
			tf.mipmaps = 1; // 1 or 0?

			mm.half_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

			Vector<RID> half_fb;
			half_fb.push_back(mm.half_texture);
			mm.half_fb = RD::get_singleton()->framebuffer_create(half_fb);
		}

		rb->blur[0].mipmaps.push_back(mm);

		if (i > 0) {
			mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->blur[1].texture, 0, i - 1);

			if (!_render_buffers_can_be_storage()) {
				Vector<RID> fb;
				fb.push_back(mm.texture);

				mm.fb = RD::get_singleton()->framebuffer_create(fb);

				// We can re-use the half texture here as it is an intermediate result
			}

			rb->blur[1].mipmaps.push_back(mm);
		}

		base_width = MAX(1, base_width >> 1);
		base_height = MAX(1, base_height >> 1);
	}

	if (!_render_buffers_can_be_storage()) {
		// create 4 weight textures, 2 full size, 2 half size

		tf.format = RD::DATA_FORMAT_R16_SFLOAT; // We could probably use DATA_FORMAT_R8_SNORM if we don't pre-multiply by blur_size but that depends on whether we can remove DEPTH_GAP
		tf.width = rb->internal_width;
		tf.height = rb->internal_height;
		tf.texture_type = rb->view_count > 1 ? RD::TEXTURE_TYPE_2D_ARRAY : RD::TEXTURE_TYPE_2D;
		tf.array_layers = rb->view_count;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		tf.mipmaps = 1;
		for (uint32_t i = 0; i < 4; i++) {
			// associated blur texture
			RID texture;
			if (i == 0) {
				texture = rb->texture;
			} else if (i == 1) {
				texture = rb->blur[0].mipmaps[0].texture;
			} else if (i == 2) {
				texture = rb->blur[1].mipmaps[0].texture;
			} else if (i == 3) {
				texture = rb->blur[0].mipmaps[1].texture;
			}

			// create weight texture
			rb->weight_buffers[i].weight = RD::get_singleton()->texture_create(tf, RD::TextureView());

			// create frame buffer
			Vector<RID> fb;
			fb.push_back(texture);
			fb.push_back(rb->weight_buffers[i].weight);
			rb->weight_buffers[i].fb = RD::get_singleton()->framebuffer_create(fb);

			if (i == 1) {
				// next 2 are half size
				tf.width = MAX(1, tf.width >> 1);
				tf.height = MAX(1, tf.height >> 1);
			}
		}

		{
			// and finally an FB for just our base weights
			Vector<RID> fb;
			fb.push_back(rb->weight_buffers[0].weight);
			rb->base_weight_fb = RD::get_singleton()->framebuffer_create(fb);
		}
	}
}

void RendererSceneRenderRD::_allocate_depth_backbuffer_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->depth_back_texture.is_null());

	{
		RD::TextureFormat tf;
		if (rb->view_count > 1) {
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		}
		// We're not using this as a depth stencil, just copying our data into this. May need to look into using a different format on mobile, maybe R16?
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;

		tf.width = rb->width;
		tf.height = rb->height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
		tf.array_layers = rb->view_count; // create a layer for every view

		tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		tf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT; // set this as color attachment because we're copying data into it, it's not actually used as a depth buffer

		rb->depth_back_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	if (!_render_buffers_can_be_storage()) {
		// create framebuffer so we can write into this...

		Vector<RID> fb;
		fb.push_back(rb->depth_back_texture);

		rb->depth_back_fb = RD::get_singleton()->framebuffer_create(fb, RD::INVALID_ID, rb->view_count);
	}
}

void RendererSceneRenderRD::_allocate_luminance_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->luminance.current.is_null());

	int w = rb->internal_width;
	int h = rb->internal_height;

	while (true) {
		w = MAX(w / 8, 1);
		h = MAX(h / 8, 1);

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = w;
		tf.height = h;

		bool final = w == 1 && h == 1;

		if (_render_buffers_can_be_storage()) {
			tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;
			if (final) {
				tf.usage_bits |= RD::TEXTURE_USAGE_SAMPLING_BIT;
			}
		} else {
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
		}

		RID texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		rb->luminance.reduce.push_back(texture);
		if (!_render_buffers_can_be_storage()) {
			Vector<RID> fb;
			fb.push_back(texture);

			rb->luminance.fb.push_back(RD::get_singleton()->framebuffer_create(fb));
		}

		if (final) {
			rb->luminance.current = RD::get_singleton()->texture_create(tf, RD::TextureView());

			if (!_render_buffers_can_be_storage()) {
				Vector<RID> fb;
				fb.push_back(rb->luminance.current);

				rb->luminance.current_fb = RD::get_singleton()->framebuffer_create(fb);
			}
			break;
		}
	}
}

void RendererSceneRenderRD::_free_render_buffer_data(RenderBuffers *rb) {
	if (rb->texture_fb.is_valid()) {
		RD::get_singleton()->free(rb->texture_fb);
		rb->texture_fb = RID();
	}

	if (rb->internal_texture == rb->texture && rb->internal_texture.is_valid()) {
		RD::get_singleton()->free(rb->internal_texture);
		rb->texture = RID();
		rb->internal_texture = RID();
		rb->upscale_texture = RID();
	} else {
		if (rb->texture.is_valid()) {
			RD::get_singleton()->free(rb->texture);
			rb->texture = RID();
		}

		if (rb->internal_texture.is_valid()) {
			RD::get_singleton()->free(rb->internal_texture);
			rb->internal_texture = RID();
		}

		if (rb->upscale_texture.is_valid()) {
			RD::get_singleton()->free(rb->upscale_texture);
			rb->upscale_texture = RID();
		}
	}

	if (rb->depth_texture.is_valid()) {
		RD::get_singleton()->free(rb->depth_texture);
		rb->depth_texture = RID();
	}

	if (rb->depth_back_fb.is_valid()) {
		RD::get_singleton()->free(rb->depth_back_fb);
		rb->depth_back_fb = RID();
	}

	if (rb->depth_back_texture.is_valid()) {
		RD::get_singleton()->free(rb->depth_back_texture);
		rb->depth_back_texture = RID();
	}

	if (rb->sss_texture.is_valid()) {
		RD::get_singleton()->free(rb->sss_texture);
		rb->sss_texture = RID();
	}

	for (int i = 0; i < 2; i++) {
		for (int m = 0; m < rb->blur[i].mipmaps.size(); m++) {
			// do we free the texture slice here? or is it enough to free the main texture?

			// do free the mobile extra stuff
			if (rb->blur[i].mipmaps[m].fb.is_valid()) {
				RD::get_singleton()->free(rb->blur[i].mipmaps[m].fb);
			}
			if (rb->blur[i].mipmaps[m].half_fb.is_valid()) {
				RD::get_singleton()->free(rb->blur[i].mipmaps[m].half_fb);
			}
			if (rb->blur[i].mipmaps[m].half_texture.is_valid()) {
				RD::get_singleton()->free(rb->blur[i].mipmaps[m].half_texture);
			}
		}
		rb->blur[i].mipmaps.clear();

		if (rb->blur[i].texture.is_valid()) {
			RD::get_singleton()->free(rb->blur[i].texture);
			rb->blur[i].texture = RID();
		}
	}

	for (int i = 0; i < rb->luminance.fb.size(); i++) {
		RD::get_singleton()->free(rb->luminance.fb[i]);
	}
	rb->luminance.fb.clear();

	for (int i = 0; i < rb->luminance.reduce.size(); i++) {
		RD::get_singleton()->free(rb->luminance.reduce[i]);
	}
	rb->luminance.reduce.clear();

	if (rb->luminance.current_fb.is_valid()) {
		RD::get_singleton()->free(rb->luminance.current_fb);
		rb->luminance.current_fb = RID();
	}

	if (rb->luminance.current.is_valid()) {
		RD::get_singleton()->free(rb->luminance.current);
		rb->luminance.current = RID();
	}

	if (rb->ss_effects.linear_depth.is_valid()) {
		RD::get_singleton()->free(rb->ss_effects.linear_depth);
		rb->ss_effects.linear_depth = RID();
		rb->ss_effects.linear_depth_slices.clear();
	}

	if (rb->ss_effects.ssao.ao_final.is_valid()) {
		RD::get_singleton()->free(rb->ss_effects.ssao.ao_deinterleaved);
		RD::get_singleton()->free(rb->ss_effects.ssao.ao_pong);
		RD::get_singleton()->free(rb->ss_effects.ssao.ao_final);

		RD::get_singleton()->free(rb->ss_effects.ssao.importance_map[0]);
		RD::get_singleton()->free(rb->ss_effects.ssao.importance_map[1]);

		rb->ss_effects.ssao.ao_deinterleaved = RID();
		rb->ss_effects.ssao.ao_pong = RID();
		rb->ss_effects.ssao.ao_final = RID();
		rb->ss_effects.ssao.importance_map[0] = RID();
		rb->ss_effects.ssao.importance_map[1] = RID();

		rb->ss_effects.ssao.ao_deinterleaved_slices.clear();
		rb->ss_effects.ssao.ao_pong_slices.clear();
	}

	if (rb->ss_effects.ssil.ssil_final.is_valid()) {
		RD::get_singleton()->free(rb->ss_effects.ssil.ssil_final);
		RD::get_singleton()->free(rb->ss_effects.ssil.deinterleaved);
		RD::get_singleton()->free(rb->ss_effects.ssil.pong);
		RD::get_singleton()->free(rb->ss_effects.ssil.edges);
		RD::get_singleton()->free(rb->ss_effects.ssil.importance_map[0]);
		RD::get_singleton()->free(rb->ss_effects.ssil.importance_map[1]);

		rb->ss_effects.ssil.ssil_final = RID();
		rb->ss_effects.ssil.deinterleaved = RID();
		rb->ss_effects.ssil.pong = RID();
		rb->ss_effects.ssil.edges = RID();
		rb->ss_effects.ssil.deinterleaved_slices.clear();
		rb->ss_effects.ssil.pong_slices.clear();
		rb->ss_effects.ssil.edges_slices.clear();
		rb->ss_effects.ssil.importance_map[0] = RID();
		rb->ss_effects.ssil.importance_map[1] = RID();

		RD::get_singleton()->free(rb->ss_effects.last_frame);
		rb->ss_effects.last_frame = RID();
		rb->ss_effects.last_frame_slices.clear();
	}

	if (rb->ssr.blur_radius[0].is_valid()) {
		RD::get_singleton()->free(rb->ssr.blur_radius[0]);
		RD::get_singleton()->free(rb->ssr.blur_radius[1]);
		rb->ssr.blur_radius[0] = RID();
		rb->ssr.blur_radius[1] = RID();
	}

	if (rb->ssr.depth_scaled.is_valid()) {
		RD::get_singleton()->free(rb->ssr.depth_scaled);
		rb->ssr.depth_scaled = RID();
		RD::get_singleton()->free(rb->ssr.normal_scaled);
		rb->ssr.normal_scaled = RID();
	}

	if (rb->ambient_buffer.is_valid()) {
		RD::get_singleton()->free(rb->ambient_buffer);
		RD::get_singleton()->free(rb->reflection_buffer);
		rb->ambient_buffer = RID();
		rb->reflection_buffer = RID();
	}
}

void RendererSceneRenderRD::_process_sss(RID p_render_buffers, const CameraMatrix &p_camera) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	bool can_use_effects = rb->internal_width >= 8 && rb->internal_height >= 8;

	if (!can_use_effects) {
		//just copy
		return;
	}

	if (rb->blur[0].texture.is_null()) {
		_allocate_blur_textures(rb);
	}

	storage->get_effects()->sub_surface_scattering(rb->internal_texture, rb->sss_texture, rb->depth_texture, p_camera, Size2i(rb->internal_width, rb->internal_height), sss_scale, sss_depth_scale, sss_quality);
}

void RendererSceneRenderRD::_process_ssr(RID p_render_buffers, RID p_dest_framebuffer, RID p_normal_buffer, RID p_specular_buffer, RID p_metallic, const Color &p_metallic_mask, RID p_environment, const CameraMatrix &p_projection, bool p_use_additive) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	bool can_use_effects = rb->internal_width >= 8 && rb->internal_height >= 8;

	if (!can_use_effects) {
		//just copy
		storage->get_effects()->merge_specular(p_dest_framebuffer, p_specular_buffer, p_use_additive ? RID() : rb->internal_texture, RID());
		return;
	}

	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_environment);
	ERR_FAIL_COND(!env);

	ERR_FAIL_COND(!env->ssr_enabled);

	if (rb->ssr.depth_scaled.is_null()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = rb->internal_width / 2;
		tf.height = rb->internal_height / 2;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		rb->ssr.depth_scaled = RD::get_singleton()->texture_create(tf, RD::TextureView());

		tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;

		rb->ssr.normal_scaled = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	if (ssr_roughness_quality != RS::ENV_SSR_ROUGNESS_QUALITY_DISABLED && !rb->ssr.blur_radius[0].is_valid()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R8_UNORM;
		tf.width = rb->internal_width / 2;
		tf.height = rb->internal_height / 2;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

		rb->ssr.blur_radius[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
		rb->ssr.blur_radius[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	if (rb->blur[0].texture.is_null()) {
		_allocate_blur_textures(rb);
	}

	storage->get_effects()->screen_space_reflection(rb->internal_texture, p_normal_buffer, ssr_roughness_quality, rb->ssr.blur_radius[0], rb->ssr.blur_radius[1], p_metallic, p_metallic_mask, rb->depth_texture, rb->ssr.depth_scaled, rb->ssr.normal_scaled, rb->blur[0].mipmaps[1].texture, rb->blur[1].mipmaps[0].texture, Size2i(rb->internal_width / 2, rb->internal_height / 2), env->ssr_max_steps, env->ssr_fade_in, env->ssr_fade_out, env->ssr_depth_tolerance, p_projection);
	storage->get_effects()->merge_specular(p_dest_framebuffer, p_specular_buffer, p_use_additive ? RID() : rb->internal_texture, rb->blur[0].mipmaps[1].texture);
}

void RendererSceneRenderRD::_process_ssao(RID p_render_buffers, RID p_environment, RID p_normal_buffer, const CameraMatrix &p_projection) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_environment);
	ERR_FAIL_COND(!env);

	RENDER_TIMESTAMP("Process SSAO");

	if (rb->ss_effects.ssao.ao_final.is_valid() && ssao_using_half_size != ssao_half_size) {
		RD::get_singleton()->free(rb->ss_effects.ssao.ao_deinterleaved);
		RD::get_singleton()->free(rb->ss_effects.ssao.ao_pong);
		RD::get_singleton()->free(rb->ss_effects.ssao.ao_final);

		RD::get_singleton()->free(rb->ss_effects.ssao.importance_map[0]);
		RD::get_singleton()->free(rb->ss_effects.ssao.importance_map[1]);

		rb->ss_effects.ssao.ao_deinterleaved = RID();
		rb->ss_effects.ssao.ao_pong = RID();
		rb->ss_effects.ssao.ao_final = RID();
		rb->ss_effects.ssao.importance_map[0] = RID();
		rb->ss_effects.ssao.importance_map[1] = RID();
		rb->ss_effects.ssao.ao_deinterleaved_slices.clear();
		rb->ss_effects.ssao.ao_pong_slices.clear();
	}

	int buffer_width;
	int buffer_height;
	int half_width;
	int half_height;
	if (ssao_half_size) {
		buffer_width = (rb->internal_width + 3) / 4;
		buffer_height = (rb->internal_height + 3) / 4;
		half_width = (rb->internal_width + 7) / 8;
		half_height = (rb->internal_height + 7) / 8;
	} else {
		buffer_width = (rb->internal_width + 1) / 2;
		buffer_height = (rb->internal_height + 1) / 2;
		half_width = (rb->internal_width + 3) / 4;
		half_height = (rb->internal_height + 3) / 4;
	}
	bool uniform_sets_are_invalid = false;
	if (rb->ss_effects.ssao.ao_deinterleaved.is_null()) {
		{
			rb->ss_effects.ssao.depth_texture_view = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.linear_depth, 0, ssao_half_size ? 1 : 0, 4, RD::TEXTURE_SLICE_2D_ARRAY);
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8_UNORM;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = buffer_width;
			tf.height = buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssao.ao_deinterleaved = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssao.ao_deinterleaved, "SSAO De-interleaved Array");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.ssao.ao_deinterleaved, i, 0);
				rb->ss_effects.ssao.ao_deinterleaved_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSAO De-interleaved Array Layer " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8_UNORM;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = buffer_width;
			tf.height = buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssao.ao_pong = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssao.ao_pong, "SSAO De-interleaved Array Pong");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.ssao.ao_pong, i, 0);
				rb->ss_effects.ssao.ao_pong_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSAO De-interleaved Array Layer " + itos(i) + " Pong");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = half_width;
			tf.height = half_height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssao.importance_map[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssao.importance_map[0], "SSAO Importance Map");
			rb->ss_effects.ssao.importance_map[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssao.importance_map[1], "SSAO Importance Map Pong");
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = rb->internal_width;
			tf.height = rb->internal_height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssao.ao_final = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssao.ao_final, "SSAO Final");
		}
		ssao_using_half_size = ssao_half_size;
		uniform_sets_are_invalid = true;
	}

	EffectsRD::SSAOSettings settings;
	settings.radius = env->ssao_radius;
	settings.intensity = env->ssao_intensity;
	settings.power = env->ssao_power;
	settings.detail = env->ssao_detail;
	settings.horizon = env->ssao_horizon;
	settings.sharpness = env->ssao_sharpness;

	settings.quality = ssao_quality;
	settings.half_size = ssao_half_size;
	settings.adaptive_target = ssao_adaptive_target;
	settings.blur_passes = ssao_blur_passes;
	settings.fadeout_from = ssao_fadeout_from;
	settings.fadeout_to = ssao_fadeout_to;
	settings.full_screen_size = Size2i(rb->internal_width, rb->internal_height);
	settings.half_screen_size = Size2i(buffer_width, buffer_height);
	settings.quarter_screen_size = Size2i(half_width, half_height);

	storage->get_effects()->generate_ssao(p_normal_buffer, rb->ss_effects.ssao.depth_texture_view, rb->ss_effects.ssao.ao_deinterleaved, rb->ss_effects.ssao.ao_deinterleaved_slices, rb->ss_effects.ssao.ao_pong, rb->ss_effects.ssao.ao_pong_slices, rb->ss_effects.ssao.ao_final, rb->ss_effects.ssao.importance_map[0], rb->ss_effects.ssao.importance_map[1], p_projection, settings, uniform_sets_are_invalid, rb->ss_effects.ssao.gather_uniform_set, rb->ss_effects.ssao.importance_map_uniform_set);
}

void RendererSceneRenderRD::_process_ssil(RID p_render_buffers, RID p_environment, RID p_normal_buffer, const CameraMatrix &p_projection, const Transform3D &p_transform) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_environment);
	ERR_FAIL_COND(!env);

	RENDER_TIMESTAMP("Process SSIL");

	if (rb->ss_effects.ssil.ssil_final.is_valid() && ssil_using_half_size != ssil_half_size) {
		RD::get_singleton()->free(rb->ss_effects.ssil.ssil_final);
		RD::get_singleton()->free(rb->ss_effects.ssil.deinterleaved);
		RD::get_singleton()->free(rb->ss_effects.ssil.pong);
		RD::get_singleton()->free(rb->ss_effects.ssil.edges);
		RD::get_singleton()->free(rb->ss_effects.ssil.importance_map[0]);
		RD::get_singleton()->free(rb->ss_effects.ssil.importance_map[1]);

		rb->ss_effects.ssil.ssil_final = RID();
		rb->ss_effects.ssil.deinterleaved = RID();
		rb->ss_effects.ssil.pong = RID();
		rb->ss_effects.ssil.edges = RID();
		rb->ss_effects.ssil.deinterleaved_slices.clear();
		rb->ss_effects.ssil.pong_slices.clear();
		rb->ss_effects.ssil.edges_slices.clear();
		rb->ss_effects.ssil.importance_map[0] = RID();
		rb->ss_effects.ssil.importance_map[1] = RID();
	}

	int buffer_width;
	int buffer_height;
	int half_width;
	int half_height;
	if (ssil_half_size) {
		buffer_width = (rb->width + 3) / 4;
		buffer_height = (rb->height + 3) / 4;
		half_width = (rb->width + 7) / 8;
		half_height = (rb->height + 7) / 8;
	} else {
		buffer_width = (rb->width + 1) / 2;
		buffer_height = (rb->height + 1) / 2;
		half_width = (rb->width + 3) / 4;
		half_height = (rb->height + 3) / 4;
	}
	bool uniform_sets_are_invalid = false;
	if (rb->ss_effects.ssil.ssil_final.is_null()) {
		{
			rb->ss_effects.ssil.depth_texture_view = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.linear_depth, 0, ssil_half_size ? 1 : 0, 4, RD::TEXTURE_SLICE_2D_ARRAY);
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.width = rb->width;
			tf.height = rb->height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
			rb->ss_effects.ssil.ssil_final = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssil.ssil_final, "SSIL texture");
			RD::get_singleton()->texture_clear(rb->ss_effects.ssil.ssil_final, Color(0, 0, 0, 0), 0, 1, 0, 1);
			if (rb->ss_effects.last_frame.is_null()) {
				tf.mipmaps = 6;
				rb->ss_effects.last_frame = RD::get_singleton()->texture_create(tf, RD::TextureView());
				RD::get_singleton()->set_resource_name(rb->ss_effects.last_frame, "Last Frame Radiance");
				RD::get_singleton()->texture_clear(rb->ss_effects.last_frame, Color(0, 0, 0, 0), 0, tf.mipmaps, 0, 1);
				for (uint32_t i = 0; i < 6; i++) {
					RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.last_frame, 0, i);
					rb->ss_effects.last_frame_slices.push_back(slice);
					RD::get_singleton()->set_resource_name(slice, "Last Frame Radiance Mip " + itos(i) + " ");
				}
			}
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = buffer_width;
			tf.height = buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssil.deinterleaved = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssil.deinterleaved, "SSIL deinterleaved buffer");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.ssil.deinterleaved, i, 0);
				rb->ss_effects.ssil.deinterleaved_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSIL deinterleaved buffer array " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = buffer_width;
			tf.height = buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssil.pong = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssil.pong, "SSIL deinterleaved pong buffer");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.ssil.pong, i, 0);
				rb->ss_effects.ssil.pong_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSIL deinterleaved buffer pong array " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
			tf.width = buffer_width;
			tf.height = buffer_height;
			tf.array_layers = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssil.edges = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssil.edges, "SSIL edges buffer");
			for (uint32_t i = 0; i < 4; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.ssil.edges, i, 0);
				rb->ss_effects.ssil.edges_slices.push_back(slice);
				RD::get_singleton()->set_resource_name(slice, "SSIL edges buffer slice " + itos(i) + " ");
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = half_width;
			tf.height = half_height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ss_effects.ssil.importance_map[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssil.importance_map[0], "SSIL Importance Map");
			rb->ss_effects.ssil.importance_map[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(rb->ss_effects.ssil.importance_map[1], "SSIL Importance Map Pong");
		}
		uniform_sets_are_invalid = true;
		ssil_using_half_size = ssil_half_size;
	}

	EffectsRD::SSILSettings settings;
	settings.radius = env->ssil_radius;
	settings.intensity = env->ssil_intensity;
	settings.sharpness = env->ssil_sharpness;
	settings.normal_rejection = env->ssil_normal_rejection;

	settings.quality = ssil_quality;
	settings.half_size = ssil_half_size;
	settings.adaptive_target = ssil_adaptive_target;
	settings.blur_passes = ssil_blur_passes;
	settings.fadeout_from = ssil_fadeout_from;
	settings.fadeout_to = ssil_fadeout_to;
	settings.full_screen_size = Size2i(rb->width, rb->height);
	settings.half_screen_size = Size2i(buffer_width, buffer_height);
	settings.quarter_screen_size = Size2i(half_width, half_height);

	CameraMatrix correction;
	correction.set_depth_correction(true);
	CameraMatrix projection = correction * p_projection;
	Transform3D transform = p_transform;
	transform.set_origin(Vector3(0.0, 0.0, 0.0));
	CameraMatrix last_frame_projection = rb->ss_effects.last_frame_projection * CameraMatrix(rb->ss_effects.last_frame_transform.affine_inverse()) * CameraMatrix(transform) * projection.inverse();

	storage->get_effects()->screen_space_indirect_lighting(rb->ss_effects.last_frame, rb->ss_effects.ssil.ssil_final, p_normal_buffer, rb->ss_effects.ssil.depth_texture_view, rb->ss_effects.ssil.deinterleaved, rb->ss_effects.ssil.deinterleaved_slices, rb->ss_effects.ssil.pong, rb->ss_effects.ssil.pong_slices, rb->ss_effects.ssil.importance_map[0], rb->ss_effects.ssil.importance_map[1], rb->ss_effects.ssil.edges, rb->ss_effects.ssil.edges_slices, p_projection, last_frame_projection, settings, uniform_sets_are_invalid, rb->ss_effects.ssil.gather_uniform_set, rb->ss_effects.ssil.importance_map_uniform_set, rb->ss_effects.ssil.projection_uniform_set);
	rb->ss_effects.last_frame_projection = projection;
	rb->ss_effects.last_frame_transform = transform;
}

void RendererSceneRenderRD::_copy_framebuffer_to_ssil(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (rb->ss_effects.last_frame.is_valid()) {
		storage->get_effects()->copy_to_rect(rb->texture, rb->ss_effects.last_frame, Rect2i(0, 0, rb->width, rb->height));

		int width = rb->width;
		int height = rb->height;
		for (int i = 0; i < rb->ss_effects.last_frame_slices.size() - 1; i++) {
			width = MAX(1, width >> 1);
			height = MAX(1, height >> 1);
			storage->get_effects()->make_mipmap(rb->ss_effects.last_frame_slices[i], rb->ss_effects.last_frame_slices[i + 1], Size2i(width, height));
		}
	}
}

void RendererSceneRenderRD::_render_buffers_copy_screen_texture(const RenderDataRD *p_render_data) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
	ERR_FAIL_COND(!rb);

	RD::get_singleton()->draw_command_begin_label("Copy screen texture");

	if (rb->blur[0].texture.is_null()) {
		_allocate_blur_textures(rb);
	}

	// @TODO IMPLEMENT MULTIVIEW, all effects need to support stereo buffers or effects are only applied to the left eye

	bool can_use_storage = _render_buffers_can_be_storage();

	if (can_use_storage) {
		storage->get_effects()->copy_to_rect(rb->texture, rb->blur[0].mipmaps[0].texture, Rect2i(0, 0, rb->width, rb->height));
		for (int i = 1; i < rb->blur[0].mipmaps.size(); i++) {
			storage->get_effects()->make_mipmap(rb->blur[0].mipmaps[i - 1].texture, rb->blur[0].mipmaps[i].texture, Size2i(rb->blur[0].mipmaps[i].width, rb->blur[0].mipmaps[i].height));
		}
	} else {
		storage->get_effects()->copy_to_fb_rect(rb->texture, rb->blur[0].mipmaps[0].fb, Rect2i(0, 0, rb->width, rb->height));
		for (int i = 1; i < rb->blur[0].mipmaps.size(); i++) {
			storage->get_effects()->make_mipmap_raster(rb->blur[0].mipmaps[i - 1].texture, rb->blur[0].mipmaps[i].fb, Size2i(rb->blur[0].mipmaps[i].width, rb->blur[0].mipmaps[i].height));
		}
	}

	RD::get_singleton()->draw_command_end_label();
}

void RendererSceneRenderRD::_render_buffers_copy_depth_texture(const RenderDataRD *p_render_data) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
	ERR_FAIL_COND(!rb);

	RD::get_singleton()->draw_command_begin_label("Copy depth texture");

	if (rb->depth_back_texture.is_null()) {
		_allocate_depth_backbuffer_textures(rb);
	}

	// @TODO IMPLEMENT MULTIVIEW, all effects need to support stereo buffers or effects are only applied to the left eye

	bool can_use_storage = _render_buffers_can_be_storage();

	if (can_use_storage) {
		storage->get_effects()->copy_to_rect(rb->depth_texture, rb->depth_back_texture, Rect2i(0, 0, rb->width, rb->height));
	} else {
		storage->get_effects()->copy_to_fb_rect(rb->depth_texture, rb->depth_back_fb, Rect2i(0, 0, rb->width, rb->height));
	}

	RD::get_singleton()->draw_command_end_label();
}

void RendererSceneRenderRD::_render_buffers_post_process_and_tonemap(const RenderDataRD *p_render_data) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
	ERR_FAIL_COND(!rb);

	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_render_data->environment);
	// Glow and override exposure (if enabled).
	CameraEffects *camfx = camera_effects_owner.get_or_null(p_render_data->camera_effects);

	bool can_use_effects = rb->width >= 8 && rb->height >= 8;
	bool can_use_storage = _render_buffers_can_be_storage();

	// @TODO IMPLEMENT MULTIVIEW, all effects need to support stereo buffers or effects are only applied to the left eye

	if (can_use_effects && camfx && (camfx->dof_blur_near_enabled || camfx->dof_blur_far_enabled) && camfx->dof_blur_amount > 0.0) {
		RD::get_singleton()->draw_command_begin_label("DOF");
		if (rb->blur[0].texture.is_null()) {
			_allocate_blur_textures(rb);
		}

		EffectsRD::BokehBuffers buffers;

		// Textures we use
		buffers.base_texture_size = Size2i(rb->internal_width, rb->internal_height);
		buffers.base_texture = rb->internal_texture;
		buffers.depth_texture = rb->depth_texture;
		buffers.secondary_texture = rb->blur[0].mipmaps[0].texture;
		buffers.half_texture[0] = rb->blur[1].mipmaps[0].texture;
		buffers.half_texture[1] = rb->blur[0].mipmaps[1].texture;

		float bokeh_size = camfx->dof_blur_amount * 64.0;
		if (can_use_storage) {
			storage->get_effects()->bokeh_dof(buffers, camfx->dof_blur_far_enabled, camfx->dof_blur_far_distance, camfx->dof_blur_far_transition, camfx->dof_blur_near_enabled, camfx->dof_blur_near_distance, camfx->dof_blur_near_transition, bokeh_size, dof_blur_bokeh_shape, dof_blur_quality, dof_blur_use_jitter, p_render_data->z_near, p_render_data->z_far, p_render_data->cam_ortogonal);
		} else {
			// Set framebuffers.
			buffers.base_fb = rb->texture_fb;
			buffers.secondary_fb = rb->weight_buffers[1].fb;
			buffers.half_fb[0] = rb->weight_buffers[2].fb;
			buffers.half_fb[1] = rb->weight_buffers[3].fb;
			buffers.weight_texture[0] = rb->weight_buffers[0].weight;
			buffers.weight_texture[1] = rb->weight_buffers[1].weight;
			buffers.weight_texture[2] = rb->weight_buffers[2].weight;
			buffers.weight_texture[3] = rb->weight_buffers[3].weight;

			// Set weight buffers.
			buffers.base_weight_fb = rb->base_weight_fb;

			storage->get_effects()->bokeh_dof_raster(buffers, camfx->dof_blur_far_enabled, camfx->dof_blur_far_distance, camfx->dof_blur_far_transition, camfx->dof_blur_near_enabled, camfx->dof_blur_near_distance, camfx->dof_blur_near_transition, bokeh_size, dof_blur_bokeh_shape, dof_blur_quality, p_render_data->z_near, p_render_data->z_far, p_render_data->cam_ortogonal);
		}
		RD::get_singleton()->draw_command_end_label();
	}

	if (can_use_effects && env && env->auto_exposure) {
		RD::get_singleton()->draw_command_begin_label("Auto exposure");
		if (rb->luminance.current.is_null()) {
			_allocate_luminance_textures(rb);
		}

		bool set_immediate = env->auto_exposure_version != rb->auto_exposure_version;
		rb->auto_exposure_version = env->auto_exposure_version;

		double step = env->auto_exp_speed * time_step;
		if (can_use_storage) {
			storage->get_effects()->luminance_reduction(rb->internal_texture, Size2i(rb->internal_width, rb->internal_height), rb->luminance.reduce, rb->luminance.current, env->min_luminance, env->max_luminance, step, set_immediate);
		} else {
			storage->get_effects()->luminance_reduction_raster(rb->internal_texture, Size2i(rb->internal_width, rb->internal_height), rb->luminance.reduce, rb->luminance.fb, rb->luminance.current, env->min_luminance, env->max_luminance, step, set_immediate);
		}
		// Swap final reduce with prev luminance.
		SWAP(rb->luminance.current, rb->luminance.reduce.write[rb->luminance.reduce.size() - 1]);
		if (!can_use_storage) {
			SWAP(rb->luminance.current_fb, rb->luminance.fb.write[rb->luminance.fb.size() - 1]);
		}

		RenderingServerDefault::redraw_request(); // Redraw all the time if auto exposure rendering is on.
		RD::get_singleton()->draw_command_end_label();
	}

	int max_glow_level = -1;

	if (can_use_effects && env && env->glow_enabled) {
		RD::get_singleton()->draw_command_begin_label("Gaussian Glow");

		/* see that blur textures are allocated */

		if (rb->blur[1].texture.is_null()) {
			_allocate_blur_textures(rb);
		}

		for (int i = 0; i < RS::MAX_GLOW_LEVELS; i++) {
			if (env->glow_levels[i] > 0.0) {
				if (i >= rb->blur[1].mipmaps.size()) {
					max_glow_level = rb->blur[1].mipmaps.size() - 1;
				} else {
					max_glow_level = i;
				}
			}
		}

		for (int i = 0; i < (max_glow_level + 1); i++) {
			int vp_w = rb->blur[1].mipmaps[i].width;
			int vp_h = rb->blur[1].mipmaps[i].height;

			if (i == 0) {
				RID luminance_texture;
				if (env->auto_exposure && rb->luminance.current.is_valid()) {
					luminance_texture = rb->luminance.current;
				}
				if (can_use_storage) {
					storage->get_effects()->gaussian_glow(rb->internal_texture, rb->blur[1].mipmaps[i].texture, Size2i(vp_w, vp_h), env->glow_strength, glow_high_quality, true, env->glow_hdr_luminance_cap, env->exposure, env->glow_bloom, env->glow_hdr_bleed_threshold, env->glow_hdr_bleed_scale, luminance_texture, env->auto_exp_scale);
				} else {
					storage->get_effects()->gaussian_glow_raster(rb->internal_texture, rb->blur[1].mipmaps[i].half_fb, rb->blur[1].mipmaps[i].half_texture, rb->blur[1].mipmaps[i].fb, Size2i(vp_w, vp_h), env->glow_strength, glow_high_quality, true, env->glow_hdr_luminance_cap, env->exposure, env->glow_bloom, env->glow_hdr_bleed_threshold, env->glow_hdr_bleed_scale, luminance_texture, env->auto_exp_scale);
				}
			} else {
				if (can_use_storage) {
					storage->get_effects()->gaussian_glow(rb->blur[1].mipmaps[i - 1].texture, rb->blur[1].mipmaps[i].texture, Size2i(vp_w, vp_h), env->glow_strength, glow_high_quality);
				} else {
					storage->get_effects()->gaussian_glow_raster(rb->blur[1].mipmaps[i - 1].texture, rb->blur[1].mipmaps[i].half_fb, rb->blur[1].mipmaps[i].half_texture, rb->blur[1].mipmaps[i].fb, Vector2(1.0 / vp_w, 1.0 / vp_h), env->glow_strength, glow_high_quality);
				}
			}
		}

		RD::get_singleton()->draw_command_end_label();
	}

	{
		RD::get_singleton()->draw_command_begin_label("Tonemap");

		EffectsRD::TonemapSettings tonemap;

		if (can_use_effects && env && env->auto_exposure && rb->luminance.current.is_valid()) {
			tonemap.use_auto_exposure = true;
			tonemap.exposure_texture = rb->luminance.current;
			tonemap.auto_exposure_grey = env->auto_exp_scale;
		} else {
			tonemap.exposure_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}

		if (can_use_effects && env && env->glow_enabled) {
			tonemap.use_glow = true;
			tonemap.glow_mode = EffectsRD::TonemapSettings::GlowMode(env->glow_blend_mode);
			tonemap.glow_intensity = env->glow_blend_mode == RS::ENV_GLOW_BLEND_MODE_MIX ? env->glow_mix : env->glow_intensity;
			for (int i = 0; i < RS::MAX_GLOW_LEVELS; i++) {
				tonemap.glow_levels[i] = env->glow_levels[i];
			}
			tonemap.glow_texture_size.x = rb->blur[1].mipmaps[0].width;
			tonemap.glow_texture_size.y = rb->blur[1].mipmaps[0].height;
			tonemap.glow_use_bicubic_upscale = glow_bicubic_upscale;
			tonemap.glow_texture = rb->blur[1].texture;
		} else {
			tonemap.glow_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_BLACK);
		}

		if (rb->screen_space_aa == RS::VIEWPORT_SCREEN_SPACE_AA_FXAA) {
			tonemap.use_fxaa = true;
		}

		tonemap.use_debanding = rb->use_debanding;
		tonemap.texture_size = Vector2i(rb->internal_width, rb->internal_height);

		if (env) {
			tonemap.tonemap_mode = env->tone_mapper;
			tonemap.white = env->white;
			tonemap.exposure = env->exposure;
		}

		if (camfx && camfx->override_exposure_enabled) {
			tonemap.exposure = camfx->override_exposure;
		}

		tonemap.use_color_correction = false;
		tonemap.use_1d_color_correction = false;
		tonemap.color_correction_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);

		if (can_use_effects && env) {
			tonemap.use_bcs = env->adjustments_enabled;
			tonemap.brightness = env->adjustments_brightness;
			tonemap.contrast = env->adjustments_contrast;
			tonemap.saturation = env->adjustments_saturation;
			if (env->adjustments_enabled && env->color_correction.is_valid()) {
				tonemap.use_color_correction = true;
				tonemap.use_1d_color_correction = env->use_1d_color_correction;
				tonemap.color_correction_texture = storage->texture_get_rd_texture(env->color_correction);
			}
		}

		tonemap.luminance_multiplier = _render_buffers_get_luminance_multiplier();
		tonemap.view_count = p_render_data->view_count;

		storage->get_effects()->tonemapper(rb->internal_texture, storage->render_target_get_rd_framebuffer(rb->render_target), tonemap);

		RD::get_singleton()->draw_command_end_label();
	}

	if (can_use_effects && can_use_storage && (rb->internal_width != rb->width || rb->internal_height != rb->height)) {
		RD::get_singleton()->draw_command_begin_label("FSR Upscale");

		storage->get_effects()->fsr_upscale(rb->internal_texture, rb->upscale_texture, rb->texture, Size2i(rb->internal_width, rb->internal_height), Size2i(rb->width, rb->height), rb->fsr_sharpness);

		RD::get_singleton()->draw_command_end_label();
	}

	storage->render_target_disable_clear_request(rb->render_target);
}

void RendererSceneRenderRD::_post_process_subpass(RID p_source_texture, RID p_framebuffer, const RenderDataRD *p_render_data) {
	RD::get_singleton()->draw_command_begin_label("Post Process Subpass");

	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
	ERR_FAIL_COND(!rb);

	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_render_data->environment);
	// Override exposure (if enabled).
	CameraEffects *camfx = camera_effects_owner.get_or_null(p_render_data->camera_effects);

	bool can_use_effects = rb->width >= 8 && rb->height >= 8;

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_switch_to_next_pass();

	EffectsRD::TonemapSettings tonemap;

	if (env) {
		tonemap.tonemap_mode = env->tone_mapper;
		tonemap.exposure = env->exposure;
		tonemap.white = env->white;
	}

	if (camfx && camfx->override_exposure_enabled) {
		tonemap.exposure = camfx->override_exposure;
	}

	// We don't support glow or auto exposure here, if they are needed, don't use subpasses!
	// The problem is that we need to use the result so far and process them before we can
	// apply this to our results.
	if (can_use_effects && env && env->glow_enabled) {
		ERR_FAIL_MSG("Glow is not supported when using subpasses.");
	}
	if (can_use_effects && env && env->auto_exposure) {
		ERR_FAIL_MSG("Glow is not supported when using subpasses.");
	}

	tonemap.use_glow = false;
	tonemap.glow_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_BLACK);
	tonemap.use_auto_exposure = false;
	tonemap.exposure_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE);

	tonemap.use_color_correction = false;
	tonemap.use_1d_color_correction = false;
	tonemap.color_correction_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);

	if (can_use_effects && env) {
		tonemap.use_bcs = env->adjustments_enabled;
		tonemap.brightness = env->adjustments_brightness;
		tonemap.contrast = env->adjustments_contrast;
		tonemap.saturation = env->adjustments_saturation;
		if (env->adjustments_enabled && env->color_correction.is_valid()) {
			tonemap.use_color_correction = true;
			tonemap.use_1d_color_correction = env->use_1d_color_correction;
			tonemap.color_correction_texture = storage->texture_get_rd_texture(env->color_correction);
		}
	}

	tonemap.use_debanding = rb->use_debanding;
	tonemap.texture_size = Vector2i(rb->width, rb->height);

	tonemap.luminance_multiplier = _render_buffers_get_luminance_multiplier();
	tonemap.view_count = p_render_data->view_count;

	storage->get_effects()->tonemapper(draw_list, p_source_texture, RD::get_singleton()->framebuffer_get_format(p_framebuffer), tonemap);

	RD::get_singleton()->draw_command_end_label();
}

void RendererSceneRenderRD::_disable_clear_request(const RenderDataRD *p_render_data) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
	ERR_FAIL_COND(!rb);

	storage->render_target_disable_clear_request(rb->render_target);
}

void RendererSceneRenderRD::_render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas, RID p_occlusion_buffer) {
	EffectsRD *effects = storage->get_effects();

	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SHADOW_ATLAS) {
		if (p_shadow_atlas.is_valid()) {
			RID shadow_atlas_texture = shadow_atlas_get_texture(p_shadow_atlas);

			if (shadow_atlas_texture.is_null()) {
				shadow_atlas_texture = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_BLACK);
			}

			Size2 rtsize = storage->render_target_get_size(rb->render_target);
			effects->copy_to_fb_rect(shadow_atlas_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize / 2), false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS) {
		if (directional_shadow_get_texture().is_valid()) {
			RID shadow_atlas_texture = directional_shadow_get_texture();
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(shadow_atlas_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize / 2), false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_DECAL_ATLAS) {
		RID decal_atlas = storage->decal_atlas_get_texture();

		if (decal_atlas.is_valid()) {
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(decal_atlas, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize / 2), false, false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SCENE_LUMINANCE) {
		if (rb->luminance.current.is_valid()) {
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_fb_rect(rb->luminance.current, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize / 8), false, true);
		}
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SSAO && rb->ss_effects.ssao.ao_final.is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		effects->copy_to_fb_rect(rb->ss_effects.ssao.ao_final, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, true);
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SSIL && rb->ss_effects.ssil.ssil_final.is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		effects->copy_to_fb_rect(rb->ss_effects.ssil.ssil_final, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, false);
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_NORMAL_BUFFER && _render_buffers_get_normal_texture(p_render_buffers).is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		effects->copy_to_fb_rect(_render_buffers_get_normal_texture(p_render_buffers), storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, false);
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_GI_BUFFER && rb->ambient_buffer.is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		RID ambient_texture = rb->ambient_buffer;
		RID reflection_texture = rb->reflection_buffer;
		effects->copy_to_fb_rect(ambient_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, false, false, true, reflection_texture);
	}

	if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_OCCLUDERS) {
		if (p_occlusion_buffer.is_valid()) {
			Size2 rtsize = storage->render_target_get_size(rb->render_target);
			effects->copy_to_fb_rect(storage->texture_get_rd_texture(p_occlusion_buffer), storage->render_target_get_rd_framebuffer(rb->render_target), Rect2i(Vector2(), rtsize), true, false);
		}
	}
}

void RendererSceneRenderRD::environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) {
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);

	env->adjustments_enabled = p_enable;
	env->adjustments_brightness = p_brightness;
	env->adjustments_contrast = p_contrast;
	env->adjustments_saturation = p_saturation;
	env->use_1d_color_correction = p_use_1d_color_correction;
	env->color_correction = p_color_correction;
}

RID RendererSceneRenderRD::render_buffers_get_back_buffer_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	if (!rb->blur[0].texture.is_valid()) {
		return RID(); //not valid at the moment
	}
	return rb->blur[0].texture;
}

RID RendererSceneRenderRD::render_buffers_get_back_depth_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	if (!rb->depth_back_texture.is_valid()) {
		return RID(); //not valid at the moment
	}
	return rb->depth_back_texture;
}

RID RendererSceneRenderRD::render_buffers_get_depth_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	return rb->depth_texture;
}

RID RendererSceneRenderRD::render_buffers_get_ao_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	return rb->ss_effects.ssao.ao_final;
}
RID RendererSceneRenderRD::render_buffers_get_ssil_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	return rb->ss_effects.ssil.ssil_final;
}

RID RendererSceneRenderRD::render_buffers_get_voxel_gi_buffer(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	if (rb->gi.voxel_gi_buffer.is_null()) {
		rb->gi.voxel_gi_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(RendererSceneGIRD::VoxelGIData) * RendererSceneGIRD::MAX_VOXEL_GI_INSTANCES);
	}
	return rb->gi.voxel_gi_buffer;
}

RID RendererSceneRenderRD::render_buffers_get_default_voxel_gi_buffer() {
	return gi.default_voxel_gi_buffer;
}

RID RendererSceneRenderRD::render_buffers_get_gi_ambient_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	return rb->ambient_buffer;
}
RID RendererSceneRenderRD::render_buffers_get_gi_reflection_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	return rb->reflection_buffer;
}

uint32_t RendererSceneRenderRD::render_buffers_get_sdfgi_cascade_count(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->cascades.size();
}
bool RendererSceneRenderRD::render_buffers_is_sdfgi_enabled(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, false);

	return rb->sdfgi != nullptr;
}
RID RendererSceneRenderRD::render_buffers_get_sdfgi_irradiance_probes(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	ERR_FAIL_COND_V(!rb->sdfgi, RID());

	return rb->sdfgi->lightprobe_texture;
}

Vector3 RendererSceneRenderRD::render_buffers_get_sdfgi_cascade_offset(RID p_render_buffers, uint32_t p_cascade) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, Vector3());
	ERR_FAIL_COND_V(!rb->sdfgi, Vector3());
	ERR_FAIL_UNSIGNED_INDEX_V(p_cascade, rb->sdfgi->cascades.size(), Vector3());

	return Vector3((Vector3i(1, 1, 1) * -int32_t(rb->sdfgi->cascade_size >> 1) + rb->sdfgi->cascades[p_cascade].position)) * rb->sdfgi->cascades[p_cascade].cell_size;
}

Vector3i RendererSceneRenderRD::render_buffers_get_sdfgi_cascade_probe_offset(RID p_render_buffers, uint32_t p_cascade) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, Vector3i());
	ERR_FAIL_COND_V(!rb->sdfgi, Vector3i());
	ERR_FAIL_UNSIGNED_INDEX_V(p_cascade, rb->sdfgi->cascades.size(), Vector3i());
	int32_t probe_divisor = rb->sdfgi->cascade_size / RendererSceneGIRD::SDFGI::PROBE_DIVISOR;

	return rb->sdfgi->cascades[p_cascade].position / probe_divisor;
}

float RendererSceneRenderRD::render_buffers_get_sdfgi_normal_bias(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->normal_bias;
}
float RendererSceneRenderRD::render_buffers_get_sdfgi_cascade_probe_size(RID p_render_buffers, uint32_t p_cascade) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_cascade, rb->sdfgi->cascades.size(), 0);

	return float(rb->sdfgi->cascade_size) * rb->sdfgi->cascades[p_cascade].cell_size / float(rb->sdfgi->probe_axis_count - 1);
}
uint32_t RendererSceneRenderRD::render_buffers_get_sdfgi_cascade_probe_count(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->probe_axis_count;
}

uint32_t RendererSceneRenderRD::render_buffers_get_sdfgi_cascade_size(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0);

	return rb->sdfgi->cascade_size;
}

bool RendererSceneRenderRD::render_buffers_is_sdfgi_using_occlusion(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, false);
	ERR_FAIL_COND_V(!rb->sdfgi, false);

	return rb->sdfgi->uses_occlusion;
}

float RendererSceneRenderRD::render_buffers_get_sdfgi_energy(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, 0.0);
	ERR_FAIL_COND_V(!rb->sdfgi, 0.0);

	return rb->sdfgi->energy;
}
RID RendererSceneRenderRD::render_buffers_get_sdfgi_occlusion_texture(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	ERR_FAIL_COND_V(!rb->sdfgi, RID());

	return rb->sdfgi->occlusion_texture;
}

bool RendererSceneRenderRD::render_buffers_has_volumetric_fog(RID p_render_buffers) const {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, false);

	return rb->volumetric_fog != nullptr;
}
RID RendererSceneRenderRD::render_buffers_get_volumetric_fog_texture(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb || !rb->volumetric_fog, RID());

	return rb->volumetric_fog->fog_map;
}

RID RendererSceneRenderRD::render_buffers_get_volumetric_fog_sky_uniform_set(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	if (!rb->volumetric_fog) {
		return RID();
	}

	return rb->volumetric_fog->sky_uniform_set;
}

float RendererSceneRenderRD::render_buffers_get_volumetric_fog_end(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb || !rb->volumetric_fog, 0);
	return rb->volumetric_fog->length;
}
float RendererSceneRenderRD::render_buffers_get_volumetric_fog_detail_spread(RID p_render_buffers) {
	const RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb || !rb->volumetric_fog, 0);
	return rb->volumetric_fog->spread;
}

float RendererSceneRenderRD::_render_buffers_get_luminance_multiplier() {
	return 1.0;
}

RD::DataFormat RendererSceneRenderRD::_render_buffers_get_color_format() {
	return RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
}

bool RendererSceneRenderRD::_render_buffers_can_be_storage() {
	return true;
}

void RendererSceneRenderRD::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_internal_width, int p_internal_height, int p_width, int p_height, float p_fsr_sharpness, float p_fsr_mipmap_bias, RS::ViewportMSAA p_msaa, RenderingServer::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding, uint32_t p_view_count) {
	ERR_FAIL_COND_MSG(p_view_count == 0, "Must have at least 1 view");

	if (!_render_buffers_can_be_storage()) {
		p_internal_height = p_height;
		p_internal_width = p_width;
	}

	if (p_width != p_internal_width) {
		float fsr_mipmap_bias = -log2f(p_width / p_internal_width) + p_fsr_mipmap_bias;
		storage->sampler_rd_configure_custom(fsr_mipmap_bias);
		update_uniform_sets();
	}

	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);

	// Should we add an overrule per viewport?
	rb->internal_width = p_internal_width;
	rb->internal_height = p_internal_height;
	rb->width = p_width;
	rb->height = p_height;
	rb->fsr_sharpness = p_fsr_sharpness;
	rb->render_target = p_render_target;
	rb->msaa = p_msaa;
	rb->screen_space_aa = p_screen_space_aa;
	rb->use_debanding = p_use_debanding;
	rb->view_count = p_view_count;

	if (is_clustered_enabled()) {
		if (rb->cluster_builder == nullptr) {
			rb->cluster_builder = memnew(ClusterBuilderRD);
		}
		rb->cluster_builder->set_shared(&cluster_builder_shared);
	}

	_free_render_buffer_data(rb);

	{
		RD::TextureFormat tf;
		if (rb->view_count > 1) {
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		}
		tf.format = _render_buffers_get_color_format();
		tf.width = rb->internal_width; // If set to rb->width, msaa won't crash
		tf.height = rb->internal_height; // If set to rb->width, msaa won't crash
		tf.array_layers = rb->view_count; // create a layer for every view
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | (_render_buffers_can_be_storage() ? RD::TEXTURE_USAGE_STORAGE_BIT : 0) | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		if (rb->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
		}
		tf.usage_bits |= RD::TEXTURE_USAGE_INPUT_ATTACHMENT_BIT; // only needed when using subpasses in the mobile renderer

		rb->internal_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		if ((p_internal_width != p_width || p_internal_height != p_height)) {
			tf.width = rb->width;
			tf.height = rb->height;
			rb->texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			rb->upscale_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
		} else {
			rb->texture = rb->internal_texture;
			rb->upscale_texture = rb->internal_texture;
		}
	}

	{
		RD::TextureFormat tf;
		if (rb->view_count > 1) {
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		}
		if (rb->msaa == RS::VIEWPORT_MSAA_DISABLED) {
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
		} else {
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		}

		tf.width = rb->internal_width;
		tf.height = rb->internal_height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
		tf.array_layers = rb->view_count; // create a layer for every view

		if (rb->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		} else {
			tf.usage_bits |= RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}

		rb->depth_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	if (!_render_buffers_can_be_storage()) {
		// ONLY USED ON MOBILE RENDERER, ONLY USED FOR POST EFFECTS!
		Vector<RID> fb;
		fb.push_back(rb->internal_texture);

		rb->texture_fb = RD::get_singleton()->framebuffer_create(fb, RenderingDevice::INVALID_ID, rb->view_count);
	}

	RID target_texture = storage->render_target_get_rd_texture(rb->render_target);
	rb->data->configure(rb->internal_texture, rb->depth_texture, target_texture, p_internal_width, p_internal_height, p_msaa, p_view_count);

	if (is_clustered_enabled()) {
		rb->cluster_builder->setup(Size2i(p_internal_width, p_internal_height), max_cluster_elements, rb->depth_texture, storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED), rb->internal_texture);
	}
}

void RendererSceneRenderRD::gi_set_use_half_resolution(bool p_enable) {
	gi.half_resolution = p_enable;
}

void RendererSceneRenderRD::sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) {
	sss_quality = p_quality;
}

RS::SubSurfaceScatteringQuality RendererSceneRenderRD::sub_surface_scattering_get_quality() const {
	return sss_quality;
}

void RendererSceneRenderRD::sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) {
	sss_scale = p_scale;
	sss_depth_scale = p_depth_scale;
}

void RendererSceneRenderRD::shadows_quality_set(RS::ShadowQuality p_quality) {
	ERR_FAIL_INDEX_MSG(p_quality, RS::SHADOW_QUALITY_MAX, "Shadow quality too high, please see RenderingServer's ShadowQuality enum");

	if (shadows_quality != p_quality) {
		shadows_quality = p_quality;

		switch (shadows_quality) {
			case RS::SHADOW_QUALITY_HARD: {
				penumbra_shadow_samples = 4;
				soft_shadow_samples = 0;
				shadows_quality_radius = 1.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_VERY_LOW: {
				penumbra_shadow_samples = 4;
				soft_shadow_samples = 1;
				shadows_quality_radius = 1.5;
			} break;
			case RS::SHADOW_QUALITY_SOFT_LOW: {
				penumbra_shadow_samples = 8;
				soft_shadow_samples = 4;
				shadows_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_MEDIUM: {
				penumbra_shadow_samples = 12;
				soft_shadow_samples = 8;
				shadows_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_HIGH: {
				penumbra_shadow_samples = 24;
				soft_shadow_samples = 16;
				shadows_quality_radius = 3.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_ULTRA: {
				penumbra_shadow_samples = 32;
				soft_shadow_samples = 32;
				shadows_quality_radius = 4.0;
			} break;
			case RS::SHADOW_QUALITY_MAX:
				break;
		}
		get_vogel_disk(penumbra_shadow_kernel, penumbra_shadow_samples);
		get_vogel_disk(soft_shadow_kernel, soft_shadow_samples);
	}

	_update_shader_quality_settings();
}

void RendererSceneRenderRD::directional_shadow_quality_set(RS::ShadowQuality p_quality) {
	ERR_FAIL_INDEX_MSG(p_quality, RS::SHADOW_QUALITY_MAX, "Shadow quality too high, please see RenderingServer's ShadowQuality enum");

	if (directional_shadow_quality != p_quality) {
		directional_shadow_quality = p_quality;

		switch (directional_shadow_quality) {
			case RS::SHADOW_QUALITY_HARD: {
				directional_penumbra_shadow_samples = 4;
				directional_soft_shadow_samples = 0;
				directional_shadow_quality_radius = 1.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_VERY_LOW: {
				directional_penumbra_shadow_samples = 4;
				directional_soft_shadow_samples = 1;
				directional_shadow_quality_radius = 1.5;
			} break;
			case RS::SHADOW_QUALITY_SOFT_LOW: {
				directional_penumbra_shadow_samples = 8;
				directional_soft_shadow_samples = 4;
				directional_shadow_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_MEDIUM: {
				directional_penumbra_shadow_samples = 12;
				directional_soft_shadow_samples = 8;
				directional_shadow_quality_radius = 2.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_HIGH: {
				directional_penumbra_shadow_samples = 24;
				directional_soft_shadow_samples = 16;
				directional_shadow_quality_radius = 3.0;
			} break;
			case RS::SHADOW_QUALITY_SOFT_ULTRA: {
				directional_penumbra_shadow_samples = 32;
				directional_soft_shadow_samples = 32;
				directional_shadow_quality_radius = 4.0;
			} break;
			case RS::SHADOW_QUALITY_MAX:
				break;
		}
		get_vogel_disk(directional_penumbra_shadow_kernel, directional_penumbra_shadow_samples);
		get_vogel_disk(directional_soft_shadow_kernel, directional_soft_shadow_samples);
	}

	_update_shader_quality_settings();
}

void RendererSceneRenderRD::decals_set_filter(RenderingServer::DecalFilter p_filter) {
	if (decals_filter == p_filter) {
		return;
	}
	decals_filter = p_filter;
	_update_shader_quality_settings();
}
void RendererSceneRenderRD::light_projectors_set_filter(RenderingServer::LightProjectorFilter p_filter) {
	if (light_projectors_filter == p_filter) {
		return;
	}
	light_projectors_filter = p_filter;
	_update_shader_quality_settings();
}

int RendererSceneRenderRD::get_roughness_layers() const {
	return sky.roughness_layers;
}

bool RendererSceneRenderRD::is_using_radiance_cubemap_array() const {
	return sky.sky_use_cubemap_array;
}

RendererSceneRenderRD::RenderBufferData *RendererSceneRenderRD::render_buffers_get_data(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND_V(!rb, nullptr);
	return rb->data;
}

void RendererSceneRenderRD::_setup_reflections(const PagedArray<RID> &p_reflections, const Transform3D &p_camera_inverse_transform, RID p_environment) {
	cluster.reflection_count = 0;

	for (uint32_t i = 0; i < (uint32_t)p_reflections.size(); i++) {
		if (cluster.reflection_count == cluster.max_reflections) {
			break;
		}

		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_reflections[i]);
		if (!rpi) {
			continue;
		}

		cluster.reflection_sort[cluster.reflection_count].instance = rpi;
		cluster.reflection_sort[cluster.reflection_count].depth = -p_camera_inverse_transform.xform(rpi->transform.origin).z;
		cluster.reflection_count++;
	}

	if (cluster.reflection_count > 0) {
		SortArray<Cluster::InstanceSort<ReflectionProbeInstance>> sort_array;
		sort_array.sort(cluster.reflection_sort, cluster.reflection_count);
	}

	bool using_forward_ids = _uses_forward_ids();
	for (uint32_t i = 0; i < cluster.reflection_count; i++) {
		ReflectionProbeInstance *rpi = cluster.reflection_sort[i].instance;

		if (using_forward_ids) {
			_map_forward_id(FORWARD_ID_TYPE_REFLECTION_PROBE, rpi->forward_id, i);
		}

		RID base_probe = rpi->probe;

		Cluster::ReflectionData &reflection_ubo = cluster.reflections[i];

		Vector3 extents = storage->reflection_probe_get_extents(base_probe);

		rpi->cull_mask = storage->reflection_probe_get_cull_mask(base_probe);

		reflection_ubo.box_extents[0] = extents.x;
		reflection_ubo.box_extents[1] = extents.y;
		reflection_ubo.box_extents[2] = extents.z;
		reflection_ubo.index = rpi->atlas_index;

		Vector3 origin_offset = storage->reflection_probe_get_origin_offset(base_probe);

		reflection_ubo.box_offset[0] = origin_offset.x;
		reflection_ubo.box_offset[1] = origin_offset.y;
		reflection_ubo.box_offset[2] = origin_offset.z;
		reflection_ubo.mask = storage->reflection_probe_get_cull_mask(base_probe);

		reflection_ubo.intensity = storage->reflection_probe_get_intensity(base_probe);
		reflection_ubo.ambient_mode = storage->reflection_probe_get_ambient_mode(base_probe);

		reflection_ubo.exterior = !storage->reflection_probe_is_interior(base_probe);
		reflection_ubo.box_project = storage->reflection_probe_is_box_projection(base_probe);

		Color ambient_linear = storage->reflection_probe_get_ambient_color(base_probe).to_linear();
		float interior_ambient_energy = storage->reflection_probe_get_ambient_color_energy(base_probe);
		reflection_ubo.ambient[0] = ambient_linear.r * interior_ambient_energy;
		reflection_ubo.ambient[1] = ambient_linear.g * interior_ambient_energy;
		reflection_ubo.ambient[2] = ambient_linear.b * interior_ambient_energy;

		Transform3D transform = rpi->transform;
		Transform3D proj = (p_camera_inverse_transform * transform).inverse();
		RendererStorageRD::store_transform(proj, reflection_ubo.local_matrix);

		if (current_cluster_builder != nullptr) {
			current_cluster_builder->add_box(ClusterBuilderRD::BOX_TYPE_REFLECTION_PROBE, transform, extents);
		}

		rpi->last_pass = RSG::rasterizer->get_frame_number();
	}

	if (cluster.reflection_count) {
		RD::get_singleton()->buffer_update(cluster.reflection_buffer, 0, cluster.reflection_count * sizeof(Cluster::ReflectionData), cluster.reflections, RD::BARRIER_MASK_RASTER | RD::BARRIER_MASK_COMPUTE);
	}
}

void RendererSceneRenderRD::_setup_lights(const PagedArray<RID> &p_lights, const Transform3D &p_camera_transform, RID p_shadow_atlas, bool p_using_shadows, uint32_t &r_directional_light_count, uint32_t &r_positional_light_count, bool &r_directional_light_soft_shadows) {
	Transform3D inverse_transform = p_camera_transform.affine_inverse();

	r_directional_light_count = 0;
	r_positional_light_count = 0;
	sky.sky_scene_state.ubo.directional_light_count = 0;

	Plane camera_plane(-p_camera_transform.basis.get_axis(Vector3::AXIS_Z).normalized(), p_camera_transform.origin);

	cluster.omni_light_count = 0;
	cluster.spot_light_count = 0;

	r_directional_light_soft_shadows = false;

	for (int i = 0; i < (int)p_lights.size(); i++) {
		LightInstance *li = light_instance_owner.get_or_null(p_lights[i]);
		if (!li) {
			continue;
		}
		RID base = li->light;

		ERR_CONTINUE(base.is_null());

		RS::LightType type = storage->light_get_type(base);
		switch (type) {
			case RS::LIGHT_DIRECTIONAL: {
				//	Copy to SkyDirectionalLightData
				if (r_directional_light_count < sky.sky_scene_state.max_directional_lights) {
					RendererSceneSkyRD::SkyDirectionalLightData &sky_light_data = sky.sky_scene_state.directional_lights[r_directional_light_count];
					Transform3D light_transform = li->transform;
					Vector3 world_direction = light_transform.basis.xform(Vector3(0, 0, 1)).normalized();

					sky_light_data.direction[0] = world_direction.x;
					sky_light_data.direction[1] = world_direction.y;
					sky_light_data.direction[2] = -world_direction.z;

					float sign = storage->light_is_negative(base) ? -1 : 1;
					sky_light_data.energy = sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY);

					Color linear_col = storage->light_get_color(base).to_linear();
					sky_light_data.color[0] = linear_col.r;
					sky_light_data.color[1] = linear_col.g;
					sky_light_data.color[2] = linear_col.b;

					sky_light_data.enabled = true;

					float angular_diameter = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);
					if (angular_diameter > 0.0) {
						// I know tan(0) is 0, but let's not risk it with numerical precision.
						// technically this will keep expanding until reaching the sun, but all we care
						// is expand until we reach the radius of the near plane (there can't be more occluders than that)
						angular_diameter = Math::tan(Math::deg2rad(angular_diameter));
						if (storage->light_has_shadow(base)) {
							r_directional_light_soft_shadows = true;
						}
					} else {
						angular_diameter = 0.0;
					}
					sky_light_data.size = angular_diameter;
					sky.sky_scene_state.ubo.directional_light_count++;
				}

				if (r_directional_light_count >= cluster.max_directional_lights || storage->light_directional_is_sky_only(base)) {
					continue;
				}

				Cluster::DirectionalLightData &light_data = cluster.directional_lights[r_directional_light_count];

				Transform3D light_transform = li->transform;

				Vector3 direction = inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, 1))).normalized();

				light_data.direction[0] = direction.x;
				light_data.direction[1] = direction.y;
				light_data.direction[2] = direction.z;

				float sign = storage->light_is_negative(base) ? -1 : 1;

				light_data.energy = sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI;

				Color linear_col = storage->light_get_color(base).to_linear();
				light_data.color[0] = linear_col.r;
				light_data.color[1] = linear_col.g;
				light_data.color[2] = linear_col.b;

				light_data.specular = storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR);
				light_data.mask = storage->light_get_cull_mask(base);

				float size = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

				light_data.size = 1.0 - Math::cos(Math::deg2rad(size)); //angle to cosine offset

				Color shadow_col = storage->light_get_shadow_color(base).to_linear();

				if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_PSSM_SPLITS) {
					light_data.shadow_color1[0] = 1.0;
					light_data.shadow_color1[1] = 0.0;
					light_data.shadow_color1[2] = 0.0;
					light_data.shadow_color1[3] = 1.0;
					light_data.shadow_color2[0] = 0.0;
					light_data.shadow_color2[1] = 1.0;
					light_data.shadow_color2[2] = 0.0;
					light_data.shadow_color2[3] = 1.0;
					light_data.shadow_color3[0] = 0.0;
					light_data.shadow_color3[1] = 0.0;
					light_data.shadow_color3[2] = 1.0;
					light_data.shadow_color3[3] = 1.0;
					light_data.shadow_color4[0] = 1.0;
					light_data.shadow_color4[1] = 1.0;
					light_data.shadow_color4[2] = 0.0;
					light_data.shadow_color4[3] = 1.0;

				} else {
					light_data.shadow_color1[0] = shadow_col.r;
					light_data.shadow_color1[1] = shadow_col.g;
					light_data.shadow_color1[2] = shadow_col.b;
					light_data.shadow_color1[3] = 1.0;
					light_data.shadow_color2[0] = shadow_col.r;
					light_data.shadow_color2[1] = shadow_col.g;
					light_data.shadow_color2[2] = shadow_col.b;
					light_data.shadow_color2[3] = 1.0;
					light_data.shadow_color3[0] = shadow_col.r;
					light_data.shadow_color3[1] = shadow_col.g;
					light_data.shadow_color3[2] = shadow_col.b;
					light_data.shadow_color3[3] = 1.0;
					light_data.shadow_color4[0] = shadow_col.r;
					light_data.shadow_color4[1] = shadow_col.g;
					light_data.shadow_color4[2] = shadow_col.b;
					light_data.shadow_color4[3] = 1.0;
				}

				light_data.shadow_enabled = p_using_shadows && storage->light_has_shadow(base);

				float angular_diameter = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);
				if (angular_diameter > 0.0) {
					// I know tan(0) is 0, but let's not risk it with numerical precision.
					// technically this will keep expanding until reaching the sun, but all we care
					// is expand until we reach the radius of the near plane (there can't be more occluders than that)
					angular_diameter = Math::tan(Math::deg2rad(angular_diameter));
				} else {
					angular_diameter = 0.0;
				}

				if (light_data.shadow_enabled) {
					RS::LightDirectionalShadowMode smode = storage->light_directional_get_shadow_mode(base);

					int limit = smode == RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL ? 0 : (smode == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS ? 1 : 3);
					light_data.blend_splits = (smode != RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL) && storage->light_directional_get_blend_splits(base);
					for (int j = 0; j < 4; j++) {
						Rect2 atlas_rect = li->shadow_transform[j].atlas_rect;
						CameraMatrix matrix = li->shadow_transform[j].camera;
						float split = li->shadow_transform[MIN(limit, j)].split;

						CameraMatrix bias;
						bias.set_light_bias();
						CameraMatrix rectm;
						rectm.set_light_atlas_rect(atlas_rect);

						Transform3D modelview = (inverse_transform * li->shadow_transform[j].transform).inverse();

						CameraMatrix shadow_mtx = rectm * bias * matrix * modelview;
						light_data.shadow_split_offsets[j] = split;
						float bias_scale = li->shadow_transform[j].bias_scale;
						light_data.shadow_bias[j] = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) / 100.0 * bias_scale;
						light_data.shadow_normal_bias[j] = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * li->shadow_transform[j].shadow_texel_size;
						light_data.shadow_transmittance_bias[j] = storage->light_get_transmittance_bias(base) * bias_scale;
						light_data.shadow_z_range[j] = li->shadow_transform[j].farplane;
						light_data.shadow_range_begin[j] = li->shadow_transform[j].range_begin;
						RendererStorageRD::store_camera(shadow_mtx, light_data.shadow_matrices[j]);

						Vector2 uv_scale = li->shadow_transform[j].uv_scale;
						uv_scale *= atlas_rect.size; //adapt to atlas size
						switch (j) {
							case 0: {
								light_data.uv_scale1[0] = uv_scale.x;
								light_data.uv_scale1[1] = uv_scale.y;
							} break;
							case 1: {
								light_data.uv_scale2[0] = uv_scale.x;
								light_data.uv_scale2[1] = uv_scale.y;
							} break;
							case 2: {
								light_data.uv_scale3[0] = uv_scale.x;
								light_data.uv_scale3[1] = uv_scale.y;
							} break;
							case 3: {
								light_data.uv_scale4[0] = uv_scale.x;
								light_data.uv_scale4[1] = uv_scale.y;
							} break;
						}
					}

					float fade_start = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_FADE_START);
					light_data.fade_from = -light_data.shadow_split_offsets[3] * MIN(fade_start, 0.999); //using 1.0 would break smoothstep
					light_data.fade_to = -light_data.shadow_split_offsets[3];
					light_data.shadow_volumetric_fog_fade = 1.0 / storage->light_get_shadow_volumetric_fog_fade(base);

					light_data.soft_shadow_scale = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BLUR);
					light_data.softshadow_angle = angular_diameter;
					light_data.bake_mode = storage->light_get_bake_mode(base);

					if (angular_diameter <= 0.0) {
						light_data.soft_shadow_scale *= directional_shadow_quality_radius_get(); // Only use quality radius for PCF
					}
				}

				r_directional_light_count++;
			} break;
			case RS::LIGHT_OMNI: {
				if (cluster.omni_light_count >= cluster.max_lights) {
					continue;
				}

				cluster.omni_light_sort[cluster.omni_light_count].instance = li;
				cluster.omni_light_sort[cluster.omni_light_count].depth = camera_plane.distance_to(li->transform.origin);
				cluster.omni_light_count++;
			} break;
			case RS::LIGHT_SPOT: {
				if (cluster.spot_light_count >= cluster.max_lights) {
					continue;
				}

				cluster.spot_light_sort[cluster.spot_light_count].instance = li;
				cluster.spot_light_sort[cluster.spot_light_count].depth = camera_plane.distance_to(li->transform.origin);
				cluster.spot_light_count++;
			} break;
		}

		li->last_pass = RSG::rasterizer->get_frame_number();
	}

	if (cluster.omni_light_count) {
		SortArray<Cluster::InstanceSort<LightInstance>> sorter;
		sorter.sort(cluster.omni_light_sort, cluster.omni_light_count);
	}

	if (cluster.spot_light_count) {
		SortArray<Cluster::InstanceSort<LightInstance>> sorter;
		sorter.sort(cluster.spot_light_sort, cluster.spot_light_count);
	}

	ShadowAtlas *shadow_atlas = nullptr;

	if (p_shadow_atlas.is_valid() && p_using_shadows) {
		shadow_atlas = shadow_atlas_owner.get_or_null(p_shadow_atlas);
	}

	bool using_forward_ids = _uses_forward_ids();

	for (uint32_t i = 0; i < (cluster.omni_light_count + cluster.spot_light_count); i++) {
		uint32_t index = (i < cluster.omni_light_count) ? i : i - (cluster.omni_light_count);
		Cluster::LightData &light_data = (i < cluster.omni_light_count) ? cluster.omni_lights[index] : cluster.spot_lights[index];
		RS::LightType type = (i < cluster.omni_light_count) ? RS::LIGHT_OMNI : RS::LIGHT_SPOT;
		LightInstance *li = (i < cluster.omni_light_count) ? cluster.omni_light_sort[index].instance : cluster.spot_light_sort[index].instance;
		RID base = li->light;

		if (using_forward_ids) {
			_map_forward_id(type == RS::LIGHT_OMNI ? FORWARD_ID_TYPE_OMNI_LIGHT : FORWARD_ID_TYPE_SPOT_LIGHT, li->forward_id, index);
		}

		Transform3D light_transform = li->transform;

		float sign = storage->light_is_negative(base) ? -1 : 1;
		Color linear_col = storage->light_get_color(base).to_linear();

		light_data.attenuation = storage->light_get_param(base, RS::LIGHT_PARAM_ATTENUATION);

		float energy = sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI;

		light_data.color[0] = linear_col.r * energy;
		light_data.color[1] = linear_col.g * energy;
		light_data.color[2] = linear_col.b * energy;
		light_data.specular_amount = storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR) * 2.0;
		light_data.bake_mode = storage->light_get_bake_mode(base);

		float radius = MAX(0.001, storage->light_get_param(base, RS::LIGHT_PARAM_RANGE));
		light_data.inv_radius = 1.0 / radius;

		Vector3 pos = inverse_transform.xform(light_transform.origin);

		light_data.position[0] = pos.x;
		light_data.position[1] = pos.y;
		light_data.position[2] = pos.z;

		Vector3 direction = inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, -1))).normalized();

		light_data.direction[0] = direction.x;
		light_data.direction[1] = direction.y;
		light_data.direction[2] = direction.z;

		float size = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

		light_data.size = size;

		light_data.inv_spot_attenuation = 1.0f / storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ATTENUATION);
		float spot_angle = storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ANGLE);
		light_data.cos_spot_angle = Math::cos(Math::deg2rad(spot_angle));

		light_data.mask = storage->light_get_cull_mask(base);

		light_data.atlas_rect[0] = 0;
		light_data.atlas_rect[1] = 0;
		light_data.atlas_rect[2] = 0;
		light_data.atlas_rect[3] = 0;

		RID projector = storage->light_get_projector(base);

		if (projector.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(projector);

			if (type == RS::LIGHT_SPOT) {
				light_data.projector_rect[0] = rect.position.x;
				light_data.projector_rect[1] = rect.position.y + rect.size.height; //flip because shadow is flipped
				light_data.projector_rect[2] = rect.size.width;
				light_data.projector_rect[3] = -rect.size.height;
			} else {
				light_data.projector_rect[0] = rect.position.x;
				light_data.projector_rect[1] = rect.position.y;
				light_data.projector_rect[2] = rect.size.width;
				light_data.projector_rect[3] = rect.size.height * 0.5; //used by dp, so needs to be half
			}
		} else {
			light_data.projector_rect[0] = 0;
			light_data.projector_rect[1] = 0;
			light_data.projector_rect[2] = 0;
			light_data.projector_rect[3] = 0;
		}

		if (shadow_atlas && shadow_atlas->shadow_owners.has(li->self)) {
			// fill in the shadow information

			light_data.shadow_enabled = true;

			float shadow_texel_size = light_instance_get_shadow_texel_size(li->self, p_shadow_atlas);
			light_data.shadow_normal_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * shadow_texel_size * 10.0;

			if (type == RS::LIGHT_SPOT) {
				light_data.shadow_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) / 100.0;
			} else { //omni
				light_data.shadow_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS);
			}

			light_data.transmittance_bias = storage->light_get_transmittance_bias(base);

			Vector2i omni_offset;
			Rect2 rect = light_instance_get_shadow_atlas_rect(li->self, p_shadow_atlas, omni_offset);

			light_data.atlas_rect[0] = rect.position.x;
			light_data.atlas_rect[1] = rect.position.y;
			light_data.atlas_rect[2] = rect.size.width;
			light_data.atlas_rect[3] = rect.size.height;

			light_data.soft_shadow_scale = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BLUR);
			light_data.shadow_volumetric_fog_fade = 1.0 / storage->light_get_shadow_volumetric_fog_fade(base);

			if (type == RS::LIGHT_OMNI) {
				Transform3D proj = (inverse_transform * light_transform).inverse();

				RendererStorageRD::store_transform(proj, light_data.shadow_matrix);

				if (size > 0.0) {
					light_data.soft_shadow_size = size;
				} else {
					light_data.soft_shadow_size = 0.0;
					light_data.soft_shadow_scale *= shadows_quality_radius_get(); // Only use quality radius for PCF
				}

				light_data.direction[0] = omni_offset.x * float(rect.size.width);
				light_data.direction[1] = omni_offset.y * float(rect.size.height);
			} else if (type == RS::LIGHT_SPOT) {
				Transform3D modelview = (inverse_transform * light_transform).inverse();
				CameraMatrix bias;
				bias.set_light_bias();

				CameraMatrix shadow_mtx = bias * li->shadow_transform[0].camera * modelview;
				RendererStorageRD::store_camera(shadow_mtx, light_data.shadow_matrix);

				if (size > 0.0) {
					CameraMatrix cm = li->shadow_transform[0].camera;
					float half_np = cm.get_z_near() * Math::tan(Math::deg2rad(spot_angle));
					light_data.soft_shadow_size = (size * 0.5 / radius) / (half_np / cm.get_z_near()) * rect.size.width;
				} else {
					light_data.soft_shadow_size = 0.0;
					light_data.soft_shadow_scale *= shadows_quality_radius_get(); // Only use quality radius for PCF
				}
			}
		} else {
			light_data.shadow_enabled = false;
		}

		li->cull_mask = storage->light_get_cull_mask(base);

		if (current_cluster_builder != nullptr) {
			current_cluster_builder->add_light(type == RS::LIGHT_SPOT ? ClusterBuilderRD::LIGHT_TYPE_SPOT : ClusterBuilderRD::LIGHT_TYPE_OMNI, light_transform, radius, spot_angle);
		}

		r_positional_light_count++;
	}

	//update without barriers
	if (cluster.omni_light_count) {
		RD::get_singleton()->buffer_update(cluster.omni_light_buffer, 0, sizeof(Cluster::LightData) * cluster.omni_light_count, cluster.omni_lights, RD::BARRIER_MASK_RASTER | RD::BARRIER_MASK_COMPUTE);
	}

	if (cluster.spot_light_count) {
		RD::get_singleton()->buffer_update(cluster.spot_light_buffer, 0, sizeof(Cluster::LightData) * cluster.spot_light_count, cluster.spot_lights, RD::BARRIER_MASK_RASTER | RD::BARRIER_MASK_COMPUTE);
	}

	if (r_directional_light_count) {
		RD::get_singleton()->buffer_update(cluster.directional_light_buffer, 0, sizeof(Cluster::DirectionalLightData) * r_directional_light_count, cluster.directional_lights, RD::BARRIER_MASK_RASTER | RD::BARRIER_MASK_COMPUTE);
	}
}

void RendererSceneRenderRD::_setup_decals(const PagedArray<RID> &p_decals, const Transform3D &p_camera_inverse_xform) {
	Transform3D uv_xform;
	uv_xform.basis.scale(Vector3(2.0, 1.0, 2.0));
	uv_xform.origin = Vector3(-1.0, 0.0, -1.0);

	uint32_t decal_count = p_decals.size();

	cluster.decal_count = 0;

	for (uint32_t i = 0; i < decal_count; i++) {
		if (cluster.decal_count == cluster.max_decals) {
			break;
		}

		DecalInstance *di = decal_instance_owner.get_or_null(p_decals[i]);
		if (!di) {
			continue;
		}
		RID decal = di->decal;

		Transform3D xform = di->transform;

		real_t distance = -p_camera_inverse_xform.xform(xform.origin).z;

		if (storage->decal_is_distance_fade_enabled(decal)) {
			float fade_begin = storage->decal_get_distance_fade_begin(decal);
			float fade_length = storage->decal_get_distance_fade_length(decal);

			if (distance > fade_begin) {
				if (distance > fade_begin + fade_length) {
					continue; // do not use this decal, its invisible
				}
			}
		}

		cluster.decal_sort[cluster.decal_count].instance = di;
		cluster.decal_sort[cluster.decal_count].depth = distance;
		cluster.decal_count++;
	}

	if (cluster.decal_count > 0) {
		SortArray<Cluster::InstanceSort<DecalInstance>> sort_array;
		sort_array.sort(cluster.decal_sort, cluster.decal_count);
	}

	bool using_forward_ids = _uses_forward_ids();
	for (uint32_t i = 0; i < cluster.decal_count; i++) {
		DecalInstance *di = cluster.decal_sort[i].instance;
		RID decal = di->decal;

		if (using_forward_ids) {
			_map_forward_id(FORWARD_ID_TYPE_DECAL, di->forward_id, i);
		}

		di->cull_mask = storage->decal_get_cull_mask(decal);

		Transform3D xform = di->transform;
		float fade = 1.0;

		if (storage->decal_is_distance_fade_enabled(decal)) {
			real_t distance = -p_camera_inverse_xform.xform(xform.origin).z;
			float fade_begin = storage->decal_get_distance_fade_begin(decal);
			float fade_length = storage->decal_get_distance_fade_length(decal);

			if (distance > fade_begin) {
				fade = 1.0 - (distance - fade_begin) / fade_length;
			}
		}

		Cluster::DecalData &dd = cluster.decals[i];

		Vector3 decal_extents = storage->decal_get_extents(decal);

		Transform3D scale_xform;
		scale_xform.basis.scale(decal_extents);
		Transform3D to_decal_xform = (p_camera_inverse_xform * di->transform * scale_xform * uv_xform).affine_inverse();
		RendererStorageRD::store_transform(to_decal_xform, dd.xform);

		Vector3 normal = xform.basis.get_axis(Vector3::AXIS_Y).normalized();
		normal = p_camera_inverse_xform.basis.xform(normal); //camera is normalized, so fine

		dd.normal[0] = normal.x;
		dd.normal[1] = normal.y;
		dd.normal[2] = normal.z;
		dd.normal_fade = storage->decal_get_normal_fade(decal);

		RID albedo_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_ALBEDO);
		RID emission_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_EMISSION);
		if (albedo_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(albedo_tex);
			dd.albedo_rect[0] = rect.position.x;
			dd.albedo_rect[1] = rect.position.y;
			dd.albedo_rect[2] = rect.size.x;
			dd.albedo_rect[3] = rect.size.y;
		} else {
			if (!emission_tex.is_valid()) {
				continue; //no albedo, no emission, no decal.
			}
			dd.albedo_rect[0] = 0;
			dd.albedo_rect[1] = 0;
			dd.albedo_rect[2] = 0;
			dd.albedo_rect[3] = 0;
		}

		RID normal_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_NORMAL);

		if (normal_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(normal_tex);
			dd.normal_rect[0] = rect.position.x;
			dd.normal_rect[1] = rect.position.y;
			dd.normal_rect[2] = rect.size.x;
			dd.normal_rect[3] = rect.size.y;

			Basis normal_xform = p_camera_inverse_xform.basis * xform.basis.orthonormalized();
			RendererStorageRD::store_basis_3x4(normal_xform, dd.normal_xform);
		} else {
			dd.normal_rect[0] = 0;
			dd.normal_rect[1] = 0;
			dd.normal_rect[2] = 0;
			dd.normal_rect[3] = 0;
		}

		RID orm_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_ORM);
		if (orm_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(orm_tex);
			dd.orm_rect[0] = rect.position.x;
			dd.orm_rect[1] = rect.position.y;
			dd.orm_rect[2] = rect.size.x;
			dd.orm_rect[3] = rect.size.y;
		} else {
			dd.orm_rect[0] = 0;
			dd.orm_rect[1] = 0;
			dd.orm_rect[2] = 0;
			dd.orm_rect[3] = 0;
		}

		if (emission_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(emission_tex);
			dd.emission_rect[0] = rect.position.x;
			dd.emission_rect[1] = rect.position.y;
			dd.emission_rect[2] = rect.size.x;
			dd.emission_rect[3] = rect.size.y;
		} else {
			dd.emission_rect[0] = 0;
			dd.emission_rect[1] = 0;
			dd.emission_rect[2] = 0;
			dd.emission_rect[3] = 0;
		}

		Color modulate = storage->decal_get_modulate(decal);
		dd.modulate[0] = modulate.r;
		dd.modulate[1] = modulate.g;
		dd.modulate[2] = modulate.b;
		dd.modulate[3] = modulate.a * fade;
		dd.emission_energy = storage->decal_get_emission_energy(decal) * fade;
		dd.albedo_mix = storage->decal_get_albedo_mix(decal);
		dd.mask = storage->decal_get_cull_mask(decal);
		dd.upper_fade = storage->decal_get_upper_fade(decal);
		dd.lower_fade = storage->decal_get_lower_fade(decal);

		if (current_cluster_builder != nullptr) {
			current_cluster_builder->add_box(ClusterBuilderRD::BOX_TYPE_DECAL, xform, decal_extents);
		}
	}

	if (cluster.decal_count > 0) {
		RD::get_singleton()->buffer_update(cluster.decal_buffer, 0, sizeof(Cluster::DecalData) * cluster.decal_count, cluster.decals, RD::BARRIER_MASK_RASTER | RD::BARRIER_MASK_COMPUTE);
	}
}

////////////////////////////////////////////////////////////////////////////////
// FOG SHADER

void RendererSceneRenderRD::FogShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;
	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["fog"] = ShaderCompiler::STAGE_COMPUTE;

	uses_time = false;

	actions.usage_flag_pointers["TIME"] = &uses_time;

	actions.uniforms = &uniforms;

	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;

	Error err = scene_singleton->volumetric_fog.compiler.compile(RS::SHADER_FOG, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Fog shader compilation failed.");

	if (version.is_null()) {
		version = scene_singleton->volumetric_fog.shader.version_create();
	}

	scene_singleton->volumetric_fog.shader.version_set_compute_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_COMPUTE], gen_code.defines);
	ERR_FAIL_COND(!scene_singleton->volumetric_fog.shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	pipeline = RD::get_singleton()->compute_pipeline_create(scene_singleton->volumetric_fog.shader.version_get_shader(version, 0));

	valid = true;
}

void RendererSceneRenderRD::FogShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
	if (!p_texture.is_valid()) {
		if (default_texture_params.has(p_name) && default_texture_params[p_name].has(p_index)) {
			default_texture_params[p_name].erase(p_index);

			if (default_texture_params[p_name].is_empty()) {
				default_texture_params.erase(p_name);
			}
		}
	} else {
		if (!default_texture_params.has(p_name)) {
			default_texture_params[p_name] = Map<int, RID>();
		}
		default_texture_params[p_name][p_index] = p_texture;
	}
}

void RendererSceneRenderRD::FogShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {
		if (E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		if (E->get().texture_order >= 0) {
			order[E->get().texture_order + 100000] = E->key();
		} else {
			order[E->get().order] = E->key();
		}
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E->get()]);
		pi.name = E->get();
		p_param_list->push_back(pi);
	}
}

void RendererSceneRenderRD::FogShaderData::get_instance_param_list(List<RendererStorage::InstanceShaderParam> *p_param_list) const {
	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {
		if (E->get().scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E->get());
		p.info.name = E->key(); //supply name
		p.index = E->get().instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E->get().default_value, E->get().type, E->get().array_size, E->get().hint);
		p_param_list->push_back(p);
	}
}

bool RendererSceneRenderRD::FogShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RendererSceneRenderRD::FogShaderData::is_animated() const {
	return false;
}

bool RendererSceneRenderRD::FogShaderData::casts_shadows() const {
	return false;
}

Variant RendererSceneRenderRD::FogShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode RendererSceneRenderRD::FogShaderData::get_native_source_code() const {
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;

	return scene_singleton->volumetric_fog.shader.version_get_native_source_code(version);
}

RendererSceneRenderRD::FogShaderData::FogShaderData() {
	valid = false;
}

RendererSceneRenderRD::FogShaderData::~FogShaderData() {
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;
	ERR_FAIL_COND(!scene_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		scene_singleton->volumetric_fog.shader.version_free(version);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Fog material

bool RendererSceneRenderRD::FogMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;

	uniform_set_updated = true;

	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, scene_singleton->volumetric_fog.shader.version_get_shader(shader_data->version, 0), VolumetricFogShader::FogSet::FOG_SET_MATERIAL);
}

RendererSceneRenderRD::FogMaterialData::~FogMaterialData() {
	free_parameters_uniform_set(uniform_set);
}

RendererStorageRD::ShaderData *RendererSceneRenderRD::_create_fog_shader_func() {
	FogShaderData *shader_data = memnew(FogShaderData);
	return shader_data;
}

RendererStorageRD::ShaderData *RendererSceneRenderRD::_create_fog_shader_funcs() {
	return static_cast<RendererSceneRenderRD *>(RendererSceneRenderRD::singleton)->_create_fog_shader_func();
};

RendererStorageRD::MaterialData *RendererSceneRenderRD::_create_fog_material_func(FogShaderData *p_shader) {
	FogMaterialData *material_data = memnew(FogMaterialData);
	material_data->shader_data = p_shader;
	//update will happen later anyway so do nothing.
	return material_data;
}

RendererStorageRD::MaterialData *RendererSceneRenderRD::_create_fog_material_funcs(RendererStorageRD::ShaderData *p_shader) {
	return static_cast<RendererSceneRenderRD *>(RendererSceneRenderRD::singleton)->_create_fog_material_func(static_cast<FogShaderData *>(p_shader));
};

////////////////////////////////////////////////////////////////////////////////
// Volumetric Fog

void RendererSceneRenderRD::_volumetric_fog_erase(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->volumetric_fog);

	RD::get_singleton()->free(rb->volumetric_fog->prev_light_density_map);
	RD::get_singleton()->free(rb->volumetric_fog->light_density_map);
	RD::get_singleton()->free(rb->volumetric_fog->fog_map);
	RD::get_singleton()->free(rb->volumetric_fog->density_map);
	RD::get_singleton()->free(rb->volumetric_fog->light_map);
	RD::get_singleton()->free(rb->volumetric_fog->emissive_map);

	if (rb->volumetric_fog->fog_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->fog_uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->fog_uniform_set);
	}
	if (rb->volumetric_fog->process_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->process_uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->process_uniform_set);
	}
	if (rb->volumetric_fog->process_uniform_set2.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->process_uniform_set2)) {
		RD::get_singleton()->free(rb->volumetric_fog->process_uniform_set2);
	}
	if (rb->volumetric_fog->sdfgi_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->sdfgi_uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->sdfgi_uniform_set);
	}
	if (rb->volumetric_fog->sky_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->sky_uniform_set)) {
		RD::get_singleton()->free(rb->volumetric_fog->sky_uniform_set);
	}

	memdelete(rb->volumetric_fog);

	rb->volumetric_fog = nullptr;
}

Vector3i RendererSceneRenderRD::_point_get_position_in_froxel_volume(const Vector3 &p_point, float fog_end, const Vector2 &fog_near_size, const Vector2 &fog_far_size, float volumetric_fog_detail_spread, const Vector3 &fog_size, const Transform3D &p_cam_transform) {
	Vector3 view_position = p_cam_transform.affine_inverse().xform(p_point);
	view_position.z = MIN(view_position.z, -0.01); // Clamp to the front of camera
	Vector3 fog_position = Vector3(0, 0, 0);

	view_position.y = -view_position.y;
	fog_position.z = -view_position.z / fog_end;
	fog_position.x = (view_position.x / (2 * (fog_near_size.x * (1.0 - fog_position.z) + fog_far_size.x * fog_position.z))) + 0.5;
	fog_position.y = (view_position.y / (2 * (fog_near_size.y * (1.0 - fog_position.z) + fog_far_size.y * fog_position.z))) + 0.5;
	fog_position.z = Math::pow(float(fog_position.z), float(1.0 / volumetric_fog_detail_spread));
	fog_position = fog_position * fog_size - Vector3(0.5, 0.5, 0.5);

	fog_position.x = CLAMP(fog_position.x, 0.0, fog_size.x);
	fog_position.y = CLAMP(fog_position.y, 0.0, fog_size.y);
	fog_position.z = CLAMP(fog_position.z, 0.0, fog_size.z);

	return Vector3i(fog_position);
}

void RendererSceneRenderRD::_update_volumetric_fog(RID p_render_buffers, RID p_environment, const CameraMatrix &p_cam_projection, const Transform3D &p_cam_transform, RID p_shadow_atlas, int p_directional_light_count, bool p_use_directional_shadows, int p_positional_light_count, int p_voxel_gi_count, const PagedArray<RID> &p_fog_volumes) {
	ERR_FAIL_COND(!is_clustered_enabled()); // can't use volumetric fog without clustered
	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);
	RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_environment);

	float ratio = float(rb->width) / float((rb->width + rb->height) / 2);
	uint32_t target_width = uint32_t(float(volumetric_fog_size) * ratio);
	uint32_t target_height = uint32_t(float(volumetric_fog_size) / ratio);

	if (rb->volumetric_fog) {
		//validate
		if (!env || !env->volumetric_fog_enabled || rb->volumetric_fog->width != target_width || rb->volumetric_fog->height != target_height || rb->volumetric_fog->depth != volumetric_fog_depth) {
			_volumetric_fog_erase(rb);
		}
	}

	if (!env || !env->volumetric_fog_enabled) {
		//no reason to enable or update, bye
		return;
	}

	RENDER_TIMESTAMP(">Volumetric Fog");
	RD::get_singleton()->draw_command_begin_label("Volumetric Fog");

	if (env && env->volumetric_fog_enabled && !rb->volumetric_fog) {
		//required volumetric fog but not existing, create
		rb->volumetric_fog = memnew(VolumetricFog);
		rb->volumetric_fog->width = target_width;
		rb->volumetric_fog->height = target_height;
		rb->volumetric_fog->depth = volumetric_fog_depth;

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.width = target_width;
		tf.height = target_height;
		tf.depth = volumetric_fog_depth;
		tf.texture_type = RD::TEXTURE_TYPE_3D;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		rb->volumetric_fog->light_density_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->light_density_map, "Fog light-density map");

		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

		rb->volumetric_fog->prev_light_density_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->prev_light_density_map, "Fog previous light-density map");
		RD::get_singleton()->texture_clear(rb->volumetric_fog->prev_light_density_map, Color(0, 0, 0, 0), 0, 1, 0, 1);

		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

		rb->volumetric_fog->fog_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->fog_map, "Fog map");

#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
		Vector<uint8_t> dm;
		dm.resize(target_width * target_height * volumetric_fog_depth * 4);
		dm.fill(0);

		rb->volumetric_fog->density_map = RD::get_singleton()->storage_buffer_create(dm.size(), dm);
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->density_map, "Fog density map");
		rb->volumetric_fog->light_map = RD::get_singleton()->storage_buffer_create(dm.size(), dm);
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->light_map, "Fog light map");
		rb->volumetric_fog->emissive_map = RD::get_singleton()->storage_buffer_create(dm.size(), dm);
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->emissive_map, "Fog emissive map");
#else
		tf.format = RD::DATA_FORMAT_R32_UINT;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
		rb->volumetric_fog->density_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->density_map, "Fog density map");
		RD::get_singleton()->texture_clear(rb->volumetric_fog->density_map, Color(0, 0, 0, 0), 0, 1, 0, 1);
		rb->volumetric_fog->light_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->light_map, "Fog light map");
		RD::get_singleton()->texture_clear(rb->volumetric_fog->light_map, Color(0, 0, 0, 0), 0, 1, 0, 1);
		rb->volumetric_fog->emissive_map = RD::get_singleton()->texture_create(tf, RD::TextureView());
		RD::get_singleton()->set_resource_name(rb->volumetric_fog->emissive_map, "Fog emissive map");
		RD::get_singleton()->texture_clear(rb->volumetric_fog->emissive_map, Color(0, 0, 0, 0), 0, 1, 0, 1);
#endif

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(rb->volumetric_fog->fog_map);
			uniforms.push_back(u);
		}

		rb->volumetric_fog->sky_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky.sky_shader.default_shader_rd, RendererSceneSkyRD::SKY_SET_FOG);
	}

	if (p_fog_volumes.size() > 0) {
		RD::get_singleton()->draw_command_begin_label("Render Volumetric Fog Volumes");

		RENDER_TIMESTAMP("Render Fog Volumes");

		VolumetricFogShader::VolumeUBO params;

		Vector2 frustum_near_size = p_cam_projection.get_viewport_half_extents();
		Vector2 frustum_far_size = p_cam_projection.get_far_plane_half_extents();
		float z_near = p_cam_projection.get_z_near();
		float z_far = p_cam_projection.get_z_far();
		float fog_end = env->volumetric_fog_length;

		Vector2 fog_far_size = frustum_near_size.lerp(frustum_far_size, (fog_end - z_near) / (z_far - z_near));
		Vector2 fog_near_size;
		if (p_cam_projection.is_orthogonal()) {
			fog_near_size = fog_far_size;
		} else {
			fog_near_size = Vector2();
		}

		params.fog_frustum_size_begin[0] = fog_near_size.x;
		params.fog_frustum_size_begin[1] = fog_near_size.y;

		params.fog_frustum_size_end[0] = fog_far_size.x;
		params.fog_frustum_size_end[1] = fog_far_size.y;

		params.fog_frustum_end = fog_end;
		params.z_near = z_near;
		params.z_far = z_far;
		params.time = time;

		params.fog_volume_size[0] = rb->volumetric_fog->width;
		params.fog_volume_size[1] = rb->volumetric_fog->height;
		params.fog_volume_size[2] = rb->volumetric_fog->depth;

		params.use_temporal_reprojection = env->volumetric_fog_temporal_reprojection;
		params.temporal_frame = RSG::rasterizer->get_frame_number() % VolumetricFog::MAX_TEMPORAL_FRAMES;
		params.detail_spread = env->volumetric_fog_detail_spread;
		params.temporal_blend = env->volumetric_fog_temporal_reprojection_amount;

		Transform3D to_prev_cam_view = rb->volumetric_fog->prev_cam_transform.affine_inverse() * p_cam_transform;
		storage->store_transform(to_prev_cam_view, params.to_prev_view);
		storage->store_transform(p_cam_transform, params.transform);

		RD::get_singleton()->buffer_update(volumetric_fog.volume_ubo, 0, sizeof(VolumetricFogShader::VolumeUBO), &params, RD::BARRIER_MASK_COMPUTE);

		if (rb->volumetric_fog->fog_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->fog_uniform_set)) {
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
#else
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
#endif
				u.binding = 1;
				u.ids.push_back(rb->volumetric_fog->emissive_map);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 2;
				u.ids.push_back(volumetric_fog.volume_ubo);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
#else
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
#endif
				u.binding = 3;
				u.ids.push_back(rb->volumetric_fog->density_map);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
#else
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
#endif
				u.binding = 4;
				u.ids.push_back(rb->volumetric_fog->light_map);
				uniforms.push_back(u);
			}

			rb->volumetric_fog->fog_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.default_shader_rd, VolumetricFogShader::FogSet::FOG_SET_UNIFORMS);
		}

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
		bool any_uses_time = false;

		for (int i = 0; i < (int)p_fog_volumes.size(); i++) {
			FogVolumeInstance *fog_volume_instance = fog_volume_instance_owner.get_or_null(p_fog_volumes[i]);
			ERR_FAIL_COND(!fog_volume_instance);
			RID fog_volume = fog_volume_instance->volume;

			RID fog_material = storage->fog_volume_get_material(fog_volume);

			FogMaterialData *material = nullptr;

			if (fog_material.is_valid()) {
				material = (FogMaterialData *)storage->material_get_data(fog_material, RendererStorageRD::SHADER_TYPE_FOG);
				if (!material || !material->shader_data->valid) {
					material = nullptr;
				}
			}

			if (!material) {
				fog_material = volumetric_fog.default_material;
				material = (FogMaterialData *)storage->material_get_data(fog_material, RendererStorageRD::SHADER_TYPE_FOG);
			}

			ERR_FAIL_COND(!material);

			FogShaderData *shader_data = material->shader_data;

			ERR_FAIL_COND(!shader_data);

			any_uses_time |= shader_data->uses_time;

			Vector3i min = Vector3i();
			Vector3i max = Vector3i();
			Vector3i kernel_size = Vector3i();

			Vector3 position = fog_volume_instance->transform.get_origin();
			RS::FogVolumeShape volume_type = storage->fog_volume_get_shape(fog_volume);
			Vector3 extents = storage->fog_volume_get_extents(fog_volume);

			if (volume_type == RS::FOG_VOLUME_SHAPE_BOX || volume_type == RS::FOG_VOLUME_SHAPE_ELLIPSOID) {
				Vector3i points[8];
				points[0] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(extents.x, extents.y, extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[1] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(-extents.x, extents.y, extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[2] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(extents.x, -extents.y, extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[3] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(-extents.x, -extents.y, extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[4] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(extents.x, extents.y, -extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[5] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(-extents.x, extents.y, -extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[6] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(extents.x, -extents.y, -extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);
				points[7] = _point_get_position_in_froxel_volume(fog_volume_instance->transform.xform(Vector3(-extents.x, -extents.y, -extents.z)), fog_end, fog_near_size, fog_far_size, env->volumetric_fog_detail_spread, Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth), p_cam_transform);

				min = Vector3i(int32_t(rb->volumetric_fog->width) - 1, int32_t(rb->volumetric_fog->height) - 1, int32_t(rb->volumetric_fog->depth) - 1);
				max = Vector3i(1, 1, 1);

				for (int j = 0; j < 8; j++) {
					min = Vector3i(MIN(min.x, points[j].x), MIN(min.y, points[j].y), MIN(min.z, points[j].z));
					max = Vector3i(MAX(max.x, points[j].x), MAX(max.y, points[j].y), MAX(max.z, points[j].z));
				}

				kernel_size = max - min;
			} else {
				// Volume type global runs on all cells
				extents = Vector3(rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth);
				min = Vector3i(0, 0, 0);
				kernel_size = Vector3i(int32_t(rb->volumetric_fog->width), int32_t(rb->volumetric_fog->height), int32_t(rb->volumetric_fog->depth));
			}

			if (kernel_size.x == 0 || kernel_size.y == 0 || kernel_size.z == 0) {
				continue;
			}

			volumetric_fog.push_constant.position[0] = position.x;
			volumetric_fog.push_constant.position[1] = position.y;
			volumetric_fog.push_constant.position[2] = position.z;
			volumetric_fog.push_constant.extents[0] = extents.x;
			volumetric_fog.push_constant.extents[1] = extents.y;
			volumetric_fog.push_constant.extents[2] = extents.z;
			volumetric_fog.push_constant.corner[0] = min.x;
			volumetric_fog.push_constant.corner[1] = min.y;
			volumetric_fog.push_constant.corner[2] = min.z;
			volumetric_fog.push_constant.shape = uint32_t(storage->fog_volume_get_shape(fog_volume));
			storage->store_transform(fog_volume_instance->transform.affine_inverse(), volumetric_fog.push_constant.transform);

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, shader_data->pipeline);

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->fog_uniform_set, VolumetricFogShader::FogSet::FOG_SET_UNIFORMS);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &volumetric_fog.push_constant, sizeof(VolumetricFogShader::FogPushConstant));
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, volumetric_fog.base_uniform_set, VolumetricFogShader::FogSet::FOG_SET_BASE);
			if (material->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(material->uniform_set)) { // Material may not have a uniform set.
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, material->uniform_set, VolumetricFogShader::FogSet::FOG_SET_MATERIAL);
			}

			RD::get_singleton()->compute_list_dispatch_threads(compute_list, kernel_size.x, kernel_size.y, kernel_size.z);
		}
		if (any_uses_time || env->volumetric_fog_temporal_reprojection) {
			RenderingServerDefault::redraw_request();
		}

		RD::get_singleton()->draw_command_end_label();

		RD::get_singleton()->compute_list_end();
	}

	if (rb->volumetric_fog->process_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->process_uniform_set)) {
		//re create uniform set if needed
		Vector<RD::Uniform> uniforms;
		Vector<RD::Uniform> copy_uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;
			ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_shadow_atlas);
			if (shadow_atlas == nullptr || shadow_atlas->depth.is_null()) {
				u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_BLACK));
			} else {
				u.ids.push_back(shadow_atlas->depth);
			}

			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;
			if (directional_shadow.depth.is_valid()) {
				u.ids.push_back(directional_shadow.depth);
			} else {
				u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_BLACK));
			}
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 3;
			u.ids.push_back(get_omni_light_buffer());
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 4;
			u.ids.push_back(get_spot_light_buffer());
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 5;
			u.ids.push_back(get_directional_light_buffer());
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 6;
			u.ids.push_back(rb->cluster_builder->get_cluster_buffer());
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 7;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 8;
			u.ids.push_back(rb->volumetric_fog->light_density_map);
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 9;
			u.ids.push_back(rb->volumetric_fog->fog_map);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 9;
			u.ids.push_back(rb->volumetric_fog->prev_light_density_map);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 10;
			u.ids.push_back(shadow_sampler);
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 11;
			u.ids.push_back(render_buffers_get_voxel_gi_buffer(p_render_buffers));
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 12;
			for (int i = 0; i < RendererSceneGIRD::MAX_VOXEL_GI_INSTANCES; i++) {
				u.ids.push_back(rb->gi.voxel_gi_textures[i]);
			}
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 13;
			u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 14;
			u.ids.push_back(volumetric_fog.params_ubo);
			uniforms.push_back(u);
			copy_uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 15;
			u.ids.push_back(rb->volumetric_fog->prev_light_density_map);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
#else
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
#endif
			u.binding = 16;
			u.ids.push_back(rb->volumetric_fog->density_map);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
#else
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
#endif
			u.binding = 17;
			u.ids.push_back(rb->volumetric_fog->light_map);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
#else
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
#endif
			u.binding = 18;
			u.ids.push_back(rb->volumetric_fog->emissive_map);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 19;
			RID radiance_texture = storage->texture_rd_get_default(is_using_radiance_cubemap_array() ? RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK : RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK);
			RID sky_texture = env->sky.is_valid() ? sky.sky_get_radiance_texture_rd(env->sky) : RID();
			u.ids.push_back(sky_texture.is_valid() ? sky_texture : radiance_texture);
			uniforms.push_back(u);
		}

		rb->volumetric_fog->copy_uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, volumetric_fog.process_shader.version_get_shader(volumetric_fog.process_shader_version, VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_COPY), 0);

		rb->volumetric_fog->process_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.process_shader.version_get_shader(volumetric_fog.process_shader_version, VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_DENSITY), 0);

		SWAP(uniforms.write[7].ids.write[0], uniforms.write[8].ids.write[0]);

		rb->volumetric_fog->process_uniform_set2 = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.process_shader.version_get_shader(volumetric_fog.process_shader_version, 0), 0);
	}

	bool using_sdfgi = env->volumetric_fog_gi_inject > 0.0001 && env->sdfgi_enabled && (rb->sdfgi != nullptr);

	if (using_sdfgi) {
		if (rb->volumetric_fog->sdfgi_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->volumetric_fog->sdfgi_uniform_set)) {
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 0;
				u.ids.push_back(gi.sdfgi_ubo);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				u.ids.push_back(rb->sdfgi->ambient_texture);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 2;
				u.ids.push_back(rb->sdfgi->occlusion_texture);
				uniforms.push_back(u);
			}

			rb->volumetric_fog->sdfgi_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.process_shader.version_get_shader(volumetric_fog.process_shader_version, VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_DENSITY_WITH_SDFGI), 1);
		}
	}

	rb->volumetric_fog->length = env->volumetric_fog_length;
	rb->volumetric_fog->spread = env->volumetric_fog_detail_spread;

	VolumetricFogShader::ParamsUBO params;

	Vector2 frustum_near_size = p_cam_projection.get_viewport_half_extents();
	Vector2 frustum_far_size = p_cam_projection.get_far_plane_half_extents();
	float z_near = p_cam_projection.get_z_near();
	float z_far = p_cam_projection.get_z_far();
	float fog_end = env->volumetric_fog_length;

	Vector2 fog_far_size = frustum_near_size.lerp(frustum_far_size, (fog_end - z_near) / (z_far - z_near));
	Vector2 fog_near_size;
	if (p_cam_projection.is_orthogonal()) {
		fog_near_size = fog_far_size;
	} else {
		fog_near_size = Vector2();
	}

	params.fog_frustum_size_begin[0] = fog_near_size.x;
	params.fog_frustum_size_begin[1] = fog_near_size.y;

	params.fog_frustum_size_end[0] = fog_far_size.x;
	params.fog_frustum_size_end[1] = fog_far_size.y;

	params.ambient_inject = env->volumetric_fog_ambient_inject * env->ambient_light_energy;
	params.z_far = z_far;

	params.fog_frustum_end = fog_end;

	Color ambient_color = env->ambient_light.to_linear();
	params.ambient_color[0] = ambient_color.r;
	params.ambient_color[1] = ambient_color.g;
	params.ambient_color[2] = ambient_color.b;
	params.sky_contribution = env->ambient_sky_contribution;

	params.fog_volume_size[0] = rb->volumetric_fog->width;
	params.fog_volume_size[1] = rb->volumetric_fog->height;
	params.fog_volume_size[2] = rb->volumetric_fog->depth;

	params.directional_light_count = p_directional_light_count;

	Color emission = env->volumetric_fog_emission.to_linear();
	params.base_emission[0] = emission.r * env->volumetric_fog_emission_energy;
	params.base_emission[1] = emission.g * env->volumetric_fog_emission_energy;
	params.base_emission[2] = emission.b * env->volumetric_fog_emission_energy;
	params.base_density = env->volumetric_fog_density;

	Color base_scattering = env->volumetric_fog_scattering.to_linear();
	params.base_scattering[0] = base_scattering.r;
	params.base_scattering[1] = base_scattering.g;
	params.base_scattering[2] = base_scattering.b;
	params.phase_g = env->volumetric_fog_anisotropy;

	params.detail_spread = env->volumetric_fog_detail_spread;
	params.gi_inject = env->volumetric_fog_gi_inject;

	params.cam_rotation[0] = p_cam_transform.basis[0][0];
	params.cam_rotation[1] = p_cam_transform.basis[1][0];
	params.cam_rotation[2] = p_cam_transform.basis[2][0];
	params.cam_rotation[3] = 0;
	params.cam_rotation[4] = p_cam_transform.basis[0][1];
	params.cam_rotation[5] = p_cam_transform.basis[1][1];
	params.cam_rotation[6] = p_cam_transform.basis[2][1];
	params.cam_rotation[7] = 0;
	params.cam_rotation[8] = p_cam_transform.basis[0][2];
	params.cam_rotation[9] = p_cam_transform.basis[1][2];
	params.cam_rotation[10] = p_cam_transform.basis[2][2];
	params.cam_rotation[11] = 0;
	params.filter_axis = 0;
	params.max_voxel_gi_instances = env->volumetric_fog_gi_inject > 0.001 ? p_voxel_gi_count : 0;
	params.temporal_frame = RSG::rasterizer->get_frame_number() % VolumetricFog::MAX_TEMPORAL_FRAMES;

	Transform3D to_prev_cam_view = rb->volumetric_fog->prev_cam_transform.affine_inverse() * p_cam_transform;
	storage->store_transform(to_prev_cam_view, params.to_prev_view);

	params.use_temporal_reprojection = env->volumetric_fog_temporal_reprojection;
	params.temporal_blend = env->volumetric_fog_temporal_reprojection_amount;

	{
		uint32_t cluster_size = rb->cluster_builder->get_cluster_size();
		params.cluster_shift = get_shift_from_power_of_2(cluster_size);

		uint32_t cluster_screen_width = (rb->width - 1) / cluster_size + 1;
		uint32_t cluster_screen_height = (rb->height - 1) / cluster_size + 1;
		params.max_cluster_element_count_div_32 = max_cluster_elements / 32;
		params.cluster_type_size = cluster_screen_width * cluster_screen_height * (params.max_cluster_element_count_div_32 + 32);
		params.cluster_width = cluster_screen_width;

		params.screen_size[0] = rb->width;
		params.screen_size[1] = rb->height;
	}

	Basis sky_transform = env->sky_orientation;
	sky_transform = sky_transform.inverse() * p_cam_transform.basis;
	RendererStorageRD::store_transform_3x3(sky_transform, params.radiance_inverse_xform);

	RD::get_singleton()->draw_command_begin_label("Render Volumetric Fog");

	RENDER_TIMESTAMP("Render Fog");
	RD::get_singleton()->buffer_update(volumetric_fog.params_ubo, 0, sizeof(VolumetricFogShader::ParamsUBO), &params, RD::BARRIER_MASK_COMPUTE);

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.process_pipelines[using_sdfgi ? VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_DENSITY_WITH_SDFGI : VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_DENSITY]);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->process_uniform_set, 0);

	if (using_sdfgi) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->sdfgi_uniform_set, 1);
	}
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	// Copy fog to history buffer
	if (env->volumetric_fog_temporal_reprojection) {
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.process_pipelines[VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_COPY]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->copy_uniform_set, 0);
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth);
		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}
	RD::get_singleton()->draw_command_end_label();

	if (volumetric_fog_filter_active) {
		RD::get_singleton()->draw_command_begin_label("Filter Fog");

		RENDER_TIMESTAMP("Filter Fog");

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.process_pipelines[VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_FILTER]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->process_uniform_set, 0);
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth);

		RD::get_singleton()->compute_list_end();
		//need restart for buffer update

		params.filter_axis = 1;
		RD::get_singleton()->buffer_update(volumetric_fog.params_ubo, 0, sizeof(VolumetricFogShader::ParamsUBO), &params);

		compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.process_pipelines[VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_FILTER]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->process_uniform_set2, 0);
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, rb->volumetric_fog->depth);

		RD::get_singleton()->compute_list_add_barrier(compute_list);
		RD::get_singleton()->draw_command_end_label();
	}

	RENDER_TIMESTAMP("Integrate Fog");
	RD::get_singleton()->draw_command_begin_label("Integrate Fog");

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, volumetric_fog.process_pipelines[VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_FOG]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rb->volumetric_fog->process_uniform_set, 0);
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, rb->volumetric_fog->width, rb->volumetric_fog->height, 1);

	RD::get_singleton()->compute_list_end(RD::BARRIER_MASK_RASTER);

	RENDER_TIMESTAMP("<Volumetric Fog");
	RD::get_singleton()->draw_command_end_label();
	RD::get_singleton()->draw_command_end_label();

	rb->volumetric_fog->prev_cam_transform = p_cam_transform;
}

bool RendererSceneRenderRD::_needs_post_prepass_render(RenderDataRD *p_render_data, bool p_use_gi) {
	if (p_render_data->render_buffers.is_valid()) {
		RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
		if (rb->sdfgi != nullptr) {
			return true;
		}
	}
	return false;
}

void RendererSceneRenderRD::_post_prepass_render(RenderDataRD *p_render_data, bool p_use_gi) {
	if (p_render_data->render_buffers.is_valid()) {
		if (p_use_gi) {
			RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
			ERR_FAIL_COND(rb == nullptr);
			if (rb->sdfgi == nullptr) {
				return;
			}

			RendererSceneEnvironmentRD *env = environment_owner.get_or_null(p_render_data->environment);
			rb->sdfgi->update_probes(env, sky.sky_owner.get_or_null(env->sky));
		}
	}
}

void RendererSceneRenderRD::_pre_resolve_render(RenderDataRD *p_render_data, bool p_use_gi) {
	if (p_render_data->render_buffers.is_valid()) {
		if (p_use_gi) {
			RD::get_singleton()->compute_list_end();
		}
	}
}

void RendererSceneRenderRD::_pre_opaque_render(RenderDataRD *p_render_data, bool p_use_ssao, bool p_use_ssil, bool p_use_gi, RID p_normal_roughness_buffer, RID p_voxel_gi_buffer) {
	// Render shadows while GI is rendering, due to how barriers are handled, this should happen at the same time

	if (p_render_data->render_buffers.is_valid() && p_use_gi) {
		RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
		ERR_FAIL_COND(rb == nullptr);
		if (rb->sdfgi != nullptr) {
			rb->sdfgi->store_probes();
		}
	}

	render_state.cube_shadows.clear();
	render_state.shadows.clear();
	render_state.directional_shadows.clear();

	Plane camera_plane(-p_render_data->cam_transform.basis.get_axis(Vector3::AXIS_Z), p_render_data->cam_transform.origin);
	float lod_distance_multiplier = p_render_data->cam_projection.get_lod_multiplier();
	{
		for (int i = 0; i < render_state.render_shadow_count; i++) {
			LightInstance *li = light_instance_owner.get_or_null(render_state.render_shadows[i].light);

			if (storage->light_get_type(li->light) == RS::LIGHT_DIRECTIONAL) {
				render_state.directional_shadows.push_back(i);
			} else if (storage->light_get_type(li->light) == RS::LIGHT_OMNI && storage->light_omni_get_shadow_mode(li->light) == RS::LIGHT_OMNI_SHADOW_CUBE) {
				render_state.cube_shadows.push_back(i);
			} else {
				render_state.shadows.push_back(i);
			}
		}

		//cube shadows are rendered in their own way
		for (uint32_t i = 0; i < render_state.cube_shadows.size(); i++) {
			_render_shadow_pass(render_state.render_shadows[render_state.cube_shadows[i]].light, p_render_data->shadow_atlas, render_state.render_shadows[render_state.cube_shadows[i]].pass, render_state.render_shadows[render_state.cube_shadows[i]].instances, camera_plane, lod_distance_multiplier, p_render_data->screen_mesh_lod_threshold, true, true, true, p_render_data->render_info);
		}

		if (render_state.directional_shadows.size()) {
			//open the pass for directional shadows
			_update_directional_shadow_atlas();
			RD::get_singleton()->draw_list_begin(directional_shadow.fb, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_DISCARD, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_CONTINUE);
			RD::get_singleton()->draw_list_end();
		}
	}

	// Render GI

	bool render_shadows = render_state.directional_shadows.size() || render_state.shadows.size();
	bool render_gi = p_render_data->render_buffers.is_valid() && p_use_gi;

	if (render_shadows && render_gi) {
		RENDER_TIMESTAMP("Render GI + Render Shadows (parallel)");
	} else if (render_shadows) {
		RENDER_TIMESTAMP("Render Shadows");
	} else if (render_gi) {
		RENDER_TIMESTAMP("Render GI");
	}

	//prepare shadow rendering
	if (render_shadows) {
		_render_shadow_begin();

		//render directional shadows
		for (uint32_t i = 0; i < render_state.directional_shadows.size(); i++) {
			_render_shadow_pass(render_state.render_shadows[render_state.directional_shadows[i]].light, p_render_data->shadow_atlas, render_state.render_shadows[render_state.directional_shadows[i]].pass, render_state.render_shadows[render_state.directional_shadows[i]].instances, camera_plane, lod_distance_multiplier, p_render_data->screen_mesh_lod_threshold, false, i == render_state.directional_shadows.size() - 1, false, p_render_data->render_info);
		}
		//render positional shadows
		for (uint32_t i = 0; i < render_state.shadows.size(); i++) {
			_render_shadow_pass(render_state.render_shadows[render_state.shadows[i]].light, p_render_data->shadow_atlas, render_state.render_shadows[render_state.shadows[i]].pass, render_state.render_shadows[render_state.shadows[i]].instances, camera_plane, lod_distance_multiplier, p_render_data->screen_mesh_lod_threshold, i == 0, i == render_state.shadows.size() - 1, true, p_render_data->render_info);
		}

		_render_shadow_process();
	}

	//start GI
	if (render_gi) {
		gi.process_gi(p_render_data->render_buffers, p_normal_roughness_buffer, p_voxel_gi_buffer, p_render_data->environment, p_render_data->cam_projection, p_render_data->cam_transform, *p_render_data->voxel_gi_instances, this);
	}

	//Do shadow rendering (in parallel with GI)
	if (render_shadows) {
		_render_shadow_end(RD::BARRIER_MASK_NO_BARRIER);
	}

	if (render_gi) {
		RD::get_singleton()->compute_list_end(RD::BARRIER_MASK_NO_BARRIER); //use a later barrier
	}

	if (p_render_data->render_buffers.is_valid()) {
		if (p_use_ssao || p_use_ssil) {
			RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_data->render_buffers);
			ERR_FAIL_COND(!rb);

			bool invalidate_uniform_set = false;
			if (rb->ss_effects.linear_depth.is_null()) {
				RD::TextureFormat tf;
				tf.format = RD::DATA_FORMAT_R16_SFLOAT;
				tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
				tf.width = (rb->width + 1) / 2;
				tf.height = (rb->height + 1) / 2;
				tf.mipmaps = 5;
				tf.array_layers = 4;
				tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
				rb->ss_effects.linear_depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
				RD::get_singleton()->set_resource_name(rb->ss_effects.linear_depth, "SS Effects Depth");
				for (uint32_t i = 0; i < tf.mipmaps; i++) {
					RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ss_effects.linear_depth, 0, i, 1, RD::TEXTURE_SLICE_2D_ARRAY);
					rb->ss_effects.linear_depth_slices.push_back(slice);
					RD::get_singleton()->set_resource_name(slice, "SS Effects Depth Mip " + itos(i) + " ");
				}
				invalidate_uniform_set = true;
			}

			storage->get_effects()->downsample_depth(rb->depth_texture, rb->ss_effects.linear_depth_slices, ssao_quality, ssil_quality, invalidate_uniform_set, ssao_half_size, ssil_half_size, Size2i(rb->width, rb->height), p_render_data->cam_projection);
		}

		if (p_use_ssao) {
			_process_ssao(p_render_data->render_buffers, p_render_data->environment, p_normal_roughness_buffer, p_render_data->cam_projection);
		}

		if (p_use_ssil) {
			_process_ssil(p_render_data->render_buffers, p_render_data->environment, p_normal_roughness_buffer, p_render_data->cam_projection, p_render_data->cam_transform);
		}
	}

	//full barrier here, we need raster, transfer and compute and it depends from the previous work
	RD::get_singleton()->barrier(RD::BARRIER_MASK_ALL, RD::BARRIER_MASK_ALL);

	if (current_cluster_builder) {
		current_cluster_builder->begin(p_render_data->cam_transform, p_render_data->cam_projection, !p_render_data->reflection_probe.is_valid());
	}

	bool using_shadows = true;

	if (p_render_data->reflection_probe.is_valid()) {
		if (!storage->reflection_probe_renders_shadows(reflection_probe_instance_get_probe(p_render_data->reflection_probe))) {
			using_shadows = false;
		}
	} else {
		//do not render reflections when rendering a reflection probe
		_setup_reflections(*p_render_data->reflection_probes, p_render_data->cam_transform.affine_inverse(), p_render_data->environment);
	}

	uint32_t directional_light_count = 0;
	uint32_t positional_light_count = 0;
	_setup_lights(*p_render_data->lights, p_render_data->cam_transform, p_render_data->shadow_atlas, using_shadows, directional_light_count, positional_light_count, p_render_data->directional_light_soft_shadows);
	_setup_decals(*p_render_data->decals, p_render_data->cam_transform.affine_inverse());

	p_render_data->directional_light_count = directional_light_count;

	if (current_cluster_builder) {
		current_cluster_builder->bake_cluster();
	}

	if (p_render_data->render_buffers.is_valid()) {
		bool directional_shadows = false;
		for (uint32_t i = 0; i < directional_light_count; i++) {
			if (cluster.directional_lights[i].shadow_enabled) {
				directional_shadows = true;
				break;
			}
		}
		if (is_volumetric_supported()) {
			_update_volumetric_fog(p_render_data->render_buffers, p_render_data->environment, p_render_data->cam_projection, p_render_data->cam_transform, p_render_data->shadow_atlas, directional_light_count, directional_shadows, positional_light_count, render_state.voxel_gi_count, *p_render_data->fog_volumes);
		}
	}
}

void RendererSceneRenderRD::render_scene(RID p_render_buffers, const CameraData *p_camera_data, const PagedArray<GeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_mesh_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data, RendererScene::RenderInfo *r_render_info) {
	// getting this here now so we can direct call a bunch of things more easily
	RenderBuffers *rb = nullptr;
	if (p_render_buffers.is_valid()) {
		rb = render_buffers_owner.get_or_null(p_render_buffers);
		ERR_FAIL_COND(!rb);
	}

	//assign render data
	RenderDataRD render_data;
	{
		render_data.render_buffers = p_render_buffers;

		// Our first camera is used by default
		render_data.cam_transform = p_camera_data->main_transform;
		render_data.cam_projection = p_camera_data->main_projection;
		render_data.view_projection[0] = p_camera_data->main_projection;
		render_data.cam_ortogonal = p_camera_data->is_ortogonal;

		render_data.view_count = p_camera_data->view_count;
		for (uint32_t v = 0; v < p_camera_data->view_count; v++) {
			render_data.view_projection[v] = p_camera_data->view_projection[v];
		}

		render_data.z_near = p_camera_data->main_projection.get_z_near();
		render_data.z_far = p_camera_data->main_projection.get_z_far();

		render_data.instances = &p_instances;
		render_data.lights = &p_lights;
		render_data.reflection_probes = &p_reflection_probes;
		render_data.voxel_gi_instances = &p_voxel_gi_instances;
		render_data.decals = &p_decals;
		render_data.lightmaps = &p_lightmaps;
		render_data.fog_volumes = &p_fog_volumes;
		render_data.environment = p_environment;
		render_data.camera_effects = p_camera_effects;
		render_data.shadow_atlas = p_shadow_atlas;
		render_data.reflection_atlas = p_reflection_atlas;
		render_data.reflection_probe = p_reflection_probe;
		render_data.reflection_probe_pass = p_reflection_probe_pass;

		// this should be the same for all cameras..
		render_data.lod_distance_multiplier = p_camera_data->main_projection.get_lod_multiplier();
		render_data.lod_camera_plane = Plane(-p_camera_data->main_transform.basis.get_axis(Vector3::AXIS_Z), p_camera_data->main_transform.get_origin());

		if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_DISABLE_LOD) {
			render_data.screen_mesh_lod_threshold = 0.0;
		} else {
			render_data.screen_mesh_lod_threshold = p_screen_mesh_lod_threshold;
		}

		render_state.render_shadows = p_render_shadows;
		render_state.render_shadow_count = p_render_shadow_count;
		render_state.render_sdfgi_regions = p_render_sdfgi_regions;
		render_state.render_sdfgi_region_count = p_render_sdfgi_region_count;
		render_state.sdfgi_update_data = p_sdfgi_update_data;
		render_data.render_info = r_render_info;
	}

	PagedArray<RID> empty;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {
		render_data.lights = &empty;
		render_data.reflection_probes = &empty;
		render_data.voxel_gi_instances = &empty;
	}

	//sdfgi first
	if (rb != nullptr && rb->sdfgi != nullptr) {
		for (int i = 0; i < render_state.render_sdfgi_region_count; i++) {
			rb->sdfgi->render_region(p_render_buffers, render_state.render_sdfgi_regions[i].region, render_state.render_sdfgi_regions[i].instances, this);
		}
		if (render_state.sdfgi_update_data->update_static) {
			rb->sdfgi->render_static_lights(p_render_buffers, render_state.sdfgi_update_data->static_cascade_count, p_sdfgi_update_data->static_cascade_indices, render_state.sdfgi_update_data->static_positional_lights, this);
		}
	}

	Color clear_color;
	if (p_render_buffers.is_valid()) {
		clear_color = storage->render_target_get_clear_request_color(rb->render_target);
	} else {
		clear_color = storage->get_default_clear_color();
	}

	//assign render indices to voxel_gi_instances
	if (is_dynamic_gi_supported()) {
		for (uint32_t i = 0; i < (uint32_t)p_voxel_gi_instances.size(); i++) {
			RendererSceneGIRD::VoxelGIInstance *voxel_gi_inst = gi.voxel_gi_instance_owner.get_or_null(p_voxel_gi_instances[i]);
			if (voxel_gi_inst) {
				voxel_gi_inst->render_index = i;
			}
		}
	}

	if (render_buffers_owner.owns(render_data.render_buffers)) {
		// render_data.render_buffers == p_render_buffers so we can use our already retrieved rb
		current_cluster_builder = rb->cluster_builder;
	} else if (reflection_probe_instance_owner.owns(render_data.reflection_probe)) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(render_data.reflection_probe);
		ReflectionAtlas *ra = reflection_atlas_owner.get_or_null(rpi->atlas);
		if (!ra) {
			ERR_PRINT("reflection probe has no reflection atlas! Bug?");
			current_cluster_builder = nullptr;
		} else {
			current_cluster_builder = ra->cluster_builder;
		}
	} else {
		ERR_PRINT("No render buffer nor reflection atlas, bug"); //should never happen, will crash
		current_cluster_builder = nullptr;
	}

	render_state.voxel_gi_count = 0;

	if (rb != nullptr && is_dynamic_gi_supported()) {
		if (rb->sdfgi) {
			rb->sdfgi->update_cascades();
			rb->sdfgi->pre_process_gi(render_data.cam_transform, &render_data, this);
			rb->sdfgi->update_light();
		}

		gi.setup_voxel_gi_instances(render_data.render_buffers, render_data.cam_transform, *render_data.voxel_gi_instances, render_state.voxel_gi_count, this);
	}

	render_state.depth_prepass_used = false;
	//calls _pre_opaque_render between depth pre-pass and opaque pass
	if (current_cluster_builder != nullptr) {
		render_data.cluster_buffer = current_cluster_builder->get_cluster_buffer();
		render_data.cluster_size = current_cluster_builder->get_cluster_size();
		render_data.cluster_max_elements = current_cluster_builder->get_max_cluster_elements();
	}

	_render_scene(&render_data, clear_color);

	if (p_render_buffers.is_valid()) {
		/*
		_debug_draw_cluster(p_render_buffers);
		RENDER_TIMESTAMP("Tonemap");
		_render_buffers_post_process_and_tonemap(&render_data);
		*/

		_render_buffers_debug_draw(p_render_buffers, p_shadow_atlas, p_occluder_debug_tex);
		if (debug_draw == RS::VIEWPORT_DEBUG_DRAW_SDFGI && rb != nullptr && rb->sdfgi != nullptr) {
			rb->sdfgi->debug_draw(render_data.cam_projection, render_data.cam_transform, rb->width, rb->height, rb->render_target, rb->texture);
		}
	}
}

void RendererSceneRenderRD::_debug_draw_cluster(RID p_render_buffers) {
	if (p_render_buffers.is_valid() && current_cluster_builder != nullptr) {
		RS::ViewportDebugDraw dd = get_debug_draw_mode();

		if (dd == RS::VIEWPORT_DEBUG_DRAW_CLUSTER_OMNI_LIGHTS || dd == RS::VIEWPORT_DEBUG_DRAW_CLUSTER_SPOT_LIGHTS || dd == RS::VIEWPORT_DEBUG_DRAW_CLUSTER_DECALS || dd == RS::VIEWPORT_DEBUG_DRAW_CLUSTER_REFLECTION_PROBES) {
			ClusterBuilderRD::ElementType elem_type = ClusterBuilderRD::ELEMENT_TYPE_MAX;
			switch (dd) {
				case RS::VIEWPORT_DEBUG_DRAW_CLUSTER_OMNI_LIGHTS:
					elem_type = ClusterBuilderRD::ELEMENT_TYPE_OMNI_LIGHT;
					break;
				case RS::VIEWPORT_DEBUG_DRAW_CLUSTER_SPOT_LIGHTS:
					elem_type = ClusterBuilderRD::ELEMENT_TYPE_SPOT_LIGHT;
					break;
				case RS::VIEWPORT_DEBUG_DRAW_CLUSTER_DECALS:
					elem_type = ClusterBuilderRD::ELEMENT_TYPE_DECAL;
					break;
				case RS::VIEWPORT_DEBUG_DRAW_CLUSTER_REFLECTION_PROBES:
					elem_type = ClusterBuilderRD::ELEMENT_TYPE_REFLECTION_PROBE;
					break;
				default: {
				}
			}
			current_cluster_builder->debug(elem_type);
		}
	}
}

void RendererSceneRenderRD::_render_shadow_pass(RID p_light, RID p_shadow_atlas, int p_pass, const PagedArray<GeometryInstance *> &p_instances, const Plane &p_camera_plane, float p_lod_distance_multiplier, float p_screen_mesh_lod_threshold, bool p_open_pass, bool p_close_pass, bool p_clear_region, RendererScene::RenderInfo *p_render_info) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light_instance);

	Rect2i atlas_rect;
	uint32_t atlas_size;
	RID atlas_fb;

	bool using_dual_paraboloid = false;
	bool using_dual_paraboloid_flip = false;
	Vector2i dual_paraboloid_offset;
	RID render_fb;
	RID render_texture;
	float zfar;

	bool use_pancake = false;
	bool render_cubemap = false;
	bool finalize_cubemap = false;

	bool flip_y = false;

	CameraMatrix light_projection;
	Transform3D light_transform;

	if (storage->light_get_type(light_instance->light) == RS::LIGHT_DIRECTIONAL) {
		//set pssm stuff
		if (light_instance->last_scene_shadow_pass != scene_pass) {
			light_instance->directional_rect = _get_directional_shadow_rect(directional_shadow.size, directional_shadow.light_count, directional_shadow.current_light);
			directional_shadow.current_light++;
			light_instance->last_scene_shadow_pass = scene_pass;
		}

		use_pancake = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE) > 0;
		light_projection = light_instance->shadow_transform[p_pass].camera;
		light_transform = light_instance->shadow_transform[p_pass].transform;

		atlas_rect = light_instance->directional_rect;

		if (storage->light_directional_get_shadow_mode(light_instance->light) == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {
			atlas_rect.size.width /= 2;
			atlas_rect.size.height /= 2;

			if (p_pass == 1) {
				atlas_rect.position.x += atlas_rect.size.width;
			} else if (p_pass == 2) {
				atlas_rect.position.y += atlas_rect.size.height;
			} else if (p_pass == 3) {
				atlas_rect.position += atlas_rect.size;
			}
		} else if (storage->light_directional_get_shadow_mode(light_instance->light) == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {
			atlas_rect.size.height /= 2;

			if (p_pass == 0) {
			} else {
				atlas_rect.position.y += atlas_rect.size.height;
			}
		}

		light_instance->shadow_transform[p_pass].atlas_rect = atlas_rect;

		light_instance->shadow_transform[p_pass].atlas_rect.position /= directional_shadow.size;
		light_instance->shadow_transform[p_pass].atlas_rect.size /= directional_shadow.size;

		zfar = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_RANGE);

		render_fb = directional_shadow.fb;
		render_texture = RID();
		flip_y = true;

	} else {
		//set from shadow atlas

		ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas->shadow_owners.has(p_light));

		_update_shadow_atlas(shadow_atlas);

		uint32_t key = shadow_atlas->shadow_owners[p_light];

		uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

		ERR_FAIL_INDEX((int)shadow, shadow_atlas->quadrants[quadrant].shadows.size());

		uint32_t quadrant_size = shadow_atlas->size >> 1;

		atlas_rect.position.x = (quadrant & 1) * quadrant_size;
		atlas_rect.position.y = (quadrant >> 1) * quadrant_size;

		uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
		atlas_rect.position.x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
		atlas_rect.position.y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

		atlas_rect.size.width = shadow_size;
		atlas_rect.size.height = shadow_size;

		zfar = storage->light_get_param(light_instance->light, RS::LIGHT_PARAM_RANGE);

		if (storage->light_get_type(light_instance->light) == RS::LIGHT_OMNI) {
			bool wrap = (shadow + 1) % shadow_atlas->quadrants[quadrant].subdivision == 0;
			dual_paraboloid_offset = wrap ? Vector2i(1 - shadow_atlas->quadrants[quadrant].subdivision, 1) : Vector2i(1, 0);

			if (storage->light_omni_get_shadow_mode(light_instance->light) == RS::LIGHT_OMNI_SHADOW_CUBE) {
				ShadowCubemap *cubemap = _get_shadow_cubemap(shadow_size / 2);

				render_fb = cubemap->side_fb[p_pass];
				render_texture = cubemap->cubemap;

				light_projection = light_instance->shadow_transform[p_pass].camera;
				light_transform = light_instance->shadow_transform[p_pass].transform;
				render_cubemap = true;
				finalize_cubemap = p_pass == 5;
				atlas_fb = shadow_atlas->fb;

				atlas_size = shadow_atlas->size;

				if (p_pass == 0) {
					_render_shadow_begin();
				}

			} else {
				atlas_rect.position.x += 1;
				atlas_rect.position.y += 1;
				atlas_rect.size.x -= 2;
				atlas_rect.size.y -= 2;

				atlas_rect.position += p_pass * atlas_rect.size * dual_paraboloid_offset;

				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;

				using_dual_paraboloid = true;
				using_dual_paraboloid_flip = p_pass == 1;
				render_fb = shadow_atlas->fb;
				flip_y = true;
			}

		} else if (storage->light_get_type(light_instance->light) == RS::LIGHT_SPOT) {
			light_projection = light_instance->shadow_transform[0].camera;
			light_transform = light_instance->shadow_transform[0].transform;

			render_fb = shadow_atlas->fb;

			flip_y = true;
		}
	}

	if (render_cubemap) {
		//rendering to cubemap
		_render_shadow_append(render_fb, p_instances, light_projection, light_transform, zfar, 0, 0, false, false, use_pancake, p_camera_plane, p_lod_distance_multiplier, p_screen_mesh_lod_threshold, Rect2(), false, true, true, true, p_render_info);
		if (finalize_cubemap) {
			_render_shadow_process();
			_render_shadow_end();
			//reblit
			Rect2 atlas_rect_norm = atlas_rect;
			atlas_rect_norm.position /= float(atlas_size);
			atlas_rect_norm.size /= float(atlas_size);
			storage->get_effects()->copy_cubemap_to_dp(render_texture, atlas_fb, atlas_rect_norm, atlas_rect.size, light_projection.get_z_near(), light_projection.get_z_far(), false);
			atlas_rect_norm.position += Vector2(dual_paraboloid_offset) * atlas_rect_norm.size;
			storage->get_effects()->copy_cubemap_to_dp(render_texture, atlas_fb, atlas_rect_norm, atlas_rect.size, light_projection.get_z_near(), light_projection.get_z_far(), true);

			//restore transform so it can be properly used
			light_instance_set_shadow_transform(p_light, CameraMatrix(), light_instance->transform, zfar, 0, 0, 0);
		}

	} else {
		//render shadow
		_render_shadow_append(render_fb, p_instances, light_projection, light_transform, zfar, 0, 0, using_dual_paraboloid, using_dual_paraboloid_flip, use_pancake, p_camera_plane, p_lod_distance_multiplier, p_screen_mesh_lod_threshold, atlas_rect, flip_y, p_clear_region, p_open_pass, p_close_pass, p_render_info);
	}
}

void RendererSceneRenderRD::render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) {
	_render_material(p_cam_transform, p_cam_projection, p_cam_ortogonal, p_instances, p_framebuffer, p_region);
}

void RendererSceneRenderRD::render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) {
	ERR_FAIL_COND(!storage->particles_collision_is_heightfield(p_collider));
	Vector3 extents = storage->particles_collision_get_extents(p_collider) * p_transform.basis.get_scale();
	CameraMatrix cm;
	cm.set_orthogonal(-extents.x, extents.x, -extents.z, extents.z, 0, extents.y * 2.0);

	Vector3 cam_pos = p_transform.origin;
	cam_pos.y += extents.y;

	Transform3D cam_xform;
	cam_xform.set_look_at(cam_pos, cam_pos - p_transform.basis.get_axis(Vector3::AXIS_Y), -p_transform.basis.get_axis(Vector3::AXIS_Z).normalized());

	RID fb = storage->particles_collision_get_heightfield_framebuffer(p_collider);

	_render_particle_collider_heightfield(fb, cam_xform, cm, p_instances);
}

bool RendererSceneRenderRD::free(RID p_rid) {
	if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.get_or_null(p_rid);
		_free_render_buffer_data(rb);
		memdelete(rb->data);
		if (rb->sdfgi) {
			rb->sdfgi->erase();
			memdelete(rb->sdfgi);
			rb->sdfgi = nullptr;
		}
		if (rb->volumetric_fog) {
			_volumetric_fog_erase(rb);
		}
		if (rb->cluster_builder) {
			memdelete(rb->cluster_builder);
		}
		render_buffers_owner.free(p_rid);
	} else if (environment_owner.owns(p_rid)) {
		//not much to delete, just free it
		environment_owner.free(p_rid);
	} else if (camera_effects_owner.owns(p_rid)) {
		//not much to delete, just free it
		camera_effects_owner.free(p_rid);
	} else if (reflection_atlas_owner.owns(p_rid)) {
		reflection_atlas_set_size(p_rid, 0, 0);
		ReflectionAtlas *ra = reflection_atlas_owner.get_or_null(p_rid);
		if (ra->cluster_builder) {
			memdelete(ra->cluster_builder);
		}
		reflection_atlas_owner.free(p_rid);
	} else if (reflection_probe_instance_owner.owns(p_rid)) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_rid);
		_free_forward_id(FORWARD_ID_TYPE_REFLECTION_PROBE, rpi->forward_id);
		reflection_probe_release_atlas_index(p_rid);
		reflection_probe_instance_owner.free(p_rid);
	} else if (decal_instance_owner.owns(p_rid)) {
		DecalInstance *di = decal_instance_owner.get_or_null(p_rid);
		_free_forward_id(FORWARD_ID_TYPE_DECAL, di->forward_id);
		decal_instance_owner.free(p_rid);
	} else if (lightmap_instance_owner.owns(p_rid)) {
		lightmap_instance_owner.free(p_rid);
	} else if (gi.voxel_gi_instance_owner.owns(p_rid)) {
		RendererSceneGIRD::VoxelGIInstance *voxel_gi = gi.voxel_gi_instance_owner.get_or_null(p_rid);
		if (voxel_gi->texture.is_valid()) {
			RD::get_singleton()->free(voxel_gi->texture);
			RD::get_singleton()->free(voxel_gi->write_buffer);
		}

		for (int i = 0; i < voxel_gi->dynamic_maps.size(); i++) {
			RD::get_singleton()->free(voxel_gi->dynamic_maps[i].texture);
			RD::get_singleton()->free(voxel_gi->dynamic_maps[i].depth);
		}

		gi.voxel_gi_instance_owner.free(p_rid);
	} else if (sky.sky_owner.owns(p_rid)) {
		sky.update_dirty_skys();
		sky.free_sky(p_rid);
	} else if (light_instance_owner.owns(p_rid)) {
		LightInstance *light_instance = light_instance_owner.get_or_null(p_rid);

		//remove from shadow atlases..
		for (Set<RID>::Element *E = light_instance->shadow_atlases.front(); E; E = E->next()) {
			ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(E->get());
			ERR_CONTINUE(!shadow_atlas->shadow_owners.has(p_rid));
			uint32_t key = shadow_atlas->shadow_owners[p_rid];
			uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
			uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();

			if (key & ShadowAtlas::OMNI_LIGHT_FLAG) {
				// Omni lights use two atlas spots, make sure to clear the other as well
				shadow_atlas->quadrants[q].shadows.write[s + 1].owner = RID();
			}

			shadow_atlas->shadow_owners.erase(p_rid);
		}

		if (light_instance->light_type != RS::LIGHT_DIRECTIONAL) {
			_free_forward_id(light_instance->light_type == RS::LIGHT_OMNI ? FORWARD_ID_TYPE_OMNI_LIGHT : FORWARD_ID_TYPE_SPOT_LIGHT, light_instance->forward_id);
		}
		light_instance_owner.free(p_rid);

	} else if (shadow_atlas_owner.owns(p_rid)) {
		shadow_atlas_set_size(p_rid, 0);
		shadow_atlas_owner.free(p_rid);
	} else if (fog_volume_instance_owner.owns(p_rid)) {
		fog_volume_instance_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

void RendererSceneRenderRD::set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
}

void RendererSceneRenderRD::update() {
	sky.update_dirty_skys();
}

void RendererSceneRenderRD::set_time(double p_time, double p_step) {
	time = p_time;
	time_step = p_step;
}

void RendererSceneRenderRD::screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) {
	screen_space_roughness_limiter = p_enable;
	screen_space_roughness_limiter_amount = p_amount;
	screen_space_roughness_limiter_limit = p_limit;
}

bool RendererSceneRenderRD::screen_space_roughness_limiter_is_active() const {
	return screen_space_roughness_limiter;
}

float RendererSceneRenderRD::screen_space_roughness_limiter_get_amount() const {
	return screen_space_roughness_limiter_amount;
}

float RendererSceneRenderRD::screen_space_roughness_limiter_get_limit() const {
	return screen_space_roughness_limiter_limit;
}

TypedArray<Image> RendererSceneRenderRD::bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) {
	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	tf.width = p_image_size.width; // Always 64x64
	tf.height = p_image_size.height;
	tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	RID albedo_alpha_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RID normal_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RID orm_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	RID emission_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.format = RD::DATA_FORMAT_R32_SFLOAT;
	RID depth_write_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
	tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
	RID depth_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());

	Vector<RID> fb_tex;
	fb_tex.push_back(albedo_alpha_tex);
	fb_tex.push_back(normal_tex);
	fb_tex.push_back(orm_tex);
	fb_tex.push_back(emission_tex);
	fb_tex.push_back(depth_write_tex);
	fb_tex.push_back(depth_tex);

	RID fb = RD::get_singleton()->framebuffer_create(fb_tex);

	//RID sampled_light;

	GeometryInstance *gi = geometry_instance_create(p_base);

	uint32_t sc = RSG::storage->mesh_get_surface_count(p_base);
	Vector<RID> materials;
	materials.resize(sc);

	for (uint32_t i = 0; i < sc; i++) {
		if (i < (uint32_t)p_material_overrides.size()) {
			materials.write[i] = p_material_overrides[i];
		}
	}

	geometry_instance_set_surface_materials(gi, materials);

	if (cull_argument.size() == 0) {
		cull_argument.push_back(nullptr);
	}
	cull_argument[0] = gi;
	_render_uv2(cull_argument, fb, Rect2i(0, 0, p_image_size.width, p_image_size.height));

	geometry_instance_free(gi);

	TypedArray<Image> ret;

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(albedo_alpha_tex, 0);
		Ref<Image> img;
		img.instantiate();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBA8, data);
		RD::get_singleton()->free(albedo_alpha_tex);
		ret.push_back(img);
	}

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(normal_tex, 0);
		Ref<Image> img;
		img.instantiate();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBA8, data);
		RD::get_singleton()->free(normal_tex);
		ret.push_back(img);
	}

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(orm_tex, 0);
		Ref<Image> img;
		img.instantiate();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBA8, data);
		RD::get_singleton()->free(orm_tex);
		ret.push_back(img);
	}

	{
		PackedByteArray data = RD::get_singleton()->texture_get_data(emission_tex, 0);
		Ref<Image> img;
		img.instantiate();
		img->create(p_image_size.width, p_image_size.height, false, Image::FORMAT_RGBAH, data);
		RD::get_singleton()->free(emission_tex);
		ret.push_back(img);
	}

	RD::get_singleton()->free(depth_write_tex);
	RD::get_singleton()->free(depth_tex);

	return ret;
}

void RendererSceneRenderRD::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
	gi.sdfgi_debug_probe_pos = p_position;
	gi.sdfgi_debug_probe_dir = p_dir;
}

RendererSceneRenderRD *RendererSceneRenderRD::singleton = nullptr;

RID RendererSceneRenderRD::get_reflection_probe_buffer() {
	return cluster.reflection_buffer;
}
RID RendererSceneRenderRD::get_omni_light_buffer() {
	return cluster.omni_light_buffer;
}

RID RendererSceneRenderRD::get_spot_light_buffer() {
	return cluster.spot_light_buffer;
}

RID RendererSceneRenderRD::get_directional_light_buffer() {
	return cluster.directional_light_buffer;
}
RID RendererSceneRenderRD::get_decal_buffer() {
	return cluster.decal_buffer;
}
int RendererSceneRenderRD::get_max_directional_lights() const {
	return cluster.max_directional_lights;
}

bool RendererSceneRenderRD::is_dynamic_gi_supported() const {
	// usable by default (unless low end = true)
	return true;
}

bool RendererSceneRenderRD::is_clustered_enabled() const {
	// used by default.
	return true;
}

bool RendererSceneRenderRD::is_volumetric_supported() const {
	// usable by default (unless low end = true)
	return true;
}

uint32_t RendererSceneRenderRD::get_max_elements() const {
	return GLOBAL_GET("rendering/limits/cluster_builder/max_clustered_elements");
}

RendererSceneRenderRD::RendererSceneRenderRD(RendererStorageRD *p_storage) {
	storage = p_storage;
	singleton = this;
}

void RendererSceneRenderRD::init() {
	max_cluster_elements = get_max_elements();

	directional_shadow.size = GLOBAL_GET("rendering/shadows/directional_shadow/size");
	directional_shadow.use_16_bits = GLOBAL_GET("rendering/shadows/directional_shadow/16_bits");

	/* SKY SHADER */

	sky.init(storage);

	/* GI */

	if (is_dynamic_gi_supported()) {
		gi.init(storage, &sky);
	}

	{ //decals
		cluster.max_decals = max_cluster_elements;
		uint32_t decal_buffer_size = cluster.max_decals * sizeof(Cluster::DecalData);
		cluster.decals = memnew_arr(Cluster::DecalData, cluster.max_decals);
		cluster.decal_sort = memnew_arr(Cluster::InstanceSort<DecalInstance>, cluster.max_decals);
		cluster.decal_buffer = RD::get_singleton()->storage_buffer_create(decal_buffer_size);
	}

	{ //reflections

		cluster.max_reflections = max_cluster_elements;
		cluster.reflections = memnew_arr(Cluster::ReflectionData, cluster.max_reflections);
		cluster.reflection_sort = memnew_arr(Cluster::InstanceSort<ReflectionProbeInstance>, cluster.max_reflections);
		cluster.reflection_buffer = RD::get_singleton()->storage_buffer_create(sizeof(Cluster::ReflectionData) * cluster.max_reflections);
	}

	{ //lights
		cluster.max_lights = max_cluster_elements;

		uint32_t light_buffer_size = cluster.max_lights * sizeof(Cluster::LightData);
		cluster.omni_lights = memnew_arr(Cluster::LightData, cluster.max_lights);
		cluster.omni_light_buffer = RD::get_singleton()->storage_buffer_create(light_buffer_size);
		cluster.omni_light_sort = memnew_arr(Cluster::InstanceSort<LightInstance>, cluster.max_lights);
		cluster.spot_lights = memnew_arr(Cluster::LightData, cluster.max_lights);
		cluster.spot_light_buffer = RD::get_singleton()->storage_buffer_create(light_buffer_size);
		cluster.spot_light_sort = memnew_arr(Cluster::InstanceSort<LightInstance>, cluster.max_lights);
		//defines += "\n#define MAX_LIGHT_DATA_STRUCTS " + itos(cluster.max_lights) + "\n";

		cluster.max_directional_lights = MAX_DIRECTIONAL_LIGHTS;
		uint32_t directional_light_buffer_size = cluster.max_directional_lights * sizeof(Cluster::DirectionalLightData);
		cluster.directional_lights = memnew_arr(Cluster::DirectionalLightData, cluster.max_directional_lights);
		cluster.directional_light_buffer = RD::get_singleton()->uniform_buffer_create(directional_light_buffer_size);
	}

	if (is_volumetric_supported()) {
		{
			// Initialize local fog shader
			Vector<String> volumetric_fog_modes;
			volumetric_fog_modes.push_back("");
			volumetric_fog.shader.initialize(volumetric_fog_modes);

			storage->shader_set_data_request_function(RendererStorageRD::SHADER_TYPE_FOG, _create_fog_shader_funcs);
			storage->material_set_data_request_function(RendererStorageRD::SHADER_TYPE_FOG, _create_fog_material_funcs);
			volumetric_fog.volume_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(VolumetricFogShader::VolumeUBO));
		}

		{
			ShaderCompiler::DefaultIdentifierActions actions;

			actions.renames["TIME"] = "scene_params.time";
			actions.renames["PI"] = _MKSTR(Math_PI);
			actions.renames["TAU"] = _MKSTR(Math_TAU);
			actions.renames["E"] = _MKSTR(Math_E);
			actions.renames["WORLD_POSITION"] = "world.xyz";
			actions.renames["OBJECT_POSITION"] = "params.position";
			actions.renames["UVW"] = "uvw";
			actions.renames["EXTENTS"] = "params.extents";
			actions.renames["ALBEDO"] = "albedo";
			actions.renames["DENSITY"] = "density";
			actions.renames["EMISSION"] = "emission";
			actions.renames["SDF"] = "sdf";

			actions.usage_defines["SDF"] = "#define SDF_USED\n";
			actions.usage_defines["DENSITY"] = "#define DENSITY_USED\n";
			actions.usage_defines["ALBEDO"] = "#define ALBEDO_USED\n";
			actions.usage_defines["EMISSION"] = "#define EMISSION_USED\n";

			actions.sampler_array_name = "material_samplers";
			actions.base_texture_binding_index = 1;
			actions.texture_layout_set = VolumetricFogShader::FogSet::FOG_SET_MATERIAL;
			actions.base_uniform_string = "material.";

			actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
			actions.default_repeat = ShaderLanguage::REPEAT_DISABLE;
			actions.global_buffer_array_variable = "global_variables.data";

			volumetric_fog.compiler.initialize(actions);
		}

		{
			// default material and shader for fog shader
			volumetric_fog.default_shader = storage->shader_allocate();
			storage->shader_initialize(volumetric_fog.default_shader);
			storage->shader_set_code(volumetric_fog.default_shader, R"(
// Default fog shader.

shader_type fog;

void fog() {
	DENSITY = 1.0;
	ALBEDO = vec3(1.0);
}
)");
			volumetric_fog.default_material = storage->material_allocate();
			storage->material_initialize(volumetric_fog.default_material);
			storage->material_set_shader(volumetric_fog.default_material, volumetric_fog.default_shader);

			FogMaterialData *md = (FogMaterialData *)storage->material_get_data(volumetric_fog.default_material, RendererStorageRD::SHADER_TYPE_FOG);
			volumetric_fog.default_shader_rd = volumetric_fog.shader.version_get_shader(md->shader_data->version, 0);

			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
				u.binding = 1;
				u.ids.resize(12);
				RID *ids_ptr = u.ids.ptrw();
				ids_ptr[0] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				ids_ptr[1] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				ids_ptr[2] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				ids_ptr[3] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				ids_ptr[4] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				ids_ptr[5] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
				ids_ptr[6] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
				ids_ptr[7] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
				ids_ptr[8] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
				ids_ptr[9] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
				ids_ptr[10] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
				ids_ptr[11] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
				uniforms.push_back(u);
			}

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.ids.push_back(storage->global_variables_get_storage_buffer());
				uniforms.push_back(u);
			}

			volumetric_fog.base_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, volumetric_fog.default_shader_rd, VolumetricFogShader::FogSet::FOG_SET_BASE);
		}
		{
			String defines = "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(cluster.max_directional_lights) + "\n";
			defines += "\n#define MAX_SKY_LOD " + itos(get_roughness_layers() - 1) + ".0\n";
			if (is_using_radiance_cubemap_array()) {
				defines += "\n#define USE_RADIANCE_CUBEMAP_ARRAY \n";
			}
			Vector<String> volumetric_fog_modes;
			volumetric_fog_modes.push_back("\n#define MODE_DENSITY\n");
			volumetric_fog_modes.push_back("\n#define MODE_DENSITY\n#define ENABLE_SDFGI\n");
			volumetric_fog_modes.push_back("\n#define MODE_FILTER\n");
			volumetric_fog_modes.push_back("\n#define MODE_FOG\n");
			volumetric_fog_modes.push_back("\n#define MODE_COPY\n");

			volumetric_fog.process_shader.initialize(volumetric_fog_modes, defines);
			volumetric_fog.process_shader_version = volumetric_fog.process_shader.version_create();
			for (int i = 0; i < VolumetricFogShader::VOLUMETRIC_FOG_PROCESS_SHADER_MAX; i++) {
				volumetric_fog.process_pipelines[i] = RD::get_singleton()->compute_pipeline_create(volumetric_fog.process_shader.version_get_shader(volumetric_fog.process_shader_version, i));
			}
			volumetric_fog.params_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(VolumetricFogShader::ParamsUBO));
		}
	}

	{
		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.min_filter = RD::SAMPLER_FILTER_NEAREST;
		sampler.enable_compare = true;
		sampler.compare_op = RD::COMPARE_OP_LESS;
		shadow_sampler = RD::get_singleton()->sampler_create(sampler);
	}

	camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape(int(GLOBAL_GET("rendering/camera/depth_of_field/depth_of_field_bokeh_shape"))));
	camera_effects_set_dof_blur_quality(RS::DOFBlurQuality(int(GLOBAL_GET("rendering/camera/depth_of_field/depth_of_field_bokeh_quality"))), GLOBAL_GET("rendering/camera/depth_of_field/depth_of_field_use_jitter"));
	environment_set_ssao_quality(RS::EnvironmentSSAOQuality(int(GLOBAL_GET("rendering/environment/ssao/quality"))), GLOBAL_GET("rendering/environment/ssao/half_size"), GLOBAL_GET("rendering/environment/ssao/adaptive_target"), GLOBAL_GET("rendering/environment/ssao/blur_passes"), GLOBAL_GET("rendering/environment/ssao/fadeout_from"), GLOBAL_GET("rendering/environment/ssao/fadeout_to"));
	screen_space_roughness_limiter = GLOBAL_GET("rendering/anti_aliasing/screen_space_roughness_limiter/enabled");
	screen_space_roughness_limiter_amount = GLOBAL_GET("rendering/anti_aliasing/screen_space_roughness_limiter/amount");
	screen_space_roughness_limiter_limit = GLOBAL_GET("rendering/anti_aliasing/screen_space_roughness_limiter/limit");
	glow_bicubic_upscale = int(GLOBAL_GET("rendering/environment/glow/upscale_mode")) > 0;
	glow_high_quality = GLOBAL_GET("rendering/environment/glow/use_high_quality");
	ssr_roughness_quality = RS::EnvironmentSSRRoughnessQuality(int(GLOBAL_GET("rendering/environment/screen_space_reflection/roughness_quality")));
	sss_quality = RS::SubSurfaceScatteringQuality(int(GLOBAL_GET("rendering/environment/subsurface_scattering/subsurface_scattering_quality")));
	sss_scale = GLOBAL_GET("rendering/environment/subsurface_scattering/subsurface_scattering_scale");
	sss_depth_scale = GLOBAL_GET("rendering/environment/subsurface_scattering/subsurface_scattering_depth_scale");

	environment_set_ssil_quality(RS::EnvironmentSSILQuality(int(GLOBAL_GET("rendering/environment/ssil/quality"))), GLOBAL_GET("rendering/environment/ssil/half_size"), GLOBAL_GET("rendering/environment/ssil/adaptive_target"), GLOBAL_GET("rendering/environment/ssil/blur_passes"), GLOBAL_GET("rendering/environment/ssil/fadeout_from"), GLOBAL_GET("rendering/environment/ssil/fadeout_to"));

	directional_penumbra_shadow_kernel = memnew_arr(float, 128);
	directional_soft_shadow_kernel = memnew_arr(float, 128);
	penumbra_shadow_kernel = memnew_arr(float, 128);
	soft_shadow_kernel = memnew_arr(float, 128);
	shadows_quality_set(RS::ShadowQuality(int(GLOBAL_GET("rendering/shadows/shadows/soft_shadow_quality"))));
	directional_shadow_quality_set(RS::ShadowQuality(int(GLOBAL_GET("rendering/shadows/directional_shadow/soft_shadow_quality"))));

	environment_set_volumetric_fog_volume_size(GLOBAL_GET("rendering/environment/volumetric_fog/volume_size"), GLOBAL_GET("rendering/environment/volumetric_fog/volume_depth"));
	environment_set_volumetric_fog_filter_active(GLOBAL_GET("rendering/environment/volumetric_fog/use_filter"));

	decals_set_filter(RS::DecalFilter(int(GLOBAL_GET("rendering/textures/decals/filter"))));
	light_projectors_set_filter(RS::LightProjectorFilter(int(GLOBAL_GET("rendering/textures/light_projectors/filter"))));

	cull_argument.set_page_pool(&cull_argument_pool);
}

RendererSceneRenderRD::~RendererSceneRenderRD() {
	for (const KeyValue<int, ShadowCubemap> &E : shadow_cubemaps) {
		RD::get_singleton()->free(E.value.cubemap);
	}

	if (sky.sky_scene_state.uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(sky.sky_scene_state.uniform_set)) {
		RD::get_singleton()->free(sky.sky_scene_state.uniform_set);
	}

	if (is_dynamic_gi_supported()) {
		gi.free();
	}

	if (is_volumetric_supported()) {
		volumetric_fog.process_shader.version_free(volumetric_fog.process_shader_version);
		RD::get_singleton()->free(volumetric_fog.volume_ubo);
		RD::get_singleton()->free(volumetric_fog.params_ubo);
		storage->free(volumetric_fog.default_shader);
		storage->free(volumetric_fog.default_material);
	}

	RendererSceneSkyRD::SkyMaterialData *md = (RendererSceneSkyRD::SkyMaterialData *)storage->material_get_data(sky.sky_shader.default_material, RendererStorageRD::SHADER_TYPE_SKY);
	sky.sky_shader.shader.version_free(md->shader_data->version);
	RD::get_singleton()->free(sky.sky_scene_state.directional_light_buffer);
	RD::get_singleton()->free(sky.sky_scene_state.uniform_buffer);
	memdelete_arr(sky.sky_scene_state.directional_lights);
	memdelete_arr(sky.sky_scene_state.last_frame_directional_lights);
	storage->free(sky.sky_shader.default_shader);
	storage->free(sky.sky_shader.default_material);
	storage->free(sky.sky_scene_state.fog_shader);
	storage->free(sky.sky_scene_state.fog_material);
	memdelete_arr(directional_penumbra_shadow_kernel);
	memdelete_arr(directional_soft_shadow_kernel);
	memdelete_arr(penumbra_shadow_kernel);
	memdelete_arr(soft_shadow_kernel);

	{
		RD::get_singleton()->free(cluster.directional_light_buffer);
		RD::get_singleton()->free(cluster.omni_light_buffer);
		RD::get_singleton()->free(cluster.spot_light_buffer);
		RD::get_singleton()->free(cluster.reflection_buffer);
		RD::get_singleton()->free(cluster.decal_buffer);
		memdelete_arr(cluster.directional_lights);
		memdelete_arr(cluster.omni_lights);
		memdelete_arr(cluster.spot_lights);
		memdelete_arr(cluster.omni_light_sort);
		memdelete_arr(cluster.spot_light_sort);
		memdelete_arr(cluster.reflections);
		memdelete_arr(cluster.reflection_sort);
		memdelete_arr(cluster.decals);
		memdelete_arr(cluster.decal_sort);
	}

	RD::get_singleton()->free(shadow_sampler);

	directional_shadow_atlas_set_size(0);
	cull_argument.reset(); //avoid exit error
}

/**************************************************************************/
/*  renderer_scene_render.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "renderer_scene_render.h"

#include "core/variant/typed_array.h"

/////////////////////////////////////////////////////////////////////////////
// CameraData

void RendererSceneRender::CameraData::set_camera(const Transform3D p_transform, const Projection p_projection, bool p_is_orthogonal, bool p_is_asymmetrical, bool p_vaspect, const Vector2 &p_taa_jitter, float p_taa_frame_count, uint32_t p_visible_layers) {
	view_count = 1;
	is_orthogonal = p_is_orthogonal;
	is_asymmetrical = p_is_asymmetrical;
	vaspect = p_vaspect;

	main_transform = p_transform;
	main_projection = p_projection;

	visible_layers = p_visible_layers;
	view_offset[0] = Transform3D();
	view_projection[0] = p_projection;
	taa_jitter = p_taa_jitter;
	taa_frame_count = p_taa_frame_count;
}

void RendererSceneRender::CameraData::set_multiview_camera(const Transform3D &p_transform, const LocalVector<Transform3D> &p_offsets, const LocalVector<Projection> &p_projections, bool p_is_orthogonal, bool p_is_asymmetrical, bool p_vaspect, uint32_t p_visible_layers) {
	ERR_FAIL_COND_MSG(p_projections.size() != 2, "Incorrect view count for stereoscopic view");
	ERR_FAIL_COND(p_projections.size() != p_offsets.size());

	visible_layers = p_visible_layers;
	view_count = p_projections.size();
	is_orthogonal = p_is_orthogonal;
	is_asymmetrical = p_is_asymmetrical;
	vaspect = p_vaspect;
	Vector<Plane> planes[2];

	main_transform = p_transform;
	main_projection = Projection::create_combined_projection(p_transform, p_projections[0], p_offsets[0], p_projections[1], p_offsets[1]);

	for (uint32_t v = 0; v < view_count; v++) {
		view_offset[v] = p_offsets[v];
		view_projection[v] = p_projections[v] * Projection(view_offset[v].inverse());
	}
}

/* Compositor effect API */

RID RendererSceneRender::compositor_effect_allocate() {
	return compositor_storage.compositor_effect_allocate();
}

void RendererSceneRender::compositor_effect_initialize(RID p_rid) {
	compositor_storage.compositor_effect_initialize(p_rid);
}

void RendererSceneRender::compositor_effect_free(RID p_rid) {
	compositor_storage.compositor_effect_free(p_rid);
}

bool RendererSceneRender::is_compositor_effect(RID p_effect) const {
	return compositor_storage.is_compositor_effect(p_effect);
}

void RendererSceneRender::compositor_effect_set_enabled(RID p_effect, bool p_enabled) {
	compositor_storage.compositor_effect_set_enabled(p_effect, p_enabled);
}

void RendererSceneRender::compositor_effect_set_callback(RID p_effect, RSE::CompositorEffectCallbackType p_callback_type, const Callable &p_callback) {
	compositor_storage.compositor_effect_set_callback(p_effect, p_callback_type, p_callback);
}

void RendererSceneRender::compositor_effect_set_flag(RID p_effect, RSE::CompositorEffectFlags p_flag, bool p_set) {
	compositor_storage.compositor_effect_set_flag(p_effect, p_flag, p_set);
}

/* Compositor API */

RID RendererSceneRender::compositor_allocate() {
	return compositor_storage.compositor_allocate();
}

void RendererSceneRender::compositor_initialize(RID p_rid) {
	compositor_storage.compositor_initialize(p_rid);
}

void RendererSceneRender::compositor_free(RID p_rid) {
	compositor_storage.compositor_free(p_rid);
}

bool RendererSceneRender::is_compositor(RID p_rid) const {
	return compositor_storage.is_compositor(p_rid);
}

void RendererSceneRender::compositor_set_compositor_effects(RID p_compositor, const TypedArray<RID> &p_effects) {
	Vector<RID> rids;
	for (int i = 0; i < p_effects.size(); i++) {
		RID rid = p_effects[i];
		rids.push_back(rid);
	}

	compositor_storage.compositor_set_compositor_effects(p_compositor, rids);
}

/* Environment API */

RID RendererSceneRender::environment_allocate() {
	return environment_storage.environment_allocate();
}

void RendererSceneRender::environment_initialize(RID p_rid) {
	environment_storage.environment_initialize(p_rid);
}

void RendererSceneRender::environment_free(RID p_rid) {
	environment_storage.environment_free(p_rid);
}

bool RendererSceneRender::is_environment(RID p_rid) const {
	return environment_storage.is_environment(p_rid);
}

// background

void RendererSceneRender::environment_set_background(RID p_env, RSE::EnvironmentBG p_bg) {
	environment_storage.environment_set_background(p_env, p_bg);
}

void RendererSceneRender::environment_set_sky(RID p_env, RID p_sky) {
	environment_storage.environment_set_sky(p_env, p_sky);
}

void RendererSceneRender::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	environment_storage.environment_set_sky_custom_fov(p_env, p_scale);
}

void RendererSceneRender::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
	environment_storage.environment_set_sky_orientation(p_env, p_orientation);
}

void RendererSceneRender::environment_set_bg_color(RID p_env, const Color &p_color) {
	environment_storage.environment_set_bg_color(p_env, p_color);
}

void RendererSceneRender::environment_set_bg_energy(RID p_env, float p_multiplier, float p_exposure_value) {
	environment_storage.environment_set_bg_energy(p_env, p_multiplier, p_exposure_value);
}

void RendererSceneRender::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	environment_storage.environment_set_canvas_max_layer(p_env, p_max_layer);
}

void RendererSceneRender::environment_set_ambient_light(RID p_env, const Color &p_color, RSE::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RSE::EnvironmentReflectionSource p_reflection_source) {
	environment_storage.environment_set_ambient_light(p_env, p_color, p_ambient, p_energy, p_sky_contribution, p_reflection_source);
}

RSE::EnvironmentBG RendererSceneRender::environment_get_background(RID p_env) const {
	return environment_storage.environment_get_background(p_env);
}

RID RendererSceneRender::environment_get_sky(RID p_env) const {
	return environment_storage.environment_get_sky(p_env);
}

float RendererSceneRender::environment_get_sky_custom_fov(RID p_env) const {
	return environment_storage.environment_get_sky_custom_fov(p_env);
}

Basis RendererSceneRender::environment_get_sky_orientation(RID p_env) const {
	return environment_storage.environment_get_sky_orientation(p_env);
}

Color RendererSceneRender::environment_get_bg_color(RID p_env) const {
	return environment_storage.environment_get_bg_color(p_env);
}

float RendererSceneRender::environment_get_bg_energy_multiplier(RID p_env) const {
	return environment_storage.environment_get_bg_energy_multiplier(p_env);
}

float RendererSceneRender::environment_get_bg_intensity(RID p_env) const {
	return environment_storage.environment_get_bg_intensity(p_env);
}

int RendererSceneRender::environment_get_canvas_max_layer(RID p_env) const {
	return environment_storage.environment_get_canvas_max_layer(p_env);
}

RSE::EnvironmentAmbientSource RendererSceneRender::environment_get_ambient_source(RID p_env) const {
	return environment_storage.environment_get_ambient_source(p_env);
}

Color RendererSceneRender::environment_get_ambient_light(RID p_env) const {
	return environment_storage.environment_get_ambient_light(p_env);
}

float RendererSceneRender::environment_get_ambient_light_energy(RID p_env) const {
	return environment_storage.environment_get_ambient_light_energy(p_env);
}

float RendererSceneRender::environment_get_ambient_sky_contribution(RID p_env) const {
	return environment_storage.environment_get_ambient_sky_contribution(p_env);
}

RSE::EnvironmentReflectionSource RendererSceneRender::environment_get_reflection_source(RID p_env) const {
	return environment_storage.environment_get_reflection_source(p_env);
}

void RendererSceneRender::environment_set_camera_feed_id(RID p_env, int p_camera_feed_id) {
	environment_storage.environment_set_camera_feed_id(p_env, p_camera_feed_id);
}

int RendererSceneRender::environment_get_camera_feed_id(RID p_env) const {
	return environment_storage.environment_get_camera_feed_id(p_env);
}

// Tonemap

void RendererSceneRender::environment_set_tonemap(RID p_env, RSE::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white) {
	environment_storage.environment_set_tonemap(p_env, p_tone_mapper, p_exposure, p_white);
}

RSE::EnvironmentToneMapper RendererSceneRender::environment_get_tone_mapper(RID p_env) const {
	return environment_storage.environment_get_tone_mapper(p_env);
}

float RendererSceneRender::environment_get_exposure(RID p_env) const {
	return environment_storage.environment_get_exposure(p_env);
}

float RendererSceneRender::environment_get_white(RID p_env, bool p_limit_agx_white, float p_output_max_value) const {
	return environment_storage.environment_get_white(p_env, p_limit_agx_white, p_output_max_value);
}

void RendererSceneRender::environment_set_tonemap_agx_contrast(RID p_env, float p_agx_contrast) {
	environment_storage.environment_set_tonemap_agx_contrast(p_env, p_agx_contrast);
}

float RendererSceneRender::environment_get_tonemap_agx_contrast(RID p_env) const {
	return environment_storage.environment_get_tonemap_agx_contrast(p_env);
}

RendererEnvironmentStorage::TonemapParameters RendererSceneRender::environment_get_tonemap_parameters(RID p_env, bool p_limit_agx_white, float p_output_max_value) const {
	return environment_storage.environment_get_tonemap_parameters(p_env, p_limit_agx_white, p_output_max_value);
}

// Fog

void RendererSceneRender::environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective, float p_sky_affect, RSE::EnvironmentFogMode p_mode) {
	environment_storage.environment_set_fog(p_env, p_enable, p_light_color, p_light_energy, p_sun_scatter, p_density, p_height, p_height_density, p_aerial_perspective, p_sky_affect, p_mode);
}

bool RendererSceneRender::environment_get_fog_enabled(RID p_env) const {
	return environment_storage.environment_get_fog_enabled(p_env);
}

RSE::EnvironmentFogMode RendererSceneRender::environment_get_fog_mode(RID p_env) const {
	return environment_storage.environment_get_fog_mode(p_env);
}

Color RendererSceneRender::environment_get_fog_light_color(RID p_env) const {
	return environment_storage.environment_get_fog_light_color(p_env);
}

float RendererSceneRender::environment_get_fog_light_energy(RID p_env) const {
	return environment_storage.environment_get_fog_light_energy(p_env);
}

float RendererSceneRender::environment_get_fog_sun_scatter(RID p_env) const {
	return environment_storage.environment_get_fog_sun_scatter(p_env);
}

float RendererSceneRender::environment_get_fog_density(RID p_env) const {
	return environment_storage.environment_get_fog_density(p_env);
}

float RendererSceneRender::environment_get_fog_sky_affect(RID p_env) const {
	return environment_storage.environment_get_fog_sky_affect(p_env);
}

float RendererSceneRender::environment_get_fog_height(RID p_env) const {
	return environment_storage.environment_get_fog_height(p_env);
}

float RendererSceneRender::environment_get_fog_height_density(RID p_env) const {
	return environment_storage.environment_get_fog_height_density(p_env);
}

float RendererSceneRender::environment_get_fog_aerial_perspective(RID p_env) const {
	return environment_storage.environment_get_fog_aerial_perspective(p_env);
}

// Depth Fog

void RendererSceneRender::environment_set_fog_depth(RID p_env, float p_curve, float p_begin, float p_end) {
	environment_storage.environment_set_fog_depth(p_env, p_curve, p_begin, p_end);
}

float RendererSceneRender::environment_get_fog_depth_curve(RID p_env) const {
	return environment_storage.environment_get_fog_depth_curve(p_env);
}

float RendererSceneRender::environment_get_fog_depth_begin(RID p_env) const {
	return environment_storage.environment_get_fog_depth_begin(p_env);
}

float RendererSceneRender::environment_get_fog_depth_end(RID p_env) const {
	return environment_storage.environment_get_fog_depth_end(p_env);
}

// Volumetric Fog

void RendererSceneRender::environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject, float p_sky_affect) {
	environment_storage.environment_set_volumetric_fog(p_env, p_enable, p_density, p_albedo, p_emission, p_emission_energy, p_anisotropy, p_length, p_detail_spread, p_gi_inject, p_temporal_reprojection, p_temporal_reprojection_amount, p_ambient_inject, p_sky_affect);
}

bool RendererSceneRender::environment_get_volumetric_fog_enabled(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_enabled(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_density(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_density(p_env);
}

Color RendererSceneRender::environment_get_volumetric_fog_scattering(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_scattering(p_env);
}

Color RendererSceneRender::environment_get_volumetric_fog_emission(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_emission(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_emission_energy(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_emission_energy(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_anisotropy(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_anisotropy(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_length(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_length(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_detail_spread(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_detail_spread(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_gi_inject(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_gi_inject(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_sky_affect(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_sky_affect(p_env);
}

bool RendererSceneRender::environment_get_volumetric_fog_temporal_reprojection(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_temporal_reprojection(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_temporal_reprojection_amount(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_temporal_reprojection_amount(p_env);
}

float RendererSceneRender::environment_get_volumetric_fog_ambient_inject(RID p_env) const {
	return environment_storage.environment_get_volumetric_fog_ambient_inject(p_env);
}

// GLOW

void RendererSceneRender::environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RSE::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map) {
	environment_storage.environment_set_glow(p_env, p_enable, p_levels, p_intensity, p_strength, p_mix, p_bloom_threshold, p_blend_mode, p_hdr_bleed_threshold, p_hdr_bleed_scale, p_hdr_luminance_cap, p_glow_map_strength, p_glow_map);
}

bool RendererSceneRender::environment_get_glow_enabled(RID p_env) const {
	return environment_storage.environment_get_glow_enabled(p_env);
}

Vector<float> RendererSceneRender::environment_get_glow_levels(RID p_env) const {
	return environment_storage.environment_get_glow_levels(p_env);
}

float RendererSceneRender::environment_get_glow_intensity(RID p_env) const {
	return environment_storage.environment_get_glow_intensity(p_env);
}

float RendererSceneRender::environment_get_glow_strength(RID p_env) const {
	return environment_storage.environment_get_glow_strength(p_env);
}

float RendererSceneRender::environment_get_glow_bloom(RID p_env) const {
	return environment_storage.environment_get_glow_bloom(p_env);
}

float RendererSceneRender::environment_get_glow_mix(RID p_env) const {
	return environment_storage.environment_get_glow_mix(p_env);
}

RSE::EnvironmentGlowBlendMode RendererSceneRender::environment_get_glow_blend_mode(RID p_env) const {
	return environment_storage.environment_get_glow_blend_mode(p_env);
}

float RendererSceneRender::environment_get_glow_hdr_bleed_threshold(RID p_env) const {
	return environment_storage.environment_get_glow_hdr_bleed_threshold(p_env);
}

float RendererSceneRender::environment_get_glow_hdr_luminance_cap(RID p_env) const {
	return environment_storage.environment_get_glow_hdr_luminance_cap(p_env);
}

float RendererSceneRender::environment_get_glow_hdr_bleed_scale(RID p_env) const {
	return environment_storage.environment_get_glow_hdr_bleed_scale(p_env);
}

float RendererSceneRender::environment_get_glow_map_strength(RID p_env) const {
	return environment_storage.environment_get_glow_map_strength(p_env);
}

RID RendererSceneRender::environment_get_glow_map(RID p_env) const {
	return environment_storage.environment_get_glow_map(p_env);
}

// SSR

void RendererSceneRender::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
	environment_storage.environment_set_ssr(p_env, p_enable, p_max_steps, p_fade_int, p_fade_out, p_depth_tolerance);
}

bool RendererSceneRender::environment_get_ssr_enabled(RID p_env) const {
	return environment_storage.environment_get_ssr_enabled(p_env);
}

int RendererSceneRender::environment_get_ssr_max_steps(RID p_env) const {
	return environment_storage.environment_get_ssr_max_steps(p_env);
}

float RendererSceneRender::environment_get_ssr_fade_in(RID p_env) const {
	return environment_storage.environment_get_ssr_fade_in(p_env);
}

float RendererSceneRender::environment_get_ssr_fade_out(RID p_env) const {
	return environment_storage.environment_get_ssr_fade_out(p_env);
}

float RendererSceneRender::environment_get_ssr_depth_tolerance(RID p_env) const {
	return environment_storage.environment_get_ssr_depth_tolerance(p_env);
}

// SSAO

void RendererSceneRender::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
	environment_storage.environment_set_ssao(p_env, p_enable, p_radius, p_intensity, p_power, p_detail, p_horizon, p_sharpness, p_light_affect, p_ao_channel_affect);
}

bool RendererSceneRender::environment_get_ssao_enabled(RID p_env) const {
	return environment_storage.environment_get_ssao_enabled(p_env);
}

float RendererSceneRender::environment_get_ssao_radius(RID p_env) const {
	return environment_storage.environment_get_ssao_radius(p_env);
}

float RendererSceneRender::environment_get_ssao_intensity(RID p_env) const {
	return environment_storage.environment_get_ssao_intensity(p_env);
}

float RendererSceneRender::environment_get_ssao_power(RID p_env) const {
	return environment_storage.environment_get_ssao_power(p_env);
}

float RendererSceneRender::environment_get_ssao_detail(RID p_env) const {
	return environment_storage.environment_get_ssao_detail(p_env);
}

float RendererSceneRender::environment_get_ssao_horizon(RID p_env) const {
	return environment_storage.environment_get_ssao_horizon(p_env);
}

float RendererSceneRender::environment_get_ssao_sharpness(RID p_env) const {
	return environment_storage.environment_get_ssao_sharpness(p_env);
}

float RendererSceneRender::environment_get_ssao_direct_light_affect(RID p_env) const {
	return environment_storage.environment_get_ssao_direct_light_affect(p_env);
}

float RendererSceneRender::environment_get_ssao_ao_channel_affect(RID p_env) const {
	return environment_storage.environment_get_ssao_ao_channel_affect(p_env);
}

// SSIL

void RendererSceneRender::environment_set_ssil(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_sharpness, float p_normal_rejection) {
	environment_storage.environment_set_ssil(p_env, p_enable, p_radius, p_intensity, p_sharpness, p_normal_rejection);
}

bool RendererSceneRender::environment_get_ssil_enabled(RID p_env) const {
	return environment_storage.environment_get_ssil_enabled(p_env);
}

float RendererSceneRender::environment_get_ssil_radius(RID p_env) const {
	return environment_storage.environment_get_ssil_radius(p_env);
}

float RendererSceneRender::environment_get_ssil_intensity(RID p_env) const {
	return environment_storage.environment_get_ssil_intensity(p_env);
}

float RendererSceneRender::environment_get_ssil_sharpness(RID p_env) const {
	return environment_storage.environment_get_ssil_sharpness(p_env);
}

float RendererSceneRender::environment_get_ssil_normal_rejection(RID p_env) const {
	return environment_storage.environment_get_ssil_normal_rejection(p_env);
}

// SDFGI

void RendererSceneRender::environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RSE::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
	environment_storage.environment_set_sdfgi(p_env, p_enable, p_cascades, p_min_cell_size, p_y_scale, p_use_occlusion, p_bounce_feedback, p_read_sky, p_energy, p_normal_bias, p_probe_bias);
}

bool RendererSceneRender::environment_get_sdfgi_enabled(RID p_env) const {
	return environment_storage.environment_get_sdfgi_enabled(p_env);
}

int RendererSceneRender::environment_get_sdfgi_cascades(RID p_env) const {
	return environment_storage.environment_get_sdfgi_cascades(p_env);
}

float RendererSceneRender::environment_get_sdfgi_min_cell_size(RID p_env) const {
	return environment_storage.environment_get_sdfgi_min_cell_size(p_env);
}

bool RendererSceneRender::environment_get_sdfgi_use_occlusion(RID p_env) const {
	return environment_storage.environment_get_sdfgi_use_occlusion(p_env);
}

float RendererSceneRender::environment_get_sdfgi_bounce_feedback(RID p_env) const {
	return environment_storage.environment_get_sdfgi_bounce_feedback(p_env);
}

bool RendererSceneRender::environment_get_sdfgi_read_sky_light(RID p_env) const {
	return environment_storage.environment_get_sdfgi_read_sky_light(p_env);
}

float RendererSceneRender::environment_get_sdfgi_energy(RID p_env) const {
	return environment_storage.environment_get_sdfgi_energy(p_env);
}

float RendererSceneRender::environment_get_sdfgi_normal_bias(RID p_env) const {
	return environment_storage.environment_get_sdfgi_normal_bias(p_env);
}

float RendererSceneRender::environment_get_sdfgi_probe_bias(RID p_env) const {
	return environment_storage.environment_get_sdfgi_probe_bias(p_env);
}

RSE::EnvironmentSDFGIYScale RendererSceneRender::environment_get_sdfgi_y_scale(RID p_env) const {
	return environment_storage.environment_get_sdfgi_y_scale(p_env);
}

// Adjustments

void RendererSceneRender::environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) {
	environment_storage.environment_set_adjustment(p_env, p_enable, p_brightness, p_contrast, p_saturation, p_use_1d_color_correction, p_color_correction);
}

bool RendererSceneRender::environment_get_adjustments_enabled(RID p_env) const {
	return environment_storage.environment_get_adjustments_enabled(p_env);
}

float RendererSceneRender::environment_get_adjustments_brightness(RID p_env) const {
	return environment_storage.environment_get_adjustments_brightness(p_env);
}

float RendererSceneRender::environment_get_adjustments_contrast(RID p_env) const {
	return environment_storage.environment_get_adjustments_contrast(p_env);
}

float RendererSceneRender::environment_get_adjustments_saturation(RID p_env) const {
	return environment_storage.environment_get_adjustments_saturation(p_env);
}

bool RendererSceneRender::environment_get_use_1d_color_correction(RID p_env) const {
	return environment_storage.environment_get_use_1d_color_correction(p_env);
}

RID RendererSceneRender::environment_get_color_correction(RID p_env) const {
	return environment_storage.environment_get_color_correction(p_env);
}

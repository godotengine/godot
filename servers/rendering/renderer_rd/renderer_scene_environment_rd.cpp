/*************************************************************************/
/*  renderer_scene_environment_rd.cpp                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "servers/rendering/renderer_rd/renderer_scene_environment_rd.h"

uint64_t RendererSceneEnvironmentRD::auto_exposure_counter = 2;

void RendererSceneEnvironmentRD::set_ambient_light(const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source) {
	ambient_light = p_color;
	ambient_source = p_ambient;
	ambient_light_energy = p_energy;
	ambient_sky_contribution = p_sky_contribution;
	reflection_source = p_reflection_source;
}

void RendererSceneEnvironmentRD::set_tonemap(RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	exposure = p_exposure;
	tone_mapper = p_tone_mapper;
	if (!auto_exposure && p_auto_exposure) {
		auto_exposure_version = ++auto_exposure_counter;
	}
	auto_exposure = p_auto_exposure;
	white = p_white;
	min_luminance = p_min_luminance;
	max_luminance = p_max_luminance;
	auto_exp_speed = p_auto_exp_speed;
	auto_exp_scale = p_auto_exp_scale;
}

void RendererSceneEnvironmentRD::set_glow(bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) {
	ERR_FAIL_COND_MSG(p_levels.size() != 7, "Size of array of glow levels must be 7");
	glow_enabled = p_enable;
	glow_levels = p_levels;
	glow_intensity = p_intensity;
	glow_strength = p_strength;
	glow_mix = p_mix;
	glow_bloom = p_bloom_threshold;
	glow_blend_mode = p_blend_mode;
	glow_hdr_bleed_threshold = p_hdr_bleed_threshold;
	glow_hdr_bleed_scale = p_hdr_bleed_scale;
	glow_hdr_luminance_cap = p_hdr_luminance_cap;
}

void RendererSceneEnvironmentRD::set_sdfgi(bool p_enable, RS::EnvironmentSDFGICascades p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
	sdfgi_enabled = p_enable;
	sdfgi_cascades = p_cascades;
	sdfgi_min_cell_size = p_min_cell_size;
	sdfgi_use_occlusion = p_use_occlusion;
	sdfgi_bounce_feedback = p_bounce_feedback;
	sdfgi_read_sky_light = p_read_sky;
	sdfgi_energy = p_energy;
	sdfgi_normal_bias = p_normal_bias;
	sdfgi_probe_bias = p_probe_bias;
	sdfgi_y_scale = p_y_scale;
}

void RendererSceneEnvironmentRD::set_fog(bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_fog_aerial_perspective) {
	fog_enabled = p_enable;
	fog_light_color = p_light_color;
	fog_light_energy = p_light_energy;
	fog_sun_scatter = p_sun_scatter;
	fog_density = p_density;
	fog_height = p_height;
	fog_height_density = p_height_density;
	fog_aerial_perspective = p_fog_aerial_perspective;
}

void RendererSceneEnvironmentRD::set_volumetric_fog(bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) {
	volumetric_fog_enabled = p_enable;
	volumetric_fog_density = p_density;
	volumetric_fog_scattering = p_albedo;
	volumetric_fog_emission = p_emission;
	volumetric_fog_emission_energy = p_emission_energy;
	volumetric_fog_anisotropy = p_anisotropy,
	volumetric_fog_length = p_length;
	volumetric_fog_detail_spread = p_detail_spread;
	volumetric_fog_gi_inject = p_gi_inject;
	volumetric_fog_temporal_reprojection = p_temporal_reprojection;
	volumetric_fog_temporal_reprojection_amount = p_temporal_reprojection_amount;
	volumetric_fog_ambient_inject = p_ambient_inject;
}

void RendererSceneEnvironmentRD::set_ssr(bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
	ssr_enabled = p_enable;
	ssr_max_steps = p_max_steps;
	ssr_fade_in = p_fade_int;
	ssr_fade_out = p_fade_out;
	ssr_depth_tolerance = p_depth_tolerance;
}

void RendererSceneEnvironmentRD::set_ssao(bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
	ssao_enabled = p_enable;
	ssao_radius = p_radius;
	ssao_intensity = p_intensity;
	ssao_power = p_power;
	ssao_detail = p_detail;
	ssao_horizon = p_horizon;
	ssao_sharpness = p_sharpness;
	ssao_direct_light_affect = p_light_affect;
	ssao_ao_channel_affect = p_ao_channel_affect;
}

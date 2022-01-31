/*************************************************************************/
/*  renderer_scene_environment_rd.h                                      */
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

#ifndef RENDERING_SERVER_SCENE_ENVIRONMENT_RD_H
#define RENDERING_SERVER_SCENE_ENVIRONMENT_RD_H

#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering/rendering_device.h"

class RendererSceneEnvironmentRD {
private:
	static uint64_t auto_exposure_counter;

public:
	// BG
	RS::EnvironmentBG background = RS::ENV_BG_CLEAR_COLOR;
	RID sky;
	float sky_custom_fov = 0.0;
	Basis sky_orientation;
	Color bg_color;
	float bg_energy = 1.0;
	int canvas_max_layer = 0;
	RS::EnvironmentAmbientSource ambient_source = RS::ENV_AMBIENT_SOURCE_BG;
	Color ambient_light;
	float ambient_light_energy = 1.0;
	float ambient_sky_contribution = 1.0;
	RS::EnvironmentReflectionSource reflection_source = RS::ENV_REFLECTION_SOURCE_BG;

	/// Tonemap

	RS::EnvironmentToneMapper tone_mapper;
	float exposure = 1.0;
	float white = 1.0;
	bool auto_exposure = false;
	float min_luminance = 0.2;
	float max_luminance = 8.0;
	float auto_exp_speed = 0.2;
	float auto_exp_scale = 0.5;
	uint64_t auto_exposure_version = 0;

	// Fog
	bool fog_enabled = false;
	Color fog_light_color = Color(0.5, 0.6, 0.7);
	float fog_light_energy = 1.0;
	float fog_sun_scatter = 0.0;
	float fog_density = 0.001;
	float fog_height = 0.0;
	float fog_height_density = 0.0; //can be negative to invert effect
	float fog_aerial_perspective = 0.0;

	/// Volumetric Fog
	///
	bool volumetric_fog_enabled = false;
	float volumetric_fog_density = 0.01;
	Color volumetric_fog_scattering = Color(1, 1, 1);
	Color volumetric_fog_emission = Color(0, 0, 0);
	float volumetric_fog_emission_energy = 0.0;
	float volumetric_fog_anisotropy = 0.2;
	float volumetric_fog_length = 64.0;
	float volumetric_fog_detail_spread = 2.0;
	float volumetric_fog_gi_inject = 0.0;
	bool volumetric_fog_temporal_reprojection = true;
	float volumetric_fog_temporal_reprojection_amount = 0.9;
	float volumetric_fog_ambient_inject = 0.0;

	/// Glow

	bool glow_enabled = false;
	Vector<float> glow_levels;
	float glow_intensity = 0.8;
	float glow_strength = 1.0;
	float glow_bloom = 0.0;
	float glow_mix = 0.01;
	RS::EnvironmentGlowBlendMode glow_blend_mode = RS::ENV_GLOW_BLEND_MODE_SOFTLIGHT;
	float glow_hdr_bleed_threshold = 1.0;
	float glow_hdr_luminance_cap = 12.0;
	float glow_hdr_bleed_scale = 2.0;
	float glow_map_strength = 0.0f;
	RID glow_map = RID();

	/// SSAO

	bool ssao_enabled = false;
	float ssao_radius = 1.0;
	float ssao_intensity = 2.0;
	float ssao_power = 1.5;
	float ssao_detail = 0.5;
	float ssao_horizon = 0.06;
	float ssao_sharpness = 0.98;
	float ssao_direct_light_affect = 0.0;
	float ssao_ao_channel_affect = 0.0;

	/// SSR
	///
	bool ssr_enabled = false;
	int ssr_max_steps = 64;
	float ssr_fade_in = 0.15;
	float ssr_fade_out = 2.0;
	float ssr_depth_tolerance = 0.2;

	/// SSIL
	///
	bool ssil_enabled = false;
	float ssil_radius = 5.0;
	float ssil_intensity = 1.0;
	float ssil_sharpness = 0.98;
	float ssil_normal_rejection = 1.0;

	/// SDFGI
	bool sdfgi_enabled = false;
	int sdfgi_cascades = 6;
	float sdfgi_min_cell_size = 0.2;
	bool sdfgi_use_occlusion = false;
	float sdfgi_bounce_feedback = 0.0;
	bool sdfgi_read_sky_light = false;
	float sdfgi_energy = 1.0;
	float sdfgi_normal_bias = 1.1;
	float sdfgi_probe_bias = 1.1;
	RS::EnvironmentSDFGIYScale sdfgi_y_scale = RS::ENV_SDFGI_Y_SCALE_DISABLED;

	/// Adjustments

	bool adjustments_enabled = false;
	float adjustments_brightness = 1.0f;
	float adjustments_contrast = 1.0f;
	float adjustments_saturation = 1.0f;
	bool use_1d_color_correction = false;
	RID color_correction = RID();

	void set_ambient_light(const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source);
	void set_tonemap(RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale);
	void set_glow(bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map);
	void set_sdfgi(bool p_enable, int p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias);
	void set_fog(bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_fog_aerial_perspective);
	void set_volumetric_fog(bool p_enable, float p_density, const Color &p_scatterin, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject);
	void set_ssr(bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance);
	void set_ssao(bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect);
};

#endif /* !RENDERING_SERVER_SCENE_ENVIRONMENT_RD_H */

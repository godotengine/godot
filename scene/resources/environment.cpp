/*************************************************************************/
/*  environment.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "environment.h"
#include "global_config.h"
#include "servers/visual_server.h"
#include "texture.h"

RID Environment::get_rid() const {

	return environment;
}

void Environment::set_background(BGMode p_bg) {

	bg_mode = p_bg;
	VS::get_singleton()->environment_set_background(environment, VS::EnvironmentBG(p_bg));
	_change_notify();
}

void Environment::set_skybox(const Ref<SkyBox> &p_skybox) {

	bg_skybox = p_skybox;

	RID sb_rid;
	if (bg_skybox.is_valid())
		sb_rid = bg_skybox->get_rid();

	VS::get_singleton()->environment_set_skybox(environment, sb_rid);
}

void Environment::set_skybox_scale(float p_scale) {

	bg_skybox_scale = p_scale;
	VS::get_singleton()->environment_set_skybox_scale(environment, p_scale);
}

void Environment::set_bg_color(const Color &p_color) {

	bg_color = p_color;
	VS::get_singleton()->environment_set_bg_color(environment, p_color);
}
void Environment::set_bg_energy(float p_energy) {

	bg_energy = p_energy;
	VS::get_singleton()->environment_set_bg_energy(environment, p_energy);
}
void Environment::set_canvas_max_layer(int p_max_layer) {

	bg_canvas_max_layer = p_max_layer;
	VS::get_singleton()->environment_set_canvas_max_layer(environment, p_max_layer);
}
void Environment::set_ambient_light_color(const Color &p_color) {

	ambient_color = p_color;
	VS::get_singleton()->environment_set_ambient_light(environment, ambient_color, ambient_energy, ambient_skybox_contribution);
}
void Environment::set_ambient_light_energy(float p_energy) {

	ambient_energy = p_energy;
	VS::get_singleton()->environment_set_ambient_light(environment, ambient_color, ambient_energy, ambient_skybox_contribution);
}
void Environment::set_ambient_light_skybox_contribution(float p_energy) {

	ambient_skybox_contribution = p_energy;
	VS::get_singleton()->environment_set_ambient_light(environment, ambient_color, ambient_energy, ambient_skybox_contribution);
}

Environment::BGMode Environment::get_background() const {

	return bg_mode;
}
Ref<SkyBox> Environment::get_skybox() const {

	return bg_skybox;
}

float Environment::get_skybox_scale() const {

	return bg_skybox_scale;
}

Color Environment::get_bg_color() const {

	return bg_color;
}
float Environment::get_bg_energy() const {

	return bg_energy;
}
int Environment::get_canvas_max_layer() const {

	return bg_canvas_max_layer;
}
Color Environment::get_ambient_light_color() const {

	return ambient_color;
}
float Environment::get_ambient_light_energy() const {

	return ambient_energy;
}
float Environment::get_ambient_light_skybox_contribution() const {

	return ambient_skybox_contribution;
}

void Environment::set_tonemapper(ToneMapper p_tone_mapper) {

	tone_mapper = p_tone_mapper;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}

Environment::ToneMapper Environment::get_tonemapper() const {

	return tone_mapper;
}

void Environment::set_tonemap_exposure(float p_exposure) {

	tonemap_exposure = p_exposure;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}

float Environment::get_tonemap_exposure() const {

	return tonemap_exposure;
}

void Environment::set_tonemap_white(float p_white) {

	tonemap_white = p_white;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_white() const {

	return tonemap_white;
}

void Environment::set_tonemap_auto_exposure(bool p_enabled) {

	tonemap_auto_exposure = p_enabled;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
bool Environment::get_tonemap_auto_exposure() const {

	return tonemap_auto_exposure;
}

void Environment::set_tonemap_auto_exposure_max(float p_auto_exposure_max) {

	tonemap_auto_exposure_max = p_auto_exposure_max;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_max() const {

	return tonemap_auto_exposure_max;
}

void Environment::set_tonemap_auto_exposure_min(float p_auto_exposure_min) {

	tonemap_auto_exposure_min = p_auto_exposure_min;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_min() const {

	return tonemap_auto_exposure_min;
}

void Environment::set_tonemap_auto_exposure_speed(float p_auto_exposure_speed) {

	tonemap_auto_exposure_speed = p_auto_exposure_speed;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_speed() const {

	return tonemap_auto_exposure_speed;
}

void Environment::set_tonemap_auto_exposure_grey(float p_auto_exposure_grey) {

	tonemap_auto_exposure_grey = p_auto_exposure_grey;
	VS::get_singleton()->environment_set_tonemap(environment, VS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_grey() const {

	return tonemap_auto_exposure_grey;
}

void Environment::set_adjustment_enable(bool p_enable) {

	adjustment_enabled = p_enable;
	VS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}

bool Environment::is_adjustment_enabled() const {

	return adjustment_enabled;
}

void Environment::set_adjustment_brightness(float p_brightness) {

	adjustment_brightness = p_brightness;
	VS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
float Environment::get_adjustment_brightness() const {

	return adjustment_brightness;
}

void Environment::set_adjustment_contrast(float p_contrast) {

	adjustment_contrast = p_contrast;
	VS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
float Environment::get_adjustment_contrast() const {

	return adjustment_contrast;
}

void Environment::set_adjustment_saturation(float p_saturation) {

	adjustment_saturation = p_saturation;
	VS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
float Environment::get_adjustment_saturation() const {

	return adjustment_saturation;
}

void Environment::set_adjustment_color_correction(const Ref<Texture> &p_ramp) {

	adjustment_color_correction = p_ramp;
	VS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
Ref<Texture> Environment::get_adjustment_color_correction() const {

	return adjustment_color_correction;
}

void Environment::_validate_property(PropertyInfo &property) const {

	if (property.name == "background/skybox" || property.name == "background/skybox_scale" || property.name == "ambient_light/skybox_contribution") {
		if (bg_mode != BG_SKYBOX) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name == "background/color") {
		if (bg_mode != BG_COLOR) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name == "background/canvas_max_layer") {
		if (bg_mode != BG_CANVAS) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}
}

void Environment::set_ssr_enabled(bool p_enable) {

	ssr_enabled = p_enable;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}

bool Environment::is_ssr_enabled() const {

	return ssr_enabled;
}

void Environment::set_ssr_max_steps(int p_steps) {

	ssr_max_steps = p_steps;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}
int Environment::get_ssr_max_steps() const {

	return ssr_max_steps;
}

void Environment::set_ssr_accel(float p_accel) {

	ssr_accel = p_accel;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}
float Environment::get_ssr_accel() const {

	return ssr_accel;
}

void Environment::set_ssr_fade(float p_fade) {

	ssr_fade = p_fade;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}
float Environment::get_ssr_fade() const {

	return ssr_fade;
}

void Environment::set_ssr_depth_tolerance(float p_depth_tolerance) {

	ssr_depth_tolerance = p_depth_tolerance;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}
float Environment::get_ssr_depth_tolerance() const {

	return ssr_depth_tolerance;
}

void Environment::set_ssr_smooth(bool p_enable) {

	ssr_smooth = p_enable;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}
bool Environment::is_ssr_smooth() const {

	return ssr_smooth;
}

void Environment::set_ssr_rough(bool p_enable) {

	ssr_roughness = p_enable;
	VS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_accel, ssr_fade, ssr_depth_tolerance, ssr_smooth, ssr_roughness);
}
bool Environment::is_ssr_rough() const {

	return ssr_roughness;
}

void Environment::set_ssao_enabled(bool p_enable) {

	ssao_enabled = p_enable;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}

bool Environment::is_ssao_enabled() const {

	return ssao_enabled;
}

void Environment::set_ssao_radius(float p_radius) {

	ssao_radius = p_radius;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}
float Environment::get_ssao_radius() const {

	return ssao_radius;
}

void Environment::set_ssao_intensity(float p_intensity) {

	ssao_intensity = p_intensity;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}

float Environment::get_ssao_intensity() const {

	return ssao_intensity;
}

void Environment::set_ssao_radius2(float p_radius) {

	ssao_radius2 = p_radius;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}
float Environment::get_ssao_radius2() const {

	return ssao_radius2;
}

void Environment::set_ssao_intensity2(float p_intensity) {

	ssao_intensity2 = p_intensity;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}
float Environment::get_ssao_intensity2() const {

	return ssao_intensity2;
}

void Environment::set_ssao_bias(float p_bias) {

	ssao_bias = p_bias;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}
float Environment::get_ssao_bias() const {

	return ssao_bias;
}

void Environment::set_ssao_direct_light_affect(float p_direct_light_affect) {

	ssao_direct_light_affect = p_direct_light_affect;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}
float Environment::get_ssao_direct_light_affect() const {

	return ssao_direct_light_affect;
}

void Environment::set_ssao_color(const Color &p_color) {

	ssao_color = p_color;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}

Color Environment::get_ssao_color() const {

	return ssao_color;
}

void Environment::set_ssao_blur(bool p_enable) {

	ssao_blur = p_enable;
	VS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_radius2, ssao_intensity2, ssao_bias, ssao_direct_light_affect, ssao_color, ssao_blur);
}
bool Environment::is_ssao_blur_enabled() const {

	return ssao_blur;
}

void Environment::set_glow_enabled(bool p_enabled) {

	glow_enabled = p_enabled;
	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}

bool Environment::is_glow_enabled() const {

	return glow_enabled;
}

void Environment::set_glow_level(int p_level, bool p_enabled) {

	ERR_FAIL_INDEX(p_level, VS::MAX_GLOW_LEVELS);

	if (p_enabled)
		glow_levels |= (1 << p_level);
	else
		glow_levels &= ~(1 << p_level);

	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
bool Environment::is_glow_level_enabled(int p_level) const {

	ERR_FAIL_INDEX_V(p_level, VS::MAX_GLOW_LEVELS, false);

	return glow_levels & (1 << p_level);
}

void Environment::set_glow_intensity(float p_intensity) {

	glow_intensity = p_intensity;

	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
float Environment::get_glow_intensity() const {

	return glow_intensity;
}

void Environment::set_glow_strength(float p_strength) {

	glow_strength = p_strength;
	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
float Environment::get_glow_strength() const {

	return glow_strength;
}

void Environment::set_glow_bloom(float p_treshold) {

	glow_bloom = p_treshold;

	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
float Environment::get_glow_bloom() const {

	return glow_bloom;
}

void Environment::set_glow_blend_mode(GlowBlendMode p_mode) {

	glow_blend_mode = p_mode;

	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
Environment::GlowBlendMode Environment::get_glow_blend_mode() const {

	return glow_blend_mode;
}

void Environment::set_glow_hdr_bleed_treshold(float p_treshold) {

	glow_hdr_bleed_treshold = p_treshold;

	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
float Environment::get_glow_hdr_bleed_treshold() const {

	return glow_hdr_bleed_treshold;
}

void Environment::set_glow_hdr_bleed_scale(float p_scale) {

	glow_hdr_bleed_scale = p_scale;

	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}
float Environment::get_glow_hdr_bleed_scale() const {

	return glow_hdr_bleed_scale;
}

void Environment::set_glow_bicubic_upscale(bool p_enable) {

	glow_bicubic_upscale = p_enable;
	VS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_bloom, VS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_treshold, glow_hdr_bleed_treshold, glow_bicubic_upscale);
}

bool Environment::is_glow_bicubic_upscale_enabled() const {

	return glow_bicubic_upscale;
}

void Environment::set_dof_blur_far_enabled(bool p_enable) {

	dof_blur_far_enabled = p_enable;
	VS::get_singleton()->environment_set_dof_blur_far(environment, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_far_amount, VS::EnvironmentDOFBlurQuality(dof_blur_far_quality));
}

bool Environment::is_dof_blur_far_enabled() const {

	return dof_blur_far_enabled;
}

void Environment::set_dof_blur_far_distance(float p_distance) {

	dof_blur_far_distance = p_distance;
	VS::get_singleton()->environment_set_dof_blur_far(environment, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_far_amount, VS::EnvironmentDOFBlurQuality(dof_blur_far_quality));
}
float Environment::get_dof_blur_far_distance() const {

	return dof_blur_far_distance;
}

void Environment::set_dof_blur_far_transition(float p_distance) {

	dof_blur_far_transition = p_distance;
	VS::get_singleton()->environment_set_dof_blur_far(environment, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_far_amount, VS::EnvironmentDOFBlurQuality(dof_blur_far_quality));
}
float Environment::get_dof_blur_far_transition() const {

	return dof_blur_far_transition;
}

void Environment::set_dof_blur_far_amount(float p_amount) {

	dof_blur_far_amount = p_amount;
	VS::get_singleton()->environment_set_dof_blur_far(environment, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_far_amount, VS::EnvironmentDOFBlurQuality(dof_blur_far_quality));
}
float Environment::get_dof_blur_far_amount() const {

	return dof_blur_far_amount;
}

void Environment::set_dof_blur_far_quality(DOFBlurQuality p_quality) {

	dof_blur_far_quality = p_quality;
	VS::get_singleton()->environment_set_dof_blur_far(environment, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_far_amount, VS::EnvironmentDOFBlurQuality(dof_blur_far_quality));
}

Environment::DOFBlurQuality Environment::get_dof_blur_far_quality() const {

	return dof_blur_far_quality;
}

void Environment::set_dof_blur_near_enabled(bool p_enable) {

	dof_blur_near_enabled = p_enable;
	VS::get_singleton()->environment_set_dof_blur_near(environment, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_near_amount, VS::EnvironmentDOFBlurQuality(dof_blur_near_quality));
}

bool Environment::is_dof_blur_near_enabled() const {

	return dof_blur_near_enabled;
}

void Environment::set_dof_blur_near_distance(float p_distance) {

	dof_blur_near_distance = p_distance;
	VS::get_singleton()->environment_set_dof_blur_near(environment, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_near_amount, VS::EnvironmentDOFBlurQuality(dof_blur_near_quality));
}

float Environment::get_dof_blur_near_distance() const {

	return dof_blur_near_distance;
}

void Environment::set_dof_blur_near_transition(float p_distance) {

	dof_blur_near_transition = p_distance;
	VS::get_singleton()->environment_set_dof_blur_near(environment, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_near_amount, VS::EnvironmentDOFBlurQuality(dof_blur_near_quality));
}

float Environment::get_dof_blur_near_transition() const {

	return dof_blur_near_transition;
}

void Environment::set_dof_blur_near_amount(float p_amount) {

	dof_blur_near_amount = p_amount;
	VS::get_singleton()->environment_set_dof_blur_near(environment, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_near_amount, VS::EnvironmentDOFBlurQuality(dof_blur_near_quality));
}

float Environment::get_dof_blur_near_amount() const {

	return dof_blur_near_amount;
}

void Environment::set_dof_blur_near_quality(DOFBlurQuality p_quality) {

	dof_blur_near_quality = p_quality;
	VS::get_singleton()->environment_set_dof_blur_near(environment, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_near_amount, VS::EnvironmentDOFBlurQuality(dof_blur_near_quality));
}

Environment::DOFBlurQuality Environment::get_dof_blur_near_quality() const {

	return dof_blur_near_quality;
}

void Environment::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_background", "mode"), &Environment::set_background);
	ClassDB::bind_method(D_METHOD("set_skybox", "skybox:CubeMap"), &Environment::set_skybox);
	ClassDB::bind_method(D_METHOD("set_skybox_scale", "scale"), &Environment::set_skybox_scale);
	ClassDB::bind_method(D_METHOD("set_bg_color", "color"), &Environment::set_bg_color);
	ClassDB::bind_method(D_METHOD("set_bg_energy", "energy"), &Environment::set_bg_energy);
	ClassDB::bind_method(D_METHOD("set_canvas_max_layer", "layer"), &Environment::set_canvas_max_layer);
	ClassDB::bind_method(D_METHOD("set_ambient_light_color", "color"), &Environment::set_ambient_light_color);
	ClassDB::bind_method(D_METHOD("set_ambient_light_energy", "energy"), &Environment::set_ambient_light_energy);
	ClassDB::bind_method(D_METHOD("set_ambient_light_skybox_contribution", "energy"), &Environment::set_ambient_light_skybox_contribution);

	ClassDB::bind_method(D_METHOD("get_background"), &Environment::get_background);
	ClassDB::bind_method(D_METHOD("get_skybox:CubeMap"), &Environment::get_skybox);
	ClassDB::bind_method(D_METHOD("get_skybox_scale"), &Environment::get_skybox_scale);
	ClassDB::bind_method(D_METHOD("get_bg_color"), &Environment::get_bg_color);
	ClassDB::bind_method(D_METHOD("get_bg_energy"), &Environment::get_bg_energy);
	ClassDB::bind_method(D_METHOD("get_canvas_max_layer"), &Environment::get_canvas_max_layer);
	ClassDB::bind_method(D_METHOD("get_ambient_light_color"), &Environment::get_ambient_light_color);
	ClassDB::bind_method(D_METHOD("get_ambient_light_energy"), &Environment::get_ambient_light_energy);
	ClassDB::bind_method(D_METHOD("get_ambient_light_skybox_contribution"), &Environment::get_ambient_light_skybox_contribution);

	ADD_GROUP("Background", "background_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "background_mode", PROPERTY_HINT_ENUM, "Clear Color,Custom Color,Skybox,Canvas,Keep"), "set_background", "get_background");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "background_skybox", PROPERTY_HINT_RESOURCE_TYPE, "SkyBox"), "set_skybox", "get_skybox");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "background_skybox_scale", PROPERTY_HINT_RANGE, "0,32,0.01"), "set_skybox_scale", "get_skybox_scale");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "background_color"), "set_bg_color", "get_bg_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "background_energy", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_bg_energy", "get_bg_energy");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "background_canvas_max_layer", PROPERTY_HINT_RANGE, "-1000,1000,1"), "set_canvas_max_layer", "get_canvas_max_layer");
	ADD_GROUP("Ambient Light", "ambient_light_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_light_color"), "set_ambient_light_color", "get_ambient_light_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_light_energy", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_ambient_light_energy", "get_ambient_light_energy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_light_skybox_contribution", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_ambient_light_skybox_contribution", "get_ambient_light_skybox_contribution");

	ClassDB::bind_method(D_METHOD("set_ssr_enabled", "enabled"), &Environment::set_ssr_enabled);
	ClassDB::bind_method(D_METHOD("is_ssr_enabled"), &Environment::is_ssr_enabled);

	ClassDB::bind_method(D_METHOD("set_ssr_max_steps", "max_steps"), &Environment::set_ssr_max_steps);
	ClassDB::bind_method(D_METHOD("get_ssr_max_steps"), &Environment::get_ssr_max_steps);

	ClassDB::bind_method(D_METHOD("set_ssr_accel", "accel"), &Environment::set_ssr_accel);
	ClassDB::bind_method(D_METHOD("get_ssr_accel"), &Environment::get_ssr_accel);

	ClassDB::bind_method(D_METHOD("set_ssr_fade", "fade"), &Environment::set_ssr_fade);
	ClassDB::bind_method(D_METHOD("get_ssr_fade"), &Environment::get_ssr_fade);

	ClassDB::bind_method(D_METHOD("set_ssr_depth_tolerance", "depth_tolerance"), &Environment::set_ssr_depth_tolerance);
	ClassDB::bind_method(D_METHOD("get_ssr_depth_tolerance"), &Environment::get_ssr_depth_tolerance);

	ClassDB::bind_method(D_METHOD("set_ssr_smooth", "smooth"), &Environment::set_ssr_smooth);
	ClassDB::bind_method(D_METHOD("is_ssr_smooth"), &Environment::is_ssr_smooth);

	ClassDB::bind_method(D_METHOD("set_ssr_rough", "rough"), &Environment::set_ssr_rough);
	ClassDB::bind_method(D_METHOD("is_ssr_rough"), &Environment::is_ssr_rough);

	ADD_GROUP("SS Reflections", "ss_reflections_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ss_reflections_enabled"), "set_ssr_enabled", "is_ssr_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ss_reflections_max_steps", PROPERTY_HINT_RANGE, "1,512,1"), "set_ssr_max_steps", "get_ssr_max_steps");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ss_reflections_accel", PROPERTY_HINT_RANGE, "0,4,0.01"), "set_ssr_accel", "get_ssr_accel");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ss_reflections_fade", PROPERTY_HINT_EXP_EASING), "set_ssr_fade", "get_ssr_fade");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ss_reflections_depth_tolerance", PROPERTY_HINT_RANGE, "0.1,128,0.1"), "set_ssr_depth_tolerance", "get_ssr_depth_tolerance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ss_reflections_accel_smooth"), "set_ssr_smooth", "is_ssr_smooth");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ss_reflections_roughness"), "set_ssr_rough", "is_ssr_rough");

	ClassDB::bind_method(D_METHOD("set_ssao_enabled", "enabled"), &Environment::set_ssao_enabled);
	ClassDB::bind_method(D_METHOD("is_ssao_enabled"), &Environment::is_ssao_enabled);

	ClassDB::bind_method(D_METHOD("set_ssao_radius", "radius"), &Environment::set_ssao_radius);
	ClassDB::bind_method(D_METHOD("get_ssao_radius"), &Environment::get_ssao_radius);

	ClassDB::bind_method(D_METHOD("set_ssao_intensity", "intensity"), &Environment::set_ssao_intensity);
	ClassDB::bind_method(D_METHOD("get_ssao_intensity"), &Environment::get_ssao_intensity);

	ClassDB::bind_method(D_METHOD("set_ssao_radius2", "radius"), &Environment::set_ssao_radius2);
	ClassDB::bind_method(D_METHOD("get_ssao_radius2"), &Environment::get_ssao_radius2);

	ClassDB::bind_method(D_METHOD("set_ssao_intensity2", "intensity"), &Environment::set_ssao_intensity2);
	ClassDB::bind_method(D_METHOD("get_ssao_intensity2"), &Environment::get_ssao_intensity2);

	ClassDB::bind_method(D_METHOD("set_ssao_bias", "bias"), &Environment::set_ssao_bias);
	ClassDB::bind_method(D_METHOD("get_ssao_bias"), &Environment::get_ssao_bias);

	ClassDB::bind_method(D_METHOD("set_ssao_direct_light_affect", "amount"), &Environment::set_ssao_direct_light_affect);
	ClassDB::bind_method(D_METHOD("get_ssao_direct_light_affect"), &Environment::get_ssao_direct_light_affect);

	ClassDB::bind_method(D_METHOD("set_ssao_color", "color"), &Environment::set_ssao_color);
	ClassDB::bind_method(D_METHOD("get_ssao_color"), &Environment::get_ssao_color);

	ClassDB::bind_method(D_METHOD("set_ssao_blur", "enabled"), &Environment::set_ssao_blur);
	ClassDB::bind_method(D_METHOD("is_ssao_blur_enabled"), &Environment::is_ssao_blur_enabled);

	ADD_GROUP("SSAO", "ssao_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ssao_enabled"), "set_ssao_enabled", "is_ssao_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ssao_radius", PROPERTY_HINT_RANGE, "0.1,16,0.1"), "set_ssao_radius", "get_ssao_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ssao_intensity", PROPERTY_HINT_RANGE, "0.0,9,0.1"), "set_ssao_intensity", "get_ssao_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ssao_radius2", PROPERTY_HINT_RANGE, "0.0,16,0.1"), "set_ssao_radius2", "get_ssao_radius2");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ssao_intensity2", PROPERTY_HINT_RANGE, "0.0,9,0.1"), "set_ssao_intensity2", "get_ssao_intensity2");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ssao_bias", PROPERTY_HINT_RANGE, "0.001,8,0.001"), "set_ssao_bias", "get_ssao_bias");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ssao_light_affect", PROPERTY_HINT_RANGE, "0.00,1,0.01"), "set_ssao_direct_light_affect", "get_ssao_direct_light_affect");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ssao_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ssao_color", "get_ssao_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ssao_blur"), "set_ssao_blur", "is_ssao_blur_enabled");

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_enabled", "enabled"), &Environment::set_dof_blur_far_enabled);
	ClassDB::bind_method(D_METHOD("is_dof_blur_far_enabled"), &Environment::is_dof_blur_far_enabled);

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_distance", "intensity"), &Environment::set_dof_blur_far_distance);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_distance"), &Environment::get_dof_blur_far_distance);

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_transition", "intensity"), &Environment::set_dof_blur_far_transition);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_transition"), &Environment::get_dof_blur_far_transition);

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_amount", "intensity"), &Environment::set_dof_blur_far_amount);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_amount"), &Environment::get_dof_blur_far_amount);

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_quality", "intensity"), &Environment::set_dof_blur_far_quality);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_quality"), &Environment::get_dof_blur_far_quality);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_enabled", "enabled"), &Environment::set_dof_blur_near_enabled);
	ClassDB::bind_method(D_METHOD("is_dof_blur_near_enabled"), &Environment::is_dof_blur_near_enabled);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_distance", "intensity"), &Environment::set_dof_blur_near_distance);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_distance"), &Environment::get_dof_blur_near_distance);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_transition", "intensity"), &Environment::set_dof_blur_near_transition);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_transition"), &Environment::get_dof_blur_near_transition);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_amount", "intensity"), &Environment::set_dof_blur_near_amount);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_amount"), &Environment::get_dof_blur_near_amount);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_quality", "level"), &Environment::set_dof_blur_near_quality);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_quality"), &Environment::get_dof_blur_near_quality);

	ADD_GROUP("DOF Far Blur", "dof_blur_far_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dof_blur_far_enabled"), "set_dof_blur_far_enabled", "is_dof_blur_far_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "dof_blur_far_distance", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_far_distance", "get_dof_blur_far_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "dof_blur_far_transition", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_far_transition", "get_dof_blur_far_transition");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "dof_blur_far_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_dof_blur_far_amount", "get_dof_blur_far_amount");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dof_blur_far_quality", PROPERTY_HINT_ENUM, "Low,Medium,High"), "set_dof_blur_far_quality", "get_dof_blur_far_quality");

	ADD_GROUP("DOF Far Near", "dof_blur_near_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dof_blur_near_enabled"), "set_dof_blur_near_enabled", "is_dof_blur_near_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "dof_blur_near_distance", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_near_distance", "get_dof_blur_near_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "dof_blur_near_transition", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_near_transition", "get_dof_blur_near_transition");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "dof_blur_near_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_dof_blur_near_amount", "get_dof_blur_near_amount");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dof_blur_near_quality", PROPERTY_HINT_ENUM, "Low,Medium,High"), "set_dof_blur_near_quality", "get_dof_blur_near_quality");

	ClassDB::bind_method(D_METHOD("set_glow_enabled", "enabled"), &Environment::set_glow_enabled);
	ClassDB::bind_method(D_METHOD("is_glow_enabled"), &Environment::is_glow_enabled);

	ClassDB::bind_method(D_METHOD("set_glow_level", "idx", "enabled"), &Environment::set_glow_level);
	ClassDB::bind_method(D_METHOD("is_glow_level_enabled", "idx"), &Environment::is_glow_level_enabled);

	ClassDB::bind_method(D_METHOD("set_glow_intensity", "intensity"), &Environment::set_glow_intensity);
	ClassDB::bind_method(D_METHOD("get_glow_intensity"), &Environment::get_glow_intensity);

	ClassDB::bind_method(D_METHOD("set_glow_strength", "strength"), &Environment::set_glow_strength);
	ClassDB::bind_method(D_METHOD("get_glow_strength"), &Environment::get_glow_strength);

	ClassDB::bind_method(D_METHOD("set_glow_bloom", "amount"), &Environment::set_glow_bloom);
	ClassDB::bind_method(D_METHOD("get_glow_bloom"), &Environment::get_glow_bloom);

	ClassDB::bind_method(D_METHOD("set_glow_blend_mode", "mode"), &Environment::set_glow_blend_mode);
	ClassDB::bind_method(D_METHOD("get_glow_blend_mode"), &Environment::get_glow_blend_mode);

	ClassDB::bind_method(D_METHOD("set_glow_hdr_bleed_treshold", "treshold"), &Environment::set_glow_hdr_bleed_treshold);
	ClassDB::bind_method(D_METHOD("get_glow_hdr_bleed_treshold"), &Environment::get_glow_hdr_bleed_treshold);

	ClassDB::bind_method(D_METHOD("set_glow_hdr_bleed_scale", "scale"), &Environment::set_glow_hdr_bleed_scale);
	ClassDB::bind_method(D_METHOD("get_glow_hdr_bleed_scale"), &Environment::get_glow_hdr_bleed_scale);

	ClassDB::bind_method(D_METHOD("set_glow_bicubic_upscale", "enabled"), &Environment::set_glow_bicubic_upscale);
	ClassDB::bind_method(D_METHOD("is_glow_bicubic_upscale_enabled"), &Environment::is_glow_bicubic_upscale_enabled);

	ADD_GROUP("Glow", "glow_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "glow_enabled"), "set_glow_enabled", "is_glow_enabled");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/1"), "set_glow_level", "is_glow_level_enabled", 0);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/2"), "set_glow_level", "is_glow_level_enabled", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/3"), "set_glow_level", "is_glow_level_enabled", 2);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/4"), "set_glow_level", "is_glow_level_enabled", 3);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/5"), "set_glow_level", "is_glow_level_enabled", 4);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/6"), "set_glow_level", "is_glow_level_enabled", 5);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/7"), "set_glow_level", "is_glow_level_enabled", 6);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "glow_intensity", PROPERTY_HINT_RANGE, "0.0,8.0,0.01"), "set_glow_intensity", "get_glow_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "glow_strength", PROPERTY_HINT_RANGE, "0.0,2.0,0.01"), "set_glow_strength", "get_glow_strength");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "glow_bloom", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_glow_bloom", "get_glow_bloom");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "glow_blend_mode", PROPERTY_HINT_ENUM, "Additive,Screen,Softlight,Replace"), "set_glow_blend_mode", "get_glow_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "glow_hdr_treshold", PROPERTY_HINT_RANGE, "0.0,4.0,0.01"), "set_glow_hdr_bleed_treshold", "get_glow_hdr_bleed_treshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "glow_hdr_scale", PROPERTY_HINT_RANGE, "0.0,4.0,0.01"), "set_glow_hdr_bleed_scale", "get_glow_hdr_bleed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "glow_bicubic_upscale"), "set_glow_bicubic_upscale", "is_glow_bicubic_upscale_enabled");

	ClassDB::bind_method(D_METHOD("set_tonemapper", "mode"), &Environment::set_tonemapper);
	ClassDB::bind_method(D_METHOD("get_tonemapper"), &Environment::get_tonemapper);

	ClassDB::bind_method(D_METHOD("set_tonemap_exposure", "exposure"), &Environment::set_tonemap_exposure);
	ClassDB::bind_method(D_METHOD("get_tonemap_exposure"), &Environment::get_tonemap_exposure);

	ClassDB::bind_method(D_METHOD("set_tonemap_white", "white"), &Environment::set_tonemap_white);
	ClassDB::bind_method(D_METHOD("get_tonemap_white"), &Environment::get_tonemap_white);

	ClassDB::bind_method(D_METHOD("set_tonemap_auto_exposure", "auto_exposure"), &Environment::set_tonemap_auto_exposure);
	ClassDB::bind_method(D_METHOD("get_tonemap_auto_exposure"), &Environment::get_tonemap_auto_exposure);

	ClassDB::bind_method(D_METHOD("set_tonemap_auto_exposure_max", "exposure_max"), &Environment::set_tonemap_auto_exposure_max);
	ClassDB::bind_method(D_METHOD("get_tonemap_auto_exposure_max"), &Environment::get_tonemap_auto_exposure_max);

	ClassDB::bind_method(D_METHOD("set_tonemap_auto_exposure_min", "exposure_min"), &Environment::set_tonemap_auto_exposure_min);
	ClassDB::bind_method(D_METHOD("get_tonemap_auto_exposure_min"), &Environment::get_tonemap_auto_exposure_min);

	ClassDB::bind_method(D_METHOD("set_tonemap_auto_exposure_speed", "exposure_speed"), &Environment::set_tonemap_auto_exposure_speed);
	ClassDB::bind_method(D_METHOD("get_tonemap_auto_exposure_speed"), &Environment::get_tonemap_auto_exposure_speed);

	ClassDB::bind_method(D_METHOD("set_tonemap_auto_exposure_grey", "exposure_grey"), &Environment::set_tonemap_auto_exposure_grey);
	ClassDB::bind_method(D_METHOD("get_tonemap_auto_exposure_grey"), &Environment::get_tonemap_auto_exposure_grey);

	ADD_GROUP("Tonemap", "tonemap_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tonemap_mode", PROPERTY_HINT_ENUM, "Linear,Reindhart,Filmic,Aces"), "set_tonemapper", "get_tonemapper");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "tonemap_exposure", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_exposure", "get_tonemap_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "tonemap_white", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_white", "get_tonemap_white");
	ADD_GROUP("Auto Exposure", "auto_exposure_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_expoure_enabled"), "set_tonemap_auto_exposure", "get_tonemap_auto_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "auto_expoure_scale", PROPERTY_HINT_RANGE, "0.01,64,0.01"), "set_tonemap_auto_exposure_grey", "get_tonemap_auto_exposure_grey");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "auto_expoure_min_luma", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_auto_exposure_min", "get_tonemap_auto_exposure_min");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "auto_expoure_max_luma", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_auto_exposure_max", "get_tonemap_auto_exposure_max");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "auto_expoure_speed", PROPERTY_HINT_RANGE, "0.01,64,0.01"), "set_tonemap_auto_exposure_speed", "get_tonemap_auto_exposure_speed");

	ClassDB::bind_method(D_METHOD("set_adjustment_enable", "enabled"), &Environment::set_adjustment_enable);
	ClassDB::bind_method(D_METHOD("is_adjustment_enabled"), &Environment::is_adjustment_enabled);

	ClassDB::bind_method(D_METHOD("set_adjustment_brightness", "brightness"), &Environment::set_adjustment_brightness);
	ClassDB::bind_method(D_METHOD("get_adjustment_brightness"), &Environment::get_adjustment_brightness);

	ClassDB::bind_method(D_METHOD("set_adjustment_contrast", "contrast"), &Environment::set_adjustment_contrast);
	ClassDB::bind_method(D_METHOD("get_adjustment_contrast"), &Environment::get_adjustment_contrast);

	ClassDB::bind_method(D_METHOD("set_adjustment_saturation", "saturation"), &Environment::set_adjustment_saturation);
	ClassDB::bind_method(D_METHOD("get_adjustment_saturation"), &Environment::get_adjustment_saturation);

	ClassDB::bind_method(D_METHOD("set_adjustment_color_correction", "color_correction"), &Environment::set_adjustment_color_correction);
	ClassDB::bind_method(D_METHOD("get_adjustment_color_correction"), &Environment::get_adjustment_color_correction);

	ADD_GROUP("Adjustments", "adjustment_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "adjustment_enabled"), "set_adjustment_enable", "is_adjustment_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "adjustment_brightness", PROPERTY_HINT_RANGE, "0.01,8,0.01"), "set_adjustment_brightness", "get_adjustment_brightness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "adjustment_contrast", PROPERTY_HINT_RANGE, "0.01,8,0.01"), "set_adjustment_contrast", "get_adjustment_contrast");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "adjustment_saturation", PROPERTY_HINT_RANGE, "0.01,8,0.01"), "set_adjustment_saturation", "get_adjustment_saturation");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "adjustment_color_correction", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_adjustment_color_correction", "get_adjustment_color_correction");

	GLOBAL_DEF("rendering/skybox/irradiance_cube_resolution", 256);

	BIND_CONSTANT(BG_KEEP);
	BIND_CONSTANT(BG_CLEAR_COLOR);
	BIND_CONSTANT(BG_COLOR);
	BIND_CONSTANT(BG_SKYBOX);
	BIND_CONSTANT(BG_CANVAS);
	BIND_CONSTANT(BG_MAX);
	BIND_CONSTANT(GLOW_BLEND_MODE_ADDITIVE);
	BIND_CONSTANT(GLOW_BLEND_MODE_SCREEN);
	BIND_CONSTANT(GLOW_BLEND_MODE_SOFTLIGHT);
	BIND_CONSTANT(GLOW_BLEND_MODE_REPLACE);
	BIND_CONSTANT(TONE_MAPPER_LINEAR);
	BIND_CONSTANT(TONE_MAPPER_REINHARDT);
	BIND_CONSTANT(TONE_MAPPER_FILMIC);
	BIND_CONSTANT(TONE_MAPPER_ACES);
	BIND_CONSTANT(DOF_BLUR_QUALITY_LOW);
	BIND_CONSTANT(DOF_BLUR_QUALITY_MEDIUM);
	BIND_CONSTANT(DOF_BLUR_QUALITY_HIGH);
}

Environment::Environment() {

	bg_mode = BG_CLEAR_COLOR;
	bg_skybox_scale = 1.0;
	bg_energy = 1.0;
	bg_canvas_max_layer = 0;
	ambient_energy = 1.0;
	ambient_skybox_contribution = 0;

	tone_mapper = TONE_MAPPER_LINEAR;
	tonemap_exposure = 1.0;
	tonemap_white = 1.0;
	tonemap_auto_exposure = false;
	tonemap_auto_exposure_max = 8;
	tonemap_auto_exposure_min = 0.05;
	tonemap_auto_exposure_speed = 0.5;
	tonemap_auto_exposure_grey = 0.4;

	set_tonemapper(tone_mapper); //update

	adjustment_enabled = false;
	adjustment_contrast = 1.0;
	adjustment_saturation = 1.0;
	adjustment_brightness = 1.0;

	set_adjustment_enable(adjustment_enabled); //update

	environment = VS::get_singleton()->environment_create();

	ssr_enabled = false;
	ssr_max_steps = 64;
	ssr_accel = 0.04;
	ssr_fade = 2.0;
	ssr_depth_tolerance = 0.2;
	ssr_smooth = true;
	ssr_roughness = true;

	ssao_enabled = false;
	ssao_radius = 1;
	ssao_intensity = 1;
	ssao_radius2 = 0;
	ssao_intensity2 = 1;
	ssao_bias = 0.01;
	ssao_direct_light_affect = false;
	ssao_blur = true;

	glow_enabled = false;
	glow_levels = (1 << 2) | (1 << 4);
	glow_intensity = 0.8;
	glow_strength = 1.0;
	glow_bloom = 0.0;
	glow_blend_mode = GLOW_BLEND_MODE_SOFTLIGHT;
	glow_hdr_bleed_treshold = 1.0;
	glow_hdr_bleed_scale = 2.0;
	glow_bicubic_upscale = false;

	dof_blur_far_enabled = false;
	dof_blur_far_distance = 10;
	dof_blur_far_transition = 5;
	dof_blur_far_amount = 0.1;
	dof_blur_far_quality = DOF_BLUR_QUALITY_MEDIUM;

	dof_blur_near_enabled = false;
	dof_blur_near_distance = 2;
	dof_blur_near_transition = 1;
	dof_blur_near_amount = 0.1;
	dof_blur_near_quality = DOF_BLUR_QUALITY_MEDIUM;
}

Environment::~Environment() {

	VS::get_singleton()->free(environment);
}

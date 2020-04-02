/*************************************************************************/
/*  environment.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/project_settings.h"
#include "servers/rendering_server.h"
#include "texture.h"

RID Environment::get_rid() const {

	return environment;
}

void Environment::set_background(BGMode p_bg) {

	bg_mode = p_bg;
	RS::get_singleton()->environment_set_background(environment, RS::EnvironmentBG(p_bg));
	_change_notify();
}

void Environment::set_sky(const Ref<Sky> &p_sky) {

	bg_sky = p_sky;

	RID sb_rid;
	if (bg_sky.is_valid())
		sb_rid = bg_sky->get_rid();

	RS::get_singleton()->environment_set_sky(environment, sb_rid);
}

void Environment::set_sky_custom_fov(float p_scale) {

	bg_sky_custom_fov = p_scale;
	RS::get_singleton()->environment_set_sky_custom_fov(environment, p_scale);
}
void Environment::set_bg_color(const Color &p_color) {

	bg_color = p_color;
	RS::get_singleton()->environment_set_bg_color(environment, p_color);
}
void Environment::set_bg_energy(float p_energy) {

	bg_energy = p_energy;
	RS::get_singleton()->environment_set_bg_energy(environment, p_energy);
}
void Environment::set_canvas_max_layer(int p_max_layer) {

	bg_canvas_max_layer = p_max_layer;
	RS::get_singleton()->environment_set_canvas_max_layer(environment, p_max_layer);
}
void Environment::set_ambient_light_color(const Color &p_color) {

	ambient_color = p_color;
	RS::get_singleton()->environment_set_ambient_light(environment, ambient_color, RS::EnvironmentAmbientSource(ambient_source), ambient_energy, ambient_sky_contribution, RS::EnvironmentReflectionSource(reflection_source), ao_color);
}
void Environment::set_ambient_light_energy(float p_energy) {

	ambient_energy = p_energy;
	RS::get_singleton()->environment_set_ambient_light(environment, ambient_color, RS::EnvironmentAmbientSource(ambient_source), ambient_energy, ambient_sky_contribution, RS::EnvironmentReflectionSource(reflection_source), ao_color);
}
void Environment::set_ambient_light_sky_contribution(float p_energy) {

	ambient_sky_contribution = p_energy;
	RS::get_singleton()->environment_set_ambient_light(environment, ambient_color, RS::EnvironmentAmbientSource(ambient_source), ambient_energy, ambient_sky_contribution, RS::EnvironmentReflectionSource(reflection_source), ao_color);
}

void Environment::set_camera_feed_id(int p_camera_feed_id) {
	camera_feed_id = p_camera_feed_id;
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	RS::get_singleton()->environment_set_camera_feed_id(environment, camera_feed_id);
#endif
};

void Environment::set_ambient_source(AmbientSource p_source) {
	ambient_source = p_source;
	RS::get_singleton()->environment_set_ambient_light(environment, ambient_color, RS::EnvironmentAmbientSource(ambient_source), ambient_energy, ambient_sky_contribution, RS::EnvironmentReflectionSource(reflection_source), ao_color);
}

Environment::AmbientSource Environment::get_ambient_source() const {
	return ambient_source;
}
void Environment::set_reflection_source(ReflectionSource p_source) {
	reflection_source = p_source;
	RS::get_singleton()->environment_set_ambient_light(environment, ambient_color, RS::EnvironmentAmbientSource(ambient_source), ambient_energy, ambient_sky_contribution, RS::EnvironmentReflectionSource(reflection_source), ao_color);
}
Environment::ReflectionSource Environment::get_reflection_source() const {
	return reflection_source;
}

Environment::BGMode Environment::get_background() const {

	return bg_mode;
}
Ref<Sky> Environment::get_sky() const {

	return bg_sky;
}

float Environment::get_sky_custom_fov() const {

	return bg_sky_custom_fov;
}

void Environment::set_sky_rotation(const Vector3 &p_rotation) {
	sky_rotation = p_rotation;
	RS::get_singleton()->environment_set_sky_orientation(environment, Basis(p_rotation));
}

Vector3 Environment::get_sky_rotation() const {

	return sky_rotation;
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
float Environment::get_ambient_light_sky_contribution() const {

	return ambient_sky_contribution;
}
int Environment::get_camera_feed_id(void) const {

	return camera_feed_id;
}

void Environment::set_tonemapper(ToneMapper p_tone_mapper) {

	tone_mapper = p_tone_mapper;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}

Environment::ToneMapper Environment::get_tonemapper() const {

	return tone_mapper;
}

void Environment::set_tonemap_exposure(float p_exposure) {

	tonemap_exposure = p_exposure;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}

float Environment::get_tonemap_exposure() const {

	return tonemap_exposure;
}

void Environment::set_tonemap_white(float p_white) {

	tonemap_white = p_white;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_white() const {

	return tonemap_white;
}

void Environment::set_tonemap_auto_exposure(bool p_enabled) {

	tonemap_auto_exposure = p_enabled;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
	_change_notify();
}
bool Environment::get_tonemap_auto_exposure() const {

	return tonemap_auto_exposure;
}

void Environment::set_tonemap_auto_exposure_max(float p_auto_exposure_max) {

	tonemap_auto_exposure_max = p_auto_exposure_max;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_max() const {

	return tonemap_auto_exposure_max;
}

void Environment::set_tonemap_auto_exposure_min(float p_auto_exposure_min) {

	tonemap_auto_exposure_min = p_auto_exposure_min;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_min() const {

	return tonemap_auto_exposure_min;
}

void Environment::set_tonemap_auto_exposure_speed(float p_auto_exposure_speed) {

	tonemap_auto_exposure_speed = p_auto_exposure_speed;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_speed() const {

	return tonemap_auto_exposure_speed;
}

void Environment::set_tonemap_auto_exposure_grey(float p_auto_exposure_grey) {

	tonemap_auto_exposure_grey = p_auto_exposure_grey;
	RS::get_singleton()->environment_set_tonemap(environment, RS::EnvironmentToneMapper(tone_mapper), tonemap_exposure, tonemap_white, tonemap_auto_exposure, tonemap_auto_exposure_min, tonemap_auto_exposure_max, tonemap_auto_exposure_speed, tonemap_auto_exposure_grey);
}
float Environment::get_tonemap_auto_exposure_grey() const {

	return tonemap_auto_exposure_grey;
}

void Environment::set_adjustment_enable(bool p_enable) {

	adjustment_enabled = p_enable;
	RS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
	_change_notify();
}

bool Environment::is_adjustment_enabled() const {

	return adjustment_enabled;
}

void Environment::set_adjustment_brightness(float p_brightness) {

	adjustment_brightness = p_brightness;
	RS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
float Environment::get_adjustment_brightness() const {

	return adjustment_brightness;
}

void Environment::set_adjustment_contrast(float p_contrast) {

	adjustment_contrast = p_contrast;
	RS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
float Environment::get_adjustment_contrast() const {

	return adjustment_contrast;
}

void Environment::set_adjustment_saturation(float p_saturation) {

	adjustment_saturation = p_saturation;
	RS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
float Environment::get_adjustment_saturation() const {

	return adjustment_saturation;
}

void Environment::set_adjustment_color_correction(const Ref<Texture2D> &p_ramp) {

	adjustment_color_correction = p_ramp;
	RS::get_singleton()->environment_set_adjustment(environment, adjustment_enabled, adjustment_brightness, adjustment_contrast, adjustment_saturation, adjustment_color_correction.is_valid() ? adjustment_color_correction->get_rid() : RID());
}
Ref<Texture2D> Environment::get_adjustment_color_correction() const {

	return adjustment_color_correction;
}

void Environment::_validate_property(PropertyInfo &property) const {

	if (property.name == "sky" || property.name == "sky_custom_fov" || property.name == "sky_rotation" || property.name == "ambient_light/sky_contribution") {
		if (bg_mode != BG_SKY && ambient_source != AMBIENT_SOURCE_SKY && reflection_source != REFLECTION_SOURCE_SKY) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "glow_intensity" && glow_blend_mode == GLOW_BLEND_MODE_MIX) {
		property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
	}

	if (property.name == "glow_mix" && glow_blend_mode != GLOW_BLEND_MODE_MIX) {
		property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
	}

	if (property.name == "background_color") {
		if (bg_mode != BG_COLOR && ambient_source != AMBIENT_SOURCE_COLOR) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "background_canvas_max_layer") {
		if (bg_mode != BG_CANVAS) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "background_camera_feed_id") {
		if (bg_mode != BG_CAMERA_FEED) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}

	static const char *hide_prefixes[] = {
		"fog_",
		"auto_exposure_",
		"ss_reflections_",
		"ssao_",
		"glow_",
		"adjustment_",
		nullptr

	};

	static const char *high_end_prefixes[] = {
		"auto_exposure_",
		"tonemap_",
		"ss_reflections_",
		"ssao_",
		nullptr

	};

	const char **prefixes = hide_prefixes;
	while (*prefixes) {
		String prefix = String(*prefixes);

		String enabled = prefix + "enabled";
		if (property.name.begins_with(prefix) && property.name != enabled && !bool(get(enabled))) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
			return;
		}

		prefixes++;
	}

	if (RenderingServer::get_singleton()->is_low_end()) {
		prefixes = high_end_prefixes;
		while (*prefixes) {
			String prefix = String(*prefixes);

			if (property.name.begins_with(prefix)) {
				property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
				return;
			}

			prefixes++;
		}
	}
}

void Environment::set_ssr_enabled(bool p_enable) {

	ssr_enabled = p_enable;
	RS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_fade_in, ssr_fade_out, ssr_depth_tolerance);
	_change_notify();
}

bool Environment::is_ssr_enabled() const {

	return ssr_enabled;
}

void Environment::set_ssr_max_steps(int p_steps) {

	ssr_max_steps = p_steps;
	RS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_fade_in, ssr_fade_out, ssr_depth_tolerance);
}
int Environment::get_ssr_max_steps() const {

	return ssr_max_steps;
}

void Environment::set_ssr_fade_in(float p_fade_in) {

	ssr_fade_in = p_fade_in;
	RS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_fade_in, ssr_fade_out, ssr_depth_tolerance);
}
float Environment::get_ssr_fade_in() const {

	return ssr_fade_in;
}

void Environment::set_ssr_fade_out(float p_fade_out) {

	ssr_fade_out = p_fade_out;
	RS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_fade_in, ssr_fade_out, ssr_depth_tolerance);
}
float Environment::get_ssr_fade_out() const {

	return ssr_fade_out;
}

void Environment::set_ssr_depth_tolerance(float p_depth_tolerance) {

	ssr_depth_tolerance = p_depth_tolerance;
	RS::get_singleton()->environment_set_ssr(environment, ssr_enabled, ssr_max_steps, ssr_fade_in, ssr_fade_out, ssr_depth_tolerance);
}
float Environment::get_ssr_depth_tolerance() const {

	return ssr_depth_tolerance;
}

void Environment::set_ssao_enabled(bool p_enable) {

	ssao_enabled = p_enable;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
	_change_notify();
}

bool Environment::is_ssao_enabled() const {

	return ssao_enabled;
}

void Environment::set_ssao_radius(float p_radius) {

	ssao_radius = p_radius;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}
float Environment::get_ssao_radius() const {

	return ssao_radius;
}

void Environment::set_ssao_intensity(float p_intensity) {

	ssao_intensity = p_intensity;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}

float Environment::get_ssao_intensity() const {

	return ssao_intensity;
}

void Environment::set_ssao_bias(float p_bias) {

	ssao_bias = p_bias;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}
float Environment::get_ssao_bias() const {

	return ssao_bias;
}

void Environment::set_ssao_direct_light_affect(float p_direct_light_affect) {

	ssao_direct_light_affect = p_direct_light_affect;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}
float Environment::get_ssao_direct_light_affect() const {

	return ssao_direct_light_affect;
}

void Environment::set_ssao_ao_channel_affect(float p_ao_channel_affect) {

	ssao_ao_channel_affect = p_ao_channel_affect;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}
float Environment::get_ssao_ao_channel_affect() const {

	return ssao_ao_channel_affect;
}

void Environment::set_ao_color(const Color &p_color) {

	ao_color = p_color;
	RS::get_singleton()->environment_set_ambient_light(environment, ambient_color, RS::EnvironmentAmbientSource(ambient_source), ambient_energy, ambient_sky_contribution, RS::EnvironmentReflectionSource(reflection_source), ao_color);
}

Color Environment::get_ao_color() const {

	return ao_color;
}

void Environment::set_ssao_blur(SSAOBlur p_blur) {

	ssao_blur = p_blur;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}
Environment::SSAOBlur Environment::get_ssao_blur() const {

	return ssao_blur;
}

void Environment::set_ssao_edge_sharpness(float p_edge_sharpness) {

	ssao_edge_sharpness = p_edge_sharpness;
	RS::get_singleton()->environment_set_ssao(environment, ssao_enabled, ssao_radius, ssao_intensity, ssao_bias, ssao_direct_light_affect, ssao_ao_channel_affect, RS::EnvironmentSSAOBlur(ssao_blur), ssao_edge_sharpness);
}

float Environment::get_ssao_edge_sharpness() const {

	return ssao_edge_sharpness;
}

void Environment::set_glow_enabled(bool p_enabled) {

	glow_enabled = p_enabled;
	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
	_change_notify();
}

bool Environment::is_glow_enabled() const {

	return glow_enabled;
}

void Environment::set_glow_level(int p_level, bool p_enabled) {

	ERR_FAIL_INDEX(p_level, RS::MAX_GLOW_LEVELS);

	if (p_enabled)
		glow_levels |= (1 << p_level);
	else
		glow_levels &= ~(1 << p_level);

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
bool Environment::is_glow_level_enabled(int p_level) const {

	ERR_FAIL_INDEX_V(p_level, RS::MAX_GLOW_LEVELS, false);

	return glow_levels & (1 << p_level);
}

void Environment::set_glow_intensity(float p_intensity) {

	glow_intensity = p_intensity;

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_intensity() const {

	return glow_intensity;
}

void Environment::set_glow_strength(float p_strength) {

	glow_strength = p_strength;
	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_strength() const {

	return glow_strength;
}

void Environment::set_glow_mix(float p_mix) {

	glow_mix = p_mix;
	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_mix() const {

	return glow_mix;
}

void Environment::set_glow_bloom(float p_threshold) {

	glow_bloom = p_threshold;

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_bloom() const {

	return glow_bloom;
}

void Environment::set_glow_blend_mode(GlowBlendMode p_mode) {

	glow_blend_mode = p_mode;

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
	_change_notify();
}
Environment::GlowBlendMode Environment::get_glow_blend_mode() const {

	return glow_blend_mode;
}

void Environment::set_glow_hdr_bleed_threshold(float p_threshold) {

	glow_hdr_bleed_threshold = p_threshold;

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_hdr_bleed_threshold() const {

	return glow_hdr_bleed_threshold;
}

void Environment::set_glow_hdr_luminance_cap(float p_amount) {

	glow_hdr_luminance_cap = p_amount;

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_hdr_luminance_cap() const {

	return glow_hdr_luminance_cap;
}

void Environment::set_glow_hdr_bleed_scale(float p_scale) {

	glow_hdr_bleed_scale = p_scale;

	RS::get_singleton()->environment_set_glow(environment, glow_enabled, glow_levels, glow_intensity, glow_strength, glow_mix, glow_bloom, RS::EnvironmentGlowBlendMode(glow_blend_mode), glow_hdr_bleed_threshold, glow_hdr_bleed_threshold, glow_hdr_luminance_cap);
}
float Environment::get_glow_hdr_bleed_scale() const {

	return glow_hdr_bleed_scale;
}

void Environment::set_fog_enabled(bool p_enabled) {

	fog_enabled = p_enabled;
	RS::get_singleton()->environment_set_fog(environment, fog_enabled, fog_color, fog_sun_color, fog_sun_amount);
	_change_notify();
}

bool Environment::is_fog_enabled() const {

	return fog_enabled;
}

void Environment::set_fog_color(const Color &p_color) {

	fog_color = p_color;
	RS::get_singleton()->environment_set_fog(environment, fog_enabled, fog_color, fog_sun_color, fog_sun_amount);
}
Color Environment::get_fog_color() const {

	return fog_color;
}

void Environment::set_fog_sun_color(const Color &p_color) {

	fog_sun_color = p_color;
	RS::get_singleton()->environment_set_fog(environment, fog_enabled, fog_color, fog_sun_color, fog_sun_amount);
}
Color Environment::get_fog_sun_color() const {

	return fog_sun_color;
}

void Environment::set_fog_sun_amount(float p_amount) {

	fog_sun_amount = p_amount;
	RS::get_singleton()->environment_set_fog(environment, fog_enabled, fog_color, fog_sun_color, fog_sun_amount);
}
float Environment::get_fog_sun_amount() const {

	return fog_sun_amount;
}

void Environment::set_fog_depth_enabled(bool p_enabled) {

	fog_depth_enabled = p_enabled;
	RS::get_singleton()->environment_set_fog_depth(environment, fog_depth_enabled, fog_depth_begin, fog_depth_end, fog_depth_curve, fog_transmit_enabled, fog_transmit_curve);
}
bool Environment::is_fog_depth_enabled() const {

	return fog_depth_enabled;
}

void Environment::set_fog_depth_begin(float p_distance) {

	fog_depth_begin = p_distance;
	RS::get_singleton()->environment_set_fog_depth(environment, fog_depth_enabled, fog_depth_begin, fog_depth_end, fog_depth_curve, fog_transmit_enabled, fog_transmit_curve);
}
float Environment::get_fog_depth_begin() const {

	return fog_depth_begin;
}

void Environment::set_fog_depth_end(float p_distance) {

	fog_depth_end = p_distance;
	RS::get_singleton()->environment_set_fog_depth(environment, fog_depth_enabled, fog_depth_begin, fog_depth_end, fog_depth_curve, fog_transmit_enabled, fog_transmit_curve);
}

float Environment::get_fog_depth_end() const {

	return fog_depth_end;
}

void Environment::set_fog_depth_curve(float p_curve) {

	fog_depth_curve = p_curve;
	RS::get_singleton()->environment_set_fog_depth(environment, fog_depth_enabled, fog_depth_begin, fog_depth_end, fog_depth_curve, fog_transmit_enabled, fog_transmit_curve);
}
float Environment::get_fog_depth_curve() const {

	return fog_depth_curve;
}

void Environment::set_fog_transmit_enabled(bool p_enabled) {

	fog_transmit_enabled = p_enabled;
	RS::get_singleton()->environment_set_fog_depth(environment, fog_depth_enabled, fog_depth_begin, fog_depth_end, fog_depth_curve, fog_transmit_enabled, fog_transmit_curve);
}
bool Environment::is_fog_transmit_enabled() const {

	return fog_transmit_enabled;
}

void Environment::set_fog_transmit_curve(float p_curve) {

	fog_transmit_curve = p_curve;
	RS::get_singleton()->environment_set_fog_depth(environment, fog_depth_enabled, fog_depth_begin, fog_depth_end, fog_depth_curve, fog_transmit_enabled, fog_transmit_curve);
}
float Environment::get_fog_transmit_curve() const {

	return fog_transmit_curve;
}

void Environment::set_fog_height_enabled(bool p_enabled) {

	fog_height_enabled = p_enabled;
	RS::get_singleton()->environment_set_fog_height(environment, fog_height_enabled, fog_height_min, fog_height_max, fog_height_curve);
}
bool Environment::is_fog_height_enabled() const {

	return fog_height_enabled;
}

void Environment::set_fog_height_min(float p_distance) {

	fog_height_min = p_distance;
	RS::get_singleton()->environment_set_fog_height(environment, fog_height_enabled, fog_height_min, fog_height_max, fog_height_curve);
}
float Environment::get_fog_height_min() const {

	return fog_height_min;
}

void Environment::set_fog_height_max(float p_distance) {

	fog_height_max = p_distance;
	RS::get_singleton()->environment_set_fog_height(environment, fog_height_enabled, fog_height_min, fog_height_max, fog_height_curve);
}
float Environment::get_fog_height_max() const {

	return fog_height_max;
}

void Environment::set_fog_height_curve(float p_distance) {

	fog_height_curve = p_distance;
	RS::get_singleton()->environment_set_fog_height(environment, fog_height_enabled, fog_height_min, fog_height_max, fog_height_curve);
}
float Environment::get_fog_height_curve() const {

	return fog_height_curve;
}

#ifndef DISABLE_DEPRECATED
// Kept for compatibility from 3.x to 4.0.
bool Environment::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "background_sky") {
		set_sky(p_value);
		return true;
	} else if (p_name == "background_sky_custom_fov") {
		set_sky_custom_fov(p_value);
		return true;
	} else if (p_name == "background_sky_orientation") {
		Vector3 euler = p_value.operator Basis().get_euler();
		set_sky_rotation(euler);
		return true;
	} else {
		return false;
	}
}
#endif

void Environment::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_background", "mode"), &Environment::set_background);
	ClassDB::bind_method(D_METHOD("set_sky", "sky"), &Environment::set_sky);
	ClassDB::bind_method(D_METHOD("set_sky_custom_fov", "scale"), &Environment::set_sky_custom_fov);
	ClassDB::bind_method(D_METHOD("set_sky_rotation", "euler_radians"), &Environment::set_sky_rotation);
	ClassDB::bind_method(D_METHOD("set_bg_color", "color"), &Environment::set_bg_color);
	ClassDB::bind_method(D_METHOD("set_bg_energy", "energy"), &Environment::set_bg_energy);
	ClassDB::bind_method(D_METHOD("set_canvas_max_layer", "layer"), &Environment::set_canvas_max_layer);
	ClassDB::bind_method(D_METHOD("set_ambient_light_color", "color"), &Environment::set_ambient_light_color);
	ClassDB::bind_method(D_METHOD("set_ambient_light_energy", "energy"), &Environment::set_ambient_light_energy);
	ClassDB::bind_method(D_METHOD("set_ambient_light_sky_contribution", "energy"), &Environment::set_ambient_light_sky_contribution);
	ClassDB::bind_method(D_METHOD("set_camera_feed_id", "camera_feed_id"), &Environment::set_camera_feed_id);
	ClassDB::bind_method(D_METHOD("set_ambient_source", "source"), &Environment::set_ambient_source);
	ClassDB::bind_method(D_METHOD("set_reflection_source", "source"), &Environment::set_reflection_source);

	ClassDB::bind_method(D_METHOD("get_background"), &Environment::get_background);
	ClassDB::bind_method(D_METHOD("get_sky"), &Environment::get_sky);
	ClassDB::bind_method(D_METHOD("get_sky_custom_fov"), &Environment::get_sky_custom_fov);
	ClassDB::bind_method(D_METHOD("get_sky_rotation"), &Environment::get_sky_rotation);
	ClassDB::bind_method(D_METHOD("get_bg_color"), &Environment::get_bg_color);
	ClassDB::bind_method(D_METHOD("get_bg_energy"), &Environment::get_bg_energy);
	ClassDB::bind_method(D_METHOD("get_canvas_max_layer"), &Environment::get_canvas_max_layer);
	ClassDB::bind_method(D_METHOD("get_ambient_light_color"), &Environment::get_ambient_light_color);
	ClassDB::bind_method(D_METHOD("get_ambient_light_energy"), &Environment::get_ambient_light_energy);
	ClassDB::bind_method(D_METHOD("get_ambient_light_sky_contribution"), &Environment::get_ambient_light_sky_contribution);
	ClassDB::bind_method(D_METHOD("get_camera_feed_id"), &Environment::get_camera_feed_id);
	ClassDB::bind_method(D_METHOD("get_ambient_source"), &Environment::get_ambient_source);
	ClassDB::bind_method(D_METHOD("get_reflection_source"), &Environment::get_reflection_source);
	ClassDB::bind_method(D_METHOD("set_ao_color", "color"), &Environment::set_ao_color);
	ClassDB::bind_method(D_METHOD("get_ao_color"), &Environment::get_ao_color);

	ADD_GROUP("Background", "background_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "background_mode", PROPERTY_HINT_ENUM, "Clear Color,Custom Color,Sky,Canvas,Keep,Camera Feed"), "set_background", "get_background");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "background_canvas_max_layer", PROPERTY_HINT_RANGE, "-1000,1000,1"), "set_canvas_max_layer", "get_canvas_max_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "background_camera_feed_id", PROPERTY_HINT_RANGE, "1,10,1"), "set_camera_feed_id", "get_camera_feed_id");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "background_color"), "set_bg_color", "get_bg_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "background_energy", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_bg_energy", "get_bg_energy");
	ADD_GROUP("Sky", "sky_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "sky", PROPERTY_HINT_RESOURCE_TYPE, "Sky"), "set_sky", "get_sky");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sky_custom_fov", PROPERTY_HINT_RANGE, "0,180,0.1"), "set_sky_custom_fov", "get_sky_custom_fov");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "sky_rotation"), "set_sky_rotation", "get_sky_rotation");
	ADD_GROUP("Ambient Light", "ambient_light_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ambient_light_source", PROPERTY_HINT_ENUM, "Background,Disabled,Color,Sky"), "set_ambient_source", "get_ambient_source");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_light_color"), "set_ambient_light_color", "get_ambient_light_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ambient_light_sky_contribution", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_ambient_light_sky_contribution", "get_ambient_light_sky_contribution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ambient_light_energy", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_ambient_light_energy", "get_ambient_light_energy");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_light_occlusion_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ao_color", "get_ao_color");
	ADD_GROUP("Reflected Light", "reflected_light_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reflected_light_source", PROPERTY_HINT_ENUM, "Background,Disabled,Sky"), "set_reflection_source", "get_reflection_source");

	ClassDB::bind_method(D_METHOD("set_fog_enabled", "enabled"), &Environment::set_fog_enabled);
	ClassDB::bind_method(D_METHOD("is_fog_enabled"), &Environment::is_fog_enabled);

	ClassDB::bind_method(D_METHOD("set_fog_color", "color"), &Environment::set_fog_color);
	ClassDB::bind_method(D_METHOD("get_fog_color"), &Environment::get_fog_color);

	ClassDB::bind_method(D_METHOD("set_fog_sun_color", "color"), &Environment::set_fog_sun_color);
	ClassDB::bind_method(D_METHOD("get_fog_sun_color"), &Environment::get_fog_sun_color);

	ClassDB::bind_method(D_METHOD("set_fog_sun_amount", "amount"), &Environment::set_fog_sun_amount);
	ClassDB::bind_method(D_METHOD("get_fog_sun_amount"), &Environment::get_fog_sun_amount);

	ClassDB::bind_method(D_METHOD("set_fog_depth_enabled", "enabled"), &Environment::set_fog_depth_enabled);
	ClassDB::bind_method(D_METHOD("is_fog_depth_enabled"), &Environment::is_fog_depth_enabled);

	ClassDB::bind_method(D_METHOD("set_fog_depth_begin", "distance"), &Environment::set_fog_depth_begin);
	ClassDB::bind_method(D_METHOD("get_fog_depth_begin"), &Environment::get_fog_depth_begin);

	ClassDB::bind_method(D_METHOD("set_fog_depth_end", "distance"), &Environment::set_fog_depth_end);
	ClassDB::bind_method(D_METHOD("get_fog_depth_end"), &Environment::get_fog_depth_end);

	ClassDB::bind_method(D_METHOD("set_fog_depth_curve", "curve"), &Environment::set_fog_depth_curve);
	ClassDB::bind_method(D_METHOD("get_fog_depth_curve"), &Environment::get_fog_depth_curve);

	ClassDB::bind_method(D_METHOD("set_fog_transmit_enabled", "enabled"), &Environment::set_fog_transmit_enabled);
	ClassDB::bind_method(D_METHOD("is_fog_transmit_enabled"), &Environment::is_fog_transmit_enabled);

	ClassDB::bind_method(D_METHOD("set_fog_transmit_curve", "curve"), &Environment::set_fog_transmit_curve);
	ClassDB::bind_method(D_METHOD("get_fog_transmit_curve"), &Environment::get_fog_transmit_curve);

	ClassDB::bind_method(D_METHOD("set_fog_height_enabled", "enabled"), &Environment::set_fog_height_enabled);
	ClassDB::bind_method(D_METHOD("is_fog_height_enabled"), &Environment::is_fog_height_enabled);

	ClassDB::bind_method(D_METHOD("set_fog_height_min", "height"), &Environment::set_fog_height_min);
	ClassDB::bind_method(D_METHOD("get_fog_height_min"), &Environment::get_fog_height_min);

	ClassDB::bind_method(D_METHOD("set_fog_height_max", "height"), &Environment::set_fog_height_max);
	ClassDB::bind_method(D_METHOD("get_fog_height_max"), &Environment::get_fog_height_max);

	ClassDB::bind_method(D_METHOD("set_fog_height_curve", "curve"), &Environment::set_fog_height_curve);
	ClassDB::bind_method(D_METHOD("get_fog_height_curve"), &Environment::get_fog_height_curve);

	ADD_GROUP("Fog", "fog_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fog_enabled"), "set_fog_enabled", "is_fog_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "fog_color"), "set_fog_color", "get_fog_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "fog_sun_color"), "set_fog_sun_color", "get_fog_sun_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_sun_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_fog_sun_amount", "get_fog_sun_amount");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fog_depth_enabled"), "set_fog_depth_enabled", "is_fog_depth_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_depth_begin", PROPERTY_HINT_RANGE, "0,4000,0.1"), "set_fog_depth_begin", "get_fog_depth_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_depth_end", PROPERTY_HINT_RANGE, "0,4000,0.1,or_greater"), "set_fog_depth_end", "get_fog_depth_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_depth_curve", PROPERTY_HINT_EXP_EASING), "set_fog_depth_curve", "get_fog_depth_curve");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fog_transmit_enabled"), "set_fog_transmit_enabled", "is_fog_transmit_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_transmit_curve", PROPERTY_HINT_EXP_EASING), "set_fog_transmit_curve", "get_fog_transmit_curve");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fog_height_enabled"), "set_fog_height_enabled", "is_fog_height_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_height_min", PROPERTY_HINT_RANGE, "-4000,4000,0.1,or_lesser,or_greater"), "set_fog_height_min", "get_fog_height_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_height_max", PROPERTY_HINT_RANGE, "-4000,4000,0.1,or_lesser,or_greater"), "set_fog_height_max", "get_fog_height_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_height_curve", PROPERTY_HINT_EXP_EASING), "set_fog_height_curve", "get_fog_height_curve");

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
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tonemap_mode", PROPERTY_HINT_ENUM, "Linear,Reinhard,Filmic,Aces"), "set_tonemapper", "get_tonemapper");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tonemap_exposure", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_exposure", "get_tonemap_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tonemap_white", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_white", "get_tonemap_white");
	ADD_GROUP("Auto Exposure", "auto_exposure_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_exposure_enabled"), "set_tonemap_auto_exposure", "get_tonemap_auto_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "auto_exposure_scale", PROPERTY_HINT_RANGE, "0.01,64,0.01"), "set_tonemap_auto_exposure_grey", "get_tonemap_auto_exposure_grey");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "auto_exposure_min_luma", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_auto_exposure_min", "get_tonemap_auto_exposure_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "auto_exposure_max_luma", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_tonemap_auto_exposure_max", "get_tonemap_auto_exposure_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "auto_exposure_speed", PROPERTY_HINT_RANGE, "0.01,64,0.01"), "set_tonemap_auto_exposure_speed", "get_tonemap_auto_exposure_speed");

	ClassDB::bind_method(D_METHOD("set_ssr_enabled", "enabled"), &Environment::set_ssr_enabled);
	ClassDB::bind_method(D_METHOD("is_ssr_enabled"), &Environment::is_ssr_enabled);

	ClassDB::bind_method(D_METHOD("set_ssr_max_steps", "max_steps"), &Environment::set_ssr_max_steps);
	ClassDB::bind_method(D_METHOD("get_ssr_max_steps"), &Environment::get_ssr_max_steps);

	ClassDB::bind_method(D_METHOD("set_ssr_fade_in", "fade_in"), &Environment::set_ssr_fade_in);
	ClassDB::bind_method(D_METHOD("get_ssr_fade_in"), &Environment::get_ssr_fade_in);

	ClassDB::bind_method(D_METHOD("set_ssr_fade_out", "fade_out"), &Environment::set_ssr_fade_out);
	ClassDB::bind_method(D_METHOD("get_ssr_fade_out"), &Environment::get_ssr_fade_out);

	ClassDB::bind_method(D_METHOD("set_ssr_depth_tolerance", "depth_tolerance"), &Environment::set_ssr_depth_tolerance);
	ClassDB::bind_method(D_METHOD("get_ssr_depth_tolerance"), &Environment::get_ssr_depth_tolerance);

	ADD_GROUP("SS Reflections", "ss_reflections_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ss_reflections_enabled"), "set_ssr_enabled", "is_ssr_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ss_reflections_max_steps", PROPERTY_HINT_RANGE, "1,512,1"), "set_ssr_max_steps", "get_ssr_max_steps");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ss_reflections_fade_in", PROPERTY_HINT_EXP_EASING), "set_ssr_fade_in", "get_ssr_fade_in");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ss_reflections_fade_out", PROPERTY_HINT_EXP_EASING), "set_ssr_fade_out", "get_ssr_fade_out");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ss_reflections_depth_tolerance", PROPERTY_HINT_RANGE, "0.01,128,0.1"), "set_ssr_depth_tolerance", "get_ssr_depth_tolerance");

	ClassDB::bind_method(D_METHOD("set_ssao_enabled", "enabled"), &Environment::set_ssao_enabled);
	ClassDB::bind_method(D_METHOD("is_ssao_enabled"), &Environment::is_ssao_enabled);

	ClassDB::bind_method(D_METHOD("set_ssao_radius", "radius"), &Environment::set_ssao_radius);
	ClassDB::bind_method(D_METHOD("get_ssao_radius"), &Environment::get_ssao_radius);

	ClassDB::bind_method(D_METHOD("set_ssao_intensity", "intensity"), &Environment::set_ssao_intensity);
	ClassDB::bind_method(D_METHOD("get_ssao_intensity"), &Environment::get_ssao_intensity);

	ClassDB::bind_method(D_METHOD("set_ssao_bias", "bias"), &Environment::set_ssao_bias);
	ClassDB::bind_method(D_METHOD("get_ssao_bias"), &Environment::get_ssao_bias);

	ClassDB::bind_method(D_METHOD("set_ssao_direct_light_affect", "amount"), &Environment::set_ssao_direct_light_affect);
	ClassDB::bind_method(D_METHOD("get_ssao_direct_light_affect"), &Environment::get_ssao_direct_light_affect);

	ClassDB::bind_method(D_METHOD("set_ssao_ao_channel_affect", "amount"), &Environment::set_ssao_ao_channel_affect);
	ClassDB::bind_method(D_METHOD("get_ssao_ao_channel_affect"), &Environment::get_ssao_ao_channel_affect);

	ClassDB::bind_method(D_METHOD("set_ssao_blur", "mode"), &Environment::set_ssao_blur);
	ClassDB::bind_method(D_METHOD("get_ssao_blur"), &Environment::get_ssao_blur);

	ClassDB::bind_method(D_METHOD("set_ssao_edge_sharpness", "edge_sharpness"), &Environment::set_ssao_edge_sharpness);
	ClassDB::bind_method(D_METHOD("get_ssao_edge_sharpness"), &Environment::get_ssao_edge_sharpness);

	ADD_GROUP("SSAO", "ssao_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ssao_enabled"), "set_ssao_enabled", "is_ssao_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ssao_radius", PROPERTY_HINT_RANGE, "0.1,128,0.1"), "set_ssao_radius", "get_ssao_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ssao_intensity", PROPERTY_HINT_RANGE, "0.0,128,0.1"), "set_ssao_intensity", "get_ssao_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ssao_bias", PROPERTY_HINT_RANGE, "0.001,8,0.001"), "set_ssao_bias", "get_ssao_bias");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ssao_light_affect", PROPERTY_HINT_RANGE, "0.00,1,0.01"), "set_ssao_direct_light_affect", "get_ssao_direct_light_affect");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ssao_ao_channel_affect", PROPERTY_HINT_RANGE, "0.00,1,0.01"), "set_ssao_ao_channel_affect", "get_ssao_ao_channel_affect");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ssao_blur", PROPERTY_HINT_ENUM, "Disabled,1x1,2x2,3x3"), "set_ssao_blur", "get_ssao_blur");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ssao_edge_sharpness", PROPERTY_HINT_RANGE, "0,32,0.01"), "set_ssao_edge_sharpness", "get_ssao_edge_sharpness");

	ClassDB::bind_method(D_METHOD("set_glow_enabled", "enabled"), &Environment::set_glow_enabled);
	ClassDB::bind_method(D_METHOD("is_glow_enabled"), &Environment::is_glow_enabled);

	ClassDB::bind_method(D_METHOD("set_glow_level", "idx", "enabled"), &Environment::set_glow_level);
	ClassDB::bind_method(D_METHOD("is_glow_level_enabled", "idx"), &Environment::is_glow_level_enabled);

	ClassDB::bind_method(D_METHOD("set_glow_intensity", "intensity"), &Environment::set_glow_intensity);
	ClassDB::bind_method(D_METHOD("get_glow_intensity"), &Environment::get_glow_intensity);

	ClassDB::bind_method(D_METHOD("set_glow_strength", "strength"), &Environment::set_glow_strength);
	ClassDB::bind_method(D_METHOD("get_glow_strength"), &Environment::get_glow_strength);

	ClassDB::bind_method(D_METHOD("set_glow_mix", "mix"), &Environment::set_glow_mix);
	ClassDB::bind_method(D_METHOD("get_glow_mix"), &Environment::get_glow_mix);

	ClassDB::bind_method(D_METHOD("set_glow_bloom", "amount"), &Environment::set_glow_bloom);
	ClassDB::bind_method(D_METHOD("get_glow_bloom"), &Environment::get_glow_bloom);

	ClassDB::bind_method(D_METHOD("set_glow_blend_mode", "mode"), &Environment::set_glow_blend_mode);
	ClassDB::bind_method(D_METHOD("get_glow_blend_mode"), &Environment::get_glow_blend_mode);

	ClassDB::bind_method(D_METHOD("set_glow_hdr_bleed_threshold", "threshold"), &Environment::set_glow_hdr_bleed_threshold);
	ClassDB::bind_method(D_METHOD("get_glow_hdr_bleed_threshold"), &Environment::get_glow_hdr_bleed_threshold);

	ClassDB::bind_method(D_METHOD("set_glow_hdr_luminance_cap", "amount"), &Environment::set_glow_hdr_luminance_cap);
	ClassDB::bind_method(D_METHOD("get_glow_hdr_luminance_cap"), &Environment::get_glow_hdr_luminance_cap);

	ClassDB::bind_method(D_METHOD("set_glow_hdr_bleed_scale", "scale"), &Environment::set_glow_hdr_bleed_scale);
	ClassDB::bind_method(D_METHOD("get_glow_hdr_bleed_scale"), &Environment::get_glow_hdr_bleed_scale);

	ADD_GROUP("Glow", "glow_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "glow_enabled"), "set_glow_enabled", "is_glow_enabled");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/1"), "set_glow_level", "is_glow_level_enabled", 0);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/2"), "set_glow_level", "is_glow_level_enabled", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/3"), "set_glow_level", "is_glow_level_enabled", 2);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/4"), "set_glow_level", "is_glow_level_enabled", 3);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/5"), "set_glow_level", "is_glow_level_enabled", 4);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/6"), "set_glow_level", "is_glow_level_enabled", 5);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "glow_levels/7"), "set_glow_level", "is_glow_level_enabled", 6);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_intensity", PROPERTY_HINT_RANGE, "0.0,8.0,0.01"), "set_glow_intensity", "get_glow_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_mix", PROPERTY_HINT_RANGE, "0.0,1.0,0.001"), "set_glow_mix", "get_glow_mix");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_strength", PROPERTY_HINT_RANGE, "0.0,2.0,0.01"), "set_glow_strength", "get_glow_strength");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_bloom", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_glow_bloom", "get_glow_bloom");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "glow_blend_mode", PROPERTY_HINT_ENUM, "Additive,Screen,Softlight,Replace,Mix"), "set_glow_blend_mode", "get_glow_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_hdr_threshold", PROPERTY_HINT_RANGE, "0.0,4.0,0.01"), "set_glow_hdr_bleed_threshold", "get_glow_hdr_bleed_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_hdr_luminance_cap", PROPERTY_HINT_RANGE, "0.0,256.0,0.01"), "set_glow_hdr_luminance_cap", "get_glow_hdr_luminance_cap");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "glow_hdr_scale", PROPERTY_HINT_RANGE, "0.0,4.0,0.01"), "set_glow_hdr_bleed_scale", "get_glow_hdr_bleed_scale");

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
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adjustment_brightness", PROPERTY_HINT_RANGE, "0.01,8,0.01"), "set_adjustment_brightness", "get_adjustment_brightness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adjustment_contrast", PROPERTY_HINT_RANGE, "0.01,8,0.01"), "set_adjustment_contrast", "get_adjustment_contrast");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adjustment_saturation", PROPERTY_HINT_RANGE, "0.01,8,0.01"), "set_adjustment_saturation", "get_adjustment_saturation");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "adjustment_color_correction", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_adjustment_color_correction", "get_adjustment_color_correction");

	BIND_ENUM_CONSTANT(BG_CLEAR_COLOR);
	BIND_ENUM_CONSTANT(BG_COLOR);
	BIND_ENUM_CONSTANT(BG_SKY);
	BIND_ENUM_CONSTANT(BG_CANVAS);
	BIND_ENUM_CONSTANT(BG_KEEP);
	BIND_ENUM_CONSTANT(BG_CAMERA_FEED);
	BIND_ENUM_CONSTANT(BG_MAX);

	BIND_ENUM_CONSTANT(AMBIENT_SOURCE_BG);
	BIND_ENUM_CONSTANT(AMBIENT_SOURCE_DISABLED);
	BIND_ENUM_CONSTANT(AMBIENT_SOURCE_COLOR);
	BIND_ENUM_CONSTANT(AMBIENT_SOURCE_SKY);

	BIND_ENUM_CONSTANT(REFLECTION_SOURCE_BG);
	BIND_ENUM_CONSTANT(REFLECTION_SOURCE_DISABLED);
	BIND_ENUM_CONSTANT(REFLECTION_SOURCE_SKY);

	BIND_ENUM_CONSTANT(GLOW_BLEND_MODE_ADDITIVE);
	BIND_ENUM_CONSTANT(GLOW_BLEND_MODE_SCREEN);
	BIND_ENUM_CONSTANT(GLOW_BLEND_MODE_SOFTLIGHT);
	BIND_ENUM_CONSTANT(GLOW_BLEND_MODE_REPLACE);
	BIND_ENUM_CONSTANT(GLOW_BLEND_MODE_MIX);

	BIND_ENUM_CONSTANT(TONE_MAPPER_LINEAR);
	BIND_ENUM_CONSTANT(TONE_MAPPER_REINHARDT);
	BIND_ENUM_CONSTANT(TONE_MAPPER_FILMIC);
	BIND_ENUM_CONSTANT(TONE_MAPPER_ACES);

	BIND_ENUM_CONSTANT(SSAO_BLUR_DISABLED);
	BIND_ENUM_CONSTANT(SSAO_BLUR_1x1);
	BIND_ENUM_CONSTANT(SSAO_BLUR_2x2);
	BIND_ENUM_CONSTANT(SSAO_BLUR_3x3);
}

Environment::Environment() :
		bg_mode(BG_CLEAR_COLOR),
		tone_mapper(TONE_MAPPER_LINEAR),
		ssao_blur(SSAO_BLUR_3x3),
		glow_blend_mode(GLOW_BLEND_MODE_ADDITIVE) {

	environment = RS::get_singleton()->environment_create();

	bg_mode = BG_CLEAR_COLOR;
	bg_sky_custom_fov = 0;
	bg_energy = 1.0;
	bg_canvas_max_layer = 0;
	ambient_energy = 1.0;
	//ambient_sky_contribution = 1.0;
	ambient_source = AMBIENT_SOURCE_BG;
	reflection_source = REFLECTION_SOURCE_BG;
	set_ambient_light_sky_contribution(1.0);
	set_camera_feed_id(1);

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

	ssr_enabled = false;
	ssr_max_steps = 64;
	ssr_fade_in = 0.15;
	ssr_fade_out = 2.0;
	ssr_depth_tolerance = 0.2;

	ssao_enabled = false;
	ssao_radius = 1;
	ssao_intensity = 1;
	ssao_bias = 0.01;
	ssao_direct_light_affect = 0.0;
	ssao_ao_channel_affect = 0.0;
	ssao_blur = SSAO_BLUR_3x3;
	set_ssao_edge_sharpness(4);

	glow_enabled = false;
	glow_levels = (1 << 2) | (1 << 4);
	glow_intensity = 0.8;
	glow_strength = 1.0;
	glow_mix = 0.05;
	glow_bloom = 0.0;
	glow_blend_mode = GLOW_BLEND_MODE_SOFTLIGHT;
	glow_hdr_bleed_threshold = 1.0;
	glow_hdr_luminance_cap = 12.0;
	glow_hdr_bleed_scale = 2.0;

	fog_enabled = false;
	fog_color = Color(0.5, 0.5, 0.5);
	fog_sun_color = Color(0.8, 0.8, 0.0);
	fog_sun_amount = 0;

	fog_depth_enabled = true;

	fog_depth_begin = 10;
	fog_depth_end = 100;
	fog_depth_curve = 1;

	fog_transmit_enabled = false;
	fog_transmit_curve = 1;

	fog_height_enabled = false;
	fog_height_min = 10;
	fog_height_max = 0;
	fog_height_curve = 1;

	set_fog_color(Color(0.5, 0.6, 0.7));
	set_fog_sun_color(Color(1.0, 0.9, 0.7));
}

Environment::~Environment() {

	RS::get_singleton()->free(environment);
}

//////////////////////

void CameraEffects::set_dof_blur_far_enabled(bool p_enable) {

	dof_blur_far_enabled = p_enable;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
}

bool CameraEffects::is_dof_blur_far_enabled() const {

	return dof_blur_far_enabled;
}

void CameraEffects::set_dof_blur_far_distance(float p_distance) {

	dof_blur_far_distance = p_distance;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
}
float CameraEffects::get_dof_blur_far_distance() const {

	return dof_blur_far_distance;
}

void CameraEffects::set_dof_blur_far_transition(float p_distance) {

	dof_blur_far_transition = p_distance;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
}
float CameraEffects::get_dof_blur_far_transition() const {

	return dof_blur_far_transition;
}

void CameraEffects::set_dof_blur_near_enabled(bool p_enable) {

	dof_blur_near_enabled = p_enable;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
	_change_notify();
}

bool CameraEffects::is_dof_blur_near_enabled() const {

	return dof_blur_near_enabled;
}

void CameraEffects::set_dof_blur_near_distance(float p_distance) {

	dof_blur_near_distance = p_distance;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
}

float CameraEffects::get_dof_blur_near_distance() const {

	return dof_blur_near_distance;
}

void CameraEffects::set_dof_blur_near_transition(float p_distance) {

	dof_blur_near_transition = p_distance;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
}

float CameraEffects::get_dof_blur_near_transition() const {

	return dof_blur_near_transition;
}

void CameraEffects::set_dof_blur_amount(float p_amount) {

	dof_blur_amount = p_amount;
	RS::get_singleton()->camera_effects_set_dof_blur(camera_effects, dof_blur_far_enabled, dof_blur_far_distance, dof_blur_far_transition, dof_blur_near_enabled, dof_blur_near_distance, dof_blur_near_transition, dof_blur_amount);
}
float CameraEffects::get_dof_blur_amount() const {

	return dof_blur_amount;
}

void CameraEffects::set_override_exposure_enabled(bool p_enabled) {
	override_exposure_enabled = p_enabled;
	RS::get_singleton()->camera_effects_set_custom_exposure(camera_effects, override_exposure_enabled, override_exposure);
}

bool CameraEffects::is_override_exposure_enabled() const {
	return override_exposure_enabled;
}

void CameraEffects::set_override_exposure(float p_exposure) {
	override_exposure = p_exposure;
	RS::get_singleton()->camera_effects_set_custom_exposure(camera_effects, override_exposure_enabled, override_exposure);
}

float CameraEffects::get_override_exposure() const {
	return override_exposure;
}

RID CameraEffects::get_rid() const {
	return camera_effects;
}

void CameraEffects::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_enabled", "enabled"), &CameraEffects::set_dof_blur_far_enabled);
	ClassDB::bind_method(D_METHOD("is_dof_blur_far_enabled"), &CameraEffects::is_dof_blur_far_enabled);

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_distance", "intensity"), &CameraEffects::set_dof_blur_far_distance);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_distance"), &CameraEffects::get_dof_blur_far_distance);

	ClassDB::bind_method(D_METHOD("set_dof_blur_far_transition", "intensity"), &CameraEffects::set_dof_blur_far_transition);
	ClassDB::bind_method(D_METHOD("get_dof_blur_far_transition"), &CameraEffects::get_dof_blur_far_transition);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_enabled", "enabled"), &CameraEffects::set_dof_blur_near_enabled);
	ClassDB::bind_method(D_METHOD("is_dof_blur_near_enabled"), &CameraEffects::is_dof_blur_near_enabled);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_distance", "intensity"), &CameraEffects::set_dof_blur_near_distance);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_distance"), &CameraEffects::get_dof_blur_near_distance);

	ClassDB::bind_method(D_METHOD("set_dof_blur_near_transition", "intensity"), &CameraEffects::set_dof_blur_near_transition);
	ClassDB::bind_method(D_METHOD("get_dof_blur_near_transition"), &CameraEffects::get_dof_blur_near_transition);

	ClassDB::bind_method(D_METHOD("set_dof_blur_amount", "intensity"), &CameraEffects::set_dof_blur_amount);
	ClassDB::bind_method(D_METHOD("get_dof_blur_amount"), &CameraEffects::get_dof_blur_amount);

	ClassDB::bind_method(D_METHOD("set_override_exposure_enabled", "enable"), &CameraEffects::set_override_exposure_enabled);
	ClassDB::bind_method(D_METHOD("is_override_exposure_enabled"), &CameraEffects::is_override_exposure_enabled);

	ClassDB::bind_method(D_METHOD("set_override_exposure", "exposure"), &CameraEffects::set_override_exposure);
	ClassDB::bind_method(D_METHOD("get_override_exposure"), &CameraEffects::get_override_exposure);

	ADD_GROUP("DOF Blur", "dof_blur_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dof_blur_far_enabled"), "set_dof_blur_far_enabled", "is_dof_blur_far_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_far_distance", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_far_distance", "get_dof_blur_far_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_far_transition", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_far_transition", "get_dof_blur_far_transition");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dof_blur_near_enabled"), "set_dof_blur_near_enabled", "is_dof_blur_near_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_near_distance", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_near_distance", "get_dof_blur_near_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_near_transition", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01"), "set_dof_blur_near_transition", "get_dof_blur_near_transition");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dof_blur_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_dof_blur_amount", "get_dof_blur_amount");
	ADD_GROUP("Override Exposure", "override_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_exposure_enable"), "set_override_exposure_enabled", "is_override_exposure_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "override_exposure", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_override_exposure", "get_override_exposure");
}

CameraEffects::CameraEffects() {

	camera_effects = RS::get_singleton()->camera_effects_create();

	dof_blur_far_enabled = false;
	dof_blur_far_distance = 10;
	dof_blur_far_transition = 5;

	dof_blur_near_enabled = false;
	dof_blur_near_distance = 2;
	dof_blur_near_transition = 1;

	set_dof_blur_amount(0.1);

	override_exposure_enabled = false;
	set_override_exposure(1.0);
}

CameraEffects::~CameraEffects() {

	RS::get_singleton()->free(camera_effects);
}

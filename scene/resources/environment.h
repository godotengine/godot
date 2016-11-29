/*************************************************************************/
/*  environment.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "resource.h"
#include "servers/visual_server.h"
#include "scene/resources/texture.h"
#include "scene/resources/sky_box.h"

class Environment : public Resource {

	OBJ_TYPE(Environment,Resource);
public:

	enum BGMode {

		BG_CLEAR_COLOR,
		BG_COLOR,
		BG_SKYBOX,
		BG_CANVAS,
		BG_KEEP,
		BG_MAX
	};

	enum GlowBlendMode {
		GLOW_BLEND_MODE_ADDITIVE,
		GLOW_BLEND_MODE_SCREEN,
		GLOW_BLEND_MODE_SOFTLIGHT,
		GLOW_BLEND_MODE_DISABLED,
	};

	enum ToneMapper {
		TONE_MAPPER_LINEAR,
		TONE_MAPPER_LOG,
		TONE_MAPPER_REINHARDT,
		TONE_MAPPER_FILMIC,
		TONE_MAPPER_ACES_FILMIC
	};



private:
	RID environment;

	BGMode bg_mode;
	Ref<SkyBox> bg_skybox;
	float bg_skybox_scale;
	Color bg_color;
	float bg_energy;
	int bg_canvas_max_layer;
	Color ambient_color;
	float ambient_energy;
	float ambient_skybox_contribution;

	ToneMapper tone_mapper;
	float tonemap_exposure;
	float tonemap_white;
	bool tonemap_auto_exposure;
	float tonemap_auto_exposure_max;
	float tonemap_auto_exposure_min;
	float tonemap_auto_exposure_speed;
	float tonemap_auto_exposure_scale;

	bool adjustment_enabled;
	float adjustment_contrast;
	float adjustment_saturation;
	float adjustment_brightness;
	Ref<Texture> adjustment_color_correction;

	bool ssr_enabled;
	int ssr_max_steps;
	float ssr_accel;
	float ssr_fade;
	float ssr_depth_tolerance;
	bool ssr_smooth;
	bool ssr_roughness;

protected:

	static void _bind_methods();
	virtual void _validate_property(PropertyInfo& property) const;

public:


	void set_background(BGMode p_bg);
	void set_skybox(const Ref<SkyBox>& p_skybox);
	void set_skybox_scale(float p_scale);
	void set_bg_color(const Color& p_color);
	void set_bg_energy(float p_energy);
	void set_canvas_max_layer(int p_max_layer);
	void set_ambient_light_color(const Color& p_color);
	void set_ambient_light_energy(float p_energy);
	void set_ambient_light_skybox_contribution(float p_energy);

	BGMode get_background() const;
	Ref<SkyBox> get_skybox() const;
	float get_skybox_scale() const;
	Color get_bg_color() const;
	float get_bg_energy() const;
	int get_canvas_max_layer() const;
	Color get_ambient_light_color() const;
	float get_ambient_light_energy() const;
	float get_ambient_light_skybox_contribution() const;


	void set_tonemapper(ToneMapper p_tone_mapper);
	ToneMapper get_tonemapper() const;

	void set_tonemap_exposure(float p_exposure);
	float get_tonemap_exposure() const;

	void set_tonemap_white(float p_white);
	float get_tonemap_white() const;

	void set_tonemap_auto_exposure(bool p_enabled);
	bool get_tonemap_auto_exposure() const;

	void set_tonemap_auto_exposure_max(float p_auto_exposure_max);
	float get_tonemap_auto_exposure_max() const;

	void set_tonemap_auto_exposure_min(float p_auto_exposure_min);
	float get_tonemap_auto_exposure_min() const;

	void set_tonemap_auto_exposure_speed(float p_auto_exposure_speed);
	float get_tonemap_auto_exposure_speed() const;

	void set_tonemap_auto_exposure_scale(float p_auto_exposure_scale);
	float get_tonemap_auto_exposure_scale() const;

	void set_adjustment_enable(bool p_enable);
	bool is_adjustment_enabled() const;

	void set_adjustment_brightness(float p_brightness);
	float get_adjustment_brightness() const;

	void set_adjustment_contrast(float p_contrast);
	float get_adjustment_contrast() const;

	void set_adjustment_saturation(float p_saturation);
	float get_adjustment_saturation() const;

	void set_adjustment_color_correction(const Ref<Texture>& p_ramp);
	Ref<Texture> get_adjustment_color_correction() const;

	void set_ssr_enabled(bool p_enable);
	bool is_ssr_enabled() const;

	void set_ssr_max_steps(int p_steps);
	int get_ssr_max_steps() const;

	void set_ssr_accel(float p_accel);
	float get_ssr_accel() const;

	void set_ssr_fade(float p_fade);
	float get_ssr_fade() const;

	void set_ssr_depth_tolerance(float p_depth_tolerance);
	float get_ssr_depth_tolerance() const;

	void set_ssr_smooth(bool p_enable);
	bool is_ssr_smooth() const;

	void set_ssr_rough(bool p_enable);
	bool is_ssr_rough() const;


	virtual RID get_rid() const;

	Environment();
	~Environment();
};




VARIANT_ENUM_CAST(Environment::BGMode)
VARIANT_ENUM_CAST(Environment::GlowBlendMode)
VARIANT_ENUM_CAST(Environment::ToneMapper)


#endif // ENVIRONMENT_H

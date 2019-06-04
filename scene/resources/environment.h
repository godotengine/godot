/*************************************************************************/
/*  environment.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

class Environment : public Resource {

	OBJ_TYPE(Environment, Resource);

public:
	enum BG {

		BG_KEEP = VS::ENV_BG_KEEP,
		BG_DEFAULT_COLOR = VS::ENV_BG_DEFAULT_COLOR,
		BG_COLOR = VS::ENV_BG_COLOR,
		BG_TEXTURE = VS::ENV_BG_TEXTURE,
		BG_CUBEMAP = VS::ENV_BG_CUBEMAP,
		BG_CANVAS = VS::ENV_BG_CANVAS,
		BG_MAX = VS::ENV_BG_MAX
	};

	enum BGParam {

		BG_PARAM_CANVAS_MAX_LAYER = VS::ENV_BG_PARAM_CANVAS_MAX_LAYER,
		BG_PARAM_COLOR = VS::ENV_BG_PARAM_COLOR,
		BG_PARAM_TEXTURE = VS::ENV_BG_PARAM_TEXTURE,
		BG_PARAM_CUBEMAP = VS::ENV_BG_PARAM_CUBEMAP,
		BG_PARAM_ENERGY = VS::ENV_BG_PARAM_ENERGY,
		BG_PARAM_SCALE = VS::ENV_BG_PARAM_SCALE,
		BG_PARAM_GLOW = VS::ENV_BG_PARAM_GLOW,
		BG_PARAM_MAX = VS::ENV_BG_PARAM_MAX
	};

	enum Fx {
		FX_AMBIENT_LIGHT = VS::ENV_FX_AMBIENT_LIGHT,
		FX_FXAA = VS::ENV_FX_FXAA,
		FX_GLOW = VS::ENV_FX_GLOW,
		FX_DOF_BLUR = VS::ENV_FX_DOF_BLUR,
		FX_HDR = VS::ENV_FX_HDR,
		FX_FOG = VS::ENV_FX_FOG,
		FX_BCS = VS::ENV_FX_BCS,
		FX_SRGB = VS::ENV_FX_SRGB,
		FX_MAX = VS::ENV_FX_MAX,
	};

	enum FxBlurBlendMode {
		FX_BLUR_BLEND_MODE_ADDITIVE,
		FX_BLUR_BLEND_MODE_SCREEN,
		FX_BLUR_BLEND_MODE_SOFTLIGHT,
	};

	enum FxHDRToneMapper {
		FX_HDR_TONE_MAPPER_LINEAR,
		FX_HDR_TONE_MAPPER_LOG,
		FX_HDR_TONE_MAPPER_REINHARDT,
		FX_HDR_TONE_MAPPER_REINHARDT_AUTOWHITE,
	};

	enum FxParam {
		FX_PARAM_AMBIENT_LIGHT_COLOR = VS::ENV_FX_PARAM_AMBIENT_LIGHT_COLOR,
		FX_PARAM_AMBIENT_LIGHT_ENERGY = VS::ENV_FX_PARAM_AMBIENT_LIGHT_ENERGY,
		FX_PARAM_GLOW_BLUR_PASSES = VS::ENV_FX_PARAM_GLOW_BLUR_PASSES,
		FX_PARAM_GLOW_BLUR_SCALE = VS::ENV_FX_PARAM_GLOW_BLUR_SCALE,
		FX_PARAM_GLOW_BLUR_STRENGTH = VS::ENV_FX_PARAM_GLOW_BLUR_STRENGTH,
		FX_PARAM_GLOW_BLUR_BLEND_MODE = VS::ENV_FX_PARAM_GLOW_BLUR_BLEND_MODE,
		FX_PARAM_GLOW_BLOOM = VS::ENV_FX_PARAM_GLOW_BLOOM,
		FX_PARAM_GLOW_BLOOM_TRESHOLD = VS::ENV_FX_PARAM_GLOW_BLOOM_TRESHOLD,
		FX_PARAM_DOF_BLUR_PASSES = VS::ENV_FX_PARAM_DOF_BLUR_PASSES,
		FX_PARAM_DOF_BLUR_BEGIN = VS::ENV_FX_PARAM_DOF_BLUR_BEGIN,
		FX_PARAM_DOF_BLUR_RANGE = VS::ENV_FX_PARAM_DOF_BLUR_RANGE,
		FX_PARAM_HDR_EXPOSURE = VS::ENV_FX_PARAM_HDR_EXPOSURE,
		FX_PARAM_HDR_TONEMAPPER = VS::ENV_FX_PARAM_HDR_TONEMAPPER,
		FX_PARAM_HDR_WHITE = VS::ENV_FX_PARAM_HDR_WHITE,
		FX_PARAM_HDR_GLOW_TRESHOLD = VS::ENV_FX_PARAM_HDR_GLOW_TRESHOLD,
		FX_PARAM_HDR_GLOW_SCALE = VS::ENV_FX_PARAM_HDR_GLOW_SCALE,
		FX_PARAM_HDR_MIN_LUMINANCE = VS::ENV_FX_PARAM_HDR_MIN_LUMINANCE,
		FX_PARAM_HDR_MAX_LUMINANCE = VS::ENV_FX_PARAM_HDR_MAX_LUMINANCE,
		FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED = VS::ENV_FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED,
		FX_PARAM_FOG_BEGIN = VS::ENV_FX_PARAM_FOG_BEGIN,
		FX_PARAM_FOG_BEGIN_COLOR = VS::ENV_FX_PARAM_FOG_BEGIN_COLOR,
		FX_PARAM_FOG_END_COLOR = VS::ENV_FX_PARAM_FOG_END_COLOR,
		FX_PARAM_FOG_ATTENUATION = VS::ENV_FX_PARAM_FOG_ATTENUATION,
		FX_PARAM_FOG_BG = VS::ENV_FX_PARAM_FOG_BG,
		FX_PARAM_BCS_BRIGHTNESS = VS::ENV_FX_PARAM_BCS_BRIGHTNESS,
		FX_PARAM_BCS_CONTRAST = VS::ENV_FX_PARAM_BCS_CONTRAST,
		FX_PARAM_BCS_SATURATION = VS::ENV_FX_PARAM_BCS_SATURATION,
		FX_PARAM_MAX = VS::ENV_FX_PARAM_MAX
	};

private:
	BG bg_mode;
	Variant bg_param[BG_PARAM_MAX];
	bool fx_enabled[FX_MAX];
	Variant fx_param[FX_PARAM_MAX];
	RID environment;

protected:
	static void _bind_methods();

public:
	void set_background(BG p_bg);
	BG get_background() const;

	void set_background_param(BGParam p_param, const Variant &p_value);
	Variant get_background_param(BGParam p_param) const;

	void set_enable_fx(Fx p_effect, bool p_enabled);
	bool is_fx_enabled(Fx p_effect) const;

	void fx_set_param(FxParam p_param, const Variant &p_value);
	Variant fx_get_param(FxParam p_param) const;

	virtual RID get_rid() const;

	Environment();
	~Environment();
};

VARIANT_ENUM_CAST(Environment::BG);
VARIANT_ENUM_CAST(Environment::BGParam);
VARIANT_ENUM_CAST(Environment::Fx);
VARIANT_ENUM_CAST(Environment::FxParam);

#endif // ENVIRONMENT_H

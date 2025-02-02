/**************************************************************************/
/*  tone_mapper.h                                                         */
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

#pragma once

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/tonemap.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/tonemap_mobile.glsl.gen.h"

#include "servers/rendering/rendering_server.h"

namespace RendererRD {

class ToneMapper {
private:
	bool using_mobile_version = false;
	enum TonemapMode {
		TONEMAP_MODE_NORMAL,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER,
		TONEMAP_MODE_1D_LUT,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER_1D_LUT,

		TONEMAP_MODE_NORMAL_MULTIVIEW,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER_MULTIVIEW,
		TONEMAP_MODE_1D_LUT_MULTIVIEW,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER_1D_LUT_MULTIVIEW,

		TONEMAP_MODE_MAX
	};

	enum TonemapModeMobile {
		TONEMAP_MOBILE_MODE_NORMAL,
		TONEMAP_MOBILE_MODE_1D_LUT,
		TONEMAP_MOBILE_MODE_SUBPASS,
		TONEMAP_MOBILE_MODE_SUBPASS_1D_LUT,

		TONEMAP_MOBILE_MODE_NORMAL_MULTIVIEW,
		TONEMAP_MOBILE_MODE_1D_LUT_MULTIVIEW,
		TONEMAP_MOBILE_MODE_SUBPASS_MULTIVIEW,
		TONEMAP_MOBILE_MODE_SUBPASS_1D_LUT_MULTIVIEW,

		TONEMAP_MOBILE_MODE_MAX
	};

	enum Flags {
		TONEMAP_FLAG_USE_BCS = (1 << 0),
		TONEMAP_FLAG_USE_GLOW = (1 << 1),
		TONEMAP_FLAG_USE_AUTO_EXPOSURE = (1 << 2),
		TONEMAP_FLAG_USE_COLOR_CORRECTION = (1 << 3),
		TONEMAP_FLAG_USE_FXAA = (1 << 4),
		TONEMAP_FLAG_USE_8_BIT_DEBANDING = (1 << 5),
		TONEMAP_FLAG_CONVERT_TO_SRGB = (1 << 6),
	};

	enum FlagsMobile {
		TONEMAP_MOBILE_FLAG_USE_BCS = (1 << 0),
		TONEMAP_MOBILE_FLAG_USE_GLOW = (1 << 1),
		TONEMAP_MOBILE_FLAG_USE_GLOW_MAP = (1 << 2),
		TONEMAP_MOBILE_FLAG_USE_COLOR_CORRECTION = (1 << 3),
		TONEMAP_MOBILE_FLAG_USE_FXAA = (1 << 4),
		TONEMAP_MOBILE_FLAG_USE_8_BIT_DEBANDING = (1 << 5),
		TONEMAP_MOBILE_FLAG_USE_10_BIT_DEBANDING = (1 << 6),
		TONEMAP_MOBILE_FLAG_CONVERT_TO_SRGB = (1 << 7),

		TONEMAP_MOBILE_FLAG_TONEMAPPER_LINEAR = (1 << 8),
		TONEMAP_MOBILE_FLAG_TONEMAPPER_REINHARD = (1 << 9),
		TONEMAP_MOBILE_FLAG_TONEMAPPER_FILMIC = (1 << 10),
		TONEMAP_MOBILE_FLAG_TONEMAPPER_ACES = (1 << 11),
		TONEMAP_MOBILE_FLAG_TONEMAPPER_AGX = (1 << 12),

		TONEMAP_MOBILE_FLAG_GLOW_MODE_ADD = (1 << 13),
		TONEMAP_MOBILE_FLAG_GLOW_MODE_SCREEN = (1 << 14),
		TONEMAP_MOBILE_FLAG_GLOW_MODE_SOFTLIGHT = (1 << 15),
		TONEMAP_MOBILE_FLAG_GLOW_MODE_REPLACE = (1 << 16),
		TONEMAP_MOBILE_FLAG_GLOW_MODE_MIX = (1 << 17),
		TONEMAP_MOBILE_ADRENO_BUG = (1 << 18), // Needs to be last so we force the pipeline cache to specify specializations for all variants.
	};

	struct TonemapPushConstant {
		float bcs[3]; // 12 - 12
		uint32_t flags; //  4 - 16

		float pixel_size[2]; //  8 - 24
		uint32_t tonemapper; //  4 - 28
		float output_max_value; //  4 - 32

		uint32_t glow_texture_size[2]; //  8 - 40
		float glow_intensity; //  4 - 44
		float glow_map_strength; //  4 - 48

		uint32_t glow_mode; //  4 - 52
		float glow_levels[7]; // 28 - 80

		float exposure; //  4 - 84
		float white; //  4 - 88
		float auto_exposure_scale; //  4 - 92
		float luminance_multiplier; //  4 - 96

		float tonemapper_params[4]; //  16 - 112
	};

	struct TonemapPushConstantMobile {
		float bcs[3]; // 12 - 12
		float luminance_multiplier; //  4 - 16

		float src_pixel_size[2]; //  8 - 24
		float dest_pixel_size[2]; //  8 - 32

		float glow_intensity; //  4 - 36
		float glow_map_strength; //  4 - 40
		float exposure; //  4 - 44
		float white; //  4 - 48

		float tonemapper_params[4]; //  16 - 64
		float output_max_value; //  4 - 68
		float pad[3]; //  12 - 80
	};

	/* tonemap actually writes to a framebuffer, which is
	 * better to do using the raster pipeline rather than
	 * compute, as that framebuffer might be in different formats
	 */
	struct Tonemap {
		TonemapPushConstant push_constant;
		TonemapShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[TONEMAP_MODE_MAX];
	} tonemap;

	struct TonemapMobile {
		TonemapPushConstantMobile push_constant;
		TonemapMobileShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[TONEMAP_MOBILE_MODE_MAX];
	} tonemap_mobile;

public:
	ToneMapper(bool p_use_mobile_version);
	~ToneMapper();

	struct TonemapSettings {
		bool use_glow = false;
		RS::EnvironmentGlowBlendMode glow_mode = RS::ENV_GLOW_BLEND_MODE_SCREEN;
		float glow_intensity = 0.3;
		float glow_map_strength = 0.0f;
		float glow_levels[7] = { 1.0, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0 };
		Vector2i glow_texture_size;
		bool glow_use_bicubic_upscale = false;
		RID glow_texture;
		RID glow_map;

		RS::EnvironmentToneMapper tonemap_mode = RS::ENV_TONE_MAPPER_LINEAR;
		float tonemapper_params[4] = { 0.0, 0.0, 0.0, 0.0 };
		float exposure = 1.0;
		float white = 1.0;
		float max_value = 1.0;

		bool use_auto_exposure = false;
		float auto_exposure_scale = 0.5;
		RID exposure_texture;
		float luminance_multiplier = 1.0;

		bool use_bcs = false;
		float brightness = 1.0;
		float contrast = 1.0;
		float saturation = 1.0;

		bool use_color_correction = false;
		bool use_1d_color_correction = false;
		RID color_correction_texture;

		bool use_fxaa = false;
		enum DebandingMode {
			DEBANDING_MODE_DISABLED,
			DEBANDING_MODE_8_BIT,
			DEBANDING_MODE_10_BIT,
		};
		DebandingMode debanding_mode = DEBANDING_MODE_DISABLED;
		Vector2i texture_size;
		Vector2i dest_texture_size;
		uint32_t view_count = 1;

		bool convert_to_srgb = false;
	};

	void tonemapper(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings);
	void tonemapper_mobile(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings);
	void tonemapper_subpass(RD::DrawListID p_subpass_draw_list, RID p_source_color, RD::FramebufferFormatID p_dst_format_id, const TonemapSettings &p_settings);
};

} // namespace RendererRD

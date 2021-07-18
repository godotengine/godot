/*************************************************************************/
/*  tonemap_rd.h                                                         */
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

#ifndef TONEMAP_RD_H
#define TONEMAP_RD_H

#include "core/math/camera_matrix.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/tonemap.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"

#include "servers/rendering_server.h"

class TonemapRD {
private:
	// TODO this is shared between various effects and we may need to look into moving it into a base class instead of duplicating
	RID default_sampler;
	RID default_mipmap_sampler;
	RID index_buffer;
	RID index_array;

	Map<RID, RID> texture_to_uniform_set_cache;
	RID _get_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);

	// tonemap implementation
	enum Mode {
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

	struct PushConstant {
		float bcs[3];
		uint32_t use_bcs;

		uint32_t use_glow;
		uint32_t use_auto_exposure;
		uint32_t use_color_correction;
		uint32_t tonemapper;

		uint32_t glow_texture_size[2];
		float glow_intensity;
		uint32_t pad3;

		uint32_t glow_mode;
		float glow_levels[7];

		float exposure;
		float white;
		float auto_exposure_grey;
		uint32_t pad2;

		float pixel_size[2];
		uint32_t use_fxaa;
		uint32_t use_debanding;
	};

	/* tonemap actually writes to a framebuffer, which is
	 * better to do using the raster pipeline rather than
	 * compute, as that framebuffer might be in different formats
	 */
	PushConstant push_constant;
	TonemapShaderRD shader;
	RID shader_version;
	PipelineCacheRD pipelines[TONEMAP_MODE_MAX];

public:
	struct Settings {
		bool use_glow = false;
		enum GlowMode {
			GLOW_MODE_ADD,
			GLOW_MODE_SCREEN,
			GLOW_MODE_SOFTLIGHT,
			GLOW_MODE_REPLACE,
			GLOW_MODE_MIX
		};

		GlowMode glow_mode = GLOW_MODE_ADD;
		float glow_intensity = 1.0;
		float glow_levels[7] = { 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0 };
		Vector2i glow_texture_size;
		bool glow_use_bicubic_upscale = false;
		RID glow_texture;

		RS::EnvironmentToneMapper tonemap_mode = RS::ENV_TONE_MAPPER_LINEAR;
		float exposure = 1.0;
		float white = 1.0;

		bool use_auto_exposure = false;
		float auto_exposure_grey = 0.5;
		RID exposure_texture;

		bool use_bcs = false;
		float brightness = 1.0;
		float contrast = 1.0;
		float saturation = 1.0;

		bool use_color_correction = false;
		bool use_1d_color_correction = false;
		RID color_correction_texture;

		bool use_fxaa = false;
		bool use_debanding = false;
		Vector2i texture_size;
		uint32_t view_count = 1;
	};

	void tonemapper(RID p_source_color, RID p_dst_framebuffer, const Settings &p_settings);

	TonemapRD();
	~TonemapRD();
};

#endif // !TONEMAP_RD_H

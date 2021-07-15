/*************************************************************************/
/*  tonemap_subpass_rd.h                                                 */
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

#ifndef TONEMAP_SUBPASS_RD_H
#define TONEMAP_SUBPASS_RD_H

#include "core/math/camera_matrix.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/tonemap_subpass.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"

#include "servers/rendering_server.h"

class TonemapSubpassRD {
private:
	// TODO this is shared between various effects and we may need to look into moving it into a base class instead of duplicating
	RID default_sampler;
	RID index_buffer;
	RID index_array;

	Map<RID, RID> texture_to_uniform_set_cache;
	RID _get_uniform_set_from_texture(RD::UniformType p_type, RID p_texture, uint32_t p_set);

	// tonemap implementation
	enum Mode {
		TONEMAP_SP_MODE_NORMAL,
		TONEMAP_SP_MODE_1D_LUT,

		TONEMAP_SP_MODE_NORMAL_MULTIVIEW,
		TONEMAP_SP_MODE_1D_LUT_MULTIVIEW,

		TONEMAP_SP_MODE_MAX
	};

	struct PushConstant {
		float bcs[3];
		uint32_t use_bcs;

		uint32_t use_auto_exposure;
		uint32_t use_color_correction;
		uint32_t tonemapper;
		uint32_t pad1;

		float exposure;
		float white;
		float auto_exposure_grey;
		uint32_t use_debanding;
	};

	/* tonemap actually writes to a framebuffer, which is
	 * better to do using the raster pipeline rather than
	 * compute, as that framebuffer might be in different formats
	 */
	PushConstant push_constant;
	TonemapSubpassShaderRD shader;
	RID shader_version;
	PipelineCacheRD pipelines[TONEMAP_SP_MODE_MAX];

public:
	struct Settings {
		RD::FramebufferFormatID format_id;
		RID source_texture;

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

		bool use_debanding = false;

		uint32_t view_count = 1;
	};

	void tonemapper(RD::DrawListID p_draw_list, const Settings &p_settings);

	TonemapSubpassRD();
	~TonemapSubpassRD();
};

#endif // !TONEMAP_SUBPASS_RD_H

/**************************************************************************/
/*  motion_blur.h                                                         */
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

#include "copy_effects.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/pipeline_deferred_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_blur.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_neighbor_max.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_preprocess.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_tile_max_x.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_blur_tile_max_y.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_data_rd.h"

#define RB_SCOPE_MOTION_BLUR SNAME("motion_blur")

#define RB_TEX_CUSTOM_VELOCITY SNAME("custom_velocity")
#define RB_TEX_TILE_MAX_X SNAME("tile_max_x")
#define RB_TEX_TILE_MAX_Y SNAME("tile_max_y")
#define RB_TEX_NEIGHBOR_MAX SNAME("neighbor_max")
#define RB_TEX_BLUR_OUTPUT SNAME("blur_output")

namespace RendererRD {

class MotionBlur {
private:
	enum MotionBlurMode {
		MOTION_BLUR_PREPROCESS,
		MOTION_BLUR_TILE_MAX_X,
		MOTION_BLUR_TILE_MAX_Y,
		MOTION_BLUR_NEIGHBOR_MAX,
		MOTION_BLUR_BLUR,
		MOTION_BLUR_MAX,
	};

	struct MotionBlurPreprocessPushConstant {
		float rotation_velocity_multiplier;
		float movement_velocity_multiplier;
		float object_velocity_multiplier;
		float rotation_velocity_lower_threshold;

		float movement_velocity_lower_threshold;
		float object_velocity_lower_threshold;
		float rotation_velocity_upper_threshold;
		float movement_velocity_upper_threshold;

		float object_velocity_upper_threshold;
		float support_fsr2;
		float motion_blur_intensity;
		float pad;
	};

	struct MotionBlurBlurPushConstant {
		float motion_blur_intensity;
		int32_t sample_count;
		int32_t frame;
		int32_t clamp_velocities_to_tile;
		int32_t transparent_bg;
		int32_t pad[3];
	};

	struct {
		MotionBlurPreprocessPushConstant preprocess_push_constant;
		MotionBlurBlurPushConstant blur_push_constant;

		MotionBlurPreprocessShaderRD preprocess_shader;
		RID preprocess_shader_version;

		MotionBlurTileMaxXShaderRD tile_max_x_shader;
		RID tile_max_x_shader_version;

		MotionBlurTileMaxYShaderRD tile_max_y_shader;
		RID tile_max_y_shader_version;

		MotionBlurNeighborMaxShaderRD neighbor_max_shader;
		RID neighbor_max_shader_version;

		MotionBlurBlurShaderRD blur_shader;
		RID blur_shader_version;

		PipelineDeferredRD pipelines[MOTION_BLUR_MAX];
		RID linear_sampler;
		RID nearest_sampler;
	} motion_blur;

	struct MotionBlurBuffers {
		Size2i base_size;
		Size2i tiled_size;

		RID scene_data_uniform;

		// Textures and images
		RID base_texture;
		RID depth_texture;
		RID velocity_texture;
		RID custom_velocity_texture;
		RID tile_max_x_texture;
		RID tile_max_y_texture;
		RID neighbor_max_texture;
		RID blur_output_texture;
	};

	int tile_size;
	void motion_blur_process(const MotionBlurBuffers &p_buffers);

public:
	MotionBlur(RS::MotionBlurTileSize p_tile_size_level);
	~MotionBlur();

	void motion_blur_compute(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_camera_attributes, RenderSceneDataRD *p_scene_data, bool transparent_bg, float time_step, CopyEffects *p_copy_effects);
};
} //namespace RendererRD

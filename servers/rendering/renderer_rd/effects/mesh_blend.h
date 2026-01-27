/**************************************************************************/
/*  mesh_blend.h                                                          */
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
#include "servers/rendering/renderer_rd/shaders/effects/mesh_blend_blend.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/mesh_blend_jump_flood.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/mesh_blend_mask.glsl.gen.h"
#include "servers/rendering/rendering_device.h"

namespace RendererRD {

class MeshBlend {
public:
	struct CameraData {
		static const uint32_t MAX_CAMERAS = 2;
		float inv_view_projection[MAX_CAMERAS][16];
	};

	MeshBlend();
	~MeshBlend();

	void update_camera_data(const CameraData &p_data);

	void generate_mask(RID p_vb_vis, RID p_vb_aux, RID p_vb_depth, RID p_mask, RID p_edge_dest, const Size2i &p_size, float p_depth_tolerance, float p_neighbor_blend);
	void jump_flood(RID p_edge_src, RID p_edge_dst, RID p_mask, const Size2i &p_size, int p_spread);
	void blend(RID p_source_color, RID p_depth, RID p_mask, RID p_edges, RID p_dest_framebuffer, const Size2i &p_size, float p_edge_radius, int p_view_index, bool p_use_world_radius, float p_neighbor_blend);

private:
	struct MaskPushConstant {
		int32_t resolution[2] = { 0, 0 };
		float depth_tolerance = 0.0f;
		int32_t require_pair = 0;
	};

	struct JumpFloodPushConstant {
		int32_t resolution[2] = { 0, 0 };
		int32_t spread = 1;
		int32_t pad = 0;
	};

	struct BlendPushConstant {
		int32_t resolution[2] = { 0, 0 };
		float edge_radius = 0.0f;
		int32_t view_index = 0;
		int32_t use_world_radius = 0;
		float neighbor_blend = 0.0f;
		float pad1 = 0.0f;
		float pad2 = 0.0f;
	};

	MeshBlendMaskShaderRD mask_shader;
	MeshBlendJumpFloodShaderRD jump_flood_shader;
	MeshBlendBlendShaderRD blend_shader;

	RID mask_shader_version;
	RID jump_flood_shader_version;
	RID blend_shader_version;

	RID mask_pipeline;
	RID jump_flood_pipeline;
	PipelineCacheRD blend_pipeline;

	RID camera_ubo;

	void _ensure_camera_buffer();
};

} // namespace RendererRD

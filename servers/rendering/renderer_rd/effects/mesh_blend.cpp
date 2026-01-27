/**************************************************************************/
/*  mesh_blend.cpp                                                        */
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

#include "mesh_blend.h"

#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

MeshBlend::MeshBlend() {
	Vector<String> variants;
	variants.push_back(String());

	mask_shader.initialize(variants);
	mask_shader_version = mask_shader.version_create();
	RID mask_shader_rid = mask_shader.version_get_shader(mask_shader_version, 0);
	mask_pipeline = RD::get_singleton()->compute_pipeline_create(mask_shader_rid);

	jump_flood_shader.initialize(variants);
	jump_flood_shader_version = jump_flood_shader.version_create();
	RID jump_shader_rid = jump_flood_shader.version_get_shader(jump_flood_shader_version, 0);
	jump_flood_pipeline = RD::get_singleton()->compute_pipeline_create(jump_shader_rid);

	blend_shader.initialize(variants);
	blend_shader_version = blend_shader.version_create();
	RID blend_shader_rid = blend_shader.version_get_shader(blend_shader_version, 0);

	RD::PipelineRasterizationState raster_state;
	RD::PipelineMultisampleState multisample_state;
	RD::PipelineDepthStencilState depth_state;
	RD::PipelineColorBlendState blend_state = RD::PipelineColorBlendState::create_disabled();
	blend_pipeline.setup(blend_shader_rid, RD::RENDER_PRIMITIVE_TRIANGLES, raster_state, multisample_state, depth_state, blend_state, 0);
}

MeshBlend::~MeshBlend() {
	if (mask_pipeline.is_valid()) {
		RD::get_singleton()->free_rid(mask_pipeline);
	}
	if (jump_flood_pipeline.is_valid()) {
		RD::get_singleton()->free_rid(jump_flood_pipeline);
	}

	if (mask_shader_version.is_valid()) {
		mask_shader.version_free(mask_shader_version);
	}
	if (jump_flood_shader_version.is_valid()) {
		jump_flood_shader.version_free(jump_flood_shader_version);
	}
	if (blend_shader_version.is_valid()) {
		blend_shader.version_free(blend_shader_version);
	}

	if (camera_ubo.is_valid()) {
		RD::get_singleton()->free_rid(camera_ubo);
	}
}

void MeshBlend::_ensure_camera_buffer() {
	if (!camera_ubo.is_valid()) {
		camera_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(CameraData));
	}
}

void MeshBlend::update_camera_data(const CameraData &p_data) {
	_ensure_camera_buffer();
	RD::get_singleton()->buffer_update(camera_ubo, 0, sizeof(CameraData), &p_data);
}

void MeshBlend::generate_mask(RID p_vb_vis, RID p_vb_aux, RID p_vb_depth, RID p_mask, RID p_edge_dest, const Size2i &p_size, float p_depth_tolerance, float p_neighbor_blend) {
	if (!mask_pipeline.is_valid()) {
		return;
	}

	if (p_vb_depth.is_null()) {
		return;
	}

	UniformSetCacheRD *uniform_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_cache);

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, mask_pipeline);

	RID shader_rid = mask_shader.version_get_shader(mask_shader_version, 0);
	RD::Uniform u_vis(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_vb_vis }));
	RD::Uniform u_aux(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_vb_aux }));
	RD::Uniform u_mask(RD::UNIFORM_TYPE_IMAGE, 2, Vector<RID>({ p_mask }));
	RD::Uniform u_edge(RD::UNIFORM_TYPE_IMAGE, 3, Vector<RID>({ p_edge_dest }));
	RD::Uniform u_depth(RD::UNIFORM_TYPE_IMAGE, 4, Vector<RID>({ p_vb_depth }));

	RID uniform_set = uniform_cache->get_cache(shader_rid, 0, u_vis, u_aux, u_mask, u_edge, u_depth);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

	MaskPushConstant push_constant;
	push_constant.resolution[0] = p_size.x;
	push_constant.resolution[1] = p_size.y;
	push_constant.depth_tolerance = p_depth_tolerance;
	push_constant.require_pair = (p_neighbor_blend != 0.0f) ? 1 : 0;
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(MaskPushConstant));

	uint32_t gx = (p_size.x + 7) / 8;
	uint32_t gy = (p_size.y + 7) / 8;
	RD::get_singleton()->compute_list_dispatch(compute_list, gx, gy, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);
	RD::get_singleton()->compute_list_end();
}

void MeshBlend::jump_flood(RID p_edge_src, RID p_edge_dst, RID p_mask, const Size2i &p_size, int p_spread) {
	if (!jump_flood_pipeline.is_valid()) {
		return;
	}

	UniformSetCacheRD *uniform_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_cache);

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, jump_flood_pipeline);

	RID shader_rid = jump_flood_shader.version_get_shader(jump_flood_shader_version, 0);
	RD::Uniform u_edge_in(RD::UNIFORM_TYPE_IMAGE, 0, Vector<RID>({ p_edge_src }));
	RD::Uniform u_edge_out(RD::UNIFORM_TYPE_IMAGE, 1, Vector<RID>({ p_edge_dst }));
	RD::Uniform u_mask(RD::UNIFORM_TYPE_IMAGE, 2, Vector<RID>({ p_mask }));

	RID uniform_set = uniform_cache->get_cache(shader_rid, 0, u_edge_in, u_edge_out, u_mask);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

	JumpFloodPushConstant push_constant;
	push_constant.resolution[0] = p_size.x;
	push_constant.resolution[1] = p_size.y;
	push_constant.spread = MAX(1, p_spread);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(JumpFloodPushConstant));

	uint32_t gx = (p_size.x + 7) / 8;
	uint32_t gy = (p_size.y + 7) / 8;
	RD::get_singleton()->compute_list_dispatch(compute_list, gx, gy, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);
	RD::get_singleton()->compute_list_end();
}

void MeshBlend::blend(RID p_source_color, RID p_depth, RID p_mask, RID p_edges, RID p_dest_framebuffer, const Size2i &p_size, float p_edge_radius, int p_view_index, bool p_use_world_radius, float p_neighbor_blend) {
	ERR_FAIL_COND_MSG(!camera_ubo.is_valid(), "MeshBlend camera buffer was not initialized.");

	RID shader_rid = blend_shader.version_get_shader(blend_shader_version, 0);
	UniformSetCacheRD *uniform_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_cache);

	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	RID sampler_linear = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	RID sampler_nearest = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_camera(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, camera_ubo);
	RD::Uniform u_color(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ sampler_linear, p_source_color }));
	RD::Uniform u_depth(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, Vector<RID>({ sampler_linear, p_depth }));
	RD::Uniform u_mask(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, Vector<RID>({ sampler_linear, p_mask }));
	RD::Uniform u_edges(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, Vector<RID>({ sampler_nearest, p_edges }));

	RID set0 = uniform_cache->get_cache(shader_rid, 0, u_camera);
	RID set1 = uniform_cache->get_cache(shader_rid, 1, u_color, u_depth, u_mask, u_edges);

	RID framebuffer = p_dest_framebuffer;
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer);
	RID pipeline_rid = blend_pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(framebuffer));

	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipeline_rid);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, set0, 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, set1, 1);

	BlendPushConstant push_constant;
	push_constant.resolution[0] = p_size.x;
	push_constant.resolution[1] = p_size.y;
	push_constant.edge_radius = p_edge_radius;
	push_constant.view_index = p_view_index;
	push_constant.use_world_radius = p_use_world_radius ? 1 : 0;
	push_constant.neighbor_blend = p_neighbor_blend;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(BlendPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, false, 1, 3);
	RD::get_singleton()->draw_list_end();
}

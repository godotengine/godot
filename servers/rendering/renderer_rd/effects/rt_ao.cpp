/**************************************************************************/
/*  roughness_limiter.h                                                   */
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

#include "rt_ao.h"

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

RtAo::RtAo() {
	Vector<String> shader_modes;
	shader_modes.push_back("");
	shader.initialize(shader_modes);
	shader_version = shader.version_create();
	pipeline = RD::get_singleton()->compute_pipeline_create(shader.version_get_shader(shader_version, 0));
}

RtAo::~RtAo() {
	shader.version_free(shader_version);
}

void RtAo::generate(RID p_tlas, RID p_depth, RID p_ao_image, const Size2i &p_size, const Projection &p_projection, const Transform3D &p_cam_transform, float p_radius, uint32_t p_sample_count, uint32_t p_frame) {
	ERR_FAIL_COND(!p_tlas.is_valid());
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);

	PushConstant push_constant = {};
	MaterialStorage::store_camera(p_projection.inverse(), push_constant.inv_projection);

	// Pack the world-from-view transform as three rows: basis row r in xyz, origin component r in w.
	const Basis &basis = p_cam_transform.basis;
	const Vector3 origin = p_cam_transform.origin;
	for (int r = 0; r < 3; r++) {
		push_constant.world_from_view[r * 4 + 0] = basis.rows[r][0];
		push_constant.world_from_view[r * 4 + 1] = basis.rows[r][1];
		push_constant.world_from_view[r * 4 + 2] = basis.rows[r][2];
		push_constant.world_from_view[r * 4 + 3] = origin[r];
	}

	push_constant.radius = p_radius;
	push_constant.sample_count = p_sample_count;
	push_constant.frame = p_frame;

	RID rt_ao_shader = shader.version_get_shader(shader_version, 0);
	RD::Uniform u_depth(RD::UNIFORM_TYPE_TEXTURE, 0, Vector<RID>({ p_depth }));
	RD::Uniform u_tlas(RD::UNIFORM_TYPE_ACCELERATION_STRUCTURE, 1, Vector<RID>({ p_tlas }));
	RD::Uniform u_ao(RD::UNIFORM_TYPE_IMAGE, 2, Vector<RID>({ p_ao_image }));

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(rt_ao_shader, 0, u_depth, u_tlas, u_ao), 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_size.x, p_size.y, 1);
	RD::get_singleton()->compute_list_end();
}

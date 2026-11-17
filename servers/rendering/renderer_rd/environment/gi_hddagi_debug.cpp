/**************************************************************************/
/*  gi_hddagi_debug.cpp                                                   */
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

#include "servers/rendering/renderer_rd/effects/copy_effects.h"
#include "servers/rendering/renderer_rd/environment/gi.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

void GI::HDDAGI::debug_draw(Ref<RenderSceneBuffersRD> p_render_buffers, const Projection *p_projections, const Vector3 *p_eye_offsets, const Transform3D &p_transform) {
	ERR_FAIL_COND(p_render_buffers.is_null());
	const uint32_t view_count = p_render_buffers->get_view_count();
	ERR_FAIL_COND(view_count == 0 || view_count > 2);
	const Size2i size = p_render_buffers->get_internal_size();
	const RID sampler = MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::get_singleton()->draw_command_begin_label("Debug HDDAGI Voxels");
	for (uint32_t v = 0; v < view_count; v++) {
		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.debug_shader_version, 0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, voxel_bits_tex),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, voxel_region_tex),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, voxel_light_tex),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, sampler),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, voxel_light_neighbour_data),
				RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 6, cascades_ubo),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 7, p_render_buffers->get_internal_texture(v)));

		HDDAGIShader::DebugPushConstant push_constant = {};
		push_constant.max_cascades = cascades.size();
		push_constant.y_mult = y_mult;

		const Projection inverse = p_projections[v].inverse();
		const Vector3 near_origin = inverse.xform(Vector3(-1, -1, -1));
		const Vector3 near_x = inverse.xform(Vector3(1, -1, -1)) - near_origin;
		const Vector3 near_y = inverse.xform(Vector3(-1, 1, -1)) - near_origin;
		const bool orthogonal = p_projections[v].is_orthogonal();
		const Vector3 ray_origin = p_transform.xform(p_eye_offsets[v] + (orthogonal ? near_origin : Vector3()));
		const Vector3 ray_direction = p_transform.basis.xform(orthogonal ? Vector3(0, 0, -1) : near_origin);
		const Vector3 ray_x = p_transform.basis.xform(near_x);
		const Vector3 ray_y = p_transform.basis.xform(near_y);
		for (int axis = 0; axis < 3; axis++) {
			push_constant.grid_size[axis] = cascade_size[axis];
			push_constant.ray_origin[axis] = ray_origin[axis];
			push_constant.ray_direction[axis] = ray_direction[axis];
			push_constant.ray_x[axis] = orthogonal ? 0.0f : ray_x[axis];
			push_constant.ray_y[axis] = orthogonal ? 0.0f : ray_y[axis];
			push_constant.origin_x[axis] = orthogonal ? ray_x[axis] : 0.0f;
			push_constant.origin_y[axis] = orthogonal ? ray_y[axis] : 0.0f;
		}

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.debug_pipeline);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(push_constant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, size.x, size.y, 1);
		RD::get_singleton()->compute_list_end();
	}

	TextureStorage *texture_storage = TextureStorage::get_singleton();
	const RID render_target = p_render_buffers->get_render_target();
	const Size2i target_size = texture_storage->render_target_get_size(render_target);
	CopyEffects::get_singleton()->copy_to_fb_rect(p_render_buffers->get_internal_texture(), texture_storage->render_target_get_rd_framebuffer(render_target), Rect2i(Point2i(), target_size), true, false, false, false, RID(), view_count > 1);
	RD::get_singleton()->draw_command_end_label();
}

void GI::HDDAGI::debug_probes(RID p_framebuffer, uint32_t p_view_count, const Projection *p_camera_with_transforms) {
	ERR_FAIL_COND(p_view_count == 0 || p_view_count > 2);
	const HDDAGIShader::ProbeDebugMode mode = p_view_count > 1 ? HDDAGIShader::PROBE_DEBUG_PROBES_MULTIVIEW : HDDAGIShader::PROBE_DEBUG_PROBES;
	if (gi->hddagi_shader.debug_probes_shader_version[mode].is_null()) {
		return;
	}

	HDDAGIShader::DebugProbesSceneData scene_data = {};
	if (debug_probes_scene_data_ubo.is_null()) {
		debug_probes_scene_data_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(scene_data));
	}
	for (uint32_t v = 0; v < p_view_count; v++) {
		MaterialStorage::store_camera(p_camera_with_transforms[v], scene_data.projection[v]);
	}
	RD::get_singleton()->buffer_update(debug_probes_scene_data_ubo, 0, sizeof(scene_data), &scene_data);

	HDDAGIShader::DebugProbesPushConstant push_constant = {};
	constexpr uint32_t BAND_POINTS = 16;
	constexpr uint32_t BAND_POWER = 4;
	push_constant.band_power = BAND_POWER;
	push_constant.sections_in_band = (BAND_POINTS / 2) - 1;
	push_constant.band_mask = BAND_POINTS - 2;
	push_constant.section_arc = Math::TAU / float(push_constant.sections_in_band);
	push_constant.y_mult = y_mult;
	push_constant.oct_size = HDDAGI::LIGHTPROBE_OCT_SIZE;

	const Vector3i probe_axis_count = cascade_size / REGION_CELLS + Vector3i(1, 1, 1);
	for (int axis = 0; axis < 3; axis++) {
		push_constant.grid_size[axis] = cascade_size[axis];
		push_constant.probe_axis_size[axis] = probe_axis_count[axis];
	}
	push_constant.cascade = 0;
	const uint32_t total_points = push_constant.sections_in_band * BAND_POINTS;
	const uint32_t total_probes = probe_axis_count.x * probe_axis_count.y * probe_axis_count.z;

	RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
			gi->hddagi_shader.debug_probes_shader_version[mode], 0,
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 1, cascades_ubo),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 3, MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 4, debug_probes_scene_data_ubo),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 5, get_lightprobe_diffuse_texture()));

	RD::get_singleton()->draw_command_begin_label("Debug HDDAGI World Probes");
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, gi->hddagi_shader.debug_probes_pipeline[mode].get_render_pipeline(RD::INVALID_FORMAT_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set, 0);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(push_constant));
	RD::get_singleton()->draw_list_draw(draw_list, false, total_probes, total_points);
	RD::get_singleton()->draw_list_end();
	RD::get_singleton()->draw_command_end_label();
}

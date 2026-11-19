/**************************************************************************/
/*  gi.cpp                                                                */
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

#include "gi.h"

#include "core/config/project_settings.h"
#include "core/math/geometry_3d.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_default.h"

using namespace RendererRD;

const Vector3i GI::HDDAGI::Cascade::DIRTY_ALL = Vector3i(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);

GI *GI::singleton = nullptr;

////////////////////////////////////////////////////////////////////////////////
// VOXEL GI STORAGE

RID GI::voxel_gi_allocate() {
	return voxel_gi_owner.allocate_rid();
}

void GI::voxel_gi_free(RID p_voxel_gi) {
	voxel_gi_allocate_data(p_voxel_gi, Transform3D(), AABB(), Vector3i(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<int>()); //deallocate
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	voxel_gi->dependency.deleted_notify(p_voxel_gi);
	voxel_gi_owner.free(p_voxel_gi);
}

void GI::voxel_gi_initialize(RID p_voxel_gi) {
	voxel_gi_owner.initialize_rid(p_voxel_gi, VoxelGI());
}

void GI::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	if (voxel_gi->octree_buffer.is_valid()) {
		RD::get_singleton()->free_rid(voxel_gi->octree_buffer);
		RD::get_singleton()->free_rid(voxel_gi->data_buffer);
		if (voxel_gi->sdf_texture.is_valid()) {
			RD::get_singleton()->free_rid(voxel_gi->sdf_texture);
		}

		voxel_gi->sdf_texture = RID();
		voxel_gi->octree_buffer = RID();
		voxel_gi->data_buffer = RID();
		voxel_gi->octree_buffer_size = 0;
		voxel_gi->data_buffer_size = 0;
		voxel_gi->cell_count = 0;
	}

	voxel_gi->to_cell_xform = p_to_cell_xform;
	voxel_gi->bounds = p_aabb;
	voxel_gi->octree_size = p_octree_size;
	voxel_gi->level_counts = p_level_counts;

	if (p_octree_cells.size()) {
		ERR_FAIL_COND(p_octree_cells.size() % 32 != 0); //cells size must be a multiple of 32

		uint32_t cell_count = p_octree_cells.size() / 32;

		ERR_FAIL_COND(p_data_cells.size() != (int)cell_count * 16); //see that data size matches

		voxel_gi->cell_count = cell_count;
		voxel_gi->octree_buffer = RD::get_singleton()->storage_buffer_create(p_octree_cells.size(), p_octree_cells);
		voxel_gi->octree_buffer_size = p_octree_cells.size();
		voxel_gi->data_buffer = RD::get_singleton()->storage_buffer_create(p_data_cells.size(), p_data_cells);
		voxel_gi->data_buffer_size = p_data_cells.size();

		if (p_distance_field.size()) {
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = voxel_gi->octree_size.x;
			tf.height = voxel_gi->octree_size.y;
			tf.depth = voxel_gi->octree_size.z;
			tf.texture_type = RD::TEXTURE_TYPE_3D;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			Vector<Vector<uint8_t>> s;
			s.push_back(p_distance_field);
			voxel_gi->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView(), s);
			RD::get_singleton()->set_resource_name(voxel_gi->sdf_texture, "VoxelGI SDF Texture");
		}
	}

	voxel_gi->version++;
	voxel_gi->data_version++;

	voxel_gi->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
}

AABB GI::voxel_gi_get_bounds(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, AABB());

	return voxel_gi->bounds;
}

Vector3i GI::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, Vector3i());
	return voxel_gi->octree_size;
}

Vector<uint8_t> GI::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, Vector<uint8_t>());

	if (voxel_gi->octree_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->octree_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> GI::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->data_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> GI::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->texture_get_data(voxel_gi->sdf_texture, 0);
	}
	return Vector<uint8_t>();
}

Vector<int> GI::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, Vector<int>());

	return voxel_gi->level_counts;
}

Transform3D GI::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, Transform3D());

	return voxel_gi->to_cell_xform;
}

void GI::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->dynamic_range = p_range;
	voxel_gi->version++;
}

float GI::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);

	return voxel_gi->dynamic_range;
}

void GI::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->propagation = p_range;
	voxel_gi->version++;
}

float GI::voxel_gi_get_propagation(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->propagation;
}

void GI::voxel_gi_set_energy(RID p_voxel_gi, float p_energy) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->energy = p_energy;
}

float GI::voxel_gi_get_energy(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->energy;
}

void GI::voxel_gi_set_baked_exposure_normalization(RID p_voxel_gi, float p_baked_exposure) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->baked_exposure = p_baked_exposure;
}

float GI::voxel_gi_get_baked_exposure_normalization(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->baked_exposure;
}

void GI::voxel_gi_set_bias(RID p_voxel_gi, float p_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->bias = p_bias;
}

float GI::voxel_gi_get_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->bias;
}

void GI::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_normal_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->normal_bias = p_normal_bias;
}

float GI::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->normal_bias;
}

void GI::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->interior = p_enable;
}

void GI::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->use_two_bounces = p_enable;
	voxel_gi->version++;
}

bool GI::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, false);
	return voxel_gi->use_two_bounces;
}

bool GI::voxel_gi_is_interior(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->interior;
}

uint32_t GI::voxel_gi_get_version(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->version;
}

uint32_t GI::voxel_gi_get_data_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, 0);
	return voxel_gi->data_version;
}

RID GI::voxel_gi_get_octree_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, RID());
	return voxel_gi->octree_buffer;
}

RID GI::voxel_gi_get_data_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, RID());
	return voxel_gi->data_buffer;
}

RID GI::voxel_gi_get_sdf_texture(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, RID());

	return voxel_gi->sdf_texture;
}

Dependency *GI::voxel_gi_get_dependency(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL_V(voxel_gi, nullptr);

	return &voxel_gi->dependency;
}

void GI::hddagi_reset() {
	hddagi_current_version++;
}

////////////////////////////////////////////////////////////////////////////////
// HDDAGI

static RID create_clear_texture(const RD::TextureFormat &p_format, const String &p_name) {
	RID texture = RD::get_singleton()->texture_create(p_format, RD::TextureView());
	ERR_FAIL_COND_V_MSG(texture.is_null(), RID(), String("Cannot create texture: ") + p_name);

	RD::get_singleton()->set_resource_name(texture, p_name);
	RD::get_singleton()->texture_clear(texture, Color(0, 0, 0, 0), 0, p_format.mipmaps, 0, p_format.array_layers);

	return texture;
}

void GI::HDDAGI::create(RID p_env, const Vector3 &p_world_anchor, const Vector3 &p_planar_world_forward, float p_cascade_forward_offset, uint32_t p_requested_history_size, GI *p_gi) {
	gi = p_gi;
	version = gi->hddagi_current_version;

	cascade_size.x = cascade_size.z = CASCADE_SIZE;

	cascade_format = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_cascade_format(p_env);
	switch (cascade_format) {
		case RSE::ENV_HDDAGI_CASCADE_FORMAT_16X16X16: {
			cascade_size.y = CASCADE_SIZE;
			y_mult = 1.0;
		} break;
		case RSE::ENV_HDDAGI_CASCADE_FORMAT_16X16X16_50_PERCENT_HEIGHT: {
			cascade_size.y = CASCADE_SIZE;
			y_mult = 2.0;
		} break;
		case RSE::ENV_HDDAGI_CASCADE_FORMAT_16X16X16_75_PERCENT_HEIGHT: {
			cascade_size.y = CASCADE_SIZE;
			y_mult = 1.5;
		} break;
		case RSE::ENV_HDDAGI_CASCADE_FORMAT_16X8X16:
		default: {
			cascade_size.y = CASCADE_SIZE / 2;
			y_mult = 1.0;
		} break;
	}

	num_cascades = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_cascades(p_env);
	min_cell_size = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_min_cell_size(p_env);
	using_probe_filter = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_filter_probes(p_env);
	using_reflection_filter = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_filter_reflection(p_env);
	using_ambient_filter = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_filter_ambient(p_env);
	reflection_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_reflection_bias(p_env);
	probe_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_probe_bias(p_env);
	normal_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_normal_bias(p_env);
	frames_to_converge = p_requested_history_size;
	probe_history_ema = int(GLOBAL_GET("rendering/global_illumination/hddagi/world_probe_history_mode")) == 1;
	cascades.resize(num_cascades);

	solid_cell_ratio = gi->hddagi_solid_cell_ratio;
	solid_cell_count = uint32_t(float(cascade_size.x * cascade_size.y * cascade_size.z) * solid_cell_ratio);

	float base_cell_size = min_cell_size;

	RD::TextureFormat tf_base;
	tf_base.format = RD::DATA_FORMAT_R8_UNORM;
	tf_base.width = cascade_size.x;
	tf_base.height = cascade_size.y;
	tf_base.depth = cascade_size.z;
	tf_base.texture_type = RD::TEXTURE_TYPE_3D;
	tf_base.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	auto create_cascade_storage = [&](RD::DataFormat p_format, const Vector3i &p_divisor, const String &p_name) {
		RD::TextureFormat tf = tf_base;
		tf.format = p_format;
		tf.width /= p_divisor.x;
		tf.height /= p_divisor.y;
		tf.depth /= p_divisor.z;
		tf.height *= cascades.size();
		return create_clear_texture(tf, p_name);
	};

	voxel_bits_tex = create_cascade_storage(RD::DATA_FORMAT_R32G32_UINT, Vector3i(4, 4, 4), "HDDAGI Voxel Field");
	region_version_data = create_cascade_storage(RD::DATA_FORMAT_R16_UINT, Vector3i(REGION_CELLS, REGION_CELLS, REGION_CELLS), "HDDAGI Region Version");
	voxel_region_tex = create_cascade_storage(RD::DATA_FORMAT_R8_UINT, Vector3i(REGION_CELLS, REGION_CELLS, REGION_CELLS), "HDDAGI Voxel Regions");

	{
		voxel_pages.resize((cascade_size.x / REGION_CELLS) * (cascade_size.y / REGION_CELLS) * (cascade_size.z / REGION_CELLS) * num_cascades);
		voxel_pages.fill(0);
		voxel_page_table = RD::get_singleton()->storage_buffer_create(voxel_pages.size() * sizeof(uint32_t), voxel_pages.to_byte_array());
		voxel_data = RD::get_singleton()->storage_buffer_create(REGION_CELLS * REGION_CELLS * REGION_CELLS * 2 * sizeof(uint32_t));
		RD::get_singleton()->set_resource_name(voxel_page_table, "HDDAGI Voxel Pages");
		RD::get_singleton()->set_resource_name(voxel_data, "HDDAGI Surface Voxels");

		{ // Albedo texture, this is anisotropic (x6).
			RD::TextureFormat tf_render_albedo = tf_base;
			tf_render_albedo.width /= 2; // Albedo is half size..
			tf_render_albedo.height /= 2;
			tf_render_albedo.depth /= 2;
			tf_render_albedo.depth *= 6; // ..but anisotropic.
			tf_render_albedo.format = RD::DATA_FORMAT_R16_UINT;
			render_albedo = create_clear_texture(tf_render_albedo, "HDDAGI Render Albedo");
		}

		{ // Store emission and six directional weights separately to save space.
			RD::TextureFormat tf_render_emission = tf_base;
			tf_render_emission.width /= 2; // Emission is half size..
			tf_render_emission.height /= 2;
			tf_render_emission.depth /= 2;
			tf_render_emission.format = RD::DATA_FORMAT_R32_UINT;
			tf_render_emission.usage_bits |= RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
			render_emission = create_clear_texture(tf_render_emission, "HDDAGI Render Emission");
			render_emission_aniso = create_clear_texture(tf_render_emission, "HDDAGI Render Emission Aniso");
		}

		{ // Aniso normals
			RD::TextureFormat tf_render_aniso_normals = tf_base;
			tf_render_aniso_normals.format = RD::DATA_FORMAT_R32_UINT;
			tf_render_aniso_normals.usage_bits |= RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
			render_aniso_normals = create_clear_texture(tf_render_aniso_normals, String("HDDAGI Render Solid Bits "));
			tf_render_aniso_normals.usage_bits &= ~RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
		}
	}

	light_process_buffer_render = RD::get_singleton()->storage_buffer_create(sizeof(HDDAGI::Cascade::LightProcessCell));
	light_source_emission_buffer_render = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t));
	light_process_dispatch_buffer_render = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 4, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);

	cascades_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(HDDAGI::Cascade::UBO) * HDDAGI::MAX_CASCADES);

	// lightprobes
	Vector3i PROBE_DIVISOR = cascade_size / REGION_CELLS;

	{
		RD::TextureFormat tf_lightprobes;
		tf_lightprobes.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		tf_lightprobes.format = RD::DATA_FORMAT_R32_UINT;
		tf_lightprobes.width = (PROBE_DIVISOR.x + 1) * (LIGHTPROBE_OCT_SIZE + 2); // Include an extra probe outside the box for interpolation.
		tf_lightprobes.height = (PROBE_DIVISOR.y + 1) * (PROBE_DIVISOR.z + 1) * (LIGHTPROBE_OCT_SIZE + 2); // OctSize +2 because it needs a border to interpolate properly
		tf_lightprobes.depth = 1;
		tf_lightprobes.array_layers = cascades.size();
		tf_lightprobes.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		tf_lightprobes.shareable_formats.push_back(RD::DATA_FORMAT_R32_UINT);
		tf_lightprobes.shareable_formats.push_back(RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);

		lightprobe_specular_data = create_clear_texture(tf_lightprobes, String("HDDAGI Lighprobe Specular"));
		lightprobe_diffuse_data = create_clear_texture(tf_lightprobes, String("HDDAGI Lighprobe Diffuse"));
		lightprobe_diffuse_filter_data = create_clear_texture(tf_lightprobes, String("HDDAGI Lighprobe Diffuse Filtered"));

		RD::TextureView tv;
		tv.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
		lightprobe_diffuse_tex = RD::get_singleton()->texture_create_shared(tv, lightprobe_diffuse_data);
		lightprobe_diffuse_filter_tex = RD::get_singleton()->texture_create_shared(tv, lightprobe_diffuse_filter_data);
		lightprobe_specular_tex = RD::get_singleton()->texture_create_shared(tv, lightprobe_specular_data);

		RD::TextureFormat tf_cache_data = tf_lightprobes;
		tf_cache_data.format = RD::DATA_FORMAT_R32_UINT;
		tf_cache_data.width = (PROBE_DIVISOR.x + 1) * LIGHTPROBE_OCT_SIZE;
		tf_cache_data.height = (PROBE_DIVISOR.y + 1) * (PROBE_DIVISOR.z + 1) * LIGHTPROBE_OCT_SIZE;
		tf_cache_data.array_layers *= frames_to_converge;
		tf_cache_data.shareable_formats.clear();
		lightprobe_hit_cache_data = create_clear_texture(tf_cache_data, String("HDDAGI Lighprobe Hit Cache"));

		tf_cache_data.format = RD::DATA_FORMAT_R16_UINT;
		lightprobe_hit_cache_version_data = create_clear_texture(tf_cache_data, String("HDDAGI Lighprobe Hit Cache Version"));

		{
			tf_cache_data.format = RD::DATA_FORMAT_R32_UINT;
			if (probe_history_ema) {
				tf_cache_data.array_layers = cascades.size();
			}

			lightprobe_moving_average_history = create_clear_texture(tf_cache_data, String("HDDAGI Lighprobe Moving Average History"));

			RD::TextureFormat tf_moving_average = tf_cache_data;
			tf_moving_average.width *= 3; // no RGB32 UI so..
			tf_moving_average.array_layers = cascades.size(); // Return to just cascades, no history.
			if (probe_history_ema) {
				tf_moving_average.width = 1;
				tf_moving_average.height = 1;
			}
			lightprobe_moving_average = create_clear_texture(tf_moving_average, String("HDDAGI Lighprobe Moving Average"));
		}

		RD::TextureFormat tf_ambient = tf_lightprobes;
		tf_ambient.width = (PROBE_DIVISOR.x + 1);
		tf_ambient.height = (PROBE_DIVISOR.y + 1) * (PROBE_DIVISOR.z + 1);
		tf_ambient.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf_ambient.shareable_formats.clear();
		lightprobe_ambient_tex = create_clear_texture(tf_ambient, String("HDDAGI Ambient Light Texture"));

		RD::TextureFormat tf_neighbours;
		tf_neighbours.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		tf_neighbours.format = RD::DATA_FORMAT_R32_UINT;
		tf_neighbours.width = (PROBE_DIVISOR.x + 1);
		tf_neighbours.height = (PROBE_DIVISOR.y + 1) * (PROBE_DIVISOR.z + 1);
		tf_neighbours.depth = 1;
		tf_neighbours.array_layers = cascades.size();
		tf_neighbours.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		lightprobe_neighbour_visibility_map = create_clear_texture(tf_neighbours, String("HDDAGI Neighbor Visibility Map"));

		RD::TextureFormat tf_geometry_proximity = tf_neighbours;

		tf_geometry_proximity.format = RD::DATA_FORMAT_R8_UNORM;
		tf_geometry_proximity.usage_bits |= RD::TEXTURE_USAGE_SAMPLING_BIT;
		lightprobe_geometry_proximity_map = create_clear_texture(tf_geometry_proximity, String("HDDAGI Geometry Proximity Map"));
		lightprobe_camera_visibility_map = create_clear_texture(tf_geometry_proximity, String("HDDAGI Camera Visibility Map"));

		for (uint32_t i = 0; i < cascades.size(); i++) {
			for (int j = 0; j < 4; j++) {
				lightprobe_camera_buffers.push_back(RD::get_singleton()->uniform_buffer_create(sizeof(HDDAGIShader::IntegrateCameraUBO)));
			}
		}

		RD::TextureFormat tf_process_frame = tf_neighbours;
		tf_process_frame.format = RD::DATA_FORMAT_R32_UINT;
		lightprobe_process_frame = create_clear_texture(tf_process_frame, String("HDDAGI Lightprobe Frame"));
	}

	// Occlusion

	{
		RD::TextureFormat tf_occlusion = tf_base;
		tf_occlusion.format = gi->hddagi_shader.packed_occlusion ? RD::DATA_FORMAT_R16_UINT : RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tf_occlusion.width += 2; // Extra border for proper blending.
		tf_occlusion.height += 2; // Extra border for proper blending.
		tf_occlusion.depth += 2; // Extra border for proper blending.
		tf_occlusion.height *= cascades.size();
		// Transpose X/Y to stack cascades along texture width with less padding.
		SWAP(tf_occlusion.width, tf_occlusion.height);
		if (gi->hddagi_shader.packed_occlusion) {
			tf_occlusion.shareable_formats.push_back(RD::DATA_FORMAT_R4G4B4A4_UNORM_PACK16);
			tf_occlusion.shareable_formats.push_back(RD::DATA_FORMAT_R16_UINT);
		}
		tf_occlusion.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		for (int i = 0; i < 2; i++) {
			occlusion_data[i] = create_clear_texture(tf_occlusion, String("HDDAGI Occlusion ") + itos(i));
			if (gi->hddagi_shader.packed_occlusion) {
				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_R4G4B4A4_UNORM_PACK16;
				occlusion_tex[i] = RD::get_singleton()->texture_create_shared(tv, occlusion_data[i]);
			} else {
				occlusion_tex[i] = occlusion_data[i];
			}
		}
		// Signed receiver-only corrections; storing absolute visibility at half
		// resolution would blur thin occluders. Regenerated with voxel geometry.
		RD::TextureFormat tf_correction = tf_base;
		tf_correction.format = RD::DATA_FORMAT_R8G8B8A8_SNORM;
		tf_correction.width = cascade_size.x / 2 + 2;
		tf_correction.height = (cascade_size.y / 2 + 2) * cascades.size();
		tf_correction.depth = cascade_size.z / 2 + 2;
		SWAP(tf_correction.width, tf_correction.height);
		tf_correction.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
		for (int i = 0; i < 2; i++) {
			occlusion_correction[i] = create_clear_texture(tf_correction, String("HDDAGI Occlusion Correction ") + itos(i));
		}
	}

	for (HDDAGI::Cascade &cascade : cascades) {
		/* 3D Textures */

		cascade.light_process_buffer = RD::get_singleton()->storage_buffer_create(sizeof(HDDAGI::Cascade::LightProcessCell));
		cascade.light_source_emission_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t));
		cascade.light_process_dispatch_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 4, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
		cascade.light_process_dispatch_buffer_copy = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 4, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);

		cascade.light_position_bufer = RD::get_singleton()->storage_buffer_create(sizeof(HDDAGIShader::Light) * MAX(HDDAGI::MAX_STATIC_LIGHTS, HDDAGI::MAX_DYNAMIC_LIGHTS));
		Vector3i region_count = cascade_size / REGION_CELLS;
		cascade.light_region_buffer = RD::get_singleton()->storage_buffer_create(region_count.x * region_count.y * region_count.z * 4 * sizeof(uint32_t));

		cascade.cell_size = base_cell_size;
		Vector3 world_position = p_world_anchor + p_planar_world_forward * (cascade.cell_size * float(cascade_size.x) * p_cascade_forward_offset);
		world_position.y *= y_mult;
		cascade.blend_position = world_position;
		Vector3i probe_cells = cascade_size / REGION_CELLS;
		Vector3 probe_size = Vector3(1, 1, 1) * cascade.cell_size * Vector3(probe_cells);
		Vector3i probe_pos = Vector3i((world_position / probe_size + Vector3(0.5, 0.5, 0.5)).floor());
		cascade.position = probe_pos * probe_cells;

		cascade.dirty_regions = HDDAGI::Cascade::DIRTY_ALL;

		// lightprobes

		base_cell_size *= 2.0;
	}

	bounce_feedback = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_bounce_feedback(p_env);
	energy = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_energy(p_env);
	normal_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_normal_bias(p_env);
	reflection_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_reflection_bias(p_env);
	probe_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_probe_bias(p_env);
	reads_sky = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_read_sky_light(p_env);
}

void GI::HDDAGI::update_voxel_pages(int p_cascade, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const Vector3i &p_region_offset, const PagedArray<RenderGeometryInstance *> &p_instances) {
	const Vector3i region_size = cascade_size / REGION_CELLS;
	const uint32_t region_count = region_size.x * region_size.y * region_size.z;
	const Vector3i first = p_from / REGION_CELLS;
	const Vector3i end = (p_from + p_size + Vector3i(REGION_CELLS - 1, REGION_CELLS - 1, REGION_CELLS - 1)) / REGION_CELLS;
	const Vector3i extent = end - first;
	const Vector3 to_cell = Vector3(p_size) / p_bounds.size;
	Cascade &cascade = cascades[p_cascade];
	LocalVector<uint8_t> needed;
	needed.resize(extent.x * extent.y * extent.z);
	memset(needed.ptr(), 0, needed.size());

	for (uint32_t i = 0; i < p_instances.size(); i++) {
		const RenderGeometryInstanceBase *instance = static_cast<const RenderGeometryInstanceBase *>(p_instances[i]);
		const AABB &bounds = instance->transformed_aabb;
		Vector3 lo = (bounds.position - p_bounds.position) * to_cell + Vector3(p_from);
		Vector3 hi = (bounds.get_end() - p_bounds.position) * to_cell + Vector3(p_from);
		// The surface metadata also covers the cells immediately outside the geometry.
		Vector3i from = Vector3i(((lo - Vector3(1, 1, 1)) / REGION_CELLS).floor()).max(first);
		Vector3i to = (Vector3i(((hi + Vector3(1, 1, 1)) / REGION_CELLS).floor()) + Vector3i(1, 1, 1)).min(end);
		for (int z = from.z; z < to.z; z++) {
			for (int y = from.y; y < to.y; y++) {
				for (int x = from.x; x < to.x; x++) {
					needed[(x - first.x) + extent.x * ((y - first.y) + extent.y * (z - first.z))] = 1;
				}
			}
		}
	}

	uint32_t *pages = voxel_pages.ptrw() + p_cascade * region_count;
	LocalVector<uint32_t> new_pages;
	bool changed = false;
	for (int z = first.z; z < end.z; z++) {
		for (int y = first.y; y < end.y; y++) {
			for (int x = first.x; x < end.x; x++) {
				Vector3i local(x, y, z);
				Vector3i wrapped = local + p_region_offset;
				wrapped = Vector3i(wrapped.x & (region_size.x - 1), wrapped.y & (region_size.y - 1), wrapped.z & (region_size.z - 1));
				uint32_t &page = pages[wrapped.x + region_size.x * (wrapped.z + region_size.z * wrapped.y)];
				bool used = needed[(x - first.x) + extent.x * ((y - first.y) + extent.y * (z - first.z))] != 0;
				Vector3i corner = local * REGION_CELLS;
				Vector3i next = corner + Vector3i(REGION_CELLS, REGION_CELLS, REGION_CELLS);
				if (used && page == 0) {
					if (free_voxel_pages.is_empty()) {
						page = ++voxel_page_count;
					} else {
						page = free_voxel_pages[free_voxel_pages.size() - 1];
						free_voxel_pages.resize(free_voxel_pages.size() - 1);
					}
					new_pages.push_back(page);
					cascade.voxel_page_count++;
					changed = true;
				} else if (!used && page != 0 && corner.min(p_from) == p_from && next.max(p_from + p_size) == p_from + p_size) {
					free_voxel_pages.push_back(page);
					page = 0;
					cascade.voxel_page_count--;
					changed = true;
				}
			}
		}
	}

	RD *rd = RD::get_singleton();
	const uint32_t page_bytes = REGION_CELLS * REGION_CELLS * REGION_CELLS * 2 * sizeof(uint32_t);
	if (voxel_page_count > voxel_page_capacity) {
		uint32_t capacity = MIN(uint32_t(voxel_pages.size()), Math::next_power_of_2(voxel_page_count));
		RID data = rd->storage_buffer_create(capacity * page_bytes);
		rd->set_resource_name(data, "HDDAGI Surface Voxels");
		rd->buffer_copy(voxel_data, data, 0, 0, voxel_page_capacity * page_bytes);
		rd->free_rid(voxel_data);
		voxel_data = data;
		voxel_page_capacity = capacity;
	}
	if (changed) {
		rd->buffer_update(voxel_page_table, p_cascade * region_count * sizeof(uint32_t), region_count * sizeof(uint32_t), pages);
		new_pages.sort();
		for (uint32_t i = 0; i < new_pages.size();) {
			uint32_t begin = i++;
			while (i < new_pages.size() && new_pages[i] == new_pages[i - 1] + 1) {
				i++;
			}
			rd->buffer_clear(voxel_data, (new_pages[begin] - 1) * page_bytes, (i - begin) * page_bytes);
		}
	}

	uint32_t required = MIN(solid_cell_count, MAX(1u, cascade.voxel_page_count * REGION_CELLS * REGION_CELLS * REGION_CELLS));
	uint32_t capacity = Math::next_power_of_2(required);
	if (capacity > light_capacity_render || capacity < light_capacity_render / 2) {
		rd->free_rid(light_process_buffer_render);
		rd->free_rid(light_source_emission_buffer_render);
		light_process_buffer_render = rd->storage_buffer_create(capacity * sizeof(Cascade::LightProcessCell));
		light_source_emission_buffer_render = rd->storage_buffer_create(capacity * sizeof(uint32_t));
		light_capacity_render = capacity;
	}
}

void GI::HDDAGI::render_region(Ref<RenderSceneBuffersRD> p_render_buffers, int p_region, const PagedArray<RenderGeometryInstance *> &p_instances, float p_exposure_normalization) {
	ERR_FAIL_COND(p_render_buffers.is_null()); // we wouldn't be here if this failed but...
	AABB bounds;
	Vector3i from;
	Vector3i size;
	Vector3i scroll;
	Vector3i region_ofs;

	int cascade = get_pending_region_data(p_region, from, size, bounds, scroll, region_ofs);
	ERR_FAIL_COND(cascade < 0);

	update_voxel_pages(cascade, from, size, bounds, region_ofs, p_instances);

	// Initialize render targets.
	//@TODO: Add a clear region to RenderingDevice to optimize this part
	RD::get_singleton()->texture_clear(render_albedo, Color(0, 0, 0, 0), 0, 1, 0, 1);
	RD::get_singleton()->texture_clear(render_emission, Color(0, 0, 0, 0), 0, 1, 0, 1);
	RD::get_singleton()->texture_clear(render_emission_aniso, Color(0, 0, 0, 0), 0, 1, 0, 1);
	RD::get_singleton()->texture_clear(render_aniso_normals, Color(0, 0, 0, 0), 0, 1, 0, 1);
	const uint32_t dispatch[4] = { 0, 1, 1, 0 };
	RD::get_singleton()->buffer_update(light_process_dispatch_buffer_render, 0, sizeof(dispatch), dispatch);

	if (scroll == HDDAGI::Cascade::DIRTY_ALL) {
		RD::get_singleton()->texture_clear(lightprobe_hit_cache_data, Color(0, 0, 0, 0), 0, 1, cascade * frames_to_converge, frames_to_converge);
		RD::get_singleton()->texture_clear(lightprobe_hit_cache_version_data, Color(0, 0, 0, 0), 0, 1, cascade * frames_to_converge, frames_to_converge);
		RD::get_singleton()->texture_clear(lightprobe_moving_average_history, Color(0, 0, 0, 0), 0, 1, cascade * (probe_history_ema ? 1 : frames_to_converge), probe_history_ema ? 1 : frames_to_converge);
		RD::get_singleton()->texture_clear(lightprobe_moving_average, Color(0, 0, 0, 0), 0, 1, cascade, 1);
		RD::get_singleton()->texture_clear(lightprobe_specular_data, Color(0, 0, 0, 0), 0, 1, cascade, 1);
		RD::get_singleton()->texture_clear(lightprobe_diffuse_data, Color(0, 0, 0, 0), 0, 1, cascade, 1);
		RD::get_singleton()->texture_clear(lightprobe_process_frame, Color(0, 0, 0, 0), 0, 1, cascade, 1);
	}

	RendererSceneRenderRD::get_singleton()->_render_hddagi(p_render_buffers, from, size, bounds, p_instances, render_albedo, render_emission, render_emission_aniso, render_aniso_normals, p_exposure_normalization);

	RD::get_singleton()->draw_command_begin_label("HDDAGI Create Cascade SDF");

	RENDER_TIMESTAMP("> HDDAGI Update SDF");
	//clear dispatch indirect data

	HDDAGIShader::PreprocessPushConstant push_constant;
	memset(&push_constant, 0, sizeof(HDDAGIShader::PreprocessPushConstant));

	Vector3i dispatch_size;
	//scroll
	if (scroll != HDDAGI::Cascade::DIRTY_ALL) {
		//for scroll
		push_constant.scroll[0] = scroll.x;
		push_constant.scroll[1] = scroll.y;
		push_constant.scroll[2] = scroll.z;

		for (int i = 0; i < 3; i++) {
			if (scroll[i] > 0) {
				push_constant.offset[i] = 0;
				push_constant.limit[i] = scroll[i];
				dispatch_size[i] = scroll[i];
			} else if (scroll[i] < 0) {
				push_constant.offset[i] = cascade_size[i] + scroll[i];
				push_constant.limit[i] = cascade_size[i];
				dispatch_size[i] = -scroll[i];
			} else {
				push_constant.offset[i] = 0;
				push_constant.limit[i] = cascade_size[i];
				dispatch_size[i] = cascade_size[i];
			}
		}

	} else {
		//for no scroll
		push_constant.scroll[0] = 0;
		push_constant.scroll[1] = 0;
		push_constant.scroll[2] = 0;

		push_constant.offset[0] = 0;
		push_constant.offset[1] = 0;
		push_constant.offset[2] = 0;

		push_constant.limit[0] = cascade_size[0];
		push_constant.limit[1] = cascade_size[1];
		push_constant.limit[2] = cascade_size[2];

		dispatch_size.x = cascade_size[0];
		dispatch_size.y = cascade_size[1];
		dispatch_size.z = cascade_size[2];
	}
	push_constant.grid_size[0] = cascade_size[0];
	push_constant.grid_size[1] = cascade_size[1];
	push_constant.grid_size[2] = cascade_size[2];

	Vector3i probe_axis_count = cascade_size / REGION_CELLS + Vector3i(1, 1, 1);

	push_constant.probe_axis_size[0] = probe_axis_count[0];
	push_constant.probe_axis_size[1] = probe_axis_count[1];
	push_constant.probe_axis_size[2] = probe_axis_count[2];

	push_constant.region_world_pos[0] = region_ofs[0];
	push_constant.region_world_pos[1] = region_ofs[1];
	push_constant.region_world_pos[2] = region_ofs[2];

	push_constant.maximum_light_cells = MIN(solid_cell_count, light_capacity_render);
	push_constant.cascade_count = cascades.size();

	push_constant.ray_hit_cache_frames = frames_to_converge;

	static const uint32_t frames_to_update_table[RSE::ENV_HDDAGI_INACTIVE_PROBE_MAX] = {
		1, 2, 4, 8
	};

	push_constant.probe_update_frames = frames_to_update_table[gi->inactive_probe_frames];
	push_constant.history_mode = probe_history_ema;

	push_constant.cascade = cascade;
	push_constant.occlusion_offset = 0;

	cascades[cascade].latest_version++;

	push_constant.region_version = cascades[cascade].latest_version;

	RENDER_TIMESTAMP("HDDAGI Scroll");

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	if (scroll != HDDAGI::Cascade::DIRTY_ALL) {
		// Scroll happened

		{ // Light Scroll
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_LIGHT_SCROLL]);
			RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
					gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_LIGHT_SCROLL],
					0,
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, voxel_data),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 12, voxel_page_table),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, light_process_dispatch_buffer_render),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, light_process_buffer_render),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 7, cascades[cascade].light_process_dispatch_buffer_copy),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, cascades[cascade].light_process_buffer),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 13, light_source_emission_buffer_render),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 14, cascades[cascade].light_source_emission_buffer)

			);

			HDDAGIShader::PreprocessPushConstant push_constant_scroll = push_constant;

			for (int i = 0; i < 3; i++) {
				if (scroll[i] > 0) {
					push_constant_scroll.limit[i] = cascade_size[i] - scroll[i];
					push_constant_scroll.offset[i] = 1; // Include one extra voxel for consistency with neighboring voxels.
				} else if (scroll[i] < 0) {
					push_constant_scroll.limit[i] = cascade_size[i] - 1; // -1 because one extra is rendered below for consistency with neighboring voxels.
					push_constant_scroll.offset[i] = -scroll[i];
				} else {
					push_constant_scroll.limit[i] = cascade_size[i];
					push_constant_scroll.offset[i] = 0;
				}
			}

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant_scroll, sizeof(HDDAGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch_indirect(compute_list, cascades[cascade].light_process_dispatch_buffer, 0);
		}

		{ // Probe Scroll
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_SCROLL]);

			RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
					gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_SCROLL],
					0,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, lightprobe_specular_data),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, lightprobe_diffuse_data),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, lightprobe_ambient_tex),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, lightprobe_hit_cache_data),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, lightprobe_moving_average_history),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, lightprobe_moving_average),
					RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 7, get_lightprobe_occlusion_textures()),
					RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 8, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

			Vector3i dispatch_cells = dispatch_size / REGION_CELLS + Vector3i(1, 1, 1);

			HDDAGIShader::PreprocessPushConstant push_constant_scroll = push_constant;
			push_constant_scroll.upper_cascade = -1;
			if (cascade + 1 < int(cascades.size())) {
				const Cascade &upper = cascades[cascade + 1];
				if (upper.dirty_regions != Cascade::DIRTY_ALL) {
					// The parent atlas still contains its pre-scroll coverage.
					const Vector3i upper_region = (upper.position + upper.dirty_regions) / REGION_CELLS;
					push_constant_scroll.upper_cascade = cascade + 1;
					for (int axis = 0; axis < 3; axis++) {
						push_constant_scroll.upper_region_world_pos[axis] = upper_region[axis];
					}
				}
			}

			for (int i = 0; i < 3; i++) {
				if (scroll[i] < 0) {
					push_constant_scroll.offset[i] += REGION_CELLS;
					dispatch_cells[i]--;
				} else if (scroll[i] > 0) {
					dispatch_cells[i]--;
				}
			}

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant_scroll, sizeof(HDDAGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch(compute_list, dispatch_cells.x, dispatch_cells.y, dispatch_cells.z);
		}

		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	{ // Occlusion
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_OCCLUSION]);

		for (int i = 0; i < 2; i++) {
			RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
					gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_OCCLUSION],
					0,
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, render_aniso_normals),
					RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, occlusion_data[i]),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, gi->hddagi_shader.integrate_lookup_buffer));

			push_constant.occlusion_offset = (i - 1) * 4; // Turns out Z needs to be swapped. I have no idea why. If you figure it out, let me know.
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::PreprocessPushConstant));
			RD::get_singleton()->compute_list_dispatch(compute_list, dispatch_size.x / REGION_CELLS, dispatch_size.y / REGION_CELLS, dispatch_size.z / REGION_CELLS);
		}
	}

	{ // Region Store
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_REGION_STORE]);

		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_REGION_STORE],
				0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, render_aniso_normals),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, voxel_bits_tex),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, voxel_region_tex),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, region_version_data));

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, dispatch_size.x, dispatch_size.y, dispatch_size.z);
	}

	RD::get_singleton()->compute_list_add_barrier(compute_list); // store needs another barrier

	RENDER_TIMESTAMP("HDDAGI Store SDF");

	{
		// Storing light happens last (after barrier) because it needs occlusion information.
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_LIGHT_STORE]);
		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_LIGHT_STORE],
				0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, render_albedo),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, render_emission),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, render_emission_aniso),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, render_aniso_normals),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, light_process_dispatch_buffer_render),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, light_process_buffer_render),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 7, lightprobe_neighbour_visibility_map),
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 8, get_lightprobe_occlusion_textures()),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 9, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 11, voxel_page_table),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 12, voxel_data),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 13, light_source_emission_buffer_render));

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

		Vector3i store_size = dispatch_size;

		if (scroll != HDDAGI::Cascade::DIRTY_ALL) {
			for (int i = 0; i < 3; i++) {
				if (scroll[i] > 0) {
					push_constant.offset[i] = 0;
					push_constant.limit[i] = scroll[i] + 1; // Include an extra voxel to store light at the boundary.
					store_size[i] += 1;
				} else if (scroll[i] < 0) {
					push_constant.offset[i] = cascade_size[i] + scroll[i] - 1;
					push_constant.limit[i] = cascade_size[i];
					store_size[i] += 1;
				} else {
					push_constant.offset[i] = 0;
					push_constant.limit[i] = cascade_size[i];
				}
			}
		}

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, store_size.x, store_size.y, store_size.z);
		// Store processed ones into cascade
		SWAP(light_process_dispatch_buffer_render, cascades[cascade].light_process_dispatch_buffer);
		SWAP(light_process_buffer_render, cascades[cascade].light_process_buffer);
		SWAP(light_capacity_render, cascades[cascade].light_capacity);
		SWAP(light_source_emission_buffer_render, cascades[cascade].light_source_emission_buffer);

		cascades[cascade].static_lights_dirty = true;
		cascades[cascade].dynamic_lights_dirty = true;
	}

	auto timestamp_store_pass = [&](const char *p_name) {
		if (RSG::utilities->capturing_timestamps) {
			RD::get_singleton()->compute_list_end();
			RSG::utilities->capture_timestamp(p_name);
			compute_list = RD::get_singleton()->compute_list_begin();
		}
	};
	timestamp_store_pass("HDDAGI Store Probe Neighbors");
	{ // Probe Neighbors (no barrier needed)
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_NEIGHBOURS]);

		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_NEIGHBOURS],
				0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, get_lightprobe_occlusion_textures()),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, lightprobe_neighbour_visibility_map)

		);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

		Vector3i dispatch_cells = dispatch_size / REGION_CELLS + Vector3i(1, 1, 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, dispatch_cells.x, dispatch_cells.y, dispatch_cells.z);
	}

	timestamp_store_pass("HDDAGI Store Probe Proximity");
	{ // Probe geometry proximity (no barrier needed)
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_GEOMETRY_PROXIMITY]);

		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_GEOMETRY_PROXIMITY],
				0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, voxel_region_tex),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, lightprobe_geometry_proximity_map),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, voxel_bits_tex));

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

		Vector3i dispatch_cells = cascade_size / REGION_CELLS + Vector3i(1, 1, 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, dispatch_cells.x, dispatch_cells.y, dispatch_cells.z);
	}

	timestamp_store_pass("HDDAGI Store Probe Frames");
	{ // Probe frames
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_UPDATE_FRAMES]);

		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_LIGHTPROBE_UPDATE_FRAMES],
				0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, lightprobe_process_frame));

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

		Vector3i dispatch_cells = cascade_size / REGION_CELLS + Vector3i(1, 1, 1);

		HDDAGIShader::PreprocessPushConstant push_constant_scroll = push_constant;

		if (scroll != HDDAGI::Cascade::DIRTY_ALL) {
			// Only edge if not all dirty
			for (int i = 0; i < 3; i++) {
				if (scroll[i] < 0) {
					push_constant_scroll.offset[i] += REGION_CELLS;
					dispatch_cells[i]--;
				} else if (scroll[i] > 0) {
					dispatch_cells[i]--;
				}
			}
		}

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant_scroll, sizeof(HDDAGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch(compute_list, dispatch_cells.x, dispatch_cells.y, dispatch_cells.z);
	}

	RD::get_singleton()->compute_list_end();

	// A separate list resolves the raw occlusion and neighbor metadata writes.
	RENDER_TIMESTAMP("HDDAGI Occlusion Correction");
	{
		RD::ComputeListID filter_list = RD::get_singleton()->compute_list_begin();
		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.preprocess_shader_version[HDDAGIShader::PRE_PROCESS_OCCLUSION_FILTER], 0,
				RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, get_lightprobe_occlusion_textures()),
				RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 2, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, voxel_data),
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, voxel_page_table),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, occlusion_correction[0]),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, occlusion_correction[1]));
		RD::get_singleton()->compute_list_bind_compute_pipeline(filter_list, gi->hddagi_shader.preprocess_pipeline[HDDAGIShader::PRE_PROCESS_OCCLUSION_FILTER]);
		RD::get_singleton()->compute_list_bind_uniform_set(filter_list, uniform_set, 0);
		Vector3i filter_size = cascade_size / 2 + Vector3i(2, 2, 2);
		if (scroll != HDDAGI::Cascade::DIRTY_ALL) {
			// Include the two-cell filter footprint on both sides of the toroidal dirty region.
			const Vector3i halo(2, 2, 2);
			const Vector3i filter_from = from / 2 - halo;
			filter_size = ((from + size) / 2 + halo - filter_from).min(cascade_size / 2);
			for (int axis = 0; axis < 3; axis++) {
				push_constant.offset[axis] = filter_from[axis];
				push_constant.limit[axis] = filter_from[axis] + filter_size[axis];
			}
		}
		RD::get_singleton()->compute_list_set_push_constant(filter_list, &push_constant, sizeof(HDDAGIShader::PreprocessPushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(filter_list, filter_size.x, filter_size.y, filter_size.z);
		RD::get_singleton()->compute_list_end();
	}

	RENDER_TIMESTAMP("HDDAGI Occlusion Correction Done");

	RD::get_singleton()->buffer_copy(cascades[cascade].light_process_dispatch_buffer, cascades[cascade].light_process_dispatch_buffer_copy, 0, 0, sizeof(uint32_t) * 4);

	cascades[cascade].baked_exposure_normalization = p_exposure_normalization;

	RD::get_singleton()->draw_command_end_label();
	RENDER_TIMESTAMP("< HDDAGI Update SDF");
}

void GI::HDDAGI::free_data() {
	// we don't free things here, we handle HDDAGI differently at the moment destructing the object when it needs to change.
}

GI::HDDAGI::~HDDAGI() {
	for (const HDDAGI::Cascade &c : cascades) {
		RD::get_singleton()->free_rid(c.light_process_buffer);
		RD::get_singleton()->free_rid(c.light_source_emission_buffer);
		RD::get_singleton()->free_rid(c.light_process_dispatch_buffer);
		RD::get_singleton()->free_rid(c.light_process_dispatch_buffer_copy);
		RD::get_singleton()->free_rid(c.light_position_bufer);
		RD::get_singleton()->free_rid(c.light_region_buffer);
	}

	RD::get_singleton()->free_rid(voxel_page_table);
	RD::get_singleton()->free_rid(voxel_data);
	RD::get_singleton()->free_rid(light_process_buffer_render);
	RD::get_singleton()->free_rid(light_source_emission_buffer_render);
	RD::get_singleton()->free_rid(light_process_dispatch_buffer_render);

	if (cascades_ubo.is_valid()) {
		RD::get_singleton()->free_rid(cascades_ubo);
	}
	if (debug_probes_scene_data_ubo.is_valid()) {
		RD::get_singleton()->free_rid(debug_probes_scene_data_ubo);
	}

	for (int i = 0; i < lightprobe_camera_buffers.size(); i++) {
		RD::get_singleton()->free_rid(lightprobe_camera_buffers[i]);
	}
	lightprobe_camera_buffers.clear();

	const RID textures[] = {
		render_albedo,
		render_aniso_normals,
		render_emission,
		render_emission_aniso,
		voxel_bits_tex,
		voxel_region_tex,
		region_version_data,
		lightprobe_specular_data,
		lightprobe_diffuse_data,
		lightprobe_ambient_tex,
		lightprobe_diffuse_filter_data,
		lightprobe_hit_cache_data,
		lightprobe_hit_cache_version_data,
		lightprobe_moving_average,
		lightprobe_moving_average_history,
		lightprobe_neighbour_visibility_map,
		lightprobe_geometry_proximity_map,
		lightprobe_camera_visibility_map,
		lightprobe_process_frame,
		occlusion_data[0],
		occlusion_data[1],
		occlusion_correction[0],
		occlusion_correction[1],
	};
	for (RID texture : textures) {
		if (texture.is_valid() && RD::get_singleton()->texture_is_valid(texture)) {
			RD::get_singleton()->free_rid(texture);
		}
	}
}

void GI::HDDAGI::update(RID p_env, const Vector3 &p_world_anchor, const Vector3 &p_planar_world_forward, float p_cascade_forward_offset) {
	bounce_feedback = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_bounce_feedback(p_env);
	energy = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_energy(p_env);
	reads_sky = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_read_sky_light(p_env);

	using_probe_filter = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_filter_probes(p_env);
	using_reflection_filter = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_filter_reflection(p_env);
	using_ambient_filter = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_filter_ambient(p_env);
	reflection_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_reflection_bias(p_env);
	normal_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_normal_bias(p_env);
	probe_bias = RendererSceneRenderRD::get_singleton()->environment_get_hddagi_probe_bias(p_env);

	int32_t drag_margin = REGION_CELLS / 2;

	int idx = 0;
	for (HDDAGI::Cascade &cascade : cascades) {
		cascade.dirty_regions = Vector3i();

		Vector3 probe_half_size = Vector3(1, 1, 1) * cascade.cell_size * float(REGION_CELLS) * 0.5;
		probe_half_size = Vector3(0, 0, 0);

		Vector3 world_position = p_world_anchor + p_planar_world_forward * (cascade.cell_size * float(cascade_size.x) * p_cascade_forward_offset);
		world_position.y *= y_mult;
		cascade.blend_position = world_position;
		Vector3i pos_in_cascade = Vector3i((world_position + probe_half_size) / cascade.cell_size);

		for (int j = 0; j < 3; j++) {
			if (pos_in_cascade[j] < cascade.position[j]) {
				while (pos_in_cascade[j] < (cascade.position[j] - drag_margin)) {
					cascade.position[j] -= drag_margin * 2;
					cascade.dirty_regions[j] += drag_margin * 2;
				}
			} else if (pos_in_cascade[j] > cascade.position[j]) {
				while (pos_in_cascade[j] > (cascade.position[j] + drag_margin)) {
					cascade.position[j] += drag_margin * 2;
					cascade.dirty_regions[j] -= drag_margin * 2;
				}
			}

			if (cascade.dirty_regions[j] == 0) {
				continue; // not dirty
			} else if (uint32_t(Math::abs(cascade.dirty_regions[j])) >= uint32_t(cascade_size[j])) {
				//moved too much, just redraw everything (make all dirty)
				cascade.dirty_regions = HDDAGI::Cascade::DIRTY_ALL;
				break;
			}
		}

		if (cascade.dirty_regions != Vector3i() && cascade.dirty_regions != HDDAGI::Cascade::DIRTY_ALL) {
			//see how much the total dirty volume represents from the total volume
			uint32_t total_volume = cascade_size.x * cascade_size.y * cascade_size.z;
			uint32_t safe_volume = 1;
			for (int j = 0; j < 3; j++) {
				safe_volume *= cascade_size[j] - Math::abs(cascade.dirty_regions[j]);
			}
			uint32_t dirty_volume = total_volume - safe_volume;
			if (dirty_volume > (safe_volume / 2)) {
				//more than half the volume is dirty, make all dirty so its only rendered once
				cascade.dirty_regions = HDDAGI::Cascade::DIRTY_ALL;
			}
		}

		if (cascade.dirty_regions != Vector3i()) {
			uint32_t dirty_mask = 0;
			for (int j = 0; j < 3; j++) {
				if (cascade.dirty_regions[j] != 0) {
					dirty_mask |= (1 << j);
				}
			}
			// Notify all cascades equal or smaller than this that some motion happened.
			// In an axis, which is used for light ray cache invalidation.

			for (int j = idx; j >= 0; j--) {
				cascades[j].motion_accum |= dirty_mask;
			}
		}

		idx++;
	}

	update_frame++;
}

void GI::HDDAGI::update_light() {
	RD::get_singleton()->draw_command_begin_label("HDDAGI Update dynamic Light");
	RENDER_TIMESTAMP("HDDAGI Surface Lighting Begin");

	/* Update dynamic light */

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	const int direct_light_mode = HDDAGIShader::DIRECT_LIGHT_MODE_DYNAMIC;
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.direct_light_pipeline[direct_light_mode]);

	HDDAGIShader::DirectLightPushConstant push_constant = {};

	push_constant.grid_size[0] = cascade_size[0];
	push_constant.grid_size[1] = cascade_size[1];
	push_constant.grid_size[2] = cascade_size[2];
	push_constant.max_cascades = cascades.size();
	push_constant.probe_axis_size[0] = cascade_size[0] / REGION_CELLS + 1;
	push_constant.probe_axis_size[1] = cascade_size[1] / REGION_CELLS + 1;
	push_constant.probe_axis_size[2] = cascade_size[2] / REGION_CELLS + 1;

	push_constant.bounce_feedback = bounce_feedback;
	push_constant.y_mult = y_mult;
	push_constant.use_occlusion = using_probe_filter;
	push_constant.probe_cell_size = REGION_CELLS;

	RID area_light_atlas_uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
			gi->hddagi_shader.direct_light_shader_version[direct_light_mode],
			1,
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, RendererRD::TextureStorage::get_singleton()->area_light_atlas_get_texture()),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));

	for (uint32_t i = 0; i < cascades.size(); i++) {
		HDDAGI::Cascade &cascade = cascades[i];
		push_constant.light_count = cascade_dynamic_light_count[i];
		push_constant.cascade = i;
		push_constant.dirty_dynamic_update = cascades[i].dynamic_lights_dirty;
		push_constant.use_light_regions = cascade.use_light_regions;

		cascades[i].dynamic_lights_dirty = false;

		if ((cascade.dynamic_light_changed && bool(GLOBAL_GET("rendering/global_illumination/hddagi/fast_light_updates"))) || gi->hddagi_frames_to_update_light == RSE::ENV_HDDAGI_UPDATE_LIGHT_IN_1_FRAME) {
			push_constant.process_offset = 0;
			push_constant.process_increment = 1;
		} else {
			static const uint32_t frames_to_update_table[RSE::ENV_HDDAGI_UPDATE_LIGHT_MAX] = {
				1, 2, 4, 8, 16
			};

			uint32_t frames_to_update = frames_to_update_table[gi->hddagi_frames_to_update_light];

			push_constant.process_offset = RSG::rasterizer->get_frame_number() % frames_to_update;
			push_constant.process_increment = frames_to_update;
		}

		LocalVector<RD::Uniform> direct_light_uniforms;
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, voxel_bits_tex));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, voxel_region_tex));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 3, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, cascade.light_process_dispatch_buffer_copy));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, cascade.light_process_buffer));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, voxel_data));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 14, voxel_page_table));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, cascades_ubo));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, cascade.light_position_bufer));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 9, lightprobe_diffuse_tex));
		direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 13, cascade.light_region_buffer));

		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache_vec(
				gi->hddagi_shader.direct_light_shader_version[direct_light_mode], 0, direct_light_uniforms);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, area_light_atlas_uniform_set, 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::DirectLightPushConstant));
		RD::get_singleton()->compute_list_dispatch_indirect(compute_list, cascade.light_process_dispatch_buffer, 0);
	}

	RD::get_singleton()->compute_list_end();
	RENDER_TIMESTAMP("HDDAGI Surface Lighting End");
	RD::get_singleton()->draw_command_end_label();
}

Color GI::get_solid_background_radiance(RID p_environment, RSE::EnvironmentBG p_background) {
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
	const Color color = p_background == RSE::ENV_BG_CLEAR_COLOR ? RSG::texture_storage->get_default_clear_color() : scene_render->environment_get_bg_color(p_environment);
	return color.srgb_to_linear() * (scene_render->environment_get_bg_energy_multiplier(p_environment) * scene_render->environment_get_bg_intensity(p_environment));
}

void GI::HDDAGI::update_probes(RID p_env, RID p_camera_attributes, SkyRD::Sky *p_sky, uint32_t p_view_count, const Projection *p_projections, const Transform3D &p_cam_transform) {
	RD::get_singleton()->draw_command_begin_label("HDDAGI Update Probes");
	RENDER_TIMESTAMP("HDDAGI World Probes Begin");

	HDDAGIShader::IntegratePushConstant push_constant = {};
	push_constant.grid_size[0] = cascade_size[0];
	push_constant.grid_size[1] = cascade_size[1];
	push_constant.grid_size[2] = cascade_size[2];
	push_constant.max_cascades = cascades.size();
	push_constant.probe_axis_size[0] = cascade_size[0] / REGION_CELLS + 1;
	push_constant.probe_axis_size[1] = cascade_size[1] / REGION_CELLS + 1;
	push_constant.probe_axis_size[2] = cascade_size[2] / REGION_CELLS + 1;

	static const uint32_t frames_to_update_table[RSE::ENV_HDDAGI_INACTIVE_PROBE_MAX] = {
		1, 2, 4, 8
	};
	push_constant.inactive_update_frames = frames_to_update_table[gi->inactive_probe_frames];
	push_constant.global_frame = RSG::rasterizer->get_frame_number();
	push_constant.history_size = frames_to_converge;
	push_constant.history_mode = probe_history_ema;
	push_constant.history_response = CLAMP(float(GLOBAL_GET("rendering/global_illumination/hddagi/world_probe_light_response")), 0.0f, 1.0f);
	push_constant.ray_bias = probe_bias;
	push_constant.store_ambient_texture = RendererSceneRenderRD::get_singleton()->environment_get_volumetric_fog_enabled(p_env);
	push_constant.y_mult = y_mult;

	const float sky_irradiance_border_size = p_sky != nullptr ? p_sky->uv_border_size : 0.0f;
	push_constant.sky_irradiance_border_size[0] = sky_irradiance_border_size;
	push_constant.sky_irradiance_border_size[1] = 1.0f - sky_irradiance_border_size * 2.0f;

	RID integrate_sky_uniform_set = gi->hddagi_shader.integrate_default_sky_uniform_set;
	int32_t probe_divisor = REGION_CELLS;

	if (reads_sky && p_env.is_valid()) {
		RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
		const RSE::EnvironmentBG background = scene_render->environment_get_background(p_env);
		if (background == RSE::ENV_BG_CLEAR_COLOR || background == RSE::ENV_BG_COLOR) {
			const Color color = get_solid_background_radiance(p_env, background);
			push_constant.sky_flags |= HDDAGIShader::IntegratePushConstant::SKY_FLAGS_MODE_COLOR;
			push_constant.sky_energy = p_camera_attributes.is_valid() ? RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_camera_attributes) : 1.0f;
			push_constant.sky_color_or_orientation[0] = color.r;
			push_constant.sky_color_or_orientation[1] = color.g;
			push_constant.sky_color_or_orientation[2] = color.b;
		} else if (background == RSE::ENV_BG_SKY) {
			if (p_sky && p_sky->radiance.is_valid()) {
				integrate_sky_uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
						gi->hddagi_shader.integrate.version_get_shader(gi->hddagi_shader.integrate_shader, 0),
						1,
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, p_sky->radiance),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));

				push_constant.sky_flags |= HDDAGIShader::IntegratePushConstant::SKY_FLAGS_MODE_SKY;
				push_constant.sky_energy = scene_render->environment_get_bg_energy_multiplier(p_env);

				// Encode sky orientation as a quaternion in the existing push constant slots.
				const Basis sky_basis = scene_render->environment_get_sky_orientation(p_env);
				const Quaternion sky_quaternion = sky_basis.get_quaternion().inverse();
				push_constant.sky_color_or_orientation[0] = sky_quaternion.x;
				push_constant.sky_color_or_orientation[1] = sky_quaternion.y;
				push_constant.sky_color_or_orientation[2] = sky_quaternion.z;
				// Sky contribution to GI is low frequency, so reconstructing W is sufficiently precise.
				push_constant.sky_flags |= HDDAGIShader::IntegratePushConstant::SKY_FLAGS_ORIENTATION_SIGN * (sky_quaternion.w < 0.0 ? 0 : 1);
			}
		}
	}

	{
		RD::get_singleton()->texture_clear(lightprobe_camera_visibility_map, Color(0, 0, 0, 0), 0, 1, 0, cascades.size());

		// Setup buffers first (must be done outside compute list).
		for (uint32_t i = 0; i < cascades.size(); i++) {
			Vector3 cascade_pos = Vector3((Vector3i(1, 1, 1) * -(cascade_size / 2) + cascades[i].position)) * cascades[i].cell_size;
			float cascade_to_cell = 1.0 / cascades[i].cell_size;
			Transform3D local_xform = p_cam_transform;
			local_xform.origin -= cascade_pos;
			local_xform.scale(Vector3(1, y_mult, 1) * cascade_to_cell);

			for (uint32_t j = 0; j < p_view_count; j++) {
				Vector<Plane> planes = p_projections[j].get_projection_planes(local_xform);
				HDDAGIShader::IntegrateCameraUBO camera_ubo;
				for (int k = 0; k < planes.size(); k++) {
					Plane plane = planes[k];
					camera_ubo.planes[k * 4 + 0] = plane.normal.x;
					camera_ubo.planes[k * 4 + 1] = plane.normal.y;
					camera_ubo.planes[k * 4 + 2] = plane.normal.z;
					camera_ubo.planes[k * 4 + 3] = plane.d;
				}
				Vector3 endpoints[8];
				p_projections[j].get_endpoints(local_xform, endpoints);
				for (int k = 0; k < 8; k++) {
					Vector3 p = endpoints[k];
					camera_ubo.points[k * 4 + 0] = p.x;
					camera_ubo.points[k * 4 + 1] = p.y;
					camera_ubo.points[k * 4 + 2] = p.z;
					camera_ubo.points[k * 4 + 3] = 1;
				}
				int buffer_index = i * 4 + j;
				RD::get_singleton()->buffer_update(lightprobe_camera_buffers[buffer_index], 0, sizeof(HDDAGIShader::IntegrateCameraUBO), &camera_ubo);
			}
		}

		// Do visibility testing (all cascades and views in parallel).

		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.integrate_pipeline[HDDAGIShader::INTEGRATE_MODE_CAMERA_VISIBILITY]);

		for (uint32_t i = 0; i < cascades.size(); i++) {
			push_constant.cascade = i;
			push_constant.world_offset[0] = cascades[i].position.x / probe_divisor;
			push_constant.world_offset[1] = cascades[i].position.y / probe_divisor;
			push_constant.world_offset[2] = cascades[i].position.z / probe_divisor;

			for (uint32_t j = 0; j < p_view_count; j++) {
				int buffer_index = i * 4 + j;

				RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
						gi->hddagi_shader.integrate_shader_version[HDDAGIShader::INTEGRATE_MODE_CAMERA_VISIBILITY],
						0,
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, lightprobe_camera_visibility_map),
						RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 2, lightprobe_camera_buffers[buffer_index]));

				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::IntegratePushConstant));
				Vector3i dispatch_threads = cascade_size / REGION_CELLS;
				RD::get_singleton()->compute_list_dispatch_threads(compute_list, dispatch_threads.x, dispatch_threads.y, dispatch_threads.z);
			}
		}
		RD::get_singleton()->compute_list_end();
	}

	Vector3i probe_axis_count = cascade_size / REGION_CELLS + Vector3i(1, 1, 1);
	RENDER_TIMESTAMP("HDDAGI World Probe Integrate/Filter Begin");
	render_pass++;

	LocalVector<RD::Uniform> integrate_uniforms = {
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, voxel_bits_tex),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, voxel_region_tex),
		RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, voxel_data),
		RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 18, voxel_page_table),
		RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 4, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, lightprobe_specular_data),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, lightprobe_diffuse_data),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 7, lightprobe_ambient_tex),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 8, lightprobe_hit_cache_data),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 9, lightprobe_hit_cache_version_data),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 10, region_version_data),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 11, lightprobe_moving_average_history),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 12, lightprobe_moving_average),
		RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 13, cascades_ubo),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 14, lightprobe_process_frame),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 15, lightprobe_geometry_proximity_map),
		RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 16, lightprobe_camera_visibility_map),
		RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 17, gi->hddagi_shader.integrate_lookup_buffer)
	};
	RID integrate_uniform_set = UniformSetCacheRD::get_singleton()->get_cache_vec(gi->hddagi_shader.integrate_shader_version[HDDAGIShader::INTEGRATE_MODE_PROCESS], 0, integrate_uniforms);

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.integrate_pipeline[HDDAGIShader::INTEGRATE_MODE_PROCESS]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, integrate_uniform_set, 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, integrate_sky_uniform_set, 1);

	for (uint32_t i = 0; i < cascades.size(); i++) {
		push_constant.cascade = i;
		push_constant.motion_accum = cascades[i].motion_accum;
		cascades[i].motion_accum = 0; // Clear after use.

		push_constant.world_offset[0] = cascades[i].position.x / probe_divisor;
		push_constant.world_offset[1] = cascades[i].position.y / probe_divisor;
		push_constant.world_offset[2] = cascades[i].position.z / probe_divisor;

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::IntegratePushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_axis_count.x * LIGHTPROBE_OCT_SIZE, probe_axis_count.y * probe_axis_count.z * LIGHTPROBE_OCT_SIZE, 1);
	}

	if (using_probe_filter) {
		RD::get_singleton()->compute_list_add_barrier(compute_list);
		RENDER_TIMESTAMP("HDDAGI World Probe Filter");

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.integrate_pipeline[HDDAGIShader::INTEGRATE_MODE_FILTER]);

		integrate_uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
				gi->hddagi_shader.integrate_shader_version[HDDAGIShader::INTEGRATE_MODE_FILTER],
				0,
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, lightprobe_diffuse_data),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, lightprobe_diffuse_filter_data),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, lightprobe_neighbour_visibility_map),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, lightprobe_geometry_proximity_map),
				RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, lightprobe_camera_visibility_map));

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, integrate_uniform_set, 0);

		for (uint32_t i = 0; i < cascades.size(); i++) {
			push_constant.cascade = i;
			push_constant.world_offset[0] = cascades[i].position.x / probe_divisor;
			push_constant.world_offset[1] = cascades[i].position.y / probe_divisor;
			push_constant.world_offset[2] = cascades[i].position.z / probe_divisor;

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HDDAGIShader::IntegratePushConstant));
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, probe_axis_count.x * LIGHTPROBE_OCT_SIZE, probe_axis_count.y * probe_axis_count.z * LIGHTPROBE_OCT_SIZE, 1);
		}
	}

	RD::get_singleton()->compute_list_end();

	RENDER_TIMESTAMP("HDDAGI World Probes End");
	RD::get_singleton()->draw_command_end_label();
}

int GI::HDDAGI::get_pending_region_count() const {
	int dirty_count = 0;
	for (const RendererRD::GI::HDDAGI::Cascade &c : cascades) {
		if (c.dirty_regions == RendererRD::GI::HDDAGI::Cascade::DIRTY_ALL) {
			dirty_count++;
		} else {
			for (int j = 0; j < 3; j++) {
				if (c.dirty_regions[j] != 0) {
					dirty_count++;
				}
			}
		}
	}

	return dirty_count;
}

int GI::HDDAGI::get_pending_region_data(int p_region, Vector3i &r_local_offset, Vector3i &r_local_size, AABB &r_bounds, Vector3i &r_scroll, Vector3i &r_region_world) const {
	// higher cascades need to be processed first
	int dirty_count = 0;
	for (uint32_t i = 0; i < cascades.size(); i++) {
		const HDDAGI::Cascade &c = cascades[i];

		if (c.dirty_regions == HDDAGI::Cascade::DIRTY_ALL) {
			if (dirty_count == p_region) {
				r_local_offset = Vector3i();
				r_local_size = cascade_size;
				r_scroll = HDDAGI::Cascade::DIRTY_ALL;
				r_region_world = c.position / REGION_CELLS;

				r_bounds.position = Vector3((Vector3i(1, 1, 1) * -(cascade_size / 2) + c.position)) * c.cell_size * Vector3(1, 1.0 / y_mult, 1);
				r_bounds.size = Vector3(r_local_size) * c.cell_size * Vector3(1, 1.0 / y_mult, 1);
				return i;
			}
			dirty_count++;
		} else {
			for (int j = 0; j < 3; j++) {
				if (c.dirty_regions[j] != 0) {
					if (dirty_count == p_region) {
						Vector3i from = Vector3i(0, 0, 0);
						Vector3i to = cascade_size;

						r_scroll = Vector3i();
						r_scroll[j] = c.dirty_regions[j];

						if (c.dirty_regions[j] > 0) {
							//fill from the beginning
							to[j] = c.dirty_regions[j] + 2; // 2 extra voxels needed to rebuild light properly
						} else {
							//fill from the end
							from[j] = to[j] + c.dirty_regions[j] - 2; // 2 extra voxels needed to rebuild light properly
						}

						r_local_offset = from;
						r_local_size = to - from;

						Vector3i cascade_position = c.position;

						// Remove next axes positions so we don't voxelize the wrong region
						for (int k = j + 1; k < 3; k++) {
							if (c.dirty_regions[k] != 0) {
								cascade_position[k] += c.dirty_regions[k];
							}
						}

						r_region_world = cascade_position / REGION_CELLS;

						r_bounds.position = Vector3(from + Vector3i(1, 1, 1) * -Vector3(cascade_size / 2) + cascade_position) * c.cell_size * Vector3(1, 1.0 / y_mult, 1);
						r_bounds.size = Vector3(r_local_size) * c.cell_size * Vector3(1, 1.0 / y_mult, 1);

						return i;
					}

					dirty_count++;
				}
			}
		}
	}
	return -1;
}

void GI::HDDAGI::update_cascades() {
	//update cascades
	HDDAGI::Cascade::UBO cascade_data[HDDAGI::MAX_CASCADES];

	for (uint32_t i = 0; i < cascades.size(); i++) {
		Vector3 pos = Vector3((Vector3i(1, 1, 1) * -(cascade_size / 2) + cascades[i].position)) * cascades[i].cell_size;

		cascade_data[i].offset[0] = pos.x;
		cascade_data[i].offset[1] = pos.y;
		cascade_data[i].offset[2] = pos.z;
		cascade_data[i].to_cell = 1.0 / cascades[i].cell_size;
		cascade_data[i].region_world_offset[0] = cascades[i].position.x / REGION_CELLS;
		cascade_data[i].region_world_offset[1] = cascades[i].position.y / REGION_CELLS;
		cascade_data[i].region_world_offset[2] = cascades[i].position.z / REGION_CELLS;
	}

	RD::get_singleton()->buffer_update(cascades_ubo, 0, sizeof(HDDAGI::Cascade::UBO) * HDDAGI::MAX_CASCADES, cascade_data);
}

void GI::HDDAGI::pre_process_gi(const Transform3D &p_transform, RenderDataRD *p_render_data) {
	RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton();
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	/* Update general HDDAGI Buffer */

	HDDAGIData hddagi_data = {};

	hddagi_data.grid_size[0] = cascade_size[0];
	hddagi_data.grid_size[1] = cascade_size[1];
	hddagi_data.grid_size[2] = cascade_size[2];

	hddagi_data.max_cascades = cascades.size();

	hddagi_data.probe_axis_size[0] = cascade_size[0] / REGION_CELLS + 1;
	hddagi_data.probe_axis_size[1] = cascade_size[1] / REGION_CELLS + 1;
	hddagi_data.probe_axis_size[2] = cascade_size[2] / REGION_CELLS + 1;

	hddagi_data.y_mult = y_mult;
	hddagi_data.normal_bias = normal_bias;
	hddagi_data.reflection_bias = reflection_bias;

	hddagi_data.energy = energy;

	for (int32_t i = 0; i < hddagi_data.max_cascades; i++) {
		HDDAGIData::ProbeCascadeData &c = hddagi_data.cascades[i];
		Vector3 pos = Vector3((Vector3i(1, 1, 1) * -(cascade_size / 2) + cascades[i].position)) * cascades[i].cell_size;
		Vector3 cam_origin = p_transform.origin;
		cam_origin.y *= y_mult;
		pos -= cam_origin; //make pos local to camera, to reduce numerical error
		c.position[0] = pos.x;
		c.position[1] = pos.y;
		c.position[2] = pos.z;
		c.to_probe = 1.0 / (float(cascade_size[0]) * cascades[i].cell_size / float(hddagi_data.probe_axis_size[0] - 1));

		c.region_world_offset[0] = cascades[i].position.x / REGION_CELLS;
		c.region_world_offset[1] = cascades[i].position.y / REGION_CELLS;
		c.region_world_offset[2] = cascades[i].position.z / REGION_CELLS;

		c.to_cell = 1.0 / cascades[i].cell_size;
		const Vector3 blend_position = cascades[i].blend_position - cam_origin;
		c.blend_position[0] = blend_position.x;
		c.blend_position[1] = blend_position.y;
		c.blend_position[2] = blend_position.z;
		c.exposure_normalization = 1.0;
		if (p_render_data->camera_attributes.is_valid()) {
			float exposure_normalization = RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes);
			c.exposure_normalization = exposure_normalization / cascades[i].baked_exposure_normalization;
		}
	}

	RD::get_singleton()->buffer_update(gi->hddagi_ubo, 0, sizeof(HDDAGIData), &hddagi_data);

	/* Update dynamic lights in HDDAGI cascades */

	for (uint32_t i = 0; i < cascades.size(); i++) {
		HDDAGI::Cascade &cascade = cascades[i];

		HDDAGIShader::Light lights[HDDAGI::MAX_DYNAMIC_LIGHTS] = {};
		uint32_t idx = 0;
		for (uint32_t j = 0; j < (uint32_t)p_render_data->hddagi_update_data->directional_lights->size(); j++) {
			if (idx == HDDAGI::MAX_DYNAMIC_LIGHTS) {
				break;
			}

			RID light_instance = p_render_data->hddagi_update_data->directional_lights->get(j);
			ERR_CONTINUE(!light_storage->owns_light_instance(light_instance));

			RID light = light_storage->light_instance_get_base_light(light_instance);
			Transform3D light_transform = light_storage->light_instance_get_base_transform(light_instance);

			if (RSG::light_storage->light_directional_get_sky_mode(light) == RSE::LIGHT_DIRECTIONAL_SKY_MODE_SKY_ONLY) {
				continue;
			}

			const Vector3 dir = -light_transform.basis.get_column(Vector3::AXIS_Z).normalized();
			lights[idx].direction[0] = dir.x;
			lights[idx].direction[1] = dir.y;
			lights[idx].direction[2] = dir.z;
			Color color = RSG::light_storage->light_get_color(light);
			color = color.srgb_to_linear();
			lights[idx].color[0] = color.r;
			lights[idx].color[1] = color.g;
			lights[idx].color[2] = color.b;
			lights[idx].type = RSE::LIGHT_DIRECTIONAL;
			lights[idx].energy = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ENERGY) * RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INDIRECT_ENERGY);
			if (RendererSceneRenderRD::get_singleton()->is_using_physical_light_units()) {
				lights[idx].energy *= RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INTENSITY);
			}

			if (p_render_data->camera_attributes.is_valid()) {
				lights[idx].energy *= RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes);
			}

			lights[idx].has_shadow = RSG::light_storage->light_has_shadow(light);

			idx++;
		}

		AABB cascade_aabb;
		cascade_aabb.position = Vector3((Vector3i(1, 1, 1) * -(cascade_size / 2) + cascade.position)) * cascade.cell_size;
		cascade_aabb.size = Vector3(1, 1, 1) * cascade_size * cascade.cell_size;
		// Cascade coordinates are stretched on Y; light bounds are world-space.
		cascade_aabb.position.y /= y_mult;
		cascade_aabb.size.y /= y_mult;

		for (uint32_t j = 0; j < p_render_data->hddagi_update_data->positional_light_count; j++) {
			if (idx == HDDAGI::MAX_DYNAMIC_LIGHTS) {
				break;
			}

			RID light_instance = p_render_data->hddagi_update_data->positional_light_instances[j];
			ERR_CONTINUE(!light_storage->owns_light_instance(light_instance));

			RID light = light_storage->light_instance_get_base_light(light_instance);
			AABB light_aabb = light_storage->light_instance_get_base_aabb(light_instance);
			Transform3D light_transform = light_storage->light_instance_get_base_transform(light_instance);

			uint32_t max_hddagi_cascade = RSG::light_storage->light_get_max_hddagi_cascade(light);
			if (i > max_hddagi_cascade) {
				continue;
			}

			if (!cascade_aabb.intersects(light_aabb)) {
				continue;
			}

			const Vector3 dir = -light_transform.basis.get_column(Vector3::AXIS_Z).normalized();
			Vector2 area_size = RSG::light_storage->light_area_get_size(light);
			lights[idx].direction[0] = dir.x;
			lights[idx].direction[1] = dir.y;
			lights[idx].direction[2] = dir.z;
			const Vector3 &pos = light_transform.origin;
			lights[idx].position[0] = pos.x;
			lights[idx].position[1] = pos.y;
			lights[idx].position[2] = pos.z;
			Color color = RSG::light_storage->light_get_color(light);
			color = color.srgb_to_linear();
			lights[idx].color[0] = color.r;
			lights[idx].color[1] = color.g;
			lights[idx].color[2] = color.b;
			lights[idx].type = RSG::light_storage->light_get_type(light);

			lights[idx].energy = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ENERGY) * RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INDIRECT_ENERGY);
			if (RendererSceneRenderRD::get_singleton()->is_using_physical_light_units()) {
				lights[idx].energy *= RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INTENSITY);

				// Convert from Luminous Power to Luminous Intensity
				if (lights[idx].type == RSE::LIGHT_OMNI) {
					lights[idx].energy *= 1.0 / (Math::PI * 4.0);
				} else if (lights[idx].type == RSE::LIGHT_AREA) {
					lights[idx].energy *= 1.0 / (Math::PI * 2.0);
				} else if (lights[idx].type == RSE::LIGHT_SPOT) {
					// Spot Lights are not physically accurate, Luminous Intensity should change in relation to the cone angle.
					// We make this assumption to keep them easy to control.
					lights[idx].energy *= 1.0 / Math::PI;
				}
			}

			if (p_render_data->camera_attributes.is_valid()) {
				lights[idx].energy *= RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes);
			}

			lights[idx].has_shadow = RSG::light_storage->light_has_shadow(light);
			lights[idx].attenuation = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ATTENUATION);
			lights[idx].radius = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_RANGE);
			lights[idx].cos_spot_angle = Math::cos(Math::deg_to_rad(RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_SPOT_ANGLE)));
			lights[idx].inv_spot_attenuation = 1.0f / RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_SPOT_ATTENUATION);

			if (lights[idx].type == RSE::LIGHT_AREA) {
				Vector3 area_vec_a = light_transform.basis.get_column(0).normalized() * area_size.x;
				Vector3 area_vec_b = light_transform.basis.get_column(1).normalized() * area_size.y;

				lights[idx].area_width[0] = area_vec_a.x;
				lights[idx].area_width[1] = area_vec_a.y;
				lights[idx].area_width[2] = area_vec_a.z;
				lights[idx].area_height[0] = area_vec_b.x;
				lights[idx].area_height[1] = area_vec_b.y;
				lights[idx].area_height[2] = area_vec_b.z;

				Rect2 proj_rect = texture_storage->area_light_atlas_get_texture_rect(RSG::light_storage->light_area_get_texture(light));
				lights[idx].area_projector_rect[0] = proj_rect.position.x;
				lights[idx].area_projector_rect[1] = proj_rect.position.y;
				lights[idx].area_projector_rect[2] = proj_rect.size.x;
				lights[idx].area_projector_rect[3] = proj_rect.size.y;

				Size2i texture_size = (proj_rect.size * texture_storage->area_light_atlas_get_size()).ceil();
				lights[idx].cos_spot_angle = MIN(Math::floor(Math::log2(MAX(MIN(texture_size.x, texture_size.y), 1.0f))), texture_storage->area_light_atlas_get_mipmaps()) - 1.0f;
				lights[idx].inv_spot_attenuation = 1.0f / (lights[idx].radius + area_size.length() / 2.0f);

				if (RSG::light_storage->light_area_get_normalize_energy(light)) {
					float surface_area = area_size.x * area_size.y;
					if (surface_area > 0.0f) {
						lights[idx].energy /= surface_area;
					}
				}
			}

			idx++;
		}

		if (idx > 0) {
			RD::get_singleton()->buffer_update(cascade.light_position_bufer, 0, idx * sizeof(HDDAGIShader::Light), lights);
		}

		const uint32_t light_bytes = idx * sizeof(HDDAGIShader::Light);
		cascade.dynamic_light_changed = cascade.previous_dynamic_lights.size() != light_bytes || (light_bytes > 0 && memcmp(cascade.previous_dynamic_lights.ptr(), lights, light_bytes) != 0);
		if (cascade.dynamic_light_changed) {
			cascade.previous_dynamic_lights.resize(light_bytes);
			if (light_bytes > 0) {
				memcpy(cascade.previous_dynamic_lights.ptrw(), lights, light_bytes);
			}
		}
		cascade.use_light_regions = idx >= 8 && bool(GLOBAL_GET("rendering/global_illumination/hddagi/light_region_culling"));
		if (cascade.use_light_regions && (cascade.dynamic_light_changed || !cascade.light_regions_valid || cascade.light_region_position != cascade.position)) {
			HDDAGIShader::LightCullPushConstant cull_constant = {};
			Vector3i region_count = cascade_size / REGION_CELLS;
			Vector3 offset = Vector3(cascade.position - cascade_size / 2) * cascade.cell_size;
			for (int axis = 0; axis < 3; axis++) {
				cull_constant.offset[axis] = offset[axis];
				cull_constant.region_count[axis] = region_count[axis];
			}
			cull_constant.region_size = cascade.cell_size * float(REGION_CELLS);
			cull_constant.light_count = idx;
			cull_constant.y_mult = y_mult;
			RID cull_set = UniformSetCacheRD::get_singleton()->get_cache(gi->hddagi_shader.light_cull_version, 0,
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, cascade.light_position_bufer),
					RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, cascade.light_region_buffer));
			RD::ComputeListID cull_list = RD::get_singleton()->compute_list_begin();
			RD::get_singleton()->compute_list_bind_compute_pipeline(cull_list, gi->hddagi_shader.light_cull_pipeline);
			RD::get_singleton()->compute_list_bind_uniform_set(cull_list, cull_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(cull_list, &cull_constant, sizeof(cull_constant));
			RD::get_singleton()->compute_list_dispatch_threads(cull_list, region_count.x * region_count.y * region_count.z * ((idx + 31) / 32), 1, 1);
			RD::get_singleton()->compute_list_end();
			cascade.light_regions_valid = true;
			cascade.light_region_position = cascade.position;
		} else if (!cascade.use_light_regions) {
			cascade.light_regions_valid = false;
		}
		cascade_dynamic_light_count[i] = idx;
	}
}

void GI::HDDAGI::render_static_lights(RenderDataRD *p_render_data, Ref<RenderSceneBuffersRD> p_render_buffers, uint32_t p_cascade_count, const uint32_t *p_cascade_indices, const PagedArray<RID> *p_positional_light_cull_result) {
	ERR_FAIL_COND(p_render_buffers.is_null()); // we wouldn't be here if this failed but...

	RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton();
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();

	RD::get_singleton()->draw_command_begin_label("HDDAGI Render Static Lights");

	update_cascades();

	HDDAGIShader::Light lights[HDDAGI::MAX_STATIC_LIGHTS] = {};
	uint32_t light_count[HDDAGI::MAX_STATIC_LIGHTS];

	for (uint32_t i = 0; i < p_cascade_count; i++) {
		ERR_CONTINUE(p_cascade_indices[i] >= cascades.size());

		HDDAGI::Cascade &cc = cascades[p_cascade_indices[i]];

		{ //fill light buffer

			AABB cascade_aabb;
			cascade_aabb.position = Vector3((Vector3i(1, 1, 1) * -(cascade_size / 2) + cc.position)) * cc.cell_size;
			cascade_aabb.size = Vector3(1, 1, 1) * cascade_size * cc.cell_size;
			// Cascade coordinates are stretched on Y; light bounds are world-space.
			cascade_aabb.position.y /= y_mult;
			cascade_aabb.size.y /= y_mult;

			int idx = 0;

			const PagedArray<RID> &cascade_lights = p_positional_light_cull_result[p_cascade_indices[i]];
			for (uint32_t j = 0; j < (uint32_t)cascade_lights.size(); j++) {
				if (idx == HDDAGI::MAX_STATIC_LIGHTS) {
					break;
				}

				RID light_instance = cascade_lights[j];
				ERR_CONTINUE(!light_storage->owns_light_instance(light_instance));

				RID light = light_storage->light_instance_get_base_light(light_instance);
				AABB light_aabb = light_storage->light_instance_get_base_aabb(light_instance);
				Transform3D light_transform = light_storage->light_instance_get_base_transform(light_instance);

				uint32_t max_hddagi_cascade = RSG::light_storage->light_get_max_hddagi_cascade(light);
				if (p_cascade_indices[i] > max_hddagi_cascade) {
					continue;
				}

				if (!cascade_aabb.intersects(light_aabb)) {
					continue;
				}

				lights[idx].type = RSG::light_storage->light_get_type(light);

				const Vector3 dir = -light_transform.basis.get_column(Vector3::AXIS_Z).normalized();
				Vector2 area_size = RSG::light_storage->light_area_get_size(light);
				lights[idx].direction[0] = dir.x;
				lights[idx].direction[1] = dir.y;
				lights[idx].direction[2] = dir.z;
				const Vector3 &pos = light_transform.origin;
				lights[idx].position[0] = pos.x;
				lights[idx].position[1] = pos.y;
				lights[idx].position[2] = pos.z;
				Color color = RSG::light_storage->light_get_color(light);
				color = color.srgb_to_linear();
				lights[idx].color[0] = color.r;
				lights[idx].color[1] = color.g;
				lights[idx].color[2] = color.b;

				lights[idx].energy = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ENERGY) * RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INDIRECT_ENERGY);
				if (RendererSceneRenderRD::get_singleton()->is_using_physical_light_units()) {
					lights[idx].energy *= RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INTENSITY);

					// Convert from Luminous Power to Luminous Intensity
					if (lights[idx].type == RSE::LIGHT_OMNI) {
						lights[idx].energy *= 1.0 / (Math::PI * 4.0);
					} else if (lights[idx].type == RSE::LIGHT_AREA) {
						lights[idx].energy *= 1.0 / (Math::PI * 2.0);
					} else if (lights[idx].type == RSE::LIGHT_SPOT) {
						// Spot Lights are not physically accurate, Luminous Intensity should change in relation to the cone angle.
						// We make this assumption to keep them easy to control.
						lights[idx].energy *= 1.0 / Math::PI;
					}
				}

				if (p_render_data->camera_attributes.is_valid()) {
					lights[idx].energy *= RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes);
				}

				lights[idx].has_shadow = RSG::light_storage->light_has_shadow(light);
				lights[idx].attenuation = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ATTENUATION);
				lights[idx].radius = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_RANGE);
				lights[idx].cos_spot_angle = Math::cos(Math::deg_to_rad(RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_SPOT_ANGLE)));
				lights[idx].inv_spot_attenuation = 1.0f / RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_SPOT_ATTENUATION);

				if (lights[idx].type == RSE::LIGHT_AREA) {
					Vector3 area_vec_a = light_transform.basis.get_column(0).normalized() * area_size.x;
					Vector3 area_vec_b = light_transform.basis.get_column(1).normalized() * area_size.y;

					lights[idx].area_width[0] = area_vec_a.x;
					lights[idx].area_width[1] = area_vec_a.y;
					lights[idx].area_width[2] = area_vec_a.z;
					lights[idx].area_height[0] = area_vec_b.x;
					lights[idx].area_height[1] = area_vec_b.y;
					lights[idx].area_height[2] = area_vec_b.z;

					Rect2 proj_rect = texture_storage->area_light_atlas_get_texture_rect(RSG::light_storage->light_area_get_texture(light));
					lights[idx].area_projector_rect[0] = proj_rect.position.x;
					lights[idx].area_projector_rect[1] = proj_rect.position.y;
					lights[idx].area_projector_rect[2] = proj_rect.size.x;
					lights[idx].area_projector_rect[3] = proj_rect.size.y;

					Size2i texture_size = (proj_rect.size * texture_storage->area_light_atlas_get_size()).ceil();
					lights[idx].cos_spot_angle = MIN(Math::floor(Math::log2(MAX(MIN(texture_size.x, texture_size.y), 1.0f))), texture_storage->area_light_atlas_get_mipmaps()) - 1.0f;
					lights[idx].inv_spot_attenuation = 1.0f / (lights[idx].radius + area_size.length() / 2.0f);

					if (RSG::light_storage->light_area_get_normalize_energy(light)) {
						float surface_area = area_size.x * area_size.y;
						if (surface_area > 0.0f) {
							lights[idx].energy /= surface_area;
						}
					}
				}

				idx++;
			}

			if (idx > 0) {
				RD::get_singleton()->buffer_update(cc.light_position_bufer, 0, idx * sizeof(HDDAGIShader::Light), lights);
			}

			light_count[i] = idx;
		}
	}

	/* Static Lights */
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->hddagi_shader.direct_light_pipeline[HDDAGIShader::DIRECT_LIGHT_MODE_STATIC]);

	HDDAGIShader::DirectLightPushConstant dl_push_constant = {};

	Vector3i probe_axis_count = cascade_size / REGION_CELLS + Vector3i(1, 1, 1);

	dl_push_constant.grid_size[0] = cascade_size[0];
	dl_push_constant.grid_size[1] = cascade_size[1];
	dl_push_constant.grid_size[2] = cascade_size[2];
	dl_push_constant.max_cascades = cascades.size();
	dl_push_constant.probe_axis_size[0] = probe_axis_count[0];
	dl_push_constant.probe_axis_size[1] = probe_axis_count[1];
	dl_push_constant.probe_axis_size[2] = probe_axis_count[2];
	dl_push_constant.bounce_feedback = 0.0; // this is static light, do not multibounce yet
	dl_push_constant.y_mult = y_mult;
	dl_push_constant.use_occlusion = using_probe_filter;
	dl_push_constant.probe_cell_size = REGION_CELLS;
	dl_push_constant.dirty_dynamic_update = 0;

	//all must be processed
	dl_push_constant.process_offset = 0;
	dl_push_constant.process_increment = 1;

	RID area_light_atlas_uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
			gi->hddagi_shader.direct_light_shader_version[HDDAGIShader::DIRECT_LIGHT_MODE_STATIC],
			1,
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, texture_storage->area_light_atlas_get_texture()),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 1, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));

	for (uint32_t i = 0; i < p_cascade_count; i++) {
		ERR_CONTINUE(p_cascade_indices[i] >= cascades.size());

		HDDAGI::Cascade &cc = cascades[p_cascade_indices[i]];

		dl_push_constant.light_count = light_count[i];
		dl_push_constant.cascade = p_cascade_indices[i];

		if (dl_push_constant.light_count > 0) {
			LocalVector<RD::Uniform> direct_light_uniforms;
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, voxel_bits_tex));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, voxel_region_tex));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 3, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, cc.light_process_dispatch_buffer_copy));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, cc.light_process_buffer));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, voxel_data));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 14, voxel_page_table));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, cascades_ubo));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, cc.light_position_bufer));
			direct_light_uniforms.push_back(RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 11, cc.light_source_emission_buffer));
			RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache_vec(
					gi->hddagi_shader.direct_light_shader_version[HDDAGIShader::DIRECT_LIGHT_MODE_STATIC], 0, direct_light_uniforms);

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, area_light_atlas_uniform_set, 1);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &dl_push_constant, sizeof(HDDAGIShader::DirectLightPushConstant));
			RD::get_singleton()->compute_list_dispatch_indirect(compute_list, cc.light_process_dispatch_buffer, 0);
		}
	}

	RD::get_singleton()->compute_list_end();

	RD::get_singleton()->draw_command_end_label();
}

////////////////////////////////////////////////////////////////////////////////
// VoxelGIInstance

void GI::VoxelGIInstance::update(bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RenderGeometryInstance *> &p_dynamic_objects) {
	RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	uint32_t data_version = gi->voxel_gi_get_data_version(probe);

	// (RE)CREATE IF NEEDED

	if (last_probe_data_version != data_version) {
		//need to re-create everything
		free_resources();

		Vector3i octree_size = gi->voxel_gi_get_octree_size(probe);

		if (octree_size != Vector3i()) {
			//can create a 3D texture
			Vector<int> levels = gi->voxel_gi_get_level_counts(probe);

			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			tf.width = octree_size.x;
			tf.height = octree_size.y;
			tf.depth = octree_size.z;
			tf.texture_type = RD::TEXTURE_TYPE_3D;
			tf.mipmaps = levels.size();

			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

			texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			RD::get_singleton()->set_resource_name(texture, "VoxelGI Instance Texture");

			RD::get_singleton()->texture_clear(texture, Color(0, 0, 0, 0), 0, levels.size(), 0, 1);

			{
				int total_elements = 0;
				for (int i = 0; i < levels.size(); i++) {
					total_elements += levels[i];
				}

				write_buffer = RD::get_singleton()->storage_buffer_create(total_elements * 16);
			}

			for (int i = 0; i < levels.size(); i++) {
				VoxelGIInstance::Mipmap mipmap;
				mipmap.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), texture, 0, i, 1, RD::TEXTURE_SLICE_3D);
				mipmap.level = levels.size() - i - 1;
				mipmap.cell_offset = 0;
				for (uint32_t j = 0; j < mipmap.level; j++) {
					mipmap.cell_offset += levels[j];
				}
				mipmap.cell_count = levels[mipmap.level];

				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 1;
					u.append_id(gi->voxel_gi_get_octree_buffer(probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 2;
					u.append_id(gi->voxel_gi_get_data_buffer(probe));
					uniforms.push_back(u);
				}

				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 4;
					u.append_id(write_buffer);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
					u.binding = 9;
					u.append_id(gi->voxel_gi_get_sdf_texture(probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
					u.binding = 10;
					u.append_id(material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
					uniforms.push_back(u);
				}

				{
					Vector<RD::Uniform> copy_uniforms = uniforms;
					if (i == 0) {
						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
							u.binding = 3;
							u.append_id(gi->voxel_gi_lights_uniform);
							copy_uniforms.push_back(u);
						}

						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_LIGHT], 0);

						copy_uniforms = uniforms; //restore

						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 5;
							u.append_id(texture);
							copy_uniforms.push_back(u);
						}
						mipmap.second_bounce_uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_SECOND_BOUNCE], 0);
					} else {
						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_MIPMAP], 0);
					}
				}

				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 5;
					u.append_id(mipmap.texture);
					uniforms.push_back(u);
				}

				mipmap.write_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_WRITE_TEXTURE], 0);

				mipmaps.push_back(mipmap);
			}

			{
				uint32_t dynamic_map_size = MAX(MAX(octree_size.x, octree_size.y), octree_size.z);
				uint32_t oversample = Math::nearest_power_of_2_templated(4);
				int mipmap_index = 0;

				while (mipmap_index < mipmaps.size()) {
					VoxelGIInstance::DynamicMap dmap;

					if (oversample > 0) {
						dmap.size = dynamic_map_size * (1 << oversample);
						dmap.mipmap = -1;
						oversample--;
					} else {
						dmap.size = dynamic_map_size >> mipmap_index;
						dmap.mipmap = mipmap_index;
						mipmap_index++;
					}

					RD::TextureFormat dtf;
					dtf.width = dmap.size;
					dtf.height = dmap.size;
					dtf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

					if (dynamic_maps.size() == 0) {
						dtf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
					}
					dmap.texture = RD::get_singleton()->texture_create(dtf, RD::TextureView());
					RD::get_singleton()->set_resource_name(dmap.texture, "VoxelGI Instance DMap Texture");

					if (dynamic_maps.size() == 0) {
						// Render depth for first one.
						// Use 16-bit depth when supported to improve performance.
						dtf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D16_UNORM, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D16_UNORM : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
						dtf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
						dmap.fb_depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						RD::get_singleton()->set_resource_name(dmap.fb_depth, "VoxelGI Instance DMap FB Depth");
					}

					//just use depth as-is
					dtf.format = RD::DATA_FORMAT_R32_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

					dmap.depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());
					RD::get_singleton()->set_resource_name(dmap.depth, "VoxelGI Instance DMap Depth");

					if (dynamic_maps.size() == 0) {
						dtf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
						dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
						dmap.albedo = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						RD::get_singleton()->set_resource_name(dmap.albedo, "VoxelGI Instance DMap Albedo");
						dmap.normal = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						RD::get_singleton()->set_resource_name(dmap.normal, "VoxelGI Instance DMap Normal");
						dmap.orm = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						RD::get_singleton()->set_resource_name(dmap.orm, "VoxelGI Instance DMap ORM");

						Vector<RID> fb;
						fb.push_back(dmap.albedo);
						fb.push_back(dmap.normal);
						fb.push_back(dmap.orm);
						fb.push_back(dmap.texture); //emission
						fb.push_back(dmap.depth);
						fb.push_back(dmap.fb_depth);

						dmap.fb = RD::get_singleton()->framebuffer_create(fb);

						{
							Vector<RD::Uniform> uniforms;
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
								u.binding = 3;
								u.append_id(gi->voxel_gi_lights_uniform);
								uniforms.push_back(u);
							}

							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 5;
								u.append_id(dmap.albedo);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 6;
								u.append_id(dmap.normal);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.append_id(dmap.orm);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 8;
								u.append_id(dmap.fb_depth);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 9;
								u.append_id(gi->voxel_gi_get_sdf_texture(probe));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
								u.binding = 10;
								u.append_id(material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.append_id(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 12;
								u.append_id(dmap.depth);
								uniforms.push_back(u);
							}

							dmap.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING], 0);
						}
					} else {
						bool plot = dmap.mipmap >= 0;
						bool write = dmap.mipmap < (mipmaps.size() - 1);

						Vector<RD::Uniform> uniforms;

						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 5;
							u.append_id(dynamic_maps[dynamic_maps.size() - 1].texture);
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 6;
							u.append_id(dynamic_maps[dynamic_maps.size() - 1].depth);
							uniforms.push_back(u);
						}

						if (write) {
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.append_id(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 8;
								u.append_id(dmap.depth);
								uniforms.push_back(u);
							}
						}

						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 9;
							u.append_id(gi->voxel_gi_get_sdf_texture(probe));
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
							u.binding = 10;
							u.append_id(material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
							uniforms.push_back(u);
						}

						if (plot) {
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.append_id(mipmaps[dmap.mipmap].texture);
								uniforms.push_back(u);
							}
						}

						dmap.uniform_set = RD::get_singleton()->uniform_set_create(
								uniforms,
								gi->voxel_gi_lighting_shader_version_shaders[(write && plot) ? VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT : (write ? VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE : VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_PLOT)],
								0);
					}

					dynamic_maps.push_back(dmap);
				}
			}
		}

		last_probe_data_version = data_version;
		p_update_light_instances = true; //just in case

		RendererSceneRenderRD::get_singleton()->base_uniforms_changed();
	}

	// UDPDATE TIME

	if (has_dynamic_object_data) {
		//if it has dynamic object data, it needs to be cleared
		RD::get_singleton()->texture_clear(texture, Color(0, 0, 0, 0), 0, mipmaps.size(), 0, 1);
	}

	uint32_t light_count = 0;

	if (p_update_light_instances || p_dynamic_objects.size() > 0) {
		light_count = MIN(gi->voxel_gi_max_lights, (uint32_t)p_light_instances.size());

		{
			Transform3D to_cell = gi->voxel_gi_get_to_cell_xform(probe);
			Transform3D to_probe_xform = to_cell * transform.affine_inverse();

			//update lights

			for (uint32_t i = 0; i < light_count; i++) {
				VoxelGILight &l = gi->voxel_gi_lights[i];
				RID light_instance = p_light_instances[i];
				RID light = light_storage->light_instance_get_base_light(light_instance);

				l.type = RSG::light_storage->light_get_type(light);
				if (l.type == RSE::LIGHT_DIRECTIONAL && RSG::light_storage->light_directional_get_sky_mode(light) == RSE::LIGHT_DIRECTIONAL_SKY_MODE_SKY_ONLY) {
					light_count--;
					continue;
				}

				l.attenuation = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ATTENUATION);
				l.energy = RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_ENERGY) * RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INDIRECT_ENERGY);

				if (RendererSceneRenderRD::get_singleton()->is_using_physical_light_units()) {
					l.energy *= RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_INTENSITY);

					l.energy *= gi->voxel_gi_get_baked_exposure_normalization(probe);

					// Convert from Luminous Power to Luminous Intensity
					if (l.type == RSE::LIGHT_OMNI) {
						l.energy *= 1.0 / (Math::PI * 4.0);
					} else if (l.type == RSE::LIGHT_AREA) {
						l.energy *= 1.0 / (Math::PI * 2.0);
					} else if (l.type == RSE::LIGHT_SPOT) {
						// Spot Lights are not physically accurate, Luminous Intensity should change in relation to the cone angle.
						// We make this assumption to keep them easy to control.
						l.energy *= 1.0 / Math::PI;
					}
				}

				l.radius = to_cell.basis.xform(Vector3(RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_RANGE), 0, 0)).length();
				Color color = RSG::light_storage->light_get_color(light).srgb_to_linear();
				l.color[0] = color.r;
				l.color[1] = color.g;
				l.color[2] = color.b;

				l.cos_spot_angle = Math::cos(Math::deg_to_rad(RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_SPOT_ANGLE)));
				l.inv_spot_attenuation = 1.0f / RSG::light_storage->light_get_param(light, RSE::LIGHT_PARAM_SPOT_ATTENUATION);

				Transform3D xform = light_storage->light_instance_get_base_transform(light_instance);
				Vector2 area_size = RSG::light_storage->light_area_get_size(light);

				Vector3 pos = to_probe_xform.xform(xform.origin);
				Vector3 dir = to_probe_xform.basis.xform(-xform.basis.get_column(2)).normalized();

				l.position[0] = pos.x;
				l.position[1] = pos.y;
				l.position[2] = pos.z;

				l.direction[0] = dir.x;
				l.direction[1] = dir.y;
				l.direction[2] = dir.z;

				l.has_shadow = RSG::light_storage->light_has_shadow(light);

				if (l.type == RSE::LIGHT_AREA) {
					Vector3 area_vec_a = to_probe_xform.basis.xform(xform.basis.get_column(0).normalized() * area_size.x);
					Vector3 area_vec_b = to_probe_xform.basis.xform(xform.basis.get_column(1).normalized() * area_size.y);
					l.area_width[0] = area_vec_a.x;
					l.area_width[1] = area_vec_a.y;
					l.area_width[2] = area_vec_a.z;
					l.area_height[0] = area_vec_b.x;
					l.area_height[1] = area_vec_b.y;
					l.area_height[2] = area_vec_b.z;

					Rect2 proj_rect = RendererRD::TextureStorage::get_singleton()->area_light_atlas_get_texture_rect(RSG::light_storage->light_area_get_texture(light));
					l.area_projector_rect[0] = proj_rect.position.x;
					l.area_projector_rect[1] = proj_rect.position.y;
					l.area_projector_rect[2] = proj_rect.size.width;
					l.area_projector_rect[3] = proj_rect.size.height;
					l.inv_spot_attenuation = 1.0f / (l.radius + area_size.length() / 2.0f);

					Size2i texture_size = proj_rect.size * RendererRD::TextureStorage::get_singleton()->area_light_atlas_get_size();
					l.cos_spot_angle = MIN(Math::floor(Math::log2(MAX(MIN(texture_size.x, texture_size.y), 1.0f))), RendererRD::TextureStorage::get_singleton()->area_light_atlas_get_mipmaps()) - 1.0f;
					if (RSG::light_storage->light_area_get_normalize_energy(light)) {
						float surface_area = area_size.x * area_size.y;
						l.energy /= surface_area;
					}
				}
			}

			RD::get_singleton()->buffer_update(gi->voxel_gi_lights_uniform, 0, sizeof(VoxelGILight) * light_count, gi->voxel_gi_lights);
		}
	}

	if (has_dynamic_object_data || p_update_light_instances || p_dynamic_objects.size()) {
		// PROCESS MIPMAPS
		if (mipmaps.size()) {
			//can update mipmaps

			Vector3i probe_size = gi->voxel_gi_get_octree_size(probe);
			Vector3 ps = probe_size / gi->voxel_gi_get_bounds(probe).size;
			float cell_size = 1.0 / MAX(MAX(ps.x, ps.y), ps.z);

			VoxelGIPushConstant push_constant;

			push_constant.limits[0] = probe_size.x;
			push_constant.limits[1] = probe_size.y;
			push_constant.limits[2] = probe_size.z;
			push_constant.stack_size = mipmaps.size();
			push_constant.emission_scale = 1.0;
			push_constant.propagation = gi->voxel_gi_get_propagation(probe);
			push_constant.dynamic_range = gi->voxel_gi_get_dynamic_range(probe);
			push_constant.light_count = light_count;
			push_constant.aniso_strength = 0;
			push_constant.cell_size = cell_size;

			/*		print_line("probe update to version " + itos(last_probe_version));
			print_line("propagation " + rtos(push_constant.propagation));
			print_line("dynrange " + rtos(push_constant.dynamic_range));
	*/
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

			int passes;
			if (p_update_light_instances) {
				passes = gi->voxel_gi_is_using_two_bounces(probe) ? 2 : 1;
			} else {
				passes = 1; //only re-blitting is necessary
			}
			int wg_size = 64;
			int64_t wg_limit_x = (int64_t)RD::get_singleton()->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);

			RID compute_light_area_light_atlas_uniform_set;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				u.append_id(RendererRD::TextureStorage::get_singleton()->area_light_atlas_get_texture());

				compute_light_area_light_atlas_uniform_set = UniformSetCacheRD::get_singleton()->get_cache(gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_LIGHT], 1, u);
			}

			for (int pass = 0; pass < passes; pass++) {
				if (p_update_light_instances) {
					for (int i = 0; i < mipmaps.size(); i++) {
						if (i == 0) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[pass == 0 ? VOXEL_GI_SHADER_VERSION_COMPUTE_LIGHT : VOXEL_GI_SHADER_VERSION_COMPUTE_SECOND_BOUNCE]);
						} else if (i == 1) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_COMPUTE_MIPMAP]);
						}

						if (pass == 1 || i > 0) {
							RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
						}
						if (pass == 0 || i > 0) {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mipmaps[i].uniform_set, 0);
							if (i == 0) {
								RD::get_singleton()->compute_list_bind_uniform_set(compute_list, compute_light_area_light_atlas_uniform_set, 1);
							}
						} else {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mipmaps[i].second_bounce_uniform_set, 0);
						}

						push_constant.cell_offset = mipmaps[i].cell_offset;
						push_constant.cell_count = mipmaps[i].cell_count;

						int64_t wg_todo = (mipmaps[i].cell_count + wg_size - 1) / wg_size;
						while (wg_todo) {
							int64_t wg_count = MIN(wg_todo, wg_limit_x);
							RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIPushConstant));
							RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
							wg_todo -= wg_count;
							push_constant.cell_offset += wg_count * wg_size;
						}
					}

					RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
				}

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_WRITE_TEXTURE]);

				for (int i = 0; i < mipmaps.size(); i++) {
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mipmaps[i].write_uniform_set, 0);

					push_constant.cell_offset = mipmaps[i].cell_offset;
					push_constant.cell_count = mipmaps[i].cell_count;

					int64_t wg_todo = (mipmaps[i].cell_count + wg_size - 1) / wg_size;
					while (wg_todo) {
						int64_t wg_count = MIN(wg_todo, wg_limit_x);
						RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIPushConstant));
						RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
						wg_todo -= wg_count;
						push_constant.cell_offset += wg_count * wg_size;
					}
				}
			}

			RD::get_singleton()->compute_list_end();
		}
	}

	has_dynamic_object_data = false; //clear until dynamic object data is used again

	if (p_dynamic_objects.size() && dynamic_maps.size()) {
		Vector3i octree_size = gi->voxel_gi_get_octree_size(probe);
		int multiplier = dynamic_maps[0].size / MAX(MAX(octree_size.x, octree_size.y), octree_size.z);

		Transform3D oversample_scale;
		oversample_scale.basis.scale(Vector3(multiplier, multiplier, multiplier));

		Transform3D to_cell = oversample_scale * gi->voxel_gi_get_to_cell_xform(probe);
		Transform3D to_world_xform = transform * to_cell.affine_inverse();
		Transform3D to_probe_xform = to_world_xform.affine_inverse();

		AABB probe_aabb(Vector3(), octree_size);

		RID dynamic_object_lighting_area_light_atlas_uniform_set;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 0;
			u.append_id(RendererRD::TextureStorage::get_singleton()->area_light_atlas_get_texture());

			dynamic_object_lighting_area_light_atlas_uniform_set = UniformSetCacheRD::get_singleton()->get_cache(gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING], 1, u);
		}

		//this could probably be better parallelized in compute..
		for (int i = 0; i < (int)p_dynamic_objects.size(); i++) {
			RenderGeometryInstance *instance = p_dynamic_objects[i];

			//transform aabb to voxel_gi
			AABB aabb = (to_probe_xform * instance->get_transform()).xform(instance->get_aabb());

			//this needs to wrap to grid resolution to avoid jitter
			//also extend margin a bit just in case
			Vector3i begin = aabb.position - Vector3i(1, 1, 1);
			Vector3i end = aabb.position + aabb.size + Vector3i(1, 1, 1);

			for (int j = 0; j < 3; j++) {
				if ((end[j] - begin[j]) & 1) {
					end[j]++; //for half extents split, it needs to be even
				}
				begin[j] = MAX(begin[j], 0);
				end[j] = MIN(end[j], octree_size[j] * multiplier);
			}

			//aabb = aabb.intersection(probe_aabb); //intersect
			aabb.position = begin;
			aabb.size = end - begin;

			//print_line("aabb: " + aabb);

			for (int j = 0; j < 6; j++) {
				//if (j != 0 && j != 3) {
				//	continue;
				//}
				static const Vector3 render_z[6] = {
					Vector3(1, 0, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(-1, 0, 0),
					Vector3(0, -1, 0),
					Vector3(0, 0, -1),
				};
				static const Vector3 render_up[6] = {
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
				};

				Vector3 render_dir = render_z[j];
				Vector3 up_dir = render_up[j];

				Vector3 center = aabb.get_center();
				Transform3D xform;
				xform.set_look_at(center - aabb.size * 0.5 * render_dir, center, up_dir);

				Vector3 x_dir = xform.basis.get_column(0).abs();
				int x_axis = int(Vector3(0, 1, 2).dot(x_dir));
				Vector3 y_dir = xform.basis.get_column(1).abs();
				int y_axis = int(Vector3(0, 1, 2).dot(y_dir));
				Vector3 z_dir = -xform.basis.get_column(2);
				int z_axis = int(Vector3(0, 1, 2).dot(z_dir.abs()));

				Rect2i rect(aabb.position[x_axis], aabb.position[y_axis], aabb.size[x_axis], aabb.size[y_axis]);
				bool x_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_column(0)) < 0);
				bool y_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_column(1)) < 0);
				bool z_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_column(2)) > 0);

				Projection cm;
				cm.set_orthogonal(-rect.size.width / 2, rect.size.width / 2, -rect.size.height / 2, rect.size.height / 2, 0.0001, aabb.size[z_axis]);

				if (RendererSceneRenderRD::get_singleton()->cull_argument.size() == 0) {
					RendererSceneRenderRD::get_singleton()->cull_argument.push_back(nullptr);
				}
				RendererSceneRenderRD::get_singleton()->cull_argument[0] = instance;

				float exposure_normalization = 1.0;
				if (RendererSceneRenderRD::get_singleton()->is_using_physical_light_units()) {
					exposure_normalization = gi->voxel_gi_get_baked_exposure_normalization(probe);
				}

				RendererSceneRenderRD::get_singleton()->_render_material(to_world_xform * xform, cm, true, RendererSceneRenderRD::get_singleton()->cull_argument, dynamic_maps[0].fb, Rect2i(Vector2i(), rect.size), exposure_normalization);

				Vector3 ps = octree_size / gi->voxel_gi_get_bounds(probe).size;
				float cell_size = 1.0 / MAX(MAX(ps.x, ps.y), ps.z);

				VoxelGIDynamicPushConstant push_constant;
				memset(&push_constant, 0, sizeof(VoxelGIDynamicPushConstant));
				push_constant.limits[0] = octree_size.x;
				push_constant.limits[1] = octree_size.y;
				push_constant.limits[2] = octree_size.z;
				push_constant.light_count = p_light_instances.size();
				push_constant.x_dir[0] = x_dir[0];
				push_constant.x_dir[1] = x_dir[1];
				push_constant.x_dir[2] = x_dir[2];
				push_constant.y_dir[0] = y_dir[0];
				push_constant.y_dir[1] = y_dir[1];
				push_constant.y_dir[2] = y_dir[2];
				push_constant.z_dir[0] = z_dir[0];
				push_constant.z_dir[1] = z_dir[1];
				push_constant.z_dir[2] = z_dir[2];
				push_constant.z_base = xform.origin[z_axis];
				push_constant.z_sign = (z_flip ? -1.0 : 1.0);
				push_constant.pos_multiplier = float(1.0) / multiplier;
				push_constant.dynamic_range = gi->voxel_gi_get_dynamic_range(probe);
				push_constant.flip_x = x_flip;
				push_constant.flip_y = y_flip;
				push_constant.rect_pos[0] = rect.position[0];
				push_constant.rect_pos[1] = rect.position[1];
				push_constant.rect_size[0] = rect.size[0];
				push_constant.rect_size[1] = rect.size[1];
				push_constant.prev_rect_ofs[0] = 0;
				push_constant.prev_rect_ofs[1] = 0;
				push_constant.prev_rect_size[0] = 0;
				push_constant.prev_rect_size[1] = 0;
				push_constant.on_mipmap = false;
				push_constant.propagation = gi->voxel_gi_get_propagation(probe);
				push_constant.cell_size = cell_size;
				push_constant.pad[0] = 0;
				push_constant.pad[1] = 0;

				//process lighting
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, dynamic_maps[0].uniform_set, 0);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, dynamic_object_lighting_area_light_atlas_uniform_set, 1);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIDynamicPushConstant));
				RD::get_singleton()->compute_list_dispatch(compute_list, Math::division_round_up(rect.size.x, 8), Math::division_round_up(rect.size.y, 8), 1);
				//print_line("rect: " + itos(i) + ": " + rect);

				for (int k = 1; k < dynamic_maps.size(); k++) {
					// enlarge the rect if needed so all pixels fit when downscaled,
					// this ensures downsampling is smooth and optimal because no pixels are left behind

					//x
					if (rect.position.x & 1) {
						rect.size.x++;
						push_constant.prev_rect_ofs[0] = 1; //this is used to ensure reading is also optimal
					} else {
						push_constant.prev_rect_ofs[0] = 0;
					}
					if (rect.size.x & 1) {
						rect.size.x++;
					}

					rect.position.x >>= 1;
					rect.size.x = MAX(1, rect.size.x >> 1);

					//y
					if (rect.position.y & 1) {
						rect.size.y++;
						push_constant.prev_rect_ofs[1] = 1;
					} else {
						push_constant.prev_rect_ofs[1] = 0;
					}
					if (rect.size.y & 1) {
						rect.size.y++;
					}

					rect.position.y >>= 1;
					rect.size.y = MAX(1, rect.size.y >> 1);

					//shrink limits to ensure plot does not go outside map
					if (dynamic_maps[k].mipmap > 0) {
						for (int l = 0; l < 3; l++) {
							push_constant.limits[l] = MAX(1, push_constant.limits[l] >> 1);
						}
					}

					//print_line("rect: " + itos(i) + ": " + rect);
					push_constant.rect_pos[0] = rect.position[0];
					push_constant.rect_pos[1] = rect.position[1];
					push_constant.prev_rect_size[0] = push_constant.rect_size[0];
					push_constant.prev_rect_size[1] = push_constant.rect_size[1];
					push_constant.rect_size[0] = rect.size[0];
					push_constant.rect_size[1] = rect.size[1];
					push_constant.on_mipmap = dynamic_maps[k].mipmap > 0;

					RD::get_singleton()->compute_list_add_barrier(compute_list);

					if (dynamic_maps[k].mipmap < 0) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE]);
					} else if (k < dynamic_maps.size() - 1) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT]);
					} else {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_PLOT]);
					}
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, dynamic_maps[k].uniform_set, 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIDynamicPushConstant));
					RD::get_singleton()->compute_list_dispatch(compute_list, Math::division_round_up(rect.size.x, 8), Math::division_round_up(rect.size.y, 8), 1);
				}

				RD::get_singleton()->compute_list_end();
			}
		}

		has_dynamic_object_data = true; //clear until dynamic object data is used again
	}

	last_probe_version = gi->voxel_gi_get_version(probe);
}

void GI::VoxelGIInstance::free_resources() {
	if (texture.is_valid()) {
		RD::get_singleton()->free_rid(texture);
		RD::get_singleton()->free_rid(write_buffer);

		texture = RID();
		write_buffer = RID();
		mipmaps.clear();
	}

	for (int i = 0; i < dynamic_maps.size(); i++) {
		RD::get_singleton()->free_rid(dynamic_maps[i].texture);
		RD::get_singleton()->free_rid(dynamic_maps[i].depth);

		// these only exist on the first level...
		if (dynamic_maps[i].fb_depth.is_valid()) {
			RD::get_singleton()->free_rid(dynamic_maps[i].fb_depth);
		}
		if (dynamic_maps[i].albedo.is_valid()) {
			RD::get_singleton()->free_rid(dynamic_maps[i].albedo);
		}
		if (dynamic_maps[i].normal.is_valid()) {
			RD::get_singleton()->free_rid(dynamic_maps[i].normal);
		}
		if (dynamic_maps[i].orm.is_valid()) {
			RD::get_singleton()->free_rid(dynamic_maps[i].orm);
		}
	}
	dynamic_maps.clear();
}

void GI::VoxelGIInstance::debug(RD::DrawListID p_draw_list, RID p_framebuffer, const Projection &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha) {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	if (mipmaps.size() == 0) {
		return;
	}

	Projection cam_transform = (p_camera_with_transform * Projection(transform)) * Projection(gi->voxel_gi_get_to_cell_xform(probe).affine_inverse());

	int level = 0;
	Vector3i octree_size = gi->voxel_gi_get_octree_size(probe);

	VoxelGIDebugPushConstant push_constant;
	push_constant.alpha = p_alpha;
	push_constant.dynamic_range = gi->voxel_gi_get_dynamic_range(probe);
	push_constant.cell_offset = mipmaps[level].cell_offset;
	push_constant.level = level;

	push_constant.bounds[0] = octree_size.x >> level;
	push_constant.bounds[1] = octree_size.y >> level;
	push_constant.bounds[2] = octree_size.z >> level;
	push_constant.pad = 0;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			push_constant.projection[i * 4 + j] = cam_transform.columns[i][j];
		}
	}

	if (gi->voxel_gi_debug_uniform_set.is_valid()) {
		RD::get_singleton()->free_rid(gi->voxel_gi_debug_uniform_set);
	}
	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.binding = 1;
		u.append_id(gi->voxel_gi_get_data_buffer(probe));
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 2;
		u.append_id(texture);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 3;
		u.append_id(material_storage->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
		uniforms.push_back(u);
	}

	int cell_count;
	if (!p_emission && p_lighting && has_dynamic_object_data) {
		cell_count = push_constant.bounds[0] * push_constant.bounds[1] * push_constant.bounds[2];
	} else {
		cell_count = mipmaps[level].cell_count;
	}

	gi->voxel_gi_debug_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, gi->voxel_gi_debug_shader_version_shaders[0], 0);

	int voxel_gi_debug_pipeline = VOXEL_GI_DEBUG_COLOR;
	if (p_emission) {
		voxel_gi_debug_pipeline = VOXEL_GI_DEBUG_EMISSION;
	} else if (p_lighting) {
		voxel_gi_debug_pipeline = has_dynamic_object_data ? VOXEL_GI_DEBUG_LIGHT_FULL : VOXEL_GI_DEBUG_LIGHT;
	}
	RD::get_singleton()->draw_list_bind_render_pipeline(
			p_draw_list,
			gi->voxel_gi_debug_shader_version_pipelines[voxel_gi_debug_pipeline].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, gi->voxel_gi_debug_uniform_set, 0);
	RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(VoxelGIDebugPushConstant));
	RD::get_singleton()->draw_list_draw(p_draw_list, false, cell_count, 36);
}

////////////////////////////////////////////////////////////////////////////////
// GI

GI::GI() {
	singleton = this;

	hddagi_frames_to_converge = RSE::EnvironmentHDDAGIFramesToConverge(CLAMP(int32_t(GLOBAL_GET("rendering/global_illumination/hddagi/frames_to_converge")), 0, int32_t(RSE::ENV_HDDAGI_CONVERGE_MAX - 1)));
	hddagi_frames_to_update_light = RSE::EnvironmentHDDAGIFramesToUpdateLight(CLAMP(int32_t(GLOBAL_GET("rendering/global_illumination/hddagi/frames_to_update_lights")), 0, int32_t(RSE::ENV_HDDAGI_UPDATE_LIGHT_MAX - 1)));
	inactive_probe_frames = RSE::EnvironmentHDDAGIInactiveProbeFrames(CLAMP(int32_t(GLOBAL_GET("rendering/global_illumination/hddagi/frames_to_update_inactive_probes")), 0, int32_t(RSE::ENV_HDDAGI_INACTIVE_PROBE_MAX - 1)));
}

GI::~GI() {
	singleton = nullptr;
}

void GI::init(SkyRD *p_sky) {
	sky = p_sky;

	/* GI */

	{
		//kinda complicated to compute the amount of slots, we try to use as many as we can

		voxel_gi_lights = memnew_arr(VoxelGILight, voxel_gi_max_lights);
		voxel_gi_lights_uniform = RD::get_singleton()->uniform_buffer_create(voxel_gi_max_lights * sizeof(VoxelGILight));
		voxel_gi_quality = RSE::VoxelGIQuality(CLAMP(int(GLOBAL_GET("rendering/global_illumination/voxel_gi/quality")), 0, 1));

		String defines = "\n#define MAX_LIGHTS " + itos(voxel_gi_max_lights) + "\n";

		Vector<String> versions;
		versions.push_back("\n#define MODE_COMPUTE_LIGHT\n");
		versions.push_back("\n#define MODE_SECOND_BOUNCE\n");
		versions.push_back("\n#define MODE_UPDATE_MIPMAPS\n");
		versions.push_back("\n#define MODE_WRITE_TEXTURE\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_LIGHTING\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_WRITE\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_PLOT\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_PLOT\n#define MODE_DYNAMIC_SHRINK_WRITE\n");

		voxel_gi_shader.initialize(versions, defines);
		voxel_gi_lighting_shader_version = voxel_gi_shader.version_create();
		for (int i = 0; i < VOXEL_GI_SHADER_VERSION_MAX; i++) {
			voxel_gi_lighting_shader_version_shaders[i] = voxel_gi_shader.version_get_shader(voxel_gi_lighting_shader_version, i);
			voxel_gi_lighting_shader_version_pipelines[i] = RD::get_singleton()->compute_pipeline_create(voxel_gi_lighting_shader_version_shaders[i]);
		}
	}

	{
		String defines;
		Vector<String> versions;
		versions.push_back("\n#define MODE_DEBUG_COLOR\n");
		versions.push_back("\n#define MODE_DEBUG_LIGHT\n");
		versions.push_back("\n#define MODE_DEBUG_EMISSION\n");
		versions.push_back("\n#define MODE_DEBUG_LIGHT\n#define MODE_DEBUG_LIGHT_FULL\n");

		voxel_gi_debug_shader.initialize(versions, defines);
		voxel_gi_debug_shader_version = voxel_gi_debug_shader.version_create();
		for (int i = 0; i < VOXEL_GI_DEBUG_MAX; i++) {
			voxel_gi_debug_shader_version_shaders[i] = voxel_gi_debug_shader.version_get_shader(voxel_gi_debug_shader_version, i);

			RD::PipelineRasterizationState rs;
			rs.cull_mode = RD::POLYGON_CULL_FRONT;
			RD::PipelineDepthStencilState ds;
			ds.enable_depth_test = true;
			ds.enable_depth_write = true;
			ds.depth_compare_operator = RD::COMPARE_OP_GREATER_OR_EQUAL;

			voxel_gi_debug_shader_version_pipelines[i].setup(voxel_gi_debug_shader_version_shaders[i], RD::RENDER_PRIMITIVE_TRIANGLES, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	/* HDDAGI */

	hddagi_shader.packed_occlusion = RD::get_singleton()->texture_can_share_format(RD::DATA_FORMAT_R16_UINT, RD::DATA_FORMAT_R4G4B4A4_UNORM_PACK16);

	{
		Vector<String> preprocess_modes;
		preprocess_modes.push_back("\n#define MODE_REGION_STORE\n");
		preprocess_modes.push_back("\n#define MODE_LIGHT_STORE\n");
		preprocess_modes.push_back("\n#define MODE_LIGHT_SCROLL\n");
		preprocess_modes.push_back("\n#define MODE_OCCLUSION\n");
		preprocess_modes.push_back("\n#define MODE_OCCLUSION_FILTER\n");
		preprocess_modes.push_back("\n#define MODE_LIGHTPROBE_SCROLL\n");
		preprocess_modes.push_back("\n#define MODE_LIGHTPROBE_NEIGHBOURS\n");
		preprocess_modes.push_back("\n#define MODE_LIGHTPROBE_GEOMETRY_PROXIMITY\n");
		preprocess_modes.push_back("\n#define MODE_LIGHTPROBE_UPDATE_FRAMES\n");

		String defines = "\n#define LIGHTPROBE_OCT_SIZE " + itos(HDDAGI::LIGHTPROBE_OCT_SIZE) + "\n#define OCCLUSION_OCT_SIZE " + itos(HDDAGI::OCCLUSION_OCT_SIZE) + "\n#define OCCLUSION_OCT_SIZE_HALF " + itos(HDDAGI::OCCLUSION_OCT_SIZE / 2) + "\n#define OCCLUSION_SUBPIXELS " + itos(HDDAGI::OCCLUSION_SUBPIXELS) + "\n";

		if (!hddagi_shader.packed_occlusion) {
			defines += "\n#define HDDAGI_RGBA8_OCCLUSION\n";
		}
		hddagi_shader.preprocess.initialize(preprocess_modes, defines);
		hddagi_shader.preprocess_shader = hddagi_shader.preprocess.version_create();
		for (int i = 0; i < HDDAGIShader::PRE_PROCESS_MAX; i++) {
			hddagi_shader.preprocess_shader_version[i] = hddagi_shader.preprocess.version_get_shader(hddagi_shader.preprocess_shader, i);
			hddagi_shader.preprocess_pipeline[i] = RD::get_singleton()->compute_pipeline_create(hddagi_shader.preprocess_shader_version[i]);
		}
	}

	{
		//calculate tables
		String defines = "\n#define LIGHTPROBE_OCT_SIZE " + itos(HDDAGI::LIGHTPROBE_OCT_SIZE) + "\n#define OCCLUSION_OCT_SIZE " + itos(HDDAGI::OCCLUSION_OCT_SIZE) + "\n";

		Vector<String> direct_light_modes;
		static_assert(HDDAGI::MAX_DYNAMIC_LIGHTS == 128, "The light-region shader stores four 32-bit masks per region.");
		Vector<String> cull_modes;
		cull_modes.push_back("");
		hddagi_shader.light_cull.initialize(cull_modes);
		hddagi_shader.light_cull_shader = hddagi_shader.light_cull.version_create();
		hddagi_shader.light_cull_version = hddagi_shader.light_cull.version_get_shader(hddagi_shader.light_cull_shader, 0);
		hddagi_shader.light_cull_pipeline = RD::get_singleton()->compute_pipeline_create(hddagi_shader.light_cull_version);
		direct_light_modes.push_back("\n#define MODE_PROCESS_STATIC\n");
		direct_light_modes.push_back("\n#define MODE_PROCESS_DYNAMIC\n");
		hddagi_shader.direct_light.initialize(direct_light_modes, defines);
		hddagi_shader.direct_light_shader = hddagi_shader.direct_light.version_create();
		for (int i = 0; i < HDDAGIShader::DIRECT_LIGHT_MODE_MAX; i++) {
			hddagi_shader.direct_light_shader_version[i] = hddagi_shader.direct_light.version_get_shader(hddagi_shader.direct_light_shader, i);
			hddagi_shader.direct_light_pipeline[i] = RD::get_singleton()->compute_pipeline_create(hddagi_shader.direct_light_shader_version[i]);
		}
	}

	{
		//calculate tables
		String defines = "\n#define LIGHTPROBE_OCT_SIZE " + itos(HDDAGI::LIGHTPROBE_OCT_SIZE) + "\n#define OCCLUSION_OCT_SIZE " + itos(HDDAGI::OCCLUSION_OCT_SIZE) + "\n";

		defines += "\n#define SH_SIZE " + itos(HDDAGI::SH_SIZE) + "\n";
		if (p_sky->sky_use_octmap_array) {
			defines += "\n#define USE_RADIANCE_OCTMAP_ARRAY\n";
		}

		auto create_probe_lookup_buffer = [](Span<uint32_t> p_values) {
			// Wavefront sizes/offsets and packed positions for occlusion preprocessing.
			static const uint32_t occlusion_lookup_data[] = { 1, 0, 3, 1, 6, 4, 10, 10, 15, 20, 21, 35, 28, 56, 36, 84, 42, 120, 46, 162, 48, 208, 0, 65536, 256, 1, 131072, 65792, 512, 65537, 257, 2, 196608, 131328, 66048, 768, 131073, 65793, 513, 65538, 258, 3, 262144, 196864, 131584, 66304, 1024, 196609, 131329, 66049, 769, 131074, 65794, 514, 65539, 259, 4, 327680, 262400, 197120, 131840, 66560, 1280, 262145, 196865, 131585, 66305, 1025, 196610, 131330, 66050, 770, 131075, 65795, 515, 65540, 260, 5, 393216, 327936, 262656, 197376, 132096, 66816, 1536, 327681, 262401, 197121, 131841, 66561, 1281, 262146, 196866, 131586, 66306, 1026, 196611, 131331, 66051, 771, 131076, 65796, 516, 65541, 261, 6, 458752, 393472, 328192, 262912, 197632, 132352, 67072, 1792, 393217, 327937, 262657, 197377, 132097, 66817, 1537, 327682, 262402, 197122, 131842, 66562, 1282, 262147, 196867, 131587, 66307, 1027, 196612, 131332, 66052, 772, 131077, 65797, 517, 65542, 262, 7, 459008, 393728, 328448, 263168, 197888, 132608, 67328, 458753, 393473, 328193, 262913, 197633, 132353, 67073, 1793, 393218, 327938, 262658, 197378, 132098, 66818, 1538, 327683, 262403, 197123, 131843, 66563, 1283, 262148, 196868, 131588, 66308, 1028, 196613, 131333, 66053, 773, 131078, 65798, 518, 65543, 263, 459264, 393984, 328704, 263424, 198144, 132864, 459009, 393729, 328449, 263169, 197889, 132609, 67329, 458754, 393474, 328194, 262914, 197634, 132354, 67074, 1794, 393219, 327939, 262659, 197379, 132099, 66819, 1539, 327684, 262404, 197124, 131844, 66564, 1284, 262149, 196869, 131589, 66309, 1029, 196614, 131334, 66054, 774, 131079, 65799, 519, 459520, 394240, 328960, 263680, 198400, 459265, 393985, 328705, 263425, 198145, 132865, 459010, 393730, 328450, 263170, 197890, 132610, 67330, 458755, 393475, 328195, 262915, 197635, 132355, 67075, 1795, 393220, 327940, 262660, 197380, 132100, 66820, 1540, 327685, 262405, 197125, 131845, 66565, 1285, 262150, 196870, 131590, 66310, 1030, 196615, 131335, 66055, 775 };
			static_assert(sizeof(occlusion_lookup_data) == 278 * sizeof(uint32_t));
			DEV_ASSERT((p_values.size() & 1) == 0); // Keep the following uvec2 table aligned.
			Vector<uint8_t> data;
			data.resize(p_values.size() * sizeof(uint32_t) + sizeof(occlusion_lookup_data));
			memcpy(data.ptrw(), p_values.ptr(), p_values.size() * sizeof(uint32_t));
			memcpy(data.ptrw() + p_values.size() * sizeof(uint32_t), occlusion_lookup_data, sizeof(occlusion_lookup_data));
			return RD::get_singleton()->storage_buffer_create(uint32_t(data.size()), data);
		};
		static_assert(HDDAGI::LIGHTPROBE_OCT_SIZE >= 4 && HDDAGI::LIGHTPROBE_OCT_SIZE <= 6);
		if constexpr (HDDAGI::LIGHTPROBE_OCT_SIZE == 4) {
			static const uint32_t probe_lookup_data[] = { 15544, 73563, 135085, 206971, 270171, 528301, 796795, 988221, 8569, 82130, 144347, 200892, 272100, 336249, 397500, 0, 4284, 78811, 147666, 205177, 331964, 401785, 468708, 0, 10363, 69549, 139099, 212152, 466779, 724909, 791613, 993403, 8569, 75492, 278738, 336249, 537563, 594108, 790716, 0, 73563, 135085, 270171, 343224, 403579, 528301, 600187, 660541, 69549, 139099, 338043, 408760, 466779, 595005, 665723, 724909, 141028, 205177, 401785, 475346, 659644, 734171, 987324, 0, 4284, 275419, 331964, 540882, 598393, 795001, 861924, 0, 266157, 338043, 398397, 532315, 605368, 665723, 859995, 921517, 332861, 403579, 462765, 600187, 670904, 728923, 855981, 925531, 200892, 397500, 472027, 663929, 737490, 927460, 991609, 0, 10363, 201789, 266157, 532315, 801976, 859995, 921517, 993403, 534244, 598393, 659644, 795001, 868562, 930779, 987324, 0, 594108, 663929, 730852, 790716, 865243, 934098, 991609, 0, 5181, 206971, 462765, 728923, 796795, 855981, 925531, 998584, 196611, 3, 2, 1, 0, 196608, 196608, 0, 1, 2, 3, 196611, 131072, 65536, 65537, 65538, 65539, 131075, 65536, 131072, 131073, 131074, 131075, 65539, 0, 196608, 196609, 196610, 196611, 3, 3, 196611, 196610, 196609, 196608, 0 };
			hddagi_shader.integrate_lookup_buffer = create_probe_lookup_buffer(probe_lookup_data);
		} else if constexpr (HDDAGI::LIGHTPROBE_OCT_SIZE == 5) {
			static const uint32_t probe_lookup_data[] = { 11139, 72624, 131886, 201671, 271258, 334768, 394335, 590836, 656174, 988103, 1319834, 1377268, 1579952, 0, 0, 6839, 76283, 139717, 205401, 267029, 334519, 400776, 461448, 527528, 590801, 657717, 984017, 1311697, 0, 0, 778, 74103, 141723, 205175, 262922, 330016, 400965, 466633, 532037, 592160, 655986, 723045, 789015, 854117, 918130, 4885, 74329, 139717, 207355, 268983, 328657, 396456, 461448, 531848, 596663, 919861, 1246161, 1573841, 0, 0, 9114, 70599, 131886, 203696, 273283, 328692, 525407, 596912, 918318, 1250247, 1317808, 1508340, 1581978, 0, 0, 6839, 72375, 133429, 197585, 263121, 338427, 400776, 664005, 723592, 991833, 1051816, 1315605, 1377233, 0, 0, 1026, 72722, 138504, 199687, 334866, 403430, 465362, 525422, 662792, 727506, 789836, 986119, 1049710, 0, 0, 68049, 138485, 199121, 399699, 468771, 530771, 657381, 727832, 794768, 858904, 919525, 1117965, 0, 0, 0, 68615, 138504, 203794, 263170, 394350, 465362, 534502, 597010, 789836, 858578, 924936, 1180782, 1248263, 0, 0, 977, 66513, 133429, 203447, 268983, 531848, 600571, 854664, 926149, 1182888, 1253977, 1508305, 1577749, 0, 0, 778, 67872, 131698, 336247, 400965, 460901, 666011, 728777, 789015, 991607, 1056325, 1116261, 1311498, 1378592, 1442418, 133093, 330193, 399699, 465688, 662773, 730915, 794768, 855821, 985553, 1055059, 1121048, 1443813, 0, 0, 0, 133468, 396510, 466974, 527582, 657756, 729118, 796314, 860190, 919900, 1051870, 1122334, 1182942, 1444188, 0, 0, 133093, 465688, 530771, 592337, 724749, 794768, 861987, 924917, 1121048, 1186131, 1247697, 1443813, 0, 0, 0, 131698, 198944, 262922, 460901, 532037, 598391, 789015, 859849, 928155, 1116261, 1187397, 1253751, 1442418, 1509664, 1573642, 4885, 66513, 336473, 396456, 664005, 723592, 993787, 1056136, 1317559, 1383095, 1444149, 1508305, 1573841, 0, 0, 330759, 394350, 662792, 727506, 789836, 990226, 1058790, 1120722, 1180782, 1311746, 1383442, 1449224, 1510407, 0, 0, 462605, 657381, 727832, 794768, 858904, 919525, 1055059, 1124131, 1186131, 1378769, 1449205, 1509841, 0, 0, 0, 525422, 592903, 789836, 858578, 924936, 1049710, 1120722, 1189862, 1252370, 1379335, 1449224, 1514514, 1573890, 0, 0, 197585, 267029, 527528, 598617, 854664, 926149, 1187208, 1255931, 1311697, 1377233, 1444149, 1514167, 1579703, 0, 0, 9114, 66548, 269232, 332743, 656174, 990128, 1049695, 1246196, 1321859, 1383344, 1442606, 1512391, 1581978, 0, 0, 977, 328657, 657717, 989879, 1056136, 1116808, 1182888, 1246161, 1317559, 1387003, 1450437, 1516121, 1577749, 0, 0, 655986, 723045, 789015, 854117, 918130, 985376, 1056325, 1121993, 1187397, 1247520, 1311498, 1384823, 1452443, 1515895, 1573642, 263121, 590801, 919861, 984017, 1051816, 1116808, 1187208, 1252023, 1315605, 1385049, 1450437, 1518075, 1579703, 0, 0, 7088, 197620, 271258, 594887, 918318, 984052, 1180767, 1252272, 1319834, 1381319, 1442606, 1514416, 1584003, 0, 0, 262148, 4, 3, 2, 1, 0, 262144, 262144, 0, 1, 2, 3, 4, 262148, 196608, 65536, 65537, 65538, 65539, 65540, 196612, 131072, 131072, 131073, 131074, 131075, 131076, 131076, 65536, 196608, 196609, 196610, 196611, 196612, 65540, 0, 262144, 262145, 262146, 262147, 262148, 4, 4, 262148, 262147, 262146, 262145, 262144, 0 };
			hddagi_shader.integrate_lookup_buffer = create_probe_lookup_buffer(probe_lookup_data);
		} else if constexpr (HDDAGI::LIGHTPROBE_OCT_SIZE == 6) {
			static const uint32_t probe_lookup_data[] = { 7409, 71136, 133126, 197977, 266811, 334266, 398816, 461221, 723696, 788486, 1181017, 1577531, 1902410, 1972666, 2034416, 2230090, 2299522, 0, 5357, 72623, 137015, 201808, 268219, 332144, 398278, 464109, 527325, 591849, 658038, 722921, 789403, 852980, 1180390, 1574889, 1968758, 2295545, 2062, 71772, 138508, 203472, 267601, 329055, 396334, 464939, 530524, 595281, 659485, 721675, 789292, 855086, 919566, 984415, 1049355, 0, 1375, 70993, 137936, 204044, 268380, 329742, 393995, 462877, 529745, 596060, 661547, 724014, 852747, 918879, 985102, 1051694, 1116972, 0, 4464, 71611, 136272, 202551, 269231, 333037, 395241, 461430, 526313, 592861, 660717, 725958, 1049588, 1117083, 1508070, 1902569, 1967865, 2296438, 6586, 70203, 132441, 198662, 267744, 335089, 396016, 657829, 726496, 1116166, 1508697, 1574730, 1905211, 1971842, 2033482, 2231024, 2300346, 0, 5357, 70598, 134043, 197350, 264169, 330358, 400303, 464109, 525300, 792375, 855005, 1184848, 1247209, 1578939, 1641078, 1970544, 2033641, 2295545, 2433, 71055, 137147, 200658, 264903, 398735, 466053, 529807, 592583, 792507, 857487, 919937, 1183698, 1247943, 1575623, 0, 0, 0, 68573, 137015, 201808, 264169, 394228, 464109, 531375, 595899, 658038, 789403, 857030, 922861, 987504, 1050601, 1180390, 1247209, 1313398, 1378041, 67561, 136272, 202551, 265181, 461430, 530363, 596911, 660717, 721908, 853993, 921968, 988397, 1053638, 1117083, 1312505, 1378934, 1443817, 1508070, 68295, 135122, 202683, 267663, 330113, 527047, 595343, 662661, 726415, 985473, 1054095, 1120187, 1444551, 1511378, 1903303, 0, 0, 0, 2678, 67561, 131814, 199579, 267206, 333037, 590836, 660717, 727983, 1051613, 1120055, 1443817, 1512528, 1837686, 1906619, 1967865, 2230249, 2298224, 2062, 68654, 133932, 399452, 464939, 527406, 793868, 858204, 919566, 1186512, 1250641, 1312095, 1578321, 1642525, 1704715, 1967455, 2032395, 0, 66548, 134043, 197350, 396253, 464109, 529350, 591849, 792375, 859055, 922861, 985718, 1184848, 1251259, 1315184, 1378041, 1574889, 1641078, 1705961, 133126, 197977, 461221, 529888, 594491, 788486, 857568, 924913, 989626, 1051376, 1181017, 1249851, 1317306, 1382018, 1443658, 1706736, 1771338, 0, 132441, 198662, 528955, 595424, 657829, 854768, 924090, 990449, 1054176, 1116166, 1247050, 1316482, 1382842, 1446459, 1508697, 1705802, 1772272, 0, 131814, 199579, 263156, 526313, 594886, 660717, 723933, 920182, 988397, 1055663, 1120055, 1312505, 1380720, 1447867, 1512528, 1771497, 1837686, 1902569, 199468, 265262, 329742, 592942, 661547, 727132, 985102, 1054812, 1121548, 1377631, 1447249, 1514192, 1770251, 1839133, 1906001, 2229003, 2295135, 0, 1375, 66315, 398673, 462877, 525067, 793296, 857425, 918879, 1187084, 1251420, 1312782, 1579100, 1644587, 1707054, 1968142, 2034734, 2100012, 0, 395241, 461430, 526313, 791632, 858043, 921968, 984825, 1185591, 1252271, 1316077, 1378934, 1575901, 1643757, 1708998, 1771497, 2032628, 2100123, 2163430, 527088, 591690, 787801, 856635, 924090, 988802, 1050442, 1181702, 1250784, 1318129, 1382842, 1444592, 1640869, 1709536, 1774139, 2099206, 2164057, 0, 526154, 592624, 853834, 923266, 989626, 1053243, 1115481, 1247984, 1317306, 1383665, 1447392, 1509382, 1708603, 1775072, 1837477, 2098521, 2164742, 0, 591849, 658038, 722921, 919289, 987504, 1054651, 1119312, 1313398, 1381613, 1448879, 1513271, 1705961, 1774534, 1840365, 1903581, 2097894, 2165659, 2229236, 262923, 329055, 590603, 659485, 726353, 984415, 1054033, 1120976, 1378318, 1448028, 1514764, 1772590, 1841195, 1906780, 2165548, 2231342, 2295822, 0, 4464, 67561, 329465, 399291, 461430, 791632, 853993, 1185591, 1248221, 1579951, 1643757, 1704948, 1971437, 2036678, 2100123, 2163430, 2230249, 2296438, 395975, 790482, 854727, 1185723, 1250703, 1313153, 1578383, 1645701, 1709455, 1772231, 1968513, 2037135, 2103227, 2166738, 2230983, 0, 0, 0, 787174, 853993, 920182, 984825, 1182619, 1250246, 1316077, 1380720, 1443817, 1573876, 1643757, 1711023, 1775547, 1837686, 2034653, 2103095, 2167888, 2230249, 919289, 985718, 1050601, 1114854, 1247209, 1315184, 1381613, 1446854, 1510299, 1641078, 1710011, 1776559, 1840365, 1901556, 2033641, 2102352, 2168631, 2231261, 723655, 1051335, 1118162, 1378689, 1447311, 1513403, 1706695, 1774991, 1842309, 1906063, 2034375, 2101202, 2168763, 2233743, 2296193, 0, 0, 0, 1785, 264169, 332144, 658038, 726971, 1050601, 1119312, 1444829, 1513271, 1770484, 1840365, 1907631, 1968758, 2033641, 2097894, 2165659, 2233286, 2299117, 6586, 68336, 264010, 333442, 397883, 722762, 787801, 1181702, 1578464, 1640869, 1903344, 1973489, 2037216, 2099206, 2164057, 2232891, 2300346, 0, 2678, 329465, 395241, 787174, 1182619, 1246196, 1577926, 1643757, 1706973, 1771497, 1837686, 1902569, 1971437, 2038703, 2103095, 2167888, 2234299, 2298224, 1182508, 1248302, 1312782, 1377631, 1442571, 1575982, 1644587, 1710172, 1774929, 1839133, 1901323, 1968142, 2037852, 2104588, 2169552, 2233681, 2295135, 0, 1245963, 1312095, 1378318, 1444910, 1510188, 1573643, 1642525, 1709393, 1775708, 1841195, 1903662, 1967455, 2037073, 2104016, 2170124, 2234460, 2295822, 0, 1785, 330358, 722921, 1114854, 1442804, 1510299, 1574889, 1641078, 1705961, 1772509, 1840365, 1905606, 1970544, 2037691, 2102352, 2168631, 2235311, 2299117, 5762, 67402, 264944, 334266, 395082, 725563, 1115481, 1509382, 1575664, 1837477, 1906144, 1972666, 2036283, 2098521, 2164742, 2233824, 2301169, 0, 327685, 5, 4, 3, 2, 1, 0, 327680, 327680, 0, 1, 2, 3, 4, 5, 327685, 262144, 65536, 65537, 65538, 65539, 65540, 65541, 262149, 196608, 131072, 131073, 131074, 131075, 131076, 131077, 196613, 131072, 196608, 196609, 196610, 196611, 196612, 196613, 131077, 65536, 262144, 262145, 262146, 262147, 262148, 262149, 65541, 0, 327680, 327681, 327682, 327683, 327684, 327685, 5, 5, 327685, 327684, 327683, 327682, 327681, 327680, 0 };
			hddagi_shader.integrate_lookup_buffer = create_probe_lookup_buffer(probe_lookup_data);
		}

		Vector<String> integrate_modes;
		integrate_modes.push_back("\n#define MODE_PROCESS\n");
		integrate_modes.push_back("\n#define MODE_FILTER\n");
		integrate_modes.push_back("\n#define MODE_CAMERA_VISIBILITY\n");
		hddagi_shader.integrate.initialize(integrate_modes, defines);
		hddagi_shader.integrate_shader = hddagi_shader.integrate.version_create();

		for (int i = 0; i < HDDAGIShader::INTEGRATE_MODE_MAX; i++) {
			hddagi_shader.integrate_shader_version[i] = hddagi_shader.integrate.version_get_shader(hddagi_shader.integrate_shader, i);
			hddagi_shader.integrate_pipeline[i] = RD::get_singleton()->compute_pipeline_create(hddagi_shader.integrate_shader_version[i]);
		}

		Vector<RD::Uniform> uniforms;
		uniforms.push_back(RD::Uniform(
				RD::UNIFORM_TYPE_TEXTURE,
				0,
				RendererRD::TextureStorage::get_singleton()->texture_rd_get_default(
						p_sky->sky_use_octmap_array ? RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE : RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_WHITE)));
		uniforms.push_back(RD::Uniform(
				RD::UNIFORM_TYPE_SAMPLER,
				1,
				RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)));
		hddagi_shader.integrate_default_sky_uniform_set = RD::get_singleton()->uniform_set_create(
				uniforms,
				hddagi_shader.integrate.version_get_shader(hddagi_shader.integrate_shader, 0),
				1);
	}

	{
		Vector<String> screen_probe_modes;
		screen_probe_modes.push_back("\n#define MODE_SURFACE\n");
		for (int i = 0; i < HDDAGIShader::DIRECTIONAL_TRACE_VARIANT_COUNT; i++) {
			String defines = "\n#define MODE_DIRECTIONAL_TRACE\n";
			if (i & HDDAGIShader::DIRECTIONAL_TRACE_IMPORTANCE) {
				defines += "#define MODE_DIRECTIONAL_TRACE_IMPORTANCE\n";
			}
			if (i & HDDAGIShader::DIRECTIONAL_TRACE_ADAPTIVE) {
				defines += "#define MODE_ADAPTIVE\n";
			}
			if (i & HDDAGIShader::DIRECTIONAL_TRACE_MULTISAMPLE) {
				defines += "#define MODE_MULTISAMPLE\n";
			}
			screen_probe_modes.push_back(defines);
		}
		screen_probe_modes.push_back("\n#define MODE_DIRECTIONAL_IRRADIANCE\n");
		screen_probe_modes.push_back("\n#define MODE_RESOLVE\n");
		screen_probe_modes.push_back("\n#define MODE_RESOLVE\n#define MODE_SVGF_PREPARE\n");
		screen_probe_modes.push_back("\n#define MODE_APPLY\n");
		screen_probe_modes.push_back("\n#define MODE_APPLY\n#define MODE_SVGF_APPLY\n");
		screen_probe_modes.push_back("\n#define MODE_APPLY\n#define MODE_SHORT_RANGE_AO\n");
		screen_probe_modes.push_back("\n#define MODE_APPLY\n#define MODE_SVGF_APPLY\n#define MODE_SHORT_RANGE_AO\n");
		screen_probe_modes.push_back("\n#define MODE_SPECULAR_TRACE\n");
		screen_probe_modes.push_back("\n#define MODE_SPECULAR_APPLY\n");
		screen_probe_modes.push_back("\n#define MODE_DIRECTIONAL_ADAPTIVE_MARK\n");
		screen_probe_modes.push_back("\n#define MODE_DIRECTIONAL_ADAPTIVE_SPAWN\n");
		screen_probe_modes.push_back("\n#define MODE_RESOLVE\n#define MODE_ADAPTIVE\n");
		screen_probe_modes.push_back("\n#define MODE_RESOLVE\n#define MODE_SVGF_PREPARE\n#define MODE_ADAPTIVE\n");
		DEV_ASSERT(screen_probe_modes.size() == HDDAGIShader::SCREEN_PROBE_MODE_MAX);

		String screen_probe_defines;
		if (p_sky->sky_use_octmap_array) {
			screen_probe_defines += "\n#define USE_RADIANCE_OCTMAP_ARRAY\n";
		}
		screen_probe_defines += "\n#define LIGHTPROBE_OCT_SIZE " + itos(HDDAGI::LIGHTPROBE_OCT_SIZE) + "\n";

		hddagi_shader.screen_probe.initialize(screen_probe_modes, screen_probe_defines);
		if (!HDDAGIScreenProbeSVGF::is_supported()) {
			const HDDAGIShader::ScreenProbeMode svgf_modes[] = {
				HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_SVGF,
				HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE_SVGF_ADAPTIVE,
				HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF,
				HDDAGIShader::SCREEN_PROBE_MODE_APPLY_SVGF_AO,
			};
			for (HDDAGIShader::ScreenProbeMode mode : svgf_modes) {
				hddagi_shader.screen_probe.set_variant_enabled(mode, false);
			}
		}
		hddagi_shader.screen_probe_shader = hddagi_shader.screen_probe.version_create();
		for (int i = 0; i < HDDAGIShader::SCREEN_PROBE_MODE_MAX; i++) {
			if (!hddagi_shader.screen_probe.is_variant_enabled(i)) {
				continue;
			}
			hddagi_shader.screen_probe_shader_version[i] = hddagi_shader.screen_probe.version_get_shader(hddagi_shader.screen_probe_shader, i);
			if (hddagi_shader.screen_probe_shader_version[i].is_valid()) {
				hddagi_shader.screen_probe_pipeline[i] = RD::get_singleton()->compute_pipeline_create(hddagi_shader.screen_probe_shader_version[i]);
			}
		}
		const HDDAGIShader::ScreenProbeMode required_modes[] = {
			HDDAGIShader::SCREEN_PROBE_MODE_SURFACE,
			HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_TRACE,
			HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_IRRADIANCE,
			HDDAGIShader::SCREEN_PROBE_MODE_DIRECTIONAL_RESOLVE,
			HDDAGIShader::SCREEN_PROBE_MODE_APPLY,
		};
		hddagi_shader.screen_probe_available = true;
		for (HDDAGIShader::ScreenProbeMode mode : required_modes) {
			hddagi_shader.screen_probe_available = hddagi_shader.screen_probe_available && hddagi_shader.screen_probe_pipeline[mode].is_valid();
		}
	}

	//GK
	{
		//calculate tables
		String defines = "\n#define LIGHTPROBE_OCT_SIZE " + itos(HDDAGI::LIGHTPROBE_OCT_SIZE) + "\n#define OCCLUSION_OCT_SIZE " + itos(HDDAGI::OCCLUSION_OCT_SIZE) + "\n";
		if (RendererSceneRenderRD::get_singleton()->is_vrs_supported()) {
			defines += "\n#define USE_VRS\n";
		}
		if (!RD::get_singleton()->sampler_is_format_supported_for_filter(RD::DATA_FORMAT_R8G8_UINT, RD::SAMPLER_FILTER_LINEAR)) {
			defines += "\n#define SAMPLE_VOXEL_GI_NEAREST\n";
		}

		Vector<String> gi_modes;

		gi_modes.push_back("\n#define USE_VOXEL_GI_INSTANCES\n"); // MODE_VOXEL_GI
		gi_modes.push_back("\n#define USE_HDDAGI\n"); // MODE_HDDAGI
		gi_modes.push_back("\n#define USE_HDDAGI\n\n#define USE_VOXEL_GI_INSTANCES\n"); // MODE_COMBINED
		gi_modes.push_back("\n#define USE_HDDAGI\n\n#define USE_AMBIENT_BLEND\n"); // MODE_HDDAGI
		gi_modes.push_back("\n#define USE_HDDAGI\n\n#define USE_VOXEL_GI_INSTANCES\n\n#define USE_AMBIENT_BLEND\n"); // MODE_COMBINED

		shader.initialize(gi_modes, defines);
		shader_version = shader.version_create();

		Vector<RD::PipelineSpecializationConstant> specialization_constants;

		{
			RD::PipelineSpecializationConstant sc;
			sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			sc.constant_id = 0; // SHADER_SPECIALIZATION_HALF_RES
			sc.bool_value = false;
			specialization_constants.push_back(sc);

			sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			sc.constant_id = 1; // SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX
			sc.bool_value = false;
			specialization_constants.push_back(sc);

			sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			sc.constant_id = 2; // SHADER_SPECIALIZATION_USE_VRS
			sc.bool_value = false;
			specialization_constants.push_back(sc);

			sc.constant_id = 3; // SHADER_SPECIALIZATION_HALF_REFLECTIONS
			specialization_constants.push_back(sc);

			sc.constant_id = 4; // SHADER_SPECIALIZATION_DIFFUSE_ONLY
			specialization_constants.push_back(sc);
		}

		for (int v = 0; v < SHADER_SPECIALIZATION_VARIATIONS; v++) {
			specialization_constants.ptrw()[0].bool_value = (v & SHADER_SPECIALIZATION_HALF_RES) ? true : false;
			specialization_constants.ptrw()[1].bool_value = (v & SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX) ? true : false;
			specialization_constants.ptrw()[2].bool_value = (v & SHADER_SPECIALIZATION_USE_VRS) ? true : false;
			specialization_constants.ptrw()[3].bool_value = (v & SHADER_SPECIALIZATION_HALF_REFLECTIONS) ? true : false;
			specialization_constants.ptrw()[4].bool_value = v & SHADER_SPECIALIZATION_DIFFUSE_ONLY;
			for (int i = 0; i < MODE_MAX; i++) {
				if (specialization_constants[4].bool_value && ((v & SHADER_SPECIALIZATION_HALF_REFLECTIONS) || (i != MODE_HDDAGI && i != MODE_HDDAGI_BLEND_AMBIENT))) {
					continue;
				}
				pipelines[v][i] = RD::get_singleton()->compute_pipeline_create(shader.version_get_shader(shader_version, i), specialization_constants);
			}
		}

		hddagi_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(HDDAGIData));

		RD::TextureFormat tf_default_hddagi;
		tf_default_hddagi.width = 1;
		tf_default_hddagi.height = 1;
		tf_default_hddagi.depth = 1;
		tf_default_hddagi.texture_type = RD::TEXTURE_TYPE_3D;
		tf_default_hddagi.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		tf_default_hddagi.format = RD::DATA_FORMAT_R32G32_UINT;
		default_hddagi_voxel_bits_texture = create_clear_texture(tf_default_hddagi, "Default HDDAGI Voxel Field");

		tf_default_hddagi.format = RD::DATA_FORMAT_R8_UINT;
		default_hddagi_voxel_region_texture = create_clear_texture(tf_default_hddagi, "Default HDDAGI Voxel Regions");
	}
	{
		String defines;
		//calculate tables
		if (RendererSceneRenderRD::get_singleton()->is_vrs_supported()) {
			defines += "\n#define USE_VRS\n";
		}
		Vector<String> filter_modes;
		filter_modes.push_back("\n#define MODE_BILATERAL_FILTER\n");
		filter_modes.push_back("\n#define MODE_BILATERAL_FILTER\n#define HALF_SIZE\n");
		filter_shader.initialize(filter_modes, defines);
		filter_shader_version = filter_shader.version_create();

		Vector<RD::PipelineSpecializationConstant> specialization_constants;

		{
			RD::PipelineSpecializationConstant sc;
			sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			sc.constant_id = 0; // FILTER_SHADER_SPECIALIZATION_HALF_RES
			sc.bool_value = false;
			specialization_constants.push_back(sc);

			sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			sc.constant_id = 1; // FILTER_SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX
			sc.bool_value = false;
			specialization_constants.push_back(sc);
		}

		for (int v = 0; v < FILTER_SHADER_SPECIALIZATION_VARIATIONS; v++) {
			specialization_constants.ptrw()[0].bool_value = (v & FILTER_SHADER_SPECIALIZATION_HALF_RES) ? true : false;
			specialization_constants.ptrw()[1].bool_value = (v & FILTER_SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX) ? true : false;
			for (int i = 0; i < FILTER_MODE_MAX; i++) {
				filter_pipelines[v][i] = RD::get_singleton()->compute_pipeline_create(filter_shader.version_get_shader(filter_shader_version, i), specialization_constants);
			}
		}
	}

	{
		Vector<String> versions;
		versions.push_back("");
		hddagi_shader.debug.initialize(versions);
		hddagi_shader.debug_shader = hddagi_shader.debug.version_create();
		hddagi_shader.debug_shader_version = hddagi_shader.debug.version_get_shader(hddagi_shader.debug_shader, 0);
		hddagi_shader.debug_pipeline = RD::get_singleton()->compute_pipeline_create(hddagi_shader.debug_shader_version);
	}
	{
		Vector<String> versions;
		versions.push_back("");
		versions.push_back("\n#define USE_MULTIVIEW\n");
		hddagi_shader.debug_probes.initialize(versions);
		hddagi_shader.debug_probes.set_variant_enabled(HDDAGIShader::PROBE_DEBUG_PROBES_MULTIVIEW, RD::get_singleton()->has_feature(RD::SUPPORTS_MULTIVIEW));
		hddagi_shader.debug_probes_shader = hddagi_shader.debug_probes.version_create();

		RD::PipelineRasterizationState rs;
		rs.cull_mode = RD::POLYGON_CULL_DISABLED;
		RD::PipelineDepthStencilState ds;
		ds.enable_depth_test = true;
		ds.enable_depth_write = true;
		ds.depth_compare_operator = RD::COMPARE_OP_GREATER_OR_EQUAL;
		for (int i = 0; i < HDDAGIShader::PROBE_DEBUG_MAX; i++) {
			if (!hddagi_shader.debug_probes.is_variant_enabled(i)) {
				continue;
			}
			hddagi_shader.debug_probes_shader_version[i] = hddagi_shader.debug_probes.version_get_shader(hddagi_shader.debug_probes_shader, i);
			hddagi_shader.debug_probes_pipeline[i].setup(hddagi_shader.debug_probes_shader_version[i], RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	default_voxel_gi_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(VoxelGIData) * MAX_VOXEL_GI_INSTANCES);
	half_resolution = GLOBAL_GET("rendering/global_illumination/gi/use_half_resolution");
}

void GI::free() {
	if (default_voxel_gi_buffer.is_valid()) {
		RD::get_singleton()->free_rid(default_voxel_gi_buffer);
	}
	if (voxel_gi_lights_uniform.is_valid()) {
		RD::get_singleton()->free_rid(voxel_gi_lights_uniform);
	}
	if (hddagi_ubo.is_valid()) {
		RD::get_singleton()->free_rid(hddagi_ubo);
	}
	if (default_hddagi_voxel_bits_texture.is_valid()) {
		RD::get_singleton()->free_rid(default_hddagi_voxel_bits_texture);
	}
	if (default_hddagi_voxel_region_texture.is_valid()) {
		RD::get_singleton()->free_rid(default_hddagi_voxel_region_texture);
	}

	if (voxel_gi_debug_shader_version.is_valid()) {
		voxel_gi_debug_shader.version_free(voxel_gi_debug_shader_version);
	}
	if (voxel_gi_lighting_shader_version.is_valid()) {
		voxel_gi_shader.version_free(voxel_gi_lighting_shader_version);
	}
	if (shader_version.is_valid()) {
		shader.version_free(shader_version);
	}
	if (filter_shader_version.is_valid()) {
		filter_shader.version_free(filter_shader_version);
	}

	if (hddagi_shader.direct_light_shader.is_valid()) {
		hddagi_shader.direct_light.version_free(hddagi_shader.direct_light_shader);
	}
	if (hddagi_shader.integrate_lookup_buffer.is_valid()) {
		RD::get_singleton()->free_rid(hddagi_shader.integrate_lookup_buffer);
	}
	if (hddagi_shader.integrate_shader.is_valid()) {
		hddagi_shader.integrate.version_free(hddagi_shader.integrate_shader);
	}
	if (hddagi_shader.light_cull_shader.is_valid()) {
		hddagi_shader.light_cull.version_free(hddagi_shader.light_cull_shader);
	}
	if (hddagi_shader.screen_probe_shader.is_valid()) {
		hddagi_shader.screen_probe.version_free(hddagi_shader.screen_probe_shader);
	}
	if (hddagi_shader.debug_probes_shader.is_valid()) {
		hddagi_shader.debug_probes.version_free(hddagi_shader.debug_probes_shader);
	}
	if (hddagi_shader.debug_shader.is_valid()) {
		hddagi_shader.debug.version_free(hddagi_shader.debug_shader);
	}

	if (hddagi_shader.preprocess_shader.is_valid()) {
		hddagi_shader.preprocess.version_free(hddagi_shader.preprocess_shader);
	}

	if (voxel_gi_lights) {
		memdelete_arr(voxel_gi_lights);
	}
}

Ref<GI::HDDAGI> GI::create_hddagi(RID p_env, const Vector3 &p_world_anchor, const Vector3 &p_planar_world_forward, float p_cascade_forward_offset, uint32_t p_requested_history_size) {
	Ref<HDDAGI> hddagi;
	hddagi.instantiate();

	hddagi->create(p_env, p_world_anchor, p_planar_world_forward, p_cascade_forward_offset, p_requested_history_size, this);

	return hddagi;
}

void GI::setup_voxel_gi_instances(RenderDataRD *p_render_data, Ref<RenderSceneBuffersRD> p_render_buffers, const Transform3D &p_transform, const PagedArray<RID> &p_voxel_gi_instances, uint32_t &r_voxel_gi_instances_used) {
	ERR_FAIL_COND(p_render_buffers.is_null());

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	ERR_FAIL_NULL(texture_storage);

	r_voxel_gi_instances_used = 0;

	Ref<RenderBuffersGI> rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
	ERR_FAIL_COND(rbgi.is_null());

	RID voxel_gi_buffer = rbgi->get_voxel_gi_buffer();
	VoxelGIData voxel_gi_data[MAX_VOXEL_GI_INSTANCES];

	Transform3D to_camera;
	to_camera.origin = p_transform.origin; //only translation, make local

	for (int i = 0; i < MAX_VOXEL_GI_INSTANCES; i++) {
		RID texture;
		if (i < (int)p_voxel_gi_instances.size()) {
			VoxelGIInstance *gipi = voxel_gi_instance_owner.get_or_null(p_voxel_gi_instances[i]);

			if (gipi) {
				texture = gipi->texture;
				VoxelGIData &gipd = voxel_gi_data[i];

				RID base_probe = gipi->probe;

				Transform3D to_cell = voxel_gi_get_to_cell_xform(base_probe);
				Transform3D to_world_cell = to_cell * gipi->transform.affine_inverse() * to_camera;

				gipd.xform[0] = to_world_cell.basis.rows[0][0];
				gipd.xform[1] = to_world_cell.basis.rows[1][0];
				gipd.xform[2] = to_world_cell.basis.rows[2][0];
				gipd.xform[3] = 0;
				gipd.xform[4] = to_world_cell.basis.rows[0][1];
				gipd.xform[5] = to_world_cell.basis.rows[1][1];
				gipd.xform[6] = to_world_cell.basis.rows[2][1];
				gipd.xform[7] = 0;
				gipd.xform[8] = to_world_cell.basis.rows[0][2];
				gipd.xform[9] = to_world_cell.basis.rows[1][2];
				gipd.xform[10] = to_world_cell.basis.rows[2][2];
				gipd.xform[11] = 0;
				gipd.xform[12] = to_world_cell.origin.x;
				gipd.xform[13] = to_world_cell.origin.y;
				gipd.xform[14] = to_world_cell.origin.z;
				gipd.xform[15] = 1;

				Vector3 bounds = voxel_gi_get_bounds(base_probe).size;
				Vector3 cell_bounds = to_cell.basis.xform(bounds);

				gipd.bounds[0] = cell_bounds.x;
				gipd.bounds[1] = cell_bounds.y;
				gipd.bounds[2] = cell_bounds.z;

				Vector3 octree_size = voxel_gi_get_octree_size(base_probe);

				gipd.octree_size[0] = octree_size.x;
				gipd.octree_size[1] = octree_size.y;
				gipd.octree_size[2] = octree_size.z;

				gipd.dynamic_range = voxel_gi_get_dynamic_range(base_probe) * voxel_gi_get_energy(base_probe);
				gipd.bias = voxel_gi_get_bias(base_probe);
				gipd.normal_bias = voxel_gi_get_normal_bias(base_probe);
				gipd.blend_ambient = !voxel_gi_is_interior(base_probe);
				gipd.mipmaps = gipi->mipmaps.size();
				gipd.exposure_normalization = 1.0;
				if (p_render_data->camera_attributes.is_valid()) {
					float exposure_normalization = RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(p_render_data->camera_attributes);
					gipd.exposure_normalization = exposure_normalization / voxel_gi_get_baked_exposure_normalization(base_probe);
				}
			}

			r_voxel_gi_instances_used++;
		}

		if (texture == RID()) {
			texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_3D_WHITE);
		}

		if (texture != rbgi->voxel_gi_textures[i]) {
			rbgi->voxel_gi_textures[i] = texture;
		}
	}

	if (p_voxel_gi_instances.size() > 0) {
		RD::get_singleton()->draw_command_begin_label("VoxelGIs Setup");

		RD::get_singleton()->buffer_update(voxel_gi_buffer, 0, sizeof(VoxelGIData) * MIN((uint64_t)MAX_VOXEL_GI_INSTANCES, p_voxel_gi_instances.size()), voxel_gi_data);

		RD::get_singleton()->draw_command_end_label();
	}
}

RID GI::RenderBuffersGI::get_voxel_gi_buffer() {
	if (voxel_gi_buffer.is_null()) {
		voxel_gi_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(GI::VoxelGIData) * GI::MAX_VOXEL_GI_INSTANCES);
	}
	return voxel_gi_buffer;
}

void GI::RenderBuffersGI::free_data() {
	screen_probe_svgf.clear();
	screen_probe_specular_svgf.clear();
	hddagi_specular_reflection_valid = false;

	if (screen_probe_scene_data_ubo.is_valid()) {
		RD::get_singleton()->free_rid(screen_probe_scene_data_ubo);
		screen_probe_scene_data_ubo = RID();
	}

	if (scene_data_ubo.is_valid()) {
		RD::get_singleton()->free_rid(scene_data_ubo);
		scene_data_ubo = RID();
	}

	if (voxel_gi_buffer.is_valid()) {
		RD::get_singleton()->free_rid(voxel_gi_buffer);
		voxel_gi_buffer = RID();
	}
}

void GI::process_gi(Ref<RenderSceneBuffersRD> p_render_buffers, bool p_use_hddagi, const RID *p_normal_roughness_slices, const RID *p_voxel_gi_buffer_slices, uint32_t p_view_count, const Projection *p_projections, const Vector3 *p_eye_offsets, const Transform3D &p_cam_transform, const PagedArray<RID> &p_voxel_gi_instances) {
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();

	ERR_FAIL_NULL(texture_storage);
	ERR_FAIL_COND(p_render_buffers.is_null());
	ERR_FAIL_COND_MSG(p_view_count == 0 || p_view_count > 2, "Processing GI requires between 1 and 2 views.");
	ERR_FAIL_NULL(p_normal_roughness_slices);
	ERR_FAIL_NULL(p_projections);
	ERR_FAIL_NULL(p_eye_offsets);

	Ref<RenderBuffersGI> rbgi = p_render_buffers->get_custom_data(RB_SCOPE_GI);
	ERR_FAIL_COND(rbgi.is_null());

	RD::get_singleton()->draw_command_begin_label("GI Render");
	RENDER_TIMESTAMP("GI Receiver Processing Begin");

	Size2i internal_size = p_render_buffers->get_internal_size();

	if (rbgi->using_half_size_gi != half_resolution) {
		p_render_buffers->clear_context(RB_SCOPE_GI);
	}
	Size2i gi_size = internal_size;
	if (half_resolution) {
		gi_size.x >>= 1;
		gi_size.y >>= 1;
	}
	if (p_render_buffers->has_texture(RB_SCOPE_GI, RB_TEX_AMBIENT_U32)) {
		const RD::TextureFormat ambient_format = p_render_buffers->get_texture_format(RB_SCOPE_GI, RB_TEX_AMBIENT_U32);
		if (ambient_format.width != uint32_t(gi_size.x) || ambient_format.height != uint32_t(gi_size.y) || ambient_format.array_layers != p_view_count) {
			p_render_buffers->clear_context(RB_SCOPE_GI);
		}
	}

	if (!p_render_buffers->has_texture(RB_SCOPE_GI, RB_TEX_AMBIENT)) {
		Size2i size = gi_size;

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_UINT;
		if (p_view_count > 1) {
			tf.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		}
		tf.width = size.x;
		tf.height = size.y;
		tf.depth = 1;
		tf.array_layers = p_view_count;
		tf.shareable_formats.push_back(RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);
		tf.shareable_formats.push_back(RD::DATA_FORMAT_R32_UINT);
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

		RD::TextureFormat tf_blend;
		tf_blend.format = RD::DATA_FORMAT_R8G8_UNORM;
		if (p_view_count > 1) {
			tf_blend.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		}

		tf_blend.width = size.x;
		tf_blend.height = size.y;
		tf_blend.depth = 1;
		tf_blend.array_layers = p_view_count;
		tf_blend.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

		p_render_buffers->create_texture_from_format(RB_SCOPE_GI, RB_TEX_AMBIENT_U32, tf);
		p_render_buffers->create_texture_from_format(RB_SCOPE_GI, RB_TEX_REFLECTION_U32, tf);
		p_render_buffers->create_texture_from_format(RB_SCOPE_GI, RB_TEX_AMBIENT_REFLECTION_BLEND, tf_blend);

		p_render_buffers->create_texture_from_format(RB_SCOPE_GI, RB_TEX_REFLECTION_U32_FILTERED, tf);
		p_render_buffers->create_texture_from_format(RB_SCOPE_GI, RB_TEX_AMBIENT_REFLECTION_BLEND_FILTERED, tf_blend);

		RD::TextureView tv;
		tv.format_override = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
		p_render_buffers->create_texture_view(RB_SCOPE_GI, RB_TEX_AMBIENT_U32, RB_TEX_AMBIENT, tv);
		p_render_buffers->create_texture_view(RB_SCOPE_GI, RB_TEX_REFLECTION_U32, RB_TEX_REFLECTION, tv);
		p_render_buffers->create_texture_view(RB_SCOPE_GI, RB_TEX_REFLECTION_U32_FILTERED, RB_TEX_REFLECTION_FILTERED, tv);

		rbgi->using_half_size_gi = half_resolution;
	}

	// Setup our scene data
	{
		SceneData scene_data;

		if (rbgi->scene_data_ubo.is_null()) {
			rbgi->scene_data_ubo = RD::get_singleton()->uniform_buffer_create(sizeof(SceneData));
		}

		Projection correction;
		correction.set_depth_correction(true);

		for (uint32_t v = 0; v < p_view_count; v++) {
			Projection temp = correction * p_projections[v];

			RendererRD::MaterialStorage::store_camera(temp.inverse(), scene_data.inv_projection[v]);
			scene_data.eye_offset[v][0] = p_eye_offsets[v].x;
			scene_data.eye_offset[v][1] = p_eye_offsets[v].y;
			scene_data.eye_offset[v][2] = p_eye_offsets[v].z;
			scene_data.eye_offset[v][3] = 0.0;
		}

		// Note that we will be ignoring the origin of this transform.
		RendererRD::MaterialStorage::store_transform(p_cam_transform, scene_data.cam_transform);

		scene_data.screen_size[0] = internal_size.x;
		scene_data.screen_size[1] = internal_size.y;

		RD::get_singleton()->buffer_update(rbgi->scene_data_ubo, 0, sizeof(SceneData), &scene_data);
	}

	// Now compute the contents of our buffers.
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	// Render each eye separately.
	// We need to look into whether we can make our compute shader use Multiview but not sure that works or makes a difference..

	// setup our push constant

	PushConstant push_constant;

	push_constant.max_voxel_gi_instances = MIN((uint64_t)MAX_VOXEL_GI_INSTANCES, p_voxel_gi_instances.size());
	push_constant.high_quality_vct = voxel_gi_quality == RSE::VOXEL_GI_QUALITY_HIGH;

	// these should be the same for all views
	push_constant.orthogonal = p_projections[0].is_orthogonal();
	push_constant.z_near = p_projections[0].get_z_near();
	push_constant.z_far = p_projections[0].get_z_far();

	// these are only used if we have 1 view, else we use the projections in our scene data
	push_constant.proj_info[0] = -2.0f / (internal_size.x * p_projections[0].columns[0][0]);
	push_constant.proj_info[1] = -2.0f / (internal_size.y * p_projections[0].columns[1][1]);
	push_constant.proj_info[2] = (1.0f - p_projections[0].columns[2][0]) / p_projections[0].columns[0][0];
	push_constant.proj_info[3] = (1.0f + p_projections[0].columns[2][1]) / p_projections[0].columns[1][1];

	bool use_voxel_gi_instances = push_constant.max_voxel_gi_instances > 0;

	Ref<HDDAGI> hddagi;
	if (p_render_buffers->has_custom_data(RB_SCOPE_HDDAGI)) {
		hddagi = p_render_buffers->get_custom_data(RB_SCOPE_HDDAGI);
	}
	bool use_hddagi = p_use_hddagi && hddagi.is_valid() && !hddagi->cascades.is_empty() &&
			hddagi->voxel_bits_tex.is_valid() && hddagi->voxel_region_tex.is_valid() && hddagi->voxel_data.is_valid() &&
			hddagi->lightprobe_specular_tex.is_valid() && hddagi->get_lightprobe_diffuse_texture().is_valid() &&
			hddagi->voxel_page_table.is_valid();

	uint32_t pipeline_specialization = 0;
	if (rbgi->using_half_size_gi) {
		pipeline_specialization |= SHADER_SPECIALIZATION_HALF_RES;
	}
	if (p_view_count > 1) {
		pipeline_specialization |= SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX;
	}
	bool has_vrs_texture = p_render_buffers->has_texture(RB_SCOPE_VRS, RB_TEXTURE);
	if (has_vrs_texture) {
		pipeline_specialization |= SHADER_SPECIALIZATION_USE_VRS;
	}

	Mode mode;

	if (use_hddagi) {
		if (use_voxel_gi_instances) {
			mode = hddagi->using_ambient_filter ? MODE_COMBINED_BLEND_AMBIENT : MODE_COMBINED;
		} else {
			mode = hddagi->using_ambient_filter ? MODE_HDDAGI_BLEND_AMBIENT : MODE_HDDAGI;
		}
		push_constant.world_gi_flags = CLAMP(int(GLOBAL_GET("rendering/global_illumination/hddagi/reflection_quality")), 0, 2);
		if (push_constant.world_gi_flags == 0 && !use_voxel_gi_instances && !has_vrs_texture && bool(GLOBAL_GET("rendering/global_illumination/hddagi/reflection_half_resolution"))) {
			pipeline_specialization |= SHADER_SPECIALIZATION_HALF_REFLECTIONS;
		}
		if (push_constant.world_gi_flags == 2 && !use_voxel_gi_instances) {
			pipeline_specialization |= SHADER_SPECIALIZATION_DIFFUSE_ONLY;
		}
		if (bool(GLOBAL_GET("rendering/global_illumination/hddagi/world_probe_validity"))) {
			push_constant.world_gi_flags |= 8;
		}
	} else {
		mode = MODE_VOXEL_GI;
	}

	RID hddagi_voxel_bits_texture = default_hddagi_voxel_bits_texture;
	RID hddagi_voxel_region_texture = default_hddagi_voxel_region_texture;
	RID hddagi_voxel_data = RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer();
	RID hddagi_lightprobe_specular_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_BLACK);
	RID hddagi_lightprobe_diffuse_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_BLACK);
	Vector<RID> hddagi_occlusion_textures = { texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_3D_BLACK), texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_3D_BLACK) };
	Vector<RID> hddagi_occlusion_correction_textures = hddagi_occlusion_textures;
	RID hddagi_voxel_pages = hddagi_voxel_data;

	if (use_hddagi) {
		hddagi_voxel_bits_texture = hddagi->voxel_bits_tex;
		hddagi_voxel_region_texture = hddagi->voxel_region_tex;
		hddagi_voxel_data = hddagi->voxel_data;
		hddagi_lightprobe_specular_texture = hddagi->lightprobe_specular_tex;
		hddagi_lightprobe_diffuse_texture = hddagi->using_probe_filter ? hddagi->lightprobe_diffuse_filter_tex : hddagi->lightprobe_diffuse_tex;
		hddagi_occlusion_textures = hddagi->get_lightprobe_occlusion_textures();
		hddagi_occlusion_correction_textures = { hddagi->occlusion_correction[0], hddagi->occlusion_correction[1] };
		hddagi_voxel_pages = hddagi->voxel_page_table;
	}

	for (uint32_t v = 0; v < p_view_count; v++) {
		push_constant.view_index = v;

		// setup our uniform set
		RD::Uniform vgiu;
		vgiu.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		vgiu.binding = 17;
		for (int i = 0; i < MAX_VOXEL_GI_INSTANCES; i++) {
			vgiu.append_id(rbgi->voxel_gi_textures[i]);
		}

		LocalVector<RD::Uniform> gi_uniforms = {
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, hddagi_voxel_bits_texture),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, hddagi_voxel_region_texture),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, hddagi_voxel_data),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 4, hddagi_lightprobe_specular_texture),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 5, hddagi_lightprobe_diffuse_texture),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 6, hddagi_occlusion_textures),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 7, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 8, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 10, hddagi_voxel_pages),

			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 12, p_render_buffers->get_depth_texture(v)),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 13, p_normal_roughness_slices[v]),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 14, p_voxel_gi_buffer_slices[v].is_valid() ? p_voxel_gi_buffer_slices[v] : texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_UINT)),

			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 15, hddagi_ubo),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 16, rbgi->get_voxel_gi_buffer()),
			vgiu, // 17
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 18, rbgi->scene_data_ubo),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 19, has_vrs_texture ? p_render_buffers->get_texture_slice(RB_SCOPE_VRS, RB_TEXTURE, v, 0) : texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_VRS)),

			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 20, p_render_buffers->get_texture_slice(RB_SCOPE_GI, RB_TEX_AMBIENT_U32, v, 0)),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 21, p_render_buffers->get_texture_slice(RB_SCOPE_GI, RB_TEX_REFLECTION_U32, v, 0)),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 22, p_render_buffers->get_texture_slice(RB_SCOPE_GI, RB_TEX_AMBIENT_REFLECTION_BLEND, v, 0)),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 23, use_hddagi ? hddagi->lightprobe_geometry_proximity_map : texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE)),
			RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 24, hddagi_occlusion_correction_textures)
		};
		RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache_vec(shader.version_get_shader(shader_version, 0), 0, gi_uniforms);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, pipelines[pipeline_specialization][mode]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));

		if (rbgi->using_half_size_gi) {
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, internal_size.x >> 1, internal_size.y >> 1, 1);
		} else {
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, internal_size.x, internal_size.y, 1);
		}
	}

	if (use_hddagi && hddagi->using_reflection_filter && (push_constant.world_gi_flags & 3) == 0) {
		uint32_t filter_pipeline_specialization = 0;
		if (rbgi->using_half_size_gi) {
			filter_pipeline_specialization |= FILTER_SHADER_SPECIALIZATION_HALF_RES;
		}
		if (p_view_count > 1) {
			filter_pipeline_specialization |= FILTER_SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX;
		}

		FilterPushConstant filter_push_constant;
		filter_push_constant.orthogonal = push_constant.orthogonal;
		filter_push_constant.z_near = push_constant.z_near;
		filter_push_constant.z_far = push_constant.z_far;
		filter_push_constant.proj_info[0] = push_constant.proj_info[0];
		filter_push_constant.proj_info[1] = push_constant.proj_info[1];
		filter_push_constant.proj_info[2] = push_constant.proj_info[2];
		filter_push_constant.proj_info[3] = push_constant.proj_info[3];

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, filter_pipelines[filter_pipeline_specialization][rbgi->using_half_size_gi ? FILTER_MODE_BILATERAL_HALF_SIZE : FILTER_MODE_BILATERAL]);

		for (int i = 0; i < 2; i++) {
			RD::get_singleton()->compute_list_add_barrier(compute_list);

			filter_push_constant.filter_dir[1] = i & 1;
			filter_push_constant.filter_dir[0] = (i + 1) & 1;

			for (uint32_t v = 0; v < p_view_count; v++) {
				RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(
						filter_shader.version_get_shader(filter_shader_version, 0),
						0,
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, p_render_buffers->get_depth_texture(v)),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, p_normal_roughness_slices[v]),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, p_render_buffers->get_texture_slice(RB_SCOPE_GI, i == 0 ? RB_TEX_REFLECTION : RB_TEX_REFLECTION_FILTERED, v, 0)),
						RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 3, p_render_buffers->get_texture_slice(RB_SCOPE_GI, i == 0 ? RB_TEX_AMBIENT_REFLECTION_BLEND : RB_TEX_AMBIENT_REFLECTION_BLEND_FILTERED, v, 0)),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, p_render_buffers->get_texture_slice(RB_SCOPE_GI, i == 0 ? RB_TEX_REFLECTION_U32_FILTERED : RB_TEX_REFLECTION_U32, v, 0)),
						RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, p_render_buffers->get_texture_slice(RB_SCOPE_GI, i == 0 ? RB_TEX_AMBIENT_REFLECTION_BLEND_FILTERED : RB_TEX_AMBIENT_REFLECTION_BLEND, v, 0)),
						RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 6, RendererRD::MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED)),
						RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 7, rbgi->scene_data_ubo));

				filter_push_constant.view_index = v;
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &filter_push_constant, sizeof(FilterPushConstant));
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

				if (rbgi->using_half_size_gi) {
					RD::get_singleton()->compute_list_dispatch_threads(compute_list, internal_size.x >> 1, internal_size.y >> 1, 1);
				} else {
					RD::get_singleton()->compute_list_dispatch_threads(compute_list, internal_size.x, internal_size.y, 1);
				}
			}
		}
	}
	RD::get_singleton()->compute_list_end();
	RENDER_TIMESTAMP("GI Receiver Processing End");
	RD::get_singleton()->draw_command_end_label();
}

RID GI::voxel_gi_instance_create(RID p_base) {
	VoxelGIInstance voxel_gi;
	voxel_gi.gi = this;
	voxel_gi.probe = p_base;
	RID rid = voxel_gi_instance_owner.make_rid(voxel_gi);
	return rid;
}

void GI::voxel_gi_instance_free(RID p_rid) {
	GI::VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_rid);
	voxel_gi->free_resources();
	voxel_gi_instance_owner.free(p_rid);
}

void GI::voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) {
	VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_probe);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->transform = p_xform;
}

bool GI::voxel_gi_needs_update(RID p_probe) const {
	VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_probe);
	ERR_FAIL_NULL_V(voxel_gi, false);

	return voxel_gi->last_probe_version != voxel_gi_get_version(voxel_gi->probe);
}

void GI::voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RenderGeometryInstance *> &p_dynamic_objects) {
	VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_probe);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->update(p_update_light_instances, p_light_instances, p_dynamic_objects);
}

void GI::debug_voxel_gi(RID p_voxel_gi, RD::DrawListID p_draw_list, RID p_framebuffer, const Projection &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha) {
	VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_NULL(voxel_gi);

	voxel_gi->debug(p_draw_list, p_framebuffer, p_camera_with_transform, p_lighting, p_emission, p_alpha);
}

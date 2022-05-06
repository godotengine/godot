/*************************************************************************/
/*  renderer_storage_rd.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "renderer_storage_rd.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/particles_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/shader_language.h"

/* FOG VOLUMES */

RID RendererStorageRD::fog_volume_allocate() {
	return fog_volume_owner.allocate_rid();
}
void RendererStorageRD::fog_volume_initialize(RID p_rid) {
	fog_volume_owner.initialize_rid(p_rid, FogVolume());
}

void RendererStorageRD::fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);

	if (p_shape == fog_volume->shape) {
		return;
	}

	fog_volume->shape = p_shape;
	fog_volume->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);

	fog_volume->extents = p_extents;
	fog_volume->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::fog_volume_set_material(RID p_fog_volume, RID p_material) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);
	fog_volume->material = p_material;
}

RID RendererStorageRD::fog_volume_get_material(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, RID());

	return fog_volume->material;
}

RS::FogVolumeShape RendererStorageRD::fog_volume_get_shape(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, RS::FOG_VOLUME_SHAPE_BOX);

	return fog_volume->shape;
}

AABB RendererStorageRD::fog_volume_get_aabb(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, AABB());

	switch (fog_volume->shape) {
		case RS::FOG_VOLUME_SHAPE_ELLIPSOID:
		case RS::FOG_VOLUME_SHAPE_BOX: {
			AABB aabb;
			aabb.position = -fog_volume->extents;
			aabb.size = fog_volume->extents * 2;
			return aabb;
		}
		default: {
			// Need some size otherwise will get culled
			return AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
		}
	}

	return AABB();
}

Vector3 RendererStorageRD::fog_volume_get_extents(RID p_fog_volume) const {
	const FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, Vector3());
	return fog_volume->extents;
}

/* VISIBILITY NOTIFIER */

RID RendererStorageRD::visibility_notifier_allocate() {
	return visibility_notifier_owner.allocate_rid();
}
void RendererStorageRD::visibility_notifier_initialize(RID p_notifier) {
	visibility_notifier_owner.initialize_rid(p_notifier, VisibilityNotifier());
}
void RendererStorageRD::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);
	vn->aabb = p_aabb;
	vn->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}
void RendererStorageRD::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);
	vn->enter_callback = p_enter_callbable;
	vn->exit_callback = p_exit_callable;
}

AABB RendererStorageRD::visibility_notifier_get_aabb(RID p_notifier) const {
	const VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND_V(!vn, AABB());
	return vn->aabb;
}
void RendererStorageRD::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);

	if (p_enter) {
		if (!vn->enter_callback.is_null()) {
			if (p_deferred) {
				vn->enter_callback.call_deferred(nullptr, 0);
			} else {
				Variant r;
				Callable::CallError ce;
				vn->enter_callback.call(nullptr, 0, r, ce);
			}
		}
	} else {
		if (!vn->exit_callback.is_null()) {
			if (p_deferred) {
				vn->exit_callback.call_deferred(nullptr, 0);
			} else {
				Variant r;
				Callable::CallError ce;
				vn->exit_callback.call(nullptr, 0, r, ce);
			}
		}
	}
}

/* VOXEL GI */

RID RendererStorageRD::voxel_gi_allocate() {
	return voxel_gi_owner.allocate_rid();
}
void RendererStorageRD::voxel_gi_initialize(RID p_voxel_gi) {
	voxel_gi_owner.initialize_rid(p_voxel_gi, VoxelGI());
}

void RendererStorageRD::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	if (voxel_gi->octree_buffer.is_valid()) {
		RD::get_singleton()->free(voxel_gi->octree_buffer);
		RD::get_singleton()->free(voxel_gi->data_buffer);
		if (voxel_gi->sdf_texture.is_valid()) {
			RD::get_singleton()->free(voxel_gi->sdf_texture);
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
		}
#if 0
			{
				RD::TextureFormat tf;
				tf.format = RD::DATA_FORMAT_R8_UNORM;
				tf.width = voxel_gi->octree_size.x;
				tf.height = voxel_gi->octree_size.y;
				tf.depth = voxel_gi->octree_size.z;
				tf.type = RD::TEXTURE_TYPE_3D;
				tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UNORM);
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UINT);
				voxel_gi->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			}
			RID shared_tex;
			{
				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_R8_UINT;
				shared_tex = RD::get_singleton()->texture_create_shared(tv, voxel_gi->sdf_texture);
			}
			//update SDF texture
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.append_id(voxel_gi->octree_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.append_id(voxel_gi->data_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 3;
				u.append_id(shared_tex);
				uniforms.push_back(u);
			}

			RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, voxel_gi_sdf_shader_version_shader, 0);

			{
				uint32_t push_constant[4] = { 0, 0, 0, 0 };

				for (int i = 0; i < voxel_gi->level_counts.size() - 1; i++) {
					push_constant[0] += voxel_gi->level_counts[i];
				}
				push_constant[1] = push_constant[0] + voxel_gi->level_counts[voxel_gi->level_counts.size() - 1];

				print_line("offset: " + itos(push_constant[0]));
				print_line("size: " + itos(push_constant[1]));
				//create SDF
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, voxel_gi_sdf_shader_pipeline);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, push_constant, sizeof(uint32_t) * 4);
				RD::get_singleton()->compute_list_dispatch(compute_list, voxel_gi->octree_size.x / 4, voxel_gi->octree_size.y / 4, voxel_gi->octree_size.z / 4);
				RD::get_singleton()->compute_list_end();
			}

			RD::get_singleton()->free(uniform_set);
			RD::get_singleton()->free(shared_tex);
		}
#endif
	}

	voxel_gi->version++;
	voxel_gi->data_version++;

	voxel_gi->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

AABB RendererStorageRD::voxel_gi_get_bounds(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, AABB());

	return voxel_gi->bounds;
}

Vector3i RendererStorageRD::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector3i());
	return voxel_gi->octree_size;
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->octree_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->octree_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->data_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->texture_get_data(voxel_gi->sdf_texture, 0);
	}
	return Vector<uint8_t>();
}

Vector<int> RendererStorageRD::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<int>());

	return voxel_gi->level_counts;
}

Transform3D RendererStorageRD::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Transform3D());

	return voxel_gi->to_cell_xform;
}

void RendererStorageRD::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->dynamic_range = p_range;
	voxel_gi->version++;
}

float RendererStorageRD::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);

	return voxel_gi->dynamic_range;
}

void RendererStorageRD::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->propagation = p_range;
	voxel_gi->version++;
}

float RendererStorageRD::voxel_gi_get_propagation(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->propagation;
}

void RendererStorageRD::voxel_gi_set_energy(RID p_voxel_gi, float p_energy) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->energy = p_energy;
}

float RendererStorageRD::voxel_gi_get_energy(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->energy;
}

void RendererStorageRD::voxel_gi_set_bias(RID p_voxel_gi, float p_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->bias = p_bias;
}

float RendererStorageRD::voxel_gi_get_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->bias;
}

void RendererStorageRD::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_normal_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->normal_bias = p_normal_bias;
}

float RendererStorageRD::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->normal_bias;
}

void RendererStorageRD::voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->anisotropy_strength = p_strength;
}

float RendererStorageRD::voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->anisotropy_strength;
}

void RendererStorageRD::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->interior = p_enable;
}

void RendererStorageRD::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->use_two_bounces = p_enable;
	voxel_gi->version++;
}

bool RendererStorageRD::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, false);
	return voxel_gi->use_two_bounces;
}

bool RendererStorageRD::voxel_gi_is_interior(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->interior;
}

uint32_t RendererStorageRD::voxel_gi_get_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->version;
}

uint32_t RendererStorageRD::voxel_gi_get_data_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->data_version;
}

RID RendererStorageRD::voxel_gi_get_octree_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());
	return voxel_gi->octree_buffer;
}

RID RendererStorageRD::voxel_gi_get_data_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());
	return voxel_gi->data_buffer;
}

RID RendererStorageRD::voxel_gi_get_sdf_texture(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());

	return voxel_gi->sdf_texture;
}

/* misc */

void RendererStorageRD::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (RendererRD::MeshStorage::get_singleton()->owns_mesh(p_base)) {
		RendererRD::Mesh *mesh = RendererRD::MeshStorage::get_singleton()->get_mesh(p_base);
		p_instance->update_dependency(&mesh->dependency);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_multimesh(p_base)) {
		RendererRD::MultiMesh *multimesh = RendererRD::MeshStorage::get_singleton()->get_multimesh(p_base);
		p_instance->update_dependency(&multimesh->dependency);
		if (multimesh->mesh.is_valid()) {
			base_update_dependency(multimesh->mesh, p_instance);
		}
	} else if (RendererRD::LightStorage::get_singleton()->owns_reflection_probe(p_base)) {
		RendererRD::ReflectionProbe *rp = RendererRD::LightStorage::get_singleton()->get_reflection_probe(p_base);
		p_instance->update_dependency(&rp->dependency);
	} else if (RendererRD::TextureStorage::get_singleton()->owns_decal(p_base)) {
		RendererRD::Decal *decal = RendererRD::TextureStorage::get_singleton()->get_decal(p_base);
		p_instance->update_dependency(&decal->dependency);
	} else if (voxel_gi_owner.owns(p_base)) {
		VoxelGI *gip = voxel_gi_owner.get_or_null(p_base);
		p_instance->update_dependency(&gip->dependency);
	} else if (RendererRD::LightStorage::get_singleton()->owns_lightmap(p_base)) {
		RendererRD::Lightmap *lm = RendererRD::LightStorage::get_singleton()->get_lightmap(p_base);
		p_instance->update_dependency(&lm->dependency);
	} else if (RendererRD::LightStorage::get_singleton()->owns_light(p_base)) {
		RendererRD::Light *l = RendererRD::LightStorage::get_singleton()->get_light(p_base);
		p_instance->update_dependency(&l->dependency);
	} else if (RendererRD::ParticlesStorage::get_singleton()->owns_particles(p_base)) {
		RendererRD::Particles *p = RendererRD::ParticlesStorage::get_singleton()->get_particles(p_base);
		p_instance->update_dependency(&p->dependency);
	} else if (RendererRD::ParticlesStorage::get_singleton()->owns_particles_collision(p_base)) {
		RendererRD::ParticlesCollision *pc = RendererRD::ParticlesStorage::get_singleton()->get_particles_collision(p_base);
		p_instance->update_dependency(&pc->dependency);
	} else if (fog_volume_owner.owns(p_base)) {
		FogVolume *fv = fog_volume_owner.get_or_null(p_base);
		p_instance->update_dependency(&fv->dependency);
	} else if (visibility_notifier_owner.owns(p_base)) {
		VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_base);
		p_instance->update_dependency(&vn->dependency);
	}
}

RS::InstanceType RendererStorageRD::get_base_type(RID p_rid) const {
	if (RendererRD::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		return RS::INSTANCE_MESH;
	}
	if (RendererRD::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	}
	if (RendererRD::LightStorage::get_singleton()->owns_reflection_probe(p_rid)) {
		return RS::INSTANCE_REFLECTION_PROBE;
	}
	if (RendererRD::TextureStorage::get_singleton()->owns_decal(p_rid)) {
		return RS::INSTANCE_DECAL;
	}
	if (voxel_gi_owner.owns(p_rid)) {
		return RS::INSTANCE_VOXEL_GI;
	}
	if (RendererRD::LightStorage::get_singleton()->owns_light(p_rid)) {
		return RS::INSTANCE_LIGHT;
	}
	if (RendererRD::LightStorage::get_singleton()->owns_lightmap(p_rid)) {
		return RS::INSTANCE_LIGHTMAP;
	}
	if (RendererRD::ParticlesStorage::get_singleton()->owns_particles(p_rid)) {
		return RS::INSTANCE_PARTICLES;
	}
	if (RendererRD::ParticlesStorage::get_singleton()->owns_particles_collision(p_rid)) {
		return RS::INSTANCE_PARTICLES_COLLISION;
	}
	if (fog_volume_owner.owns(p_rid)) {
		return RS::INSTANCE_FOG_VOLUME;
	}
	if (visibility_notifier_owner.owns(p_rid)) {
		return RS::INSTANCE_VISIBLITY_NOTIFIER;
	}

	return RS::INSTANCE_NONE;
}

void RendererStorageRD::update_dirty_resources() {
	RendererRD::MaterialStorage::get_singleton()->_update_global_variables(); //must do before materials, so it can queue them for update
	RendererRD::MaterialStorage::get_singleton()->_update_queued_materials();
	RendererRD::MeshStorage::get_singleton()->_update_dirty_multimeshes();
	RendererRD::MeshStorage::get_singleton()->_update_dirty_skeletons();
	RendererRD::TextureStorage::get_singleton()->update_decal_atlas();
}

bool RendererStorageRD::has_os_feature(const String &p_feature) const {
	if (p_feature == "rgtc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC5_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "s3tc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "bptc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC7_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if ((p_feature == "etc" || p_feature == "etc2") && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	return false;
}

bool RendererStorageRD::free(RID p_rid) {
	if (RendererRD::TextureStorage::get_singleton()->owns_texture(p_rid)) {
		RendererRD::TextureStorage::get_singleton()->texture_free(p_rid);
	} else if (RendererRD::TextureStorage::get_singleton()->owns_canvas_texture(p_rid)) {
		RendererRD::TextureStorage::get_singleton()->canvas_texture_free(p_rid);
	} else if (RendererRD::MaterialStorage::get_singleton()->owns_shader(p_rid)) {
		RendererRD::MaterialStorage::get_singleton()->shader_free(p_rid);
	} else if (RendererRD::MaterialStorage::get_singleton()->owns_material(p_rid)) {
		RendererRD::MaterialStorage::get_singleton()->material_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->mesh_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_mesh_instance(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->mesh_instance_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->multimesh_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_skeleton(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->skeleton_free(p_rid);
	} else if (RendererRD::LightStorage::get_singleton()->owns_reflection_probe(p_rid)) {
		RendererRD::LightStorage::get_singleton()->reflection_probe_free(p_rid);
	} else if (RendererRD::TextureStorage::get_singleton()->owns_decal(p_rid)) {
		RendererRD::TextureStorage::get_singleton()->decal_free(p_rid);
	} else if (voxel_gi_owner.owns(p_rid)) {
		voxel_gi_allocate_data(p_rid, Transform3D(), AABB(), Vector3i(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<int>()); //deallocate
		VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_rid);
		voxel_gi->dependency.deleted_notify(p_rid);
		voxel_gi_owner.free(p_rid);
	} else if (RendererRD::LightStorage::get_singleton()->owns_lightmap(p_rid)) {
		RendererRD::LightStorage::get_singleton()->lightmap_free(p_rid);
	} else if (RendererRD::LightStorage::get_singleton()->owns_light(p_rid)) {
		RendererRD::LightStorage::get_singleton()->light_free(p_rid);
	} else if (RendererRD::ParticlesStorage::get_singleton()->owns_particles(p_rid)) {
		RendererRD::ParticlesStorage::get_singleton()->particles_free(p_rid);
	} else if (RendererRD::ParticlesStorage::get_singleton()->owns_particles_collision(p_rid)) {
		RendererRD::ParticlesStorage::get_singleton()->particles_collision_free(p_rid);
	} else if (visibility_notifier_owner.owns(p_rid)) {
		VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_rid);
		vn->dependency.deleted_notify(p_rid);
		visibility_notifier_owner.free(p_rid);
	} else if (RendererRD::ParticlesStorage::get_singleton()->owns_particles_collision_instance(p_rid)) {
		RendererRD::ParticlesStorage::get_singleton()->particles_collision_instance_free(p_rid);
	} else if (fog_volume_owner.owns(p_rid)) {
		FogVolume *fog_volume = fog_volume_owner.get_or_null(p_rid);
		fog_volume->dependency.deleted_notify(p_rid);
		fog_volume_owner.free(p_rid);
	} else if (RendererRD::TextureStorage::get_singleton()->owns_render_target(p_rid)) {
		RendererRD::TextureStorage::get_singleton()->render_target_free(p_rid);
	} else {
		return false;
	}

	return true;
}

void RendererStorageRD::init_effects(bool p_prefer_raster_effects) {
	effects = memnew(EffectsRD(p_prefer_raster_effects));
}

EffectsRD *RendererStorageRD::get_effects() {
	ERR_FAIL_NULL_V_MSG(effects, nullptr, "Effects haven't been initialised yet.");
	return effects;
}

void RendererStorageRD::capture_timestamps_begin() {
	RD::get_singleton()->capture_timestamp("Frame Begin");
}

void RendererStorageRD::capture_timestamp(const String &p_name) {
	RD::get_singleton()->capture_timestamp(p_name);
}

uint32_t RendererStorageRD::get_captured_timestamps_count() const {
	return RD::get_singleton()->get_captured_timestamps_count();
}

uint64_t RendererStorageRD::get_captured_timestamps_frame() const {
	return RD::get_singleton()->get_captured_timestamps_frame();
}

uint64_t RendererStorageRD::get_captured_timestamp_gpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_gpu_time(p_index);
}

uint64_t RendererStorageRD::get_captured_timestamp_cpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_cpu_time(p_index);
}

String RendererStorageRD::get_captured_timestamp_name(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_name(p_index);
}

void RendererStorageRD::update_memory_info() {
	texture_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TEXTURES);
	buffer_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_BUFFERS);
	total_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
}
uint64_t RendererStorageRD::get_rendering_info(RS::RenderingInfo p_info) {
	if (p_info == RS::RENDERING_INFO_TEXTURE_MEM_USED) {
		return texture_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_BUFFER_MEM_USED) {
		return buffer_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_VIDEO_MEM_USED) {
		return total_mem_cache;
	}
	return 0;
}

String RendererStorageRD::get_video_adapter_name() const {
	return RenderingDevice::get_singleton()->get_device_name();
}

String RendererStorageRD::get_video_adapter_vendor() const {
	return RenderingDevice::get_singleton()->get_device_vendor_name();
}

RenderingDevice::DeviceType RendererStorageRD::get_video_adapter_type() const {
	return RenderingDevice::get_singleton()->get_device_type();
}

String RendererStorageRD::get_video_adapter_api_version() const {
	return RenderingDevice::get_singleton()->get_device_api_version();
}

RendererStorageRD *RendererStorageRD::base_singleton = nullptr;

RendererStorageRD::RendererStorageRD() {
	base_singleton = this;
}

RendererStorageRD::~RendererStorageRD() {
	if (effects) {
		memdelete(effects);
		effects = nullptr;
	}
}

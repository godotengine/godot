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
#include "servers/rendering/renderer_rd/environment/gi.h"
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
		case RS::FOG_VOLUME_SHAPE_CONE:
		case RS::FOG_VOLUME_SHAPE_CYLINDER:
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
	} else if (RendererRD::GI::get_singleton()->owns_voxel_gi(p_base)) {
		RendererRD::GI::VoxelGI *gip = RendererRD::GI::get_singleton()->get_voxel_gi(p_base);
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
	if (RendererRD::GI::get_singleton()->owns_voxel_gi(p_rid)) {
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
	if (!RD::get_singleton()) {
		return false;
	}

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
	} else if (RendererRD::GI::get_singleton()->owns_voxel_gi(p_rid)) {
		RendererRD::GI::get_singleton()->voxel_gi_free(p_rid);
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

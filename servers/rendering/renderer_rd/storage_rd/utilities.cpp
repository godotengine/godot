/**************************************************************************/
/*  utilities.cpp                                                         */
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

#include "utilities.h"
#include "../environment/fog.h"
#include "../environment/gi.h"
#include "light_storage.h"
#include "mesh_storage.h"
#include "particles_storage.h"
#include "texture_storage.h"

using namespace RendererRD;

Utilities *Utilities::singleton = nullptr;

Utilities::Utilities() {
	singleton = this;
}

Utilities::~Utilities() {
	singleton = nullptr;
}

/* INSTANCES */

RS::InstanceType Utilities::get_base_type(RID p_rid) const {
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
	if (RendererRD::Fog::get_singleton()->owns_fog_volume(p_rid)) {
		return RS::INSTANCE_FOG_VOLUME;
	}
	if (owns_visibility_notifier(p_rid)) {
		return RS::INSTANCE_VISIBLITY_NOTIFIER;
	}

	return RS::INSTANCE_NONE;
}

bool Utilities::free(RID p_rid) {
	if (RendererRD::LightStorage::get_singleton()->free(p_rid)) {
		return true;
	} else if (RendererRD::MaterialStorage::get_singleton()->free(p_rid)) {
		return true;
	} else if (RendererRD::MeshStorage::get_singleton()->free(p_rid)) {
		return true;
	} else if (RendererRD::ParticlesStorage::get_singleton()->free(p_rid)) {
		return true;
	} else if (RendererRD::TextureStorage::get_singleton()->free(p_rid)) {
		return true;
	} else if (RendererRD::GI::get_singleton()->owns_voxel_gi(p_rid)) {
		RendererRD::GI::get_singleton()->voxel_gi_free(p_rid);
		return true;
	} else if (RendererRD::Fog::get_singleton()->owns_fog_volume(p_rid)) {
		RendererRD::Fog::get_singleton()->fog_volume_free(p_rid);
		return true;
	} else if (owns_visibility_notifier(p_rid)) {
		visibility_notifier_free(p_rid);
		return true;
	} else {
		return false;
	}
}

/* DEPENDENCIES */

void Utilities::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (MeshStorage::get_singleton()->owns_mesh(p_base)) {
		Dependency *dependency = MeshStorage::get_singleton()->mesh_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (MeshStorage::get_singleton()->owns_multimesh(p_base)) {
		Dependency *dependency = MeshStorage::get_singleton()->multimesh_get_dependency(p_base);
		p_instance->update_dependency(dependency);

		RID mesh = MeshStorage::get_singleton()->multimesh_get_mesh(p_base);
		if (mesh.is_valid()) {
			base_update_dependency(mesh, p_instance);
		}
	} else if (LightStorage::get_singleton()->owns_reflection_probe(p_base)) {
		Dependency *dependency = LightStorage::get_singleton()->reflection_probe_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (TextureStorage::get_singleton()->owns_decal(p_base)) {
		Dependency *dependency = TextureStorage::get_singleton()->decal_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (GI::get_singleton()->owns_voxel_gi(p_base)) {
		Dependency *dependency = GI::get_singleton()->voxel_gi_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (LightStorage::get_singleton()->owns_lightmap(p_base)) {
		Dependency *dependency = LightStorage::get_singleton()->lightmap_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (LightStorage::get_singleton()->owns_light(p_base)) {
		Dependency *dependency = LightStorage::get_singleton()->light_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (ParticlesStorage::get_singleton()->owns_particles(p_base)) {
		Dependency *dependency = ParticlesStorage::get_singleton()->particles_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (ParticlesStorage::get_singleton()->owns_particles_collision(p_base)) {
		Dependency *dependency = ParticlesStorage::get_singleton()->particles_collision_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (Fog::get_singleton()->owns_fog_volume(p_base)) {
		Dependency *dependency = Fog::get_singleton()->fog_volume_get_dependency(p_base);
		p_instance->update_dependency(dependency);
	} else if (owns_visibility_notifier(p_base)) {
		VisibilityNotifier *vn = get_visibility_notifier(p_base);
		p_instance->update_dependency(&vn->dependency);
	}
}

/* VISIBILITY NOTIFIER */

RID Utilities::visibility_notifier_allocate() {
	return visibility_notifier_owner.allocate_rid();
}

void Utilities::visibility_notifier_initialize(RID p_notifier) {
	visibility_notifier_owner.initialize_rid(p_notifier, VisibilityNotifier());
}

void Utilities::visibility_notifier_free(RID p_notifier) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	vn->dependency.deleted_notify(p_notifier);
	visibility_notifier_owner.free(p_notifier);
}

void Utilities::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_NULL(vn);
	vn->aabb = p_aabb;
	vn->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_AABB);
}

void Utilities::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_NULL(vn);
	vn->enter_callback = p_enter_callbable;
	vn->exit_callback = p_exit_callable;
}

AABB Utilities::visibility_notifier_get_aabb(RID p_notifier) const {
	const VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_NULL_V(vn, AABB());
	return vn->aabb;
}

void Utilities::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_NULL(vn);

	if (p_enter) {
		if (vn->enter_callback.is_valid()) {
			if (p_deferred) {
				vn->enter_callback.call_deferred();
			} else {
				vn->enter_callback.call();
			}
		}
	} else {
		if (vn->exit_callback.is_valid()) {
			if (p_deferred) {
				vn->exit_callback.call_deferred();
			} else {
				vn->exit_callback.call();
			}
		}
	}
}

/* TIMING */

void Utilities::capture_timestamps_begin() {
	RD::get_singleton()->capture_timestamp("Frame Begin");
}

void Utilities::capture_timestamp(const String &p_name) {
	RD::get_singleton()->capture_timestamp(p_name);
}

uint32_t Utilities::get_captured_timestamps_count() const {
	return RD::get_singleton()->get_captured_timestamps_count();
}

uint64_t Utilities::get_captured_timestamps_frame() const {
	return RD::get_singleton()->get_captured_timestamps_frame();
}

uint64_t Utilities::get_captured_timestamp_gpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_gpu_time(p_index);
}

uint64_t Utilities::get_captured_timestamp_cpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_cpu_time(p_index);
}

String Utilities::get_captured_timestamp_name(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_name(p_index);
}

/* MISC */

void Utilities::update_dirty_resources() {
	MaterialStorage::get_singleton()->_update_global_shader_uniforms(); //must do before materials, so it can queue them for update
	MaterialStorage::get_singleton()->_update_queued_materials();
	MeshStorage::get_singleton()->_update_dirty_multimeshes();
	MeshStorage::get_singleton()->_update_dirty_skeletons();
	TextureStorage::get_singleton()->update_decal_atlas();
}

bool Utilities::has_os_feature(const String &p_feature) const {
	if (!RD::get_singleton()) {
		return false;
	}

	if (p_feature == "rgtc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC5_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

#if !defined(ANDROID_ENABLED) && !defined(IOS_ENABLED)
	// Some Android devices report support for S3TC but we don't expect that and don't export the textures.
	// This could be fixed but so few devices support it that it doesn't seem useful (and makes bigger APKs).
	// For good measure we do the same hack for iOS, just in case.
	if (p_feature == "s3tc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}
#endif

	if (p_feature == "bptc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC7_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "etc2" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "astc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ASTC_4x4_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "astc_hdr" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ASTC_4x4_SFLOAT_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	return false;
}

void Utilities::update_memory_info() {
	texture_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TEXTURES);
	buffer_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_BUFFERS);
	total_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
}

uint64_t Utilities::get_rendering_info(RS::RenderingInfo p_info) {
	if (p_info == RS::RENDERING_INFO_TEXTURE_MEM_USED) {
		return texture_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_BUFFER_MEM_USED) {
		return buffer_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_VIDEO_MEM_USED) {
		return total_mem_cache;
	}
	return 0;
}

String Utilities::get_video_adapter_name() const {
	return RenderingDevice::get_singleton()->get_device_name();
}

String Utilities::get_video_adapter_vendor() const {
	return RenderingDevice::get_singleton()->get_device_vendor_name();
}

RenderingDevice::DeviceType Utilities::get_video_adapter_type() const {
	return RenderingDevice::get_singleton()->get_device_type();
}

String Utilities::get_video_adapter_api_version() const {
	return RenderingDevice::get_singleton()->get_device_api_version();
}

Size2i Utilities::get_maximum_viewport_size() const {
	RenderingDevice *device = RenderingDevice::get_singleton();

	int max_x = device->limit_get(RenderingDevice::LIMIT_MAX_VIEWPORT_DIMENSIONS_X);
	int max_y = device->limit_get(RenderingDevice::LIMIT_MAX_VIEWPORT_DIMENSIONS_Y);
	return Size2i(max_x, max_y);
}

uint32_t Utilities::get_maximum_shader_varyings() const {
	return RenderingDevice::get_singleton()->limit_get(RenderingDevice::LIMIT_MAX_SHADER_VARYINGS);
}

uint64_t Utilities::get_maximum_uniform_buffer_size() const {
	return RenderingDevice::get_singleton()->limit_get(RenderingDevice::LIMIT_MAX_UNIFORM_BUFFER_SIZE);
}

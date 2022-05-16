/*************************************************************************/
/*  rasterizer_storage_gles3.cpp                                         */
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

#include "rasterizer_storage_gles3.h"

#ifdef GLES3_ENABLED

#include "core/config/project_settings.h"
#include "core/math/transform_3d.h"
// #include "rasterizer_canvas_gles3.h"
#include "rasterizer_scene_gles3.h"
#include "servers/rendering/shader_language.h"

void RasterizerStorageGLES3::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (GLES3::MeshStorage::get_singleton()->owns_mesh(p_base)) {
		GLES3::Mesh *mesh = GLES3::MeshStorage::get_singleton()->get_mesh(p_base);
		p_instance->update_dependency(&mesh->dependency);
	} else if (GLES3::MeshStorage::get_singleton()->owns_multimesh(p_base)) {
		GLES3::MultiMesh *multimesh = GLES3::MeshStorage::get_singleton()->get_multimesh(p_base);
		p_instance->update_dependency(&multimesh->dependency);
		if (multimesh->mesh.is_valid()) {
			base_update_dependency(multimesh->mesh, p_instance);
		}
	} else if (GLES3::LightStorage::get_singleton()->owns_light(p_base)) {
		GLES3::Light *l = GLES3::LightStorage::get_singleton()->get_light(p_base);
		p_instance->update_dependency(&l->dependency);
	}
}

/* VOXEL GI API */

RID RasterizerStorageGLES3::voxel_gi_allocate() {
	return RID();
}

void RasterizerStorageGLES3::voxel_gi_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
}

AABB RasterizerStorageGLES3::voxel_gi_get_bounds(RID p_voxel_gi) const {
	return AABB();
}

Vector3i RasterizerStorageGLES3::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	return Vector3i();
}

Vector<uint8_t> RasterizerStorageGLES3::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<uint8_t> RasterizerStorageGLES3::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<uint8_t> RasterizerStorageGLES3::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<int> RasterizerStorageGLES3::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	return Vector<int>();
}

Transform3D RasterizerStorageGLES3::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	return Transform3D();
}

void RasterizerStorageGLES3::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	return 0;
}

void RasterizerStorageGLES3::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_propagation(RID p_voxel_gi) const {
	return 0;
}

void RasterizerStorageGLES3::voxel_gi_set_energy(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_energy(RID p_voxel_gi) const {
	return 0.0;
}

void RasterizerStorageGLES3::voxel_gi_set_bias(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_bias(RID p_voxel_gi) const {
	return 0.0;
}

void RasterizerStorageGLES3::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	return 0.0;
}

void RasterizerStorageGLES3::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
}

bool RasterizerStorageGLES3::voxel_gi_is_interior(RID p_voxel_gi) const {
	return false;
}

void RasterizerStorageGLES3::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
}

bool RasterizerStorageGLES3::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	return false;
}

void RasterizerStorageGLES3::voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) {
}

float RasterizerStorageGLES3::voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const {
	return 0;
}

uint32_t RasterizerStorageGLES3::voxel_gi_get_version(RID p_voxel_gi) {
	return 0;
}

/* OCCLUDER */

void RasterizerStorageGLES3::occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) {
}

/* FOG */

RID RasterizerStorageGLES3::fog_volume_allocate() {
	return RID();
}

void RasterizerStorageGLES3::fog_volume_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) {
}

void RasterizerStorageGLES3::fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) {
}

void RasterizerStorageGLES3::fog_volume_set_material(RID p_fog_volume, RID p_material) {
}

AABB RasterizerStorageGLES3::fog_volume_get_aabb(RID p_fog_volume) const {
	return AABB();
}

RS::FogVolumeShape RasterizerStorageGLES3::fog_volume_get_shape(RID p_fog_volume) const {
	return RS::FOG_VOLUME_SHAPE_BOX;
}

/* VISIBILITY NOTIFIER */
RID RasterizerStorageGLES3::visibility_notifier_allocate() {
	return RID();
}

void RasterizerStorageGLES3::visibility_notifier_initialize(RID p_notifier) {
}

void RasterizerStorageGLES3::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
}

void RasterizerStorageGLES3::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
}

AABB RasterizerStorageGLES3::visibility_notifier_get_aabb(RID p_notifier) const {
	return AABB();
}

void RasterizerStorageGLES3::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
}

/* CANVAS SHADOW */

RID RasterizerStorageGLES3::canvas_light_shadow_buffer_create(int p_width) {
	CanvasLightShadow *cls = memnew(CanvasLightShadow);

	if (p_width > config->max_texture_size) {
		p_width = config->max_texture_size;
	}

	cls->size = p_width;
	cls->height = 16;

	glActiveTexture(GL_TEXTURE0);

	glGenFramebuffers(1, &cls->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, cls->fbo);

	glGenRenderbuffers(1, &cls->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, cls->depth);
	glRenderbufferStorage(GL_RENDERBUFFER, config->depth_buffer_internalformat, cls->size, cls->height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, cls->depth);

	glGenTextures(1, &cls->distance);
	glBindTexture(GL_TEXTURE_2D, cls->distance);
	if (config->use_rgba_2d_shadows) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cls->size, cls->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	} else {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, cls->size, cls->height, 0, GL_RED, GL_FLOAT, nullptr);
	}

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cls->distance, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	//printf("errnum: %x\n",status);
	glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		memdelete(cls);
		ERR_FAIL_COND_V(status != GL_FRAMEBUFFER_COMPLETE, RID());
	}

	return canvas_light_shadow_owner.make_rid(cls);
}

/* LIGHT SHADOW MAPPING */
/*

RID RasterizerStorageGLES3::canvas_light_occluder_create() {
	CanvasOccluder *co = memnew(CanvasOccluder);
	co->index_id = 0;
	co->vertex_id = 0;
	co->len = 0;

	return canvas_occluder_owner.make_rid(co);
}

void RasterizerStorageGLES3::canvas_light_occluder_set_polylines(RID p_occluder, const PoolVector<Vector2> &p_lines) {
	CanvasOccluder *co = canvas_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!co);

	co->lines = p_lines;

	if (p_lines.size() != co->len) {
		if (co->index_id) {
			glDeleteBuffers(1, &co->index_id);
		} if (co->vertex_id) {
			glDeleteBuffers(1, &co->vertex_id);
		}

		co->index_id = 0;
		co->vertex_id = 0;
		co->len = 0;
	}

	if (p_lines.size()) {
		PoolVector<float> geometry;
		PoolVector<uint16_t> indices;
		int lc = p_lines.size();

		geometry.resize(lc * 6);
		indices.resize(lc * 3);

		PoolVector<float>::Write vw = geometry.write();
		PoolVector<uint16_t>::Write iw = indices.write();

		PoolVector<Vector2>::Read lr = p_lines.read();

		const int POLY_HEIGHT = 16384;

		for (int i = 0; i < lc / 2; i++) {
			vw[i * 12 + 0] = lr[i * 2 + 0].x;
			vw[i * 12 + 1] = lr[i * 2 + 0].y;
			vw[i * 12 + 2] = POLY_HEIGHT;

			vw[i * 12 + 3] = lr[i * 2 + 1].x;
			vw[i * 12 + 4] = lr[i * 2 + 1].y;
			vw[i * 12 + 5] = POLY_HEIGHT;

			vw[i * 12 + 6] = lr[i * 2 + 1].x;
			vw[i * 12 + 7] = lr[i * 2 + 1].y;
			vw[i * 12 + 8] = -POLY_HEIGHT;

			vw[i * 12 + 9] = lr[i * 2 + 0].x;
			vw[i * 12 + 10] = lr[i * 2 + 0].y;
			vw[i * 12 + 11] = -POLY_HEIGHT;

			iw[i * 6 + 0] = i * 4 + 0;
			iw[i * 6 + 1] = i * 4 + 1;
			iw[i * 6 + 2] = i * 4 + 2;

			iw[i * 6 + 3] = i * 4 + 2;
			iw[i * 6 + 4] = i * 4 + 3;
			iw[i * 6 + 5] = i * 4 + 0;
		}

		//if same buffer len is being set, just use BufferSubData to avoid a pipeline flush

		if (!co->vertex_id) {
			glGenBuffers(1, &co->vertex_id);
			glBindBuffer(GL_ARRAY_BUFFER, co->vertex_id);
			glBufferData(GL_ARRAY_BUFFER, lc * 6 * sizeof(real_t), vw.ptr(), GL_STATIC_DRAW);
		} else {
			glBindBuffer(GL_ARRAY_BUFFER, co->vertex_id);
			glBufferSubData(GL_ARRAY_BUFFER, 0, lc * 6 * sizeof(real_t), vw.ptr());
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		if (!co->index_id) {
			glGenBuffers(1, &co->index_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, co->index_id);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, lc * 3 * sizeof(uint16_t), iw.ptr(), GL_DYNAMIC_DRAW);
		} else {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, co->index_id);
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, lc * 3 * sizeof(uint16_t), iw.ptr());
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind

		co->len = lc;
	}
}
*/

RS::InstanceType RasterizerStorageGLES3::get_base_type(RID p_rid) const {
	if (GLES3::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		return RS::INSTANCE_MESH;
	} else if (GLES3::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	} else if (GLES3::LightStorage::get_singleton()->owns_light(p_rid)) {
		return RS::INSTANCE_LIGHT;
	}
	return RS::INSTANCE_NONE;
}

bool RasterizerStorageGLES3::free(RID p_rid) {
	if (GLES3::TextureStorage::get_singleton()->owns_render_target(p_rid)) {
		GLES3::TextureStorage::get_singleton()->render_target_free(p_rid);
		return true;
	} else if (GLES3::TextureStorage::get_singleton()->owns_texture(p_rid)) {
		GLES3::TextureStorage::get_singleton()->texture_free(p_rid);
		return true;
	} else if (GLES3::TextureStorage::get_singleton()->owns_canvas_texture(p_rid)) {
		GLES3::TextureStorage::get_singleton()->canvas_texture_free(p_rid);
		return true;
	} else if (GLES3::MaterialStorage::get_singleton()->owns_shader(p_rid)) {
		GLES3::MaterialStorage::get_singleton()->shader_free(p_rid);
		return true;
	} else if (GLES3::MaterialStorage::get_singleton()->owns_material(p_rid)) {
		GLES3::MaterialStorage::get_singleton()->material_free(p_rid);
		return true;
	} else if (GLES3::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		GLES3::MeshStorage::get_singleton()->mesh_free(p_rid);
		return true;
	} else if (GLES3::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		GLES3::MeshStorage::get_singleton()->multimesh_free(p_rid);
		return true;
	} else if (GLES3::MeshStorage::get_singleton()->owns_mesh_instance(p_rid)) {
		GLES3::MeshStorage::get_singleton()->mesh_instance_free(p_rid);
		return true;
	} else if (GLES3::LightStorage::get_singleton()->owns_light(p_rid)) {
		GLES3::LightStorage::get_singleton()->light_free(p_rid);
		return true;
	} else {
		return false;
	}
	/*
	  else if (reflection_probe_owner.owns(p_rid)) {
		// delete the texture
		ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_rid);
		reflection_probe->instance_remove_deps();

		reflection_probe_owner.free(p_rid);
		memdelete(reflection_probe);

		return true;
	} else if (lightmap_capture_data_owner.owns(p_rid)) {
		// delete the texture
		LightmapCapture *lightmap_capture = lightmap_capture_data_owner.get_or_null(p_rid);
		lightmap_capture->instance_remove_deps();

		lightmap_capture_data_owner.free(p_rid);
		memdelete(lightmap_capture);
		return true;

	} else if (canvas_occluder_owner.owns(p_rid)) {
		CanvasOccluder *co = canvas_occluder_owner.get_or_null(p_rid);
		if (co->index_id) {
			glDeleteBuffers(1, &co->index_id);
		}
		if (co->vertex_id) {
			glDeleteBuffers(1, &co->vertex_id);
		}

		canvas_occluder_owner.free(p_rid);
		memdelete(co);

		return true;

	} else if (canvas_light_shadow_owner.owns(p_rid)) {
		CanvasLightShadow *cls = canvas_light_shadow_owner.get_or_null(p_rid);
		glDeleteFramebuffers(1, &cls->fbo);
		glDeleteRenderbuffers(1, &cls->depth);
		glDeleteTextures(1, &cls->distance);
		canvas_light_shadow_owner.free(p_rid);
		memdelete(cls);

		return true;
		*/
}

bool RasterizerStorageGLES3::has_os_feature(const String &p_feature) const {
	if (p_feature == "rgtc") {
		return config->rgtc_supported;
	}

	if (p_feature == "s3tc") {
		return config->s3tc_supported;
	}

	if (p_feature == "bptc") {
		return config->bptc_supported;
	}
	if (p_feature == "etc") {
		return config->etc_supported;
	}

	if (p_feature == "etc2") {
		return config->etc2_supported;
	}

	return false;
}

////////////////////////////////////////////

void RasterizerStorageGLES3::set_debug_generate_wireframes(bool p_generate) {
}

//void RasterizerStorageGLES3::render_info_begin_capture() {
//	info.snap = info.render;
//}

//void RasterizerStorageGLES3::render_info_end_capture() {
//	info.snap.object_count = info.render.object_count - info.snap.object_count;
//	info.snap.draw_call_count = info.render.draw_call_count - info.snap.draw_call_count;
//	info.snap.material_switch_count = info.render.material_switch_count - info.snap.material_switch_count;
//	info.snap.surface_switch_count = info.render.surface_switch_count - info.snap.surface_switch_count;
//	info.snap.shader_rebind_count = info.render.shader_rebind_count - info.snap.shader_rebind_count;
//	info.snap.vertices_count = info.render.vertices_count - info.snap.vertices_count;
//	info.snap._2d_item_count = info.render._2d_item_count - info.snap._2d_item_count;
//	info.snap._2d_draw_call_count = info.render._2d_draw_call_count - info.snap._2d_draw_call_count;
//}

//int RasterizerStorageGLES3::get_captured_render_info(RS::RenderInfo p_info) {
//	switch (p_info) {
//		case RS::INFO_OBJECTS_IN_FRAME: {
//			return info.snap.object_count;
//		} break;
//		case RS::INFO_VERTICES_IN_FRAME: {
//			return info.snap.vertices_count;
//		} break;
//		case RS::INFO_MATERIAL_CHANGES_IN_FRAME: {
//			return info.snap.material_switch_count;
//		} break;
//		case RS::INFO_SHADER_CHANGES_IN_FRAME: {
//			return info.snap.shader_rebind_count;
//		} break;
//		case RS::INFO_SURFACE_CHANGES_IN_FRAME: {
//			return info.snap.surface_switch_count;
//		} break;
//		case RS::INFO_DRAW_CALLS_IN_FRAME: {
//			return info.snap.draw_call_count;
//		} break;
//			/*
//		case RS::INFO_2D_ITEMS_IN_FRAME: {
//			return info.snap._2d_item_count;
//		} break;
//		case RS::INFO_2D_DRAW_CALLS_IN_FRAME: {
//			return info.snap._2d_draw_call_count;
//		} break;
//			*/
//		default: {
//			return get_render_info(p_info);
//		}
//	}
//}

//int RasterizerStorageGLES3::get_render_info(RS::RenderInfo p_info) {
//	switch (p_info) {
//		case RS::INFO_OBJECTS_IN_FRAME:
//			return info.render_final.object_count;
//		case RS::INFO_VERTICES_IN_FRAME:
//			return info.render_final.vertices_count;
//		case RS::INFO_MATERIAL_CHANGES_IN_FRAME:
//			return info.render_final.material_switch_count;
//		case RS::INFO_SHADER_CHANGES_IN_FRAME:
//			return info.render_final.shader_rebind_count;
//		case RS::INFO_SURFACE_CHANGES_IN_FRAME:
//			return info.render_final.surface_switch_count;
//		case RS::INFO_DRAW_CALLS_IN_FRAME:
//			return info.render_final.draw_call_count;
//			/*
//		case RS::INFO_2D_ITEMS_IN_FRAME:
//			return info.render_final._2d_item_count;
//		case RS::INFO_2D_DRAW_CALLS_IN_FRAME:
//			return info.render_final._2d_draw_call_count;
//*/
//		case RS::INFO_USAGE_VIDEO_MEM_TOTAL:
//			return 0; //no idea
//		case RS::INFO_VIDEO_MEM_USED:
//			return info.vertex_mem + info.texture_mem;
//		case RS::INFO_TEXTURE_MEM_USED:
//			return info.texture_mem;
//		case RS::INFO_VERTEX_MEM_USED:
//			return info.vertex_mem;
//		default:
//			return 0; //no idea either
//	}
//}

String RasterizerStorageGLES3::get_video_adapter_name() const {
	return (const char *)glGetString(GL_RENDERER);
}

String RasterizerStorageGLES3::get_video_adapter_vendor() const {
	return (const char *)glGetString(GL_VENDOR);
}

RenderingDevice::DeviceType RasterizerStorageGLES3::get_video_adapter_type() const {
	return RenderingDevice::DeviceType::DEVICE_TYPE_OTHER;
}

String RasterizerStorageGLES3::get_video_adapter_api_version() const {
	return (const char *)glGetString(GL_VERSION);
}

void RasterizerStorageGLES3::initialize() {
	config = GLES3::Config::get_singleton();

	// skeleton buffer
	{
		resources.skeleton_transform_buffer_size = 0;
		glGenBuffers(1, &resources.skeleton_transform_buffer);
	}

	// radical inverse vdc cache texture
	// used for cubemap filtering
	glGenTextures(1, &resources.radical_inverse_vdc_cache_tex);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, resources.radical_inverse_vdc_cache_tex);
	/*
	uint8_t radical_inverse[512];

	for (uint32_t i = 0; i < 512; i++) {
		uint32_t bits = i;

		bits = (bits << 16) | (bits >> 16);
		bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
		bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
		bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
		bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);

		float value = float(bits) * 2.3283064365386963e-10;
		radical_inverse[i] = uint8_t(CLAMP(value * 255.0, 0, 255));
	}

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 512, 1, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, radical_inverse);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //need this for proper sampling
	*/
	glBindTexture(GL_TEXTURE_2D, 0);

	{
		glGenFramebuffers(1, &resources.mipmap_blur_fbo);
		glGenTextures(1, &resources.mipmap_blur_color);
	}

#ifdef GLES_OVER_GL
	glEnable(GL_PROGRAM_POINT_SIZE);
#endif
}

void RasterizerStorageGLES3::finalize() {
}

void RasterizerStorageGLES3::_copy_screen() {
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}
void RasterizerStorageGLES3::update_memory_info() {
}

uint64_t RasterizerStorageGLES3::get_rendering_info(RS::RenderingInfo p_info) {
	return 0;
}

void RasterizerStorageGLES3::update_dirty_resources() {
	GLES3::MaterialStorage::get_singleton()->_update_global_variables();
	GLES3::MaterialStorage::get_singleton()->_update_queued_materials();
	//GLES3::MeshStorage::get_singleton()->_update_dirty_skeletons();
	GLES3::MeshStorage::get_singleton()->_update_dirty_multimeshes();
}

RasterizerStorageGLES3::RasterizerStorageGLES3() {
	initialize();
}

RasterizerStorageGLES3::~RasterizerStorageGLES3() {
}

#endif // GLES3_ENABLED

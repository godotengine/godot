/*************************************************************************/
/*  utilities.cpp                                                        */
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

#ifdef GLES3_ENABLED

#include "utilities.h"
#include "config.h"
#include "light_storage.h"
#include "material_storage.h"
#include "mesh_storage.h"
#include "particles_storage.h"
#include "texture_storage.h"

using namespace GLES3;

Utilities *Utilities::singleton = nullptr;

Utilities::Utilities() {
	singleton = this;
}

Utilities::~Utilities() {
	singleton = nullptr;
}

Vector<uint8_t> Utilities::buffer_get_data(GLenum p_target, GLuint p_buffer, uint32_t p_buffer_size) {
	Vector<uint8_t> ret;
	ret.resize(p_buffer_size);
	glBindBuffer(p_target, p_buffer);

#if defined(__EMSCRIPTEN__)
	{
		uint8_t *w = ret.ptrw();
		glGetBufferSubData(p_target, 0, p_buffer_size, w);
	}
#else
	void *data = glMapBufferRange(p_target, 0, p_buffer_size, GL_MAP_READ_BIT);
	ERR_FAIL_NULL_V(data, Vector<uint8_t>());
	{
		uint8_t *w = ret.ptrw();
		memcpy(w, data, p_buffer_size);
	}
	glUnmapBuffer(p_target);
#endif
	glBindBuffer(p_target, 0);
	return ret;
}

/* INSTANCES */

RS::InstanceType Utilities::get_base_type(RID p_rid) const {
	if (GLES3::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		return RS::INSTANCE_MESH;
	} else if (GLES3::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	} else if (GLES3::LightStorage::get_singleton()->owns_light(p_rid)) {
		return RS::INSTANCE_LIGHT;
	}
	return RS::INSTANCE_NONE;
}

bool Utilities::free(RID p_rid) {
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
	}
	*/
}

/* DEPENDENCIES */

void Utilities::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (MeshStorage::get_singleton()->owns_mesh(p_base)) {
		Mesh *mesh = MeshStorage::get_singleton()->get_mesh(p_base);
		p_instance->update_dependency(&mesh->dependency);
	} else if (MeshStorage::get_singleton()->owns_multimesh(p_base)) {
		MultiMesh *multimesh = MeshStorage::get_singleton()->get_multimesh(p_base);
		p_instance->update_dependency(&multimesh->dependency);
		if (multimesh->mesh.is_valid()) {
			base_update_dependency(multimesh->mesh, p_instance);
		}
	} else if (LightStorage::get_singleton()->owns_light(p_base)) {
		Light *l = LightStorage::get_singleton()->get_light(p_base);
		p_instance->update_dependency(&l->dependency);
	}
}

/* VISIBILITY NOTIFIER */

RID Utilities::visibility_notifier_allocate() {
	return RID();
}

void Utilities::visibility_notifier_initialize(RID p_notifier) {
}

void Utilities::visibility_notifier_free(RID p_notifier) {
}

void Utilities::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
}

void Utilities::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
}

AABB Utilities::visibility_notifier_get_aabb(RID p_notifier) const {
	return AABB();
}

void Utilities::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
}

/* TIMING */

//void Utilities::render_info_begin_capture() {
//	info.snap = info.render;
//}

//void Utilities::render_info_end_capture() {
//	info.snap.object_count = info.render.object_count - info.snap.object_count;
//	info.snap.draw_call_count = info.render.draw_call_count - info.snap.draw_call_count;
//	info.snap.material_switch_count = info.render.material_switch_count - info.snap.material_switch_count;
//	info.snap.surface_switch_count = info.render.surface_switch_count - info.snap.surface_switch_count;
//	info.snap.shader_rebind_count = info.render.shader_rebind_count - info.snap.shader_rebind_count;
//	info.snap.vertices_count = info.render.vertices_count - info.snap.vertices_count;
//	info.snap._2d_item_count = info.render._2d_item_count - info.snap._2d_item_count;
//	info.snap._2d_draw_call_count = info.render._2d_draw_call_count - info.snap._2d_draw_call_count;
//}

//int Utilities::get_captured_render_info(RS::RenderInfo p_info) {
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

//int Utilities::get_render_info(RS::RenderInfo p_info) {
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

/* MISC */

void Utilities::update_dirty_resources() {
	MaterialStorage::get_singleton()->_update_global_shader_uniforms();
	MaterialStorage::get_singleton()->_update_queued_materials();
	//MeshStorage::get_singleton()->_update_dirty_skeletons();
	MeshStorage::get_singleton()->_update_dirty_multimeshes();
}

void Utilities::set_debug_generate_wireframes(bool p_generate) {
}

bool Utilities::has_os_feature(const String &p_feature) const {
	Config *config = Config::get_singleton();
	if (!config) {
		return false;
	}

	if (p_feature == "rgtc") {
		return config->rgtc_supported;
	}

	if (p_feature == "s3tc") {
		return config->s3tc_supported;
	}

	if (p_feature == "bptc") {
		return config->bptc_supported;
	}

	if (p_feature == "etc" || p_feature == "etc2") {
		return config->etc2_supported;
	}

	return false;
}

void Utilities::update_memory_info() {
}

uint64_t Utilities::get_rendering_info(RS::RenderingInfo p_info) {
	return 0;
}

String Utilities::get_video_adapter_name() const {
	return (const char *)glGetString(GL_RENDERER);
}

String Utilities::get_video_adapter_vendor() const {
	return (const char *)glGetString(GL_VENDOR);
}

RenderingDevice::DeviceType Utilities::get_video_adapter_type() const {
	return RenderingDevice::DeviceType::DEVICE_TYPE_OTHER;
}

String Utilities::get_video_adapter_api_version() const {
	return (const char *)glGetString(GL_VERSION);
}

#endif // GLES3_ENABLED

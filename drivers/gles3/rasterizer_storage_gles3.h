/*************************************************************************/
/*  rasterizer_storage_gles3.h                                           */
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

#ifndef RASTERIZER_STORAGE_OPENGL_H
#define RASTERIZER_STORAGE_OPENGL_H

#ifdef GLES3_ENABLED

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_storage.h"
#include "servers/rendering/shader_compiler.h"
#include "servers/rendering/shader_language.h"
#include "storage/config.h"
#include "storage/light_storage.h"
#include "storage/material_storage.h"
#include "storage/mesh_storage.h"
#include "storage/texture_storage.h"

// class RasterizerCanvasGLES3;
// class RasterizerSceneGLES3;

class RasterizerStorageGLES3 : public RendererStorage {
public:
	// RasterizerCanvasGLES3 *canvas;
	// RasterizerSceneGLES3 *scene;

	GLES3::Config *config = nullptr;

	static _FORCE_INLINE_ void store_transform(const Transform3D &p_mtx, float *p_array) {
		p_array[0] = p_mtx.basis.rows[0][0];
		p_array[1] = p_mtx.basis.rows[1][0];
		p_array[2] = p_mtx.basis.rows[2][0];
		p_array[3] = 0;
		p_array[4] = p_mtx.basis.rows[0][1];
		p_array[5] = p_mtx.basis.rows[1][1];
		p_array[6] = p_mtx.basis.rows[2][1];
		p_array[7] = 0;
		p_array[8] = p_mtx.basis.rows[0][2];
		p_array[9] = p_mtx.basis.rows[1][2];
		p_array[10] = p_mtx.basis.rows[2][2];
		p_array[11] = 0;
		p_array[12] = p_mtx.origin.x;
		p_array[13] = p_mtx.origin.y;
		p_array[14] = p_mtx.origin.z;
		p_array[15] = 1;
	}

	static _FORCE_INLINE_ void store_transform_3x3(const Basis &p_mtx, float *p_array) {
		p_array[0] = p_mtx.rows[0][0];
		p_array[1] = p_mtx.rows[1][0];
		p_array[2] = p_mtx.rows[2][0];
		p_array[3] = 0;
		p_array[4] = p_mtx.rows[0][1];
		p_array[5] = p_mtx.rows[1][1];
		p_array[6] = p_mtx.rows[2][1];
		p_array[7] = 0;
		p_array[8] = p_mtx.rows[0][2];
		p_array[9] = p_mtx.rows[1][2];
		p_array[10] = p_mtx.rows[2][2];
		p_array[11] = 0;
	}

	static _FORCE_INLINE_ void store_camera(const CameraMatrix &p_mtx, float *p_array) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				p_array[i * 4 + j] = p_mtx.matrix[i][j];
			}
		}
	}

	// Buffer size is specified in bytes
	static Vector<uint8_t> buffer_get_data(GLenum p_target, GLuint p_buffer, uint32_t p_buffer_size);

	struct Resources {
		GLuint mipmap_blur_fbo;
		GLuint mipmap_blur_color;

		GLuint radical_inverse_vdc_cache_tex;
		bool use_rgba_2d_shadows;

		size_t skeleton_transform_buffer_size;
		GLuint skeleton_transform_buffer;
		LocalVector<float> skeleton_transform_cpu_buffer;

	} resources;

	struct Info {
		uint64_t texture_mem = 0;
		uint64_t vertex_mem = 0;

		struct Render {
			uint32_t object_count;
			uint32_t draw_call_count;
			uint32_t material_switch_count;
			uint32_t surface_switch_count;
			uint32_t shader_rebind_count;
			uint32_t vertices_count;
			uint32_t _2d_item_count;
			uint32_t _2d_draw_call_count;

			void reset() {
				object_count = 0;
				draw_call_count = 0;
				material_switch_count = 0;
				surface_switch_count = 0;
				shader_rebind_count = 0;
				vertices_count = 0;
				_2d_item_count = 0;
				_2d_draw_call_count = 0;
			}
		} render, render_final, snap;

		Info() {
			render.reset();
			render_final.reset();
		}

	} info;

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////API////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

public:
	virtual void base_update_dependency(RID p_base, DependencyTracker *p_instance) override;

	/* OCCLUDER */

	void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices);

	/* FOG VOLUMES */

	RID fog_volume_allocate() override;
	void fog_volume_initialize(RID p_rid) override;

	void fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) override;
	void fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) override;
	void fog_volume_set_material(RID p_fog_volume, RID p_material) override;
	AABB fog_volume_get_aabb(RID p_fog_volume) const override;
	RS::FogVolumeShape fog_volume_get_shape(RID p_fog_volume) const override;

	/* VISIBILITY NOTIFIER */
	RID visibility_notifier_allocate() override;
	void visibility_notifier_initialize(RID p_notifier) override;
	void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) override;
	void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) override;

	AABB visibility_notifier_get_aabb(RID p_notifier) const override;
	void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) override;

	// access from canvas
	//	GLES3::RenderTarget * render_target_get(RID p_render_target);

	/* CANVAS SHADOW */

	struct CanvasLightShadow {
		RID self;
		int size;
		int height;
		GLuint fbo;
		GLuint depth;
		GLuint distance; //for older devices
	};

	RID_PtrOwner<CanvasLightShadow> canvas_light_shadow_owner;

	RID canvas_light_shadow_buffer_create(int p_width);

	/* LIGHT SHADOW MAPPING */
	/*
	struct CanvasOccluder {
		RID self;

		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		LocalVector<Vector2> lines;
		int len;
	};

	RID_Owner<CanvasOccluder> canvas_occluder_owner;

	RID canvas_light_occluder_create();
	void canvas_light_occluder_set_polylines(RID p_occluder, const LocalVector<Vector2> &p_lines);
*/

	RS::InstanceType get_base_type(RID p_rid) const override;

	bool free(RID p_rid) override;

	void initialize();
	void finalize();

	void update_memory_info() override;
	uint64_t get_rendering_info(RS::RenderingInfo p_info) override;

	bool has_os_feature(const String &p_feature) const override;

	void update_dirty_resources() override;

	void set_debug_generate_wireframes(bool p_generate) override;

	//	void render_info_begin_capture() override;
	//	void render_info_end_capture() override;
	//	int get_captured_render_info(RS::RenderInfo p_info) override;

	//	int get_render_info(RS::RenderInfo p_info) override;
	String get_video_adapter_name() const override;
	String get_video_adapter_vendor() const override;
	RenderingDevice::DeviceType get_video_adapter_type() const override;
	String get_video_adapter_api_version() const override;

	void capture_timestamps_begin() override {}
	void capture_timestamp(const String &p_name) override {}
	uint32_t get_captured_timestamps_count() const override {
		return 0;
	}
	uint64_t get_captured_timestamps_frame() const override {
		return 0;
	}
	uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const override {
		return 0;
	}
	uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const override {
		return 0;
	}
	String get_captured_timestamp_name(uint32_t p_index) const override {
		return String();
	}

	RasterizerStorageGLES3();
	~RasterizerStorageGLES3();
};

#endif // GLES3_ENABLED

#endif // RASTERIZER_STORAGE_OPENGL_H

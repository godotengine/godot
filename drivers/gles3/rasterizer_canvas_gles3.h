/*************************************************************************/
/*  rasterizer_canvas_gles3.h                                            */
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

#ifndef RASTERIZER_CANVAS_OPENGL_H
#define RASTERIZER_CANVAS_OPENGL_H

#ifdef GLES3_ENABLED

#include "rasterizer_scene_gles3.h"
#include "rasterizer_storage_gles3.h"
#include "servers/rendering/renderer_canvas_render.h"
#include "servers/rendering/renderer_compositor.h"

#include "shaders/canvas.glsl.gen.h"

class RasterizerSceneGLES3;

class RasterizerCanvasGLES3 : public RendererCanvasRender {
	_FORCE_INLINE_ void _update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4);
	_FORCE_INLINE_ void _update_transform_2d_to_mat2x3(const Transform2D &p_transform, float *p_mat2x3);

	_FORCE_INLINE_ void _update_transform_2d_to_mat4(const Transform2D &p_transform, float *p_mat4);
	_FORCE_INLINE_ void _update_transform_to_mat4(const Transform3D &p_transform, float *p_mat4);

	enum {
		BASE_UNIFORM_BUFFER_OBJECT = 0,
		MATERIAL_UNIFORM_BUFFER_OBJECT = 1,
		TRANSFORMS_UNIFORM_BUFFER_OBJECT = 2,
		CANVAS_TEXTURE_UNIFORM_BUFFER_OBJECT = 3,
	};

	enum {

		FLAGS_INSTANCING_MASK = 0x7F,
		FLAGS_INSTANCING_HAS_COLORS = (1 << 7),
		FLAGS_INSTANCING_HAS_CUSTOM_DATA = (1 << 8),

		FLAGS_CLIP_RECT_UV = (1 << 9),
		FLAGS_TRANSPOSE_RECT = (1 << 10),

		FLAGS_NINEPACH_DRAW_CENTER = (1 << 12),
		FLAGS_USING_PARTICLES = (1 << 13),

		FLAGS_USE_SKELETON = (1 << 15),
		FLAGS_NINEPATCH_H_MODE_SHIFT = 16,
		FLAGS_NINEPATCH_V_MODE_SHIFT = 18,
		FLAGS_LIGHT_COUNT_SHIFT = 20,

		FLAGS_DEFAULT_NORMAL_MAP_USED = (1 << 26),
		FLAGS_DEFAULT_SPECULAR_MAP_USED = (1 << 27),

		FLAGS_USE_MSDF = (1 << 28),
	};

	enum {
		LIGHT_FLAGS_TEXTURE_MASK = 0xFFFF,
		LIGHT_FLAGS_BLEND_SHIFT = 16,
		LIGHT_FLAGS_BLEND_MASK = (3 << 16),
		LIGHT_FLAGS_BLEND_MODE_ADD = (0 << 16),
		LIGHT_FLAGS_BLEND_MODE_SUB = (1 << 16),
		LIGHT_FLAGS_BLEND_MODE_MIX = (2 << 16),
		LIGHT_FLAGS_BLEND_MODE_MASK = (3 << 16),
		LIGHT_FLAGS_HAS_SHADOW = (1 << 20),
		LIGHT_FLAGS_FILTER_SHIFT = 22

	};

	enum {
		MAX_RENDER_ITEMS = 256 * 1024,
		MAX_LIGHT_TEXTURES = 1024,
		MAX_LIGHTS_PER_ITEM = 16,
		DEFAULT_MAX_LIGHTS_PER_RENDER = 256,
	};

public:
	struct StateBuffer {
		float canvas_transform[16];
		float screen_transform[16];
		float canvas_normal_transform[16];
		float canvas_modulate[4];

		float screen_pixel_size[2];
		float time;
		uint32_t use_pixel_snap;

		float sdf_to_tex[4];
		float sdf_to_screen[2];
		float screen_to_sdf[2];

		uint32_t directional_light_count;
		float tex_to_sdf;
		uint32_t pad1;
		uint32_t pad2;
	};

	struct InstanceData {
		float world[6];
		float color_texture_pixel_size[2];
		union {
			//rect
			struct {
				float modulation[4];
				union {
					float msdf[4];
					float ninepatch_margins[4];
				};
				float dst_rect[4];
				float src_rect[4];
				float pad[2];
			};
			//primitive
			struct {
				float points[6]; // vec2 points[3]
				float uvs[6]; // vec2 points[3]
				uint32_t colors[6]; // colors encoded as half
			};
		};
		uint32_t flags;
		uint32_t specular_shininess;
		uint32_t lights[4];
	};

	struct Data {
		GLuint canvas_quad_vertices;
		GLuint canvas_quad_array;

		GLuint particle_quad_vertices;
		GLuint particle_quad_array;

		GLuint ninepatch_vertices;
		GLuint ninepatch_elements;
	} data;

	struct State {
		GLuint canvas_state_buffer;
		LocalVector<GLuint> canvas_instance_data_buffers;
		LocalVector<GLsync> fences;
		uint32_t current_buffer = 0;

		InstanceData *instance_data_array;
		bool canvas_texscreen_used;
		CanvasShaderGLES3 canvas_shader;
		RID canvas_shader_current_version;
		RID canvas_shader_default_version;
		//CanvasShadowShaderGLES3 canvas_shadow_shader;
		//LensDistortedShaderGLES3 lens_shader;

		bool using_texture_rect;

		bool using_ninepatch;
		bool using_skeleton;

		Transform2D skeleton_transform;
		Transform2D skeleton_transform_inverse;
		Size2i skeleton_texture_size;

		RID current_tex = RID();
		RID current_normal = RID();
		RID current_specular = RID();
		RasterizerStorageGLES3::Texture *current_tex_ptr;
		RID current_shader_version = RID();
		RS::PrimitiveType current_primitive = RS::PRIMITIVE_MAX;
		uint32_t current_primitive_points = 0;
		Item::Command::Type current_command = Item::Command::TYPE_RECT;

		bool end_batch = false;

		Transform3D vp;
		Light *using_light;
		bool using_shadow;
		bool using_transparent_rt;

		// FROM RD Renderer

		uint32_t max_lights_per_render;
		uint32_t max_lights_per_item;
		uint32_t max_instances_per_batch;

		RS::CanvasItemTextureFilter default_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT;
		RS::CanvasItemTextureRepeat default_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT;
	} state;

	Item *items[MAX_RENDER_ITEMS];

	RID default_canvas_texture;
	RID default_canvas_group_material;
	RID default_canvas_group_shader;

	typedef void Texture;

	RasterizerSceneGLES3 *scene_render;

	RasterizerStorageGLES3 *storage;

	void _set_uniforms();

	void canvas_begin();
	void canvas_end();

	//virtual void draw_window_margins(int *black_margin, RID *black_image) override;
	void draw_lens_distortion_rect(const Rect2 &p_rect, float p_k1, float p_k2, const Vector2 &p_eye_center, float p_oversample);

	virtual void reset_canvas();
	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache);

	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow) override;

	RID light_create() override;
	void light_set_texture(RID p_rid, RID p_texture) override;
	void light_set_use_shadow(RID p_rid, bool p_enable) override;
	void light_update_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) override;
	void light_update_directional_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_cull_distance, const Rect2 &p_clip_rect, LightOccluderInstance *p_occluders) override;

	void render_sdf(RID p_render_target, LightOccluderInstance *p_occluders) override;
	RID occluder_polygon_create() override;
	void occluder_polygon_set_shape(RID p_occluder, const Vector<Vector2> &p_points, bool p_closed) override;
	void occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) override;
	void set_shadow_texture_size(int p_size) override;

	bool free(RID p_rid) override;
	void update() override;

	void _bind_canvas_texture(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, uint32_t &r_index, RID &r_last_texture, Size2 &r_texpixel_size);

	struct PolygonBuffers {
		GLuint vertex_buffer;
		GLuint vertex_array;
		GLuint index_buffer;
		int count;
	};

	struct {
		HashMap<PolygonID, PolygonBuffers> polygons;
		PolygonID last_id;
	} polygon_buffers;

	RendererCanvasRender::PolygonID request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) override;
	void free_polygon(PolygonID p_polygon) override;

	void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) override;
	void _render_items(RID p_to_render_target, int p_item_count, const Transform2D &p_canvas_transform_inverse, Light *p_lights, bool p_to_backbuffer = false);
	void _render_item(RID p_render_target, const Item *p_item, const Transform2D &p_canvas_transform_inverse, Item *&current_clip, Light *p_lights, uint32_t &r_index);
	void _render_batch(uint32_t &p_max_index);
	void _end_batch(uint32_t &p_max_index);
	void _allocate_instance_data_buffer();

	void initialize();
	void finalize();
	RasterizerCanvasGLES3();
	~RasterizerCanvasGLES3();
};

#endif // GLES3_ENABLED
#endif // RASTERIZER_CANVAS_OPENGL_H

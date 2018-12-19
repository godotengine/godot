/*************************************************************************/
/*  rasterizer_canvas_gles2.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RASTERIZERCANVASGLES2_H
#define RASTERIZERCANVASGLES2_H

#include "rasterizer_storage_gles2.h"
#include "servers/visual/rasterizer.h"

#include "shaders/canvas.glsl.gen.h"
#include "shaders/lens_distorted.glsl.gen.h"

// #include "shaders/canvas_shadow.glsl.gen.h"

class RasterizerSceneGLES2;

class RasterizerCanvasGLES2 : public RasterizerCanvas {
public:
	struct Uniforms {
		Transform projection_matrix;

		Transform2D modelview_matrix;
		Transform2D extra_matrix;

		Color final_modulate;

		float time;
	};

	struct Data {

		GLuint canvas_quad_vertices;
		GLuint polygon_buffer;
		GLuint polygon_index_buffer;

		uint32_t polygon_buffer_size;

		GLuint ninepatch_vertices;
		GLuint ninepatch_elements;

	} data;

	struct State {
		Uniforms uniforms;
		bool canvas_texscreen_used;
		CanvasShaderGLES2 canvas_shader;
		// CanvasShadowShaderGLES3 canvas_shadow_shader;
		LensDistortedShaderGLES2 lens_shader;

		bool using_texture_rect;
		bool using_ninepatch;

		RID current_tex;
		RID current_normal;
		RasterizerStorageGLES2::Texture *current_tex_ptr;

		Transform vp;

	} state;

	typedef void Texture;

	RasterizerSceneGLES2 *scene_render;

	RasterizerStorageGLES2 *storage;

	virtual RID light_internal_create();
	virtual void light_internal_update(RID p_rid, Light *p_light);
	virtual void light_internal_free(RID p_rid);

	void _set_uniforms();

	virtual void canvas_begin();
	virtual void canvas_end();

	_FORCE_INLINE_ void _set_texture_rect_mode(bool p_enable, bool p_ninepatch = false);

	_FORCE_INLINE_ void _draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs);
	_FORCE_INLINE_ void _draw_polygon(const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor);
	_FORCE_INLINE_ void _draw_generic(GLuint p_primitive, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor);

	_FORCE_INLINE_ void _canvas_item_render_commands(Item *p_item, Item *current_clip, bool &reclip, RasterizerStorageGLES2::Material *p_material);
	_FORCE_INLINE_ void _copy_texscreen(const Rect2 &p_rect);

	virtual void canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform);
	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow);

	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache);

	virtual void reset_canvas();

	RasterizerStorageGLES2::Texture *_bind_canvas_texture(const RID &p_texture, const RID &p_normal_map);

	void _bind_quad_buffer();
	void draw_generic_textured_rect(const Rect2 &p_rect, const Rect2 &p_src);
	void draw_lens_distortion_rect(const Rect2 &p_rect, float p_k1, float p_k2, const Vector2 &p_eye_center, float p_oversample);

	void initialize();
	void finalize();

	virtual void draw_window_margins(int *black_margin, RID *black_image);

	RasterizerCanvasGLES2();
};

#endif // RASTERIZERCANVASGLES2_H

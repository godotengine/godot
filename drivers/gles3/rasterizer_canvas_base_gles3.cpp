/*************************************************************************/
/*  rasterizer_canvas_base_gles3.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer_canvas_base_gles3.h"
#ifdef GLES3_BACKEND_ENABLED

#include "core/os/os.h"
#include "drivers/gles3/rasterizer_asserts.h"
#include "rasterizer_scene_gles3.h"

#include "core/config/project_settings.h"
#include "servers/rendering/rendering_server_default.h"

#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif

static _FORCE_INLINE_ void store_transform3d(const Transform3D &p_mtx, float *p_array) {
	p_array[0] = p_mtx.basis.elements[0][0];
	p_array[1] = p_mtx.basis.elements[1][0];
	p_array[2] = p_mtx.basis.elements[2][0];
	p_array[3] = 0;
	p_array[4] = p_mtx.basis.elements[0][1];
	p_array[5] = p_mtx.basis.elements[1][1];
	p_array[6] = p_mtx.basis.elements[2][1];
	p_array[7] = 0;
	p_array[8] = p_mtx.basis.elements[0][2];
	p_array[9] = p_mtx.basis.elements[1][2];
	p_array[10] = p_mtx.basis.elements[2][2];
	p_array[11] = 0;
	p_array[12] = p_mtx.origin.x;
	p_array[13] = p_mtx.origin.y;
	p_array[14] = p_mtx.origin.z;
	p_array[15] = 1;
}

RID RasterizerCanvasBaseGLES3::light_internal_create() {
	return RID();
}

void RasterizerCanvasBaseGLES3::light_internal_update(RID p_rid, Light *p_light) {
}

void RasterizerCanvasBaseGLES3::light_internal_free(RID p_rid) {
}

RID RasterizerCanvasBaseGLES3::light_create() {
	return RID();
}

void RasterizerCanvasBaseGLES3::light_set_texture(RID p_rid, RID p_texture) {
}

void RasterizerCanvasBaseGLES3::light_set_use_shadow(RID p_rid, bool p_enable) {
}

void RasterizerCanvasBaseGLES3::light_update_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) {
}

void RasterizerCanvasBaseGLES3::light_update_directional_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_cull_distance, const Rect2 &p_clip_rect, LightOccluderInstance *p_occluders) {
}

void RasterizerCanvasBaseGLES3::render_sdf(RID p_render_target, LightOccluderInstance *p_occluders) {
}

RID RasterizerCanvasBaseGLES3::occluder_polygon_create() {
	return RID();
}

void RasterizerCanvasBaseGLES3::occluder_polygon_set_shape(RID p_occluder, const Vector<Vector2> &p_points, bool p_closed) {
}

void RasterizerCanvasBaseGLES3::occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) {
}

void RasterizerCanvasBaseGLES3::set_shadow_texture_size(int p_size) {
}

bool RasterizerCanvasBaseGLES3::free(RID p_rid) {
	return true;
}

void RasterizerCanvasBaseGLES3::update() {
}

void RasterizerCanvasBaseGLES3::canvas_begin() {
	state.using_transparent_rt = false;

	// always start with light_angle unset
	state.using_light_angle = false;
	state.using_large_vertex = false;
	state.using_modulate = false;

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_ATTRIB_LIGHT_ANGLE, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_ATTRIB_MODULATE, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_ATTRIB_LARGE_VERTEX, false);
	state.canvas_shader.bind();

	int viewport_x, viewport_y, viewport_width, viewport_height;

	if (storage->frame.current_rt) {
		storage->bind_framebuffer(storage->frame.current_rt->fbo);
		state.using_transparent_rt = storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT];

		if (storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_DIRECT_TO_SCREEN]) {
			// set Viewport and Scissor when rendering directly to screen
			viewport_width = storage->_dims.rt_width;
			viewport_height = storage->_dims.rt_height;
			viewport_x = storage->frame.current_rt->x;
			// FTODO
			//			viewport_y = OS::get_singleton()->get_window_size().height - viewport_height - storage->frame.current_rt->y;
			viewport_y = storage->frame.current_rt->y;

			//			viewport_x = 0;
			//			viewport_y = 0;

			glScissor(viewport_x, viewport_y, viewport_width, viewport_height);
			glViewport(viewport_x, viewport_y, viewport_width, viewport_height);
			glEnable(GL_SCISSOR_TEST);
		}
	}

	// FTODO .. this was commented out to try and get the clear color correct
	//#ifdef GODOT3
	// OLD METHOD .. now done by render target rather than frame
#if 0
	if (storage->frame.clear_request) {
		glClearColor(storage->frame.clear_request_color.r,
				storage->frame.clear_request_color.g,
				storage->frame.clear_request_color.b,
				state.using_transparent_rt ? storage->frame.clear_request_color.a : 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		storage->frame.clear_request = false;
	}
#endif

	// NEW METHOD
	if (storage->frame.current_rt && storage->frame.current_rt->clear_requested) {
		const Color &col = storage->frame.current_rt->clear_color;
		glClearColor(col.r, col.g, col.b, col.a);

		// clear EVERYTHING.
		// not clearing everything can be devastating on tiled renderers especially,
		// because if anything is preserved, often the whole frame buffer needs to be preserved.
		// Not sure if GL_ACCUM_BUFFER_BIT is needed or supported in GLES.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		storage->frame.current_rt->clear_requested = false;
	}

	//#endif

	/*
	if (storage->frame.current_rt) {
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);
		glColorMask(1, 1, 1, 1);
	}
	*/

	reset_canvas();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);
	glDisableVertexAttribArray(RS::ARRAY_COLOR);

	// set up default uniforms

	Transform3D canvas_transform;

	if (storage->frame.current_rt) {
		float csy = 1.0;
		// FTODO
		//		if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_VFLIP]) {
		//			csy = -1.0;
		//		}
		canvas_transform.translate(-(storage->frame.current_rt->width / 2.0f), -(storage->frame.current_rt->height / 2.0f), 0.0f);
		canvas_transform.scale(Vector3(2.0f / storage->frame.current_rt->width, csy * -2.0f / storage->frame.current_rt->height, 1.0f));
	} else {
		// FTODO
		//		Vector2 ssize = OS::get_singleton()->get_window_size();
		Vector2 ssize;
		ssize.x = storage->_dims.win_width;
		ssize.y = storage->_dims.win_height;

		canvas_transform.translate(-(ssize.width / 2.0f), -(ssize.height / 2.0f), 0.0f);
		canvas_transform.scale(Vector3(2.0f / ssize.width, -2.0f / ssize.height, 1.0f));
	}

	state.uniforms.projection_matrix = canvas_transform;

	state.uniforms.final_modulate = Color(1, 1, 1, 1);

	state.uniforms.modelview_matrix = Transform2D();
	state.uniforms.extra_matrix = Transform2D();

	_set_uniforms();
	_bind_quad_buffer();

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, state.canvas_item_ubo);
	glBindVertexArray(data.canvas_quad_array);
}

void RasterizerCanvasBaseGLES3::canvas_end() {
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_DIRECT_TO_SCREEN]) {
		//reset viewport to full window size
		//		int viewport_width = OS::get_singleton()->get_window_size().width;
		//		int viewport_height = OS::get_singleton()->get_window_size().height;
		int viewport_width = storage->_dims.win_width;
		int viewport_height = storage->_dims.win_height;
		glViewport(0, 0, viewport_width, viewport_height);
		glScissor(0, 0, viewport_width, viewport_height);
	}

	state.using_texture_rect = false;
	state.using_skeleton = false;
	state.using_ninepatch = false;
	state.using_transparent_rt = false;
}

void RasterizerCanvasBaseGLES3::draw_generic_textured_rect(const Rect2 &p_rect, const Rect2 &p_src) {
	state.canvas_shader.set_uniform(CanvasShaderGLES3::DST_RECT, Color(p_rect.position.x, p_rect.position.y, p_rect.size.x, p_rect.size.y));
	state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(p_src.position.x, p_src.position.y, p_src.size.x, p_src.size.y));
	_bind_quad_buffer();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

void RasterizerCanvasBaseGLES3::_set_texture_rect_mode(bool p_texture_rect, bool p_light_angle, bool p_modulate, bool p_large_vertex) {
	// always set this directly (this could be state checked)
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_TEXTURE_RECT, p_texture_rect);

	if (state.using_light_angle != p_light_angle) {
		state.using_light_angle = p_light_angle;
		state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_ATTRIB_LIGHT_ANGLE, p_light_angle);
	}

	if (state.using_modulate != p_modulate) {
		state.using_modulate = p_modulate;
		state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_ATTRIB_MODULATE, p_modulate);
	}

	if (state.using_large_vertex != p_large_vertex) {
		state.using_large_vertex = p_large_vertex;
		state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_ATTRIB_LARGE_VERTEX, p_large_vertex);
	}
}

RasterizerStorageGLES3::Texture *RasterizerCanvasBaseGLES3::_bind_canvas_texture(const RID &p_texture, const RID &p_normal_map) {
	RasterizerStorageGLES3::Texture *tex_return = NULL;

	if (p_texture.is_valid()) {
		RasterizerStorageGLES3::Texture *texture = storage->texture_owner.get_or_null(p_texture);

		if (!texture) {
			state.current_tex = RID();
			state.current_tex_ptr = NULL;

			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 1);
			glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

		} else {
			if (texture->redraw_if_visible) {
				RenderingServerDefault::redraw_request();
			}

			texture = texture->get_ptr();

			if (texture->render_target) {
				texture->render_target->used_in_frame = true;
			}

			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 1);
			glBindTexture(GL_TEXTURE_2D, texture->tex_id);

			state.current_tex = p_texture;
			state.current_tex_ptr = texture;

			// new for Godot 4. Set the texture min mag filter and repeat per item
			// we use a wrapper to avoid noop GL state changes
			texture->GLSetFilter(GL_TEXTURE_2D, state.current_filter);

			tex_return = texture;
		}
	} else {
		state.current_tex = RID();
		state.current_tex_ptr = NULL;

		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 1);
		glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
	}

	if (p_normal_map == state.current_normal) {
		//do none
		state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, state.current_normal.is_valid());

	} else if (p_normal_map.is_valid()) {
		RasterizerStorageGLES3::Texture *normal_map = storage->texture_owner.get_or_null(p_normal_map);

		if (!normal_map) {
			state.current_normal = RID();
			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 2);
			glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);
			state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, false);

		} else {
			if (normal_map->redraw_if_visible) { //check before proxy, because this is usually used with proxies
				RenderingServerDefault::redraw_request();
			}

			normal_map = normal_map->get_ptr();

			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 2);
			glBindTexture(GL_TEXTURE_2D, normal_map->tex_id);
			state.current_normal = p_normal_map;
			state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, true);
		}

	} else {
		state.current_normal = RID();
		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 2);
		glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, false);
	}

	return tex_return;
}

/*
void RasterizerCanvasBaseGLES3::draw_window_margins(int *black_margin, RID *black_image) {
	return;

	// FTODO
	int window_w = storage->_dims.rt_width;
	int window_h = storage->_dims.rt_height;
	//Vector2 window_size = Vector2(window_w, window_h);

	//	int window_h = window_size.height;
	//	int window_w = window_size.width;

	//	glBindFramebuffer(GL_FRAMEBUFFER, storage->system_fbo);
	//	glViewport(0, 0, window_size.width, window_size.height);

	canvas_begin();

	if (black_image[SIDE_LEFT].is_valid()) {
		_bind_canvas_texture(black_image[SIDE_LEFT], RID());
		Size2 sz(storage->texture_get_width(black_image[SIDE_LEFT]), storage->texture_get_height(black_image[SIDE_LEFT]));
		draw_generic_textured_rect(Rect2(0, 0, black_margin[SIDE_LEFT], window_h),
				Rect2(0, 0, (float)black_margin[SIDE_LEFT] / sz.x, (float)(window_h) / sz.y));
	} else if (black_margin[SIDE_LEFT]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(0, 0, black_margin[SIDE_LEFT], window_h), Rect2(0, 0, 1, 1));
	}

	if (black_image[SIDE_RIGHT].is_valid()) {
		_bind_canvas_texture(black_image[SIDE_RIGHT], RID());
		Size2 sz(storage->texture_get_width(black_image[SIDE_RIGHT]), storage->texture_get_height(black_image[SIDE_RIGHT]));
		draw_generic_textured_rect(Rect2(window_w - black_margin[SIDE_RIGHT], 0, black_margin[SIDE_RIGHT], window_h),
				Rect2(0, 0, (float)black_margin[SIDE_RIGHT] / sz.x, (float)window_h / sz.y));
	} else if (black_margin[SIDE_RIGHT]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(window_w - black_margin[SIDE_RIGHT], 0, black_margin[SIDE_RIGHT], window_h), Rect2(0, 0, 1, 1));
	}

	if (black_image[SIDE_TOP].is_valid()) {
		_bind_canvas_texture(black_image[SIDE_TOP], RID());

		Size2 sz(storage->texture_get_width(black_image[SIDE_TOP]), storage->texture_get_height(black_image[SIDE_TOP]));
		draw_generic_textured_rect(Rect2(0, 0, window_w, black_margin[SIDE_TOP]),
				Rect2(0, 0, (float)window_w / sz.x, (float)black_margin[SIDE_TOP] / sz.y));

	} else if (black_margin[SIDE_TOP]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(0, 0, window_w, black_margin[SIDE_TOP]), Rect2(0, 0, 1, 1));
	}

	if (black_image[SIDE_BOTTOM].is_valid()) {
		_bind_canvas_texture(black_image[SIDE_BOTTOM], RID());

		Size2 sz(storage->texture_get_width(black_image[SIDE_BOTTOM]), storage->texture_get_height(black_image[SIDE_BOTTOM]));
		draw_generic_textured_rect(Rect2(0, window_h - black_margin[SIDE_BOTTOM], window_w, black_margin[SIDE_BOTTOM]),
				Rect2(0, 0, (float)window_w / sz.x, (float)black_margin[SIDE_BOTTOM] / sz.y));

	} else if (black_margin[SIDE_BOTTOM]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(0, window_h - black_margin[SIDE_BOTTOM], window_w, black_margin[SIDE_BOTTOM]), Rect2(0, 0, 1, 1));
	}

	canvas_end();
}
*/

void RasterizerCanvasBaseGLES3::_bind_quad_buffer() {
	glBindVertexArray(data.canvas_quad_array);
}

void RasterizerCanvasBaseGLES3::_set_uniforms() {
	state.canvas_shader.set_uniform(CanvasShaderGLES3::PROJECTION_MATRIX, state.uniforms.projection_matrix);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.uniforms.extra_matrix);

	state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.uniforms.final_modulate);

	state.canvas_shader.set_uniform(CanvasShaderGLES3::TIME, storage->frame.time[0]);

	if (storage->frame.current_rt) {
		Vector2 screen_pixel_size;
		screen_pixel_size.x = 1.0 / storage->frame.current_rt->width;
		screen_pixel_size.y = 1.0 / storage->frame.current_rt->height;

		state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, screen_pixel_size);
	}

	if (state.using_skeleton) {
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SKELETON_TRANSFORM, state.skeleton_transform);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SKELETON_TRANSFORM_INVERSE, state.skeleton_transform_inverse);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SKELETON_TEXTURE_SIZE, state.skeleton_texture_size);
	}

	if (state.using_light) {
		Light *light = state.using_light;
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_MATRIX, light->light_shader_xform);
		Transform2D basis_inverse = light->light_shader_xform.affine_inverse().orthonormalized();
		basis_inverse.elements[2] = Vector2();
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_MATRIX_INVERSE, basis_inverse);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_LOCAL_MATRIX, light->xform_cache.affine_inverse());
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_COLOR, light->color * light->energy);
		//		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_POS, light->light_shader_pos);
		// FTODO
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_POS, light->light_shader_xform.elements[2]);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_HEIGHT, light->height);

		// FTODO
		//state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_OUTSIDE_ALPHA, light->mode == RS::CANVAS_LIGHT_MODE_MASK ? 1.0 : 0.0);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_OUTSIDE_ALPHA, 0.0f);

		if (state.using_shadow) {
			// FTODO
#if 0
			RasterizerStorageGLES3::CanvasLightShadow *cls = storage->canvas_light_shadow_owner.get(light->shadow_buffer);
			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 5);
			glBindTexture(GL_TEXTURE_2D, cls->distance);
			state.canvas_shader.set_uniform(CanvasShaderGLES3::SHADOW_MATRIX, light->shadow_matrix_cache);
			state.canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_SHADOW_COLOR, light->shadow_color);

			state.canvas_shader.set_uniform(CanvasShaderGLES3::SHADOWPIXEL_SIZE, (1.0 / light->shadow_buffer_size) * (1.0 + light->shadow_smooth));
			if (light->radius_cache == 0) {
				state.canvas_shader.set_uniform(CanvasShaderGLES3::SHADOW_GRADIENT, 0.0);
			} else {
				state.canvas_shader.set_uniform(CanvasShaderGLES3::SHADOW_GRADIENT, light->shadow_gradient_length / (light->radius_cache * 1.1));
			}
			state.canvas_shader.set_uniform(CanvasShaderGLES3::SHADOW_DISTANCE_MULT, light->radius_cache * 1.1);
#endif
		}
	}
}

void RasterizerCanvasBaseGLES3::reset_canvas() {
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glEnable(GL_BLEND);

	if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	// bind the back buffer to a texture so shaders can use it.
	// It should probably use texture unit -3 (as OpenGL does as well) but currently that's buggy.
	// keeping this for now as there's nothing else that uses texture unit 2
	// TODO ^
	if (storage->frame.current_rt) {
		// glActiveTexture(GL_TEXTURE0 + 2);
		// glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->copy_screen_effect.color);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasBaseGLES3::canvas_debug_viewport_shadows(Light *p_lights_with_shadow) {
}

void RasterizerCanvasBaseGLES3::_copy_texscreen(const Rect2 &p_rect) {
	state.canvas_texscreen_used = true;

	_copy_screen(p_rect);

	// back to canvas, force rebind
	state.using_texture_rect = false;
	state.canvas_shader.bind();
	_bind_canvas_texture(state.current_tex, state.current_normal);
	_set_uniforms();
}

void RasterizerCanvasBaseGLES3::_draw_polygon(const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor, const float *p_weights, const int *p_bones) {
	glBindVertexArray(data.polygon_buffer_pointer_array);
	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	uint32_t buffer_ofs = 0;
	uint32_t buffer_ofs_after = buffer_ofs + (sizeof(Vector2) * p_vertex_count);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(buffer_ofs_after > data.polygon_buffer_size);
#endif

	storage->buffer_orphan_and_upload(data.polygon_buffer_size, 0, sizeof(Vector2) * p_vertex_count, p_vertices, GL_ARRAY_BUFFER, _buffer_upload_usage_flag, true);

	glEnableVertexAttribArray(RS::ARRAY_VERTEX);
	glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), NULL);
	buffer_ofs = buffer_ofs_after;

	if (p_singlecolor) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		Color m = *p_colors;
		glVertexAttrib4f(RS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
	} else if (!p_colors) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);
	} else {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(Color) * p_vertex_count, p_colors, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttribPointer(RS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Color), CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;
	}

	if (p_uvs) {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_uvs, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
		glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;
	} else {
		glDisableVertexAttribArray(RS::ARRAY_TEX_UV);
	}

	if (p_weights && p_bones) {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(float) * 4 * p_vertex_count, p_weights, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_WEIGHTS);
		glVertexAttribPointer(RS::ARRAY_WEIGHTS, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;

		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(int) * 4 * p_vertex_count, p_bones, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_BONES);
		glVertexAttribPointer(RS::ARRAY_BONES, 4, GL_UNSIGNED_INT, GL_FALSE, sizeof(int) * 4, CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;

	} else {
		glDisableVertexAttribArray(RS::ARRAY_WEIGHTS);
		glDisableVertexAttribArray(RS::ARRAY_BONES);
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);

	if (storage->config.support_32_bits_indices) { //should check for
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND((sizeof(int) * p_index_count) > data.polygon_index_buffer_size);
#endif
		storage->buffer_orphan_and_upload(data.polygon_index_buffer_size, 0, sizeof(int) * p_index_count, p_indices, GL_ELEMENT_ARRAY_BUFFER, _buffer_upload_usage_flag, true);

		glDrawElements(GL_TRIANGLES, p_index_count, GL_UNSIGNED_INT, 0);
		storage->info.render._2d_draw_call_count++;
	} else {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND((sizeof(uint16_t) * p_index_count) > data.polygon_index_buffer_size);
#endif
		uint16_t *index16 = (uint16_t *)alloca(sizeof(uint16_t) * p_index_count);
		for (int i = 0; i < p_index_count; i++) {
			index16[i] = uint16_t(p_indices[i]);
		}
		storage->buffer_orphan_and_upload(data.polygon_index_buffer_size, 0, sizeof(uint16_t) * p_index_count, index16, GL_ELEMENT_ARRAY_BUFFER, _buffer_upload_usage_flag, true);
		glDrawElements(GL_TRIANGLES, p_index_count, GL_UNSIGNED_SHORT, 0);
		storage->info.render._2d_draw_call_count++;
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasBaseGLES3::_draw_generic(GLuint p_primitive, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {
	glBindVertexArray(data.polygon_buffer_pointer_array);
	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	uint32_t buffer_ofs = 0;
	uint32_t buffer_ofs_after = buffer_ofs + (sizeof(Vector2) * p_vertex_count);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(buffer_ofs_after > data.polygon_buffer_size);
#endif
	storage->buffer_orphan_and_upload(data.polygon_buffer_size, 0, sizeof(Vector2) * p_vertex_count, p_vertices, GL_ARRAY_BUFFER, _buffer_upload_usage_flag, true);

	glEnableVertexAttribArray(RS::ARRAY_VERTEX);
	glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), NULL);
	buffer_ofs = buffer_ofs_after;

	if (p_singlecolor) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		Color m = *p_colors;
		glVertexAttrib4f(RS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
	} else if (!p_colors) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);
	} else {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(Color) * p_vertex_count, p_colors, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttribPointer(RS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Color), CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;
	}

	if (p_uvs) {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_uvs, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
		glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;
	} else {
		glDisableVertexAttribArray(RS::ARRAY_TEX_UV);
	}

	glDrawArrays(p_primitive, 0, p_vertex_count);
	storage->info.render._2d_draw_call_count++;

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RasterizerCanvasBaseGLES3::_draw_generic_indices(GLuint p_primitive, const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {
	glBindVertexArray(data.polygon_buffer_pointer_array);
	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	uint32_t buffer_ofs = 0;
	uint32_t buffer_ofs_after = buffer_ofs + (sizeof(Vector2) * p_vertex_count);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(buffer_ofs_after > data.polygon_buffer_size);
#endif
	storage->buffer_orphan_and_upload(data.polygon_buffer_size, 0, sizeof(Vector2) * p_vertex_count, p_vertices, GL_ARRAY_BUFFER, _buffer_upload_usage_flag, true);

	glEnableVertexAttribArray(RS::ARRAY_VERTEX);
	glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), NULL);
	buffer_ofs = buffer_ofs_after;

	if (p_singlecolor) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		Color m = *p_colors;
		glVertexAttrib4f(RS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
	} else if (!p_colors) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);
	} else {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(Color) * p_vertex_count, p_colors, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttribPointer(RS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Color), CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;
	}

	if (p_uvs) {
		RAST_FAIL_COND(!storage->safe_buffer_sub_data(data.polygon_buffer_size, GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_uvs, buffer_ofs_after));
		glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
		glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), CAST_INT_TO_UCHAR_PTR(buffer_ofs));
		buffer_ofs = buffer_ofs_after;
	} else {
		glDisableVertexAttribArray(RS::ARRAY_TEX_UV);
	}

#ifdef RASTERIZER_EXTRA_CHECKS
	// very slow, do not enable in normal use
	for (int n = 0; n < p_index_count; n++) {
		RAST_DEV_DEBUG_ASSERT(p_indices[n] < p_vertex_count);
	}
#endif

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);

	if (storage->config.support_32_bits_indices) { //should check for
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND((sizeof(int) * p_index_count) > data.polygon_index_buffer_size);
#endif
		storage->buffer_orphan_and_upload(data.polygon_index_buffer_size, 0, sizeof(int) * p_index_count, p_indices, GL_ELEMENT_ARRAY_BUFFER, _buffer_upload_usage_flag, true);
		glDrawElements(p_primitive, p_index_count, GL_UNSIGNED_INT, 0);
		storage->info.render._2d_draw_call_count++;
	} else {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND((sizeof(uint16_t) * p_index_count) > data.polygon_index_buffer_size);
#endif
		uint16_t *index16 = (uint16_t *)alloca(sizeof(uint16_t) * p_index_count);
		for (int i = 0; i < p_index_count; i++) {
			index16[i] = uint16_t(p_indices[i]);
		}
		storage->buffer_orphan_and_upload(data.polygon_index_buffer_size, 0, sizeof(uint16_t) * p_index_count, index16, GL_ELEMENT_ARRAY_BUFFER, _buffer_upload_usage_flag, true);
		glDrawElements(p_primitive, p_index_count, GL_UNSIGNED_SHORT, 0);
		storage->info.render._2d_draw_call_count++;
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasBaseGLES3::_legacy_draw_poly_triangles(Item::CommandPolygon *p_poly, RasterizerStorageGLES3::Material *p_material) {
	//	return;

	const PolyData &pd = _polydata[p_poly->polygon.polygon_id];

	_set_texture_rect_mode(false);

	if (state.canvas_shader.bind()) {
		_set_uniforms();
		state.canvas_shader.use_material((void *)p_material);
	}

	// FTODO
	//RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(polygon->texture, polygon->normal_map);
	RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(p_poly->texture, RID());

	if (texture) {
		Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
	}

	_draw_polygon(pd.indices.ptr(), pd.indices.size(), pd.points.size(), pd.points.ptr(), pd.uvs.ptr(), pd.colors.ptr(), pd.colors.size() == 1, nullptr, nullptr);

//	 _draw_polygon(polygon->indices.ptr(), polygon->count, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1, polygon->weights.ptr(), polygon->bones.ptr());
#ifdef GLES_OVER_GL
#if 0
	if (polygon->antialiased) {
		glEnable(GL_LINE_SMOOTH);
		if (polygon->antialiasing_use_indices) {
			_draw_generic_indices(GL_LINE_STRIP, polygon->indices.ptr(), polygon->count, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1);
		} else {
			_draw_generic(GL_LINE_LOOP, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1);
		}
		glDisable(GL_LINE_SMOOTH);
	}
#endif
#endif
}

void RasterizerCanvasBaseGLES3::_legacy_draw_primitive(Item::CommandPrimitive *p_pr, RasterizerStorageGLES3::Material *p_material) {
	//	return;

	if (p_pr->point_count != 4)
		return; // not sure if supported

	_set_texture_rect_mode(false);

	if (state.canvas_shader.bind()) {
		_set_uniforms();
		state.canvas_shader.use_material((void *)p_material);
	}

	_bind_canvas_texture(RID(), RID());

	glDisableVertexAttribArray(RS::ARRAY_COLOR);
	glVertexAttrib4fv(RS::ARRAY_COLOR, p_pr->colors[0].components);

	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);

	_draw_gui_primitive(p_pr->point_count, p_pr->points, NULL, NULL);
}

void RasterizerCanvasBaseGLES3::_legacy_draw_line(Item::CommandPrimitive *p_pr, RasterizerStorageGLES3::Material *p_material) {
	_set_texture_rect_mode(false);

	if (state.canvas_shader.bind()) {
		_set_uniforms();
		state.canvas_shader.use_material((void *)p_material);
	}

	_bind_canvas_texture(RID(), RID());

	glDisableVertexAttribArray(RS::ARRAY_COLOR);
	glVertexAttrib4fv(RS::ARRAY_COLOR, p_pr->colors[0].components);

	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);

#ifdef GLES_OVER_GL
//		if (line->antialiased)
//			glEnable(GL_LINE_SMOOTH);
#endif
	_draw_gui_primitive(2, p_pr->points, NULL, NULL);

#ifdef GLES_OVER_GL
//		if (line->antialiased)
//			glDisable(GL_LINE_SMOOTH);
#endif
}

void RasterizerCanvasBaseGLES3::_draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs, const float *p_light_angles) {
	static const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

	int version = 0;
	int color_offset = 0;
	int uv_offset = 0;
	int light_angle_offset = 0;
	int stride = 2;

	if (p_colors) {
		version |= 1;
		color_offset = stride;
		stride += 4;
	}

	if (p_uvs) {
		version |= 2;
		uv_offset = stride;
		stride += 2;
	}

	if (p_light_angles) { //light_angles
		version |= 4;
		light_angle_offset = stride;
		stride += 1;
	}

	RAST_DEV_DEBUG_ASSERT(p_points <= 4);
	float buffer_data[(2 + 2 + 4 + 1) * 4];

	for (int i = 0; i < p_points; i++) {
		buffer_data[stride * i + 0] = p_vertices[i].x;
		buffer_data[stride * i + 1] = p_vertices[i].y;
	}

	if (p_colors) {
		for (int i = 0; i < p_points; i++) {
			buffer_data[stride * i + color_offset + 0] = p_colors[i].r;
			buffer_data[stride * i + color_offset + 1] = p_colors[i].g;
			buffer_data[stride * i + color_offset + 2] = p_colors[i].b;
			buffer_data[stride * i + color_offset + 3] = p_colors[i].a;
		}
	}

	if (p_uvs) {
		for (int i = 0; i < p_points; i++) {
			buffer_data[stride * i + uv_offset + 0] = p_uvs[i].x;
			buffer_data[stride * i + uv_offset + 1] = p_uvs[i].y;
		}
	}

	if (p_light_angles) {
		for (int i = 0; i < p_points; i++) {
			buffer_data[stride * i + light_angle_offset + 0] = p_light_angles[i];
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
	storage->buffer_orphan_and_upload(data.polygon_buffer_size, 0, p_points * stride * 4 * sizeof(float), buffer_data, GL_ARRAY_BUFFER, _buffer_upload_usage_flag, true);

	glBindVertexArray(data.polygon_buffer_quad_arrays[version]);

	glDrawArrays(prim[p_points], 0, p_points);
	storage->info.render._2d_draw_call_count++;
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RasterizerCanvasBaseGLES3::_copy_screen(const Rect2 &p_rect) {
	if (storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_DIRECT_TO_SCREEN]) {
		ERR_PRINT_ONCE("Cannot use screen texture copying in render target set to render direct to screen.");
		return;
	}

	ERR_FAIL_COND_MSG(storage->frame.current_rt->copy_screen_effect.color == 0, "Can't use screen texture copying in a render target configured without copy buffers.");

	glDisable(GL_BLEND);

	Vector2 wh(storage->frame.current_rt->width, storage->frame.current_rt->height);

	Color copy_section(p_rect.position.x / wh.x, p_rect.position.y / wh.y, p_rect.size.x / wh.x, p_rect.size.y / wh.y);

	if (p_rect != Rect2()) {
		storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_COPY_SECTION, true);
	}

	storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_NO_ALPHA, !state.using_transparent_rt);

	storage->bind_framebuffer(storage->frame.current_rt->copy_screen_effect.fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->color);

	storage->shaders.copy.bind();
	storage->shaders.copy.set_uniform(CopyShaderGLES3::COPY_SECTION, copy_section);

	const Vector2 vertpos[4] = {
		Vector2(-1, -1),
		Vector2(-1, 1),
		Vector2(1, 1),
		Vector2(1, -1),
	};

	const Vector2 uvpos[4] = {
		Vector2(0, 0),
		Vector2(0, 1),
		Vector2(1, 1),
		Vector2(1, 0)
	};

	const int indexpos[6] = {
		0, 1, 2,
		2, 3, 0
	};

	_draw_polygon(indexpos, 6, 4, vertpos, uvpos, NULL, false);

	storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_COPY_SECTION, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_NO_ALPHA, false);

	storage->bind_framebuffer(storage->frame.current_rt->fbo);
	glEnable(GL_BLEND);
}

void RasterizerCanvasBaseGLES3::canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {
#if 0
	RasterizerStorageGLES3::CanvasLightShadow *cls = storage->canvas_light_shadow_owner.get(p_buffer);
	ERR_FAIL_COND(!cls);

	glDisable(GL_BLEND);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glDisable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(true);

	glBindFramebuffer(GL_FRAMEBUFFER, cls->fbo);

	state.canvas_shadow_shader.set_conditional(CanvasShadowShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);
	state.canvas_shadow_shader.bind();

	glViewport(0, 0, cls->size, cls->height);
	glClearDepth(1.0f);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	RS::CanvasOccluderPolygonCullMode cull = RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;

	for (int i = 0; i < 4; i++) {
		//make sure it remains orthogonal, makes easy to read angle later

		Transform3D light;
		light.origin[0] = p_light_xform[2][0];
		light.origin[1] = p_light_xform[2][1];
		light.basis[0][0] = p_light_xform[0][0];
		light.basis[0][1] = p_light_xform[1][0];
		light.basis[1][0] = p_light_xform[0][1];
		light.basis[1][1] = p_light_xform[1][1];

		//light.basis.scale(Vector3(to_light.elements[0].length(),to_light.elements[1].length(),1));

		//p_near=1;
		CameraMatrix projection;
		{
			real_t fov = 90;
			real_t nearp = p_near;
			real_t farp = p_far;
			real_t aspect = 1.0;

			real_t ymax = nearp * Math::tan(Math::deg2rad(fov * 0.5));
			real_t ymin = -ymax;
			real_t xmin = ymin * aspect;
			real_t xmax = ymax * aspect;

			projection.set_frustum(xmin, xmax, ymin, ymax, nearp, farp);
		}

		Vector3 cam_target = Basis(Vector3(0, 0, Math_PI * 2 * (i / 4.0))).xform(Vector3(0, 1, 0));
		projection = projection * CameraMatrix(Transform3D().looking_at(cam_target, Vector3(0, 0, -1)).affine_inverse());

		state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::PROJECTION_MATRIX, projection);
		state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::LIGHT_MATRIX, light);
		state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::DISTANCE_NORM, 1.0 / p_far);

		if (i == 0)
			*p_xform_cache = projection;

		glViewport(0, (cls->height / 4) * i, cls->size, cls->height / 4);

		LightOccluderInstance *instance = p_occluders;

		while (instance) {
			RasterizerStorageGLES3::CanvasOccluder *cc = storage->canvas_occluder_owner.get_or_null(instance->polygon_buffer);
			if (!cc || cc->len == 0 || !(p_light_mask & instance->light_mask)) {
				instance = instance->next;
				continue;
			}

			state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::WORLD_MATRIX, instance->xform_cache);

			RS::CanvasOccluderPolygonCullMode transformed_cull_cache = instance->cull_cache;

			if (transformed_cull_cache != RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED &&
					(p_light_xform.basis_determinant() * instance->xform_cache.basis_determinant()) < 0) {
				transformed_cull_cache =
						transformed_cull_cache == RS::CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE ?
								RS::CANVAS_OCCLUDER_POLYGON_CULL_COUNTER_CLOCKWISE :
								RS::CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE;
			}

			if (cull != transformed_cull_cache) {
				cull = transformed_cull_cache;
				switch (cull) {
					case RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED: {
						glDisable(GL_CULL_FACE);

					} break;
					case RS::CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE: {
						glEnable(GL_CULL_FACE);
						glCullFace(GL_FRONT);
					} break;
					case RS::CANVAS_OCCLUDER_POLYGON_CULL_COUNTER_CLOCKWISE: {
						glEnable(GL_CULL_FACE);
						glCullFace(GL_BACK);

					} break;
				}
			}

			glBindBuffer(GL_ARRAY_BUFFER, cc->vertex_id);
			glEnableVertexAttribArray(RS::ARRAY_VERTEX);
			glVertexAttribPointer(RS::ARRAY_VERTEX, 3, GL_FLOAT, false, 0, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cc->index_id);

			glDrawElements(GL_TRIANGLES, cc->len * 3, GL_UNSIGNED_SHORT, 0);

			instance = instance->next;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
}

void RasterizerCanvasBaseGLES3::draw_lens_distortion_rect(const Rect2 &p_rect, float p_k1, float p_k2, const Vector2 &p_eye_center, float p_oversample) {
	Vector2 half_size;
	if (storage->frame.current_rt) {
		half_size = Vector2(storage->frame.current_rt->width, storage->frame.current_rt->height);
	} else {
		//		half_size = OS::get_singleton()->get_window_size();
		half_size = Vector2(storage->_dims.win_width, storage->_dims.win_height);
	}
	half_size *= 0.5;
	Vector2 offset((p_rect.position.x - half_size.x) / half_size.x, (p_rect.position.y - half_size.y) / half_size.y);
	Vector2 scale(p_rect.size.x / half_size.x, p_rect.size.y / half_size.y);

	float aspect_ratio = p_rect.size.x / p_rect.size.y;

	// setup our lens shader
	state.lens_shader.bind();
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::OFFSET, offset);
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::SCALE, scale);
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::K1, p_k1);
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::K2, p_k2);
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::EYE_CENTER, p_eye_center);
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::UPSCALE, p_oversample);
	state.lens_shader.set_uniform(LensDistortedShaderGLES3::ASPECT_RATIO, aspect_ratio);

	// bind our quad buffer
	_bind_quad_buffer();

	// and draw
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	// and cleanup
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RasterizerCanvasBaseGLES3::initialize() {
	bool flag_stream = false;
	//flag_stream = GLOBAL_GET("rendering/options/api_usage_legacy/flag_stream");
	if (flag_stream)
		_buffer_upload_usage_flag = GL_STREAM_DRAW;
	else
		_buffer_upload_usage_flag = GL_DYNAMIC_DRAW;

	// quad buffer
	{
		glGenBuffers(1, &data.canvas_quad_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, data.canvas_quad_vertices);

		const float qv[8] = {
			0, 0,
			0, 1,
			1, 1,
			1, 0
		};

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8, qv, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenVertexArrays(1, &data.canvas_quad_array);
		glBindVertexArray(data.canvas_quad_array);
		glBindBuffer(GL_ARRAY_BUFFER, data.canvas_quad_vertices);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);
		glEnableVertexAttribArray(0);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}

	{
		//particle quad buffers

		glGenBuffers(1, &data.particle_quad_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, data.particle_quad_vertices);
		{
			//quad of size 1, with pivot on the center for particles, then regular UVS. Color is general plus fetched from particle
			const float qv[16] = {
				-0.5, -0.5,
				0.0, 0.0,
				-0.5, 0.5,
				0.0, 1.0,
				0.5, 0.5,
				1.0, 1.0,
				0.5, -0.5,
				1.0, 0.0
			};

			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, qv, GL_STATIC_DRAW);
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		glGenVertexArrays(1, &data.particle_quad_array);
		glBindVertexArray(data.particle_quad_array);
		glBindBuffer(GL_ARRAY_BUFFER, data.particle_quad_vertices);
		glEnableVertexAttribArray(RS::ARRAY_VERTEX);
		glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, nullptr);
		glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
		glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, CAST_INT_TO_UCHAR_PTR(8));
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}

	// polygon buffer
	{
		uint32_t poly_size = 128; //GLOBAL_DEF_RST("rendering/limits/buffers/canvas_polygon_buffer_size_kb", 128);
		ProjectSettings::get_singleton()->set_custom_property_info("rendering/limits/buffers/canvas_polygon_buffer_size_kb", PropertyInfo(Variant::INT, "rendering/limits/buffers/canvas_polygon_buffer_size_kb", PROPERTY_HINT_RANGE, "0,256,1,or_greater"));
		poly_size = MAX(poly_size, 2); // minimum 2k, may still see anomalies in editor
		poly_size *= 1024; //kb
		glGenBuffers(1, &data.polygon_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
		glBufferData(GL_ARRAY_BUFFER, poly_size, nullptr, GL_DYNAMIC_DRAW); //allocate max size
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		data.polygon_buffer_size = poly_size;

		//quad arrays
		for (int i = 0; i < Data::NUM_QUAD_ARRAY_VARIATIONS; i++) {
			glGenVertexArrays(1, &data.polygon_buffer_quad_arrays[i]);
			glBindVertexArray(data.polygon_buffer_quad_arrays[i]);
			glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

			int uv_ofs = 0;
			int color_ofs = 0;
			int light_angle_ofs = 0;
			int stride = 2 * 4;

			if (i & 1) { //color
				color_ofs = stride;
				stride += 4 * 4;
			}

			if (i & 2) { //uv
				uv_ofs = stride;
				stride += 2 * 4;
			}

			if (i & 4) { //light_angle
				light_angle_ofs = stride;
				stride += 1 * 4;
			}

			glEnableVertexAttribArray(RS::ARRAY_VERTEX);
			glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, stride, nullptr);

			if (i & 1) {
				glEnableVertexAttribArray(RS::ARRAY_COLOR);
				glVertexAttribPointer(RS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(color_ofs));
			}

			if (i & 2) {
				glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
				glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(uv_ofs));
			}

			if (i & 4) {
				// reusing tangent for light_angle
				glEnableVertexAttribArray(RS::ARRAY_TANGENT);
				glVertexAttribPointer(RS::ARRAY_TANGENT, 1, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(light_angle_ofs));
			}

			glBindVertexArray(0);
		}

		glGenVertexArrays(1, &data.polygon_buffer_pointer_array);

		uint32_t index_size = 128; //GLOBAL_DEF_RST("rendering/limits/buffers/canvas_polygon_index_buffer_size_kb", 128);
		ProjectSettings::get_singleton()->set_custom_property_info("rendering/limits/buffers/canvas_polygon_index_buffer_size_kb", PropertyInfo(Variant::INT, "rendering/limits/buffers/canvas_polygon_index_buffer_size_kb", PROPERTY_HINT_RANGE, "0,256,1,or_greater"));
		index_size = MAX(index_size, 2);
		index_size *= 1024; //kb
		glGenBuffers(1, &data.polygon_index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, nullptr, GL_DYNAMIC_DRAW); //allocate max size
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		data.polygon_index_buffer_size = index_size;
	}

	// ninepatch buffers
	{
		// array buffer
		glGenBuffers(1, &data.ninepatch_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, data.ninepatch_vertices);

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (16 + 16) * 2, NULL, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// element buffer
		glGenBuffers(1, &data.ninepatch_elements);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.ninepatch_elements);

#define _EIDX(y, x) (y * 4 + x)
		uint8_t elems[3 * 2 * 9] = {
			// first row

			_EIDX(0, 0), _EIDX(0, 1), _EIDX(1, 1),
			_EIDX(1, 1), _EIDX(1, 0), _EIDX(0, 0),

			_EIDX(0, 1), _EIDX(0, 2), _EIDX(1, 2),
			_EIDX(1, 2), _EIDX(1, 1), _EIDX(0, 1),

			_EIDX(0, 2), _EIDX(0, 3), _EIDX(1, 3),
			_EIDX(1, 3), _EIDX(1, 2), _EIDX(0, 2),

			// second row

			_EIDX(1, 0), _EIDX(1, 1), _EIDX(2, 1),
			_EIDX(2, 1), _EIDX(2, 0), _EIDX(1, 0),

			// the center one would be here, but we'll put it at the end
			// so it's easier to disable the center and be able to use
			// one draw call for both

			_EIDX(1, 2), _EIDX(1, 3), _EIDX(2, 3),
			_EIDX(2, 3), _EIDX(2, 2), _EIDX(1, 2),

			// third row

			_EIDX(2, 0), _EIDX(2, 1), _EIDX(3, 1),
			_EIDX(3, 1), _EIDX(3, 0), _EIDX(2, 0),

			_EIDX(2, 1), _EIDX(2, 2), _EIDX(3, 2),
			_EIDX(3, 2), _EIDX(3, 1), _EIDX(2, 1),

			_EIDX(2, 2), _EIDX(2, 3), _EIDX(3, 3),
			_EIDX(3, 3), _EIDX(3, 2), _EIDX(2, 2),

			// center field

			_EIDX(1, 1), _EIDX(1, 2), _EIDX(2, 2),
			_EIDX(2, 2), _EIDX(2, 1), _EIDX(1, 1)
		};
#undef _EIDX

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elems), elems, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	store_transform3d(Transform3D(), state.canvas_item_ubo_data.projection_matrix);

	glGenBuffers(1, &state.canvas_item_ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_item_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(CanvasItemUBO), &state.canvas_item_ubo_data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.canvas_shadow_shader.init();
	state.canvas_shader.init();
	_set_texture_rect_mode(true);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);

	state.canvas_shader.bind();

	state.lens_shader.init();

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_PIXEL_SNAP, GLOBAL_DEF("rendering/quality/2d/use_pixel_snap", false));

	state.using_light = NULL;
	state.using_transparent_rt = false;
	state.using_skeleton = false;
}

RendererCanvasRender::PolygonID RasterizerCanvasBaseGLES3::request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, const Vector<int> &p_bones, const Vector<float> &p_weights) {
	uint32_t id = _polydata.alloc();
	PolyData &pd = _polydata[id];
	pd.indices = p_indices;
	pd.points = p_points;
	pd.colors = p_colors;
	pd.uvs = p_uvs;
	return id;
}
void RasterizerCanvasBaseGLES3::free_polygon(PolygonID p_polygon) {
	_polydata.free(p_polygon);
}

void RasterizerCanvasBaseGLES3::finalize() {
	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);

	glDeleteVertexArrays(1, &data.polygon_buffer_pointer_array);
}

RasterizerCanvasBaseGLES3::RasterizerCanvasBaseGLES3() {
}

#endif // GLES3_BACKEND_ENABLED

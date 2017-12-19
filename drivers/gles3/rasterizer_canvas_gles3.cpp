/*************************************************************************/
/*  rasterizer_canvas_gles3.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "rasterizer_canvas_gles3.h"
#include "os/os.h"
#include "project_settings.h"
#include "rasterizer_scene_gles3.h"
#include "servers/visual/visual_server_raster.h"
#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif

static _FORCE_INLINE_ void store_transform2d(const Transform2D &p_mtx, float *p_array) {

	p_array[0] = p_mtx.elements[0][0];
	p_array[1] = p_mtx.elements[0][1];
	p_array[2] = 0;
	p_array[3] = 0;
	p_array[4] = p_mtx.elements[1][0];
	p_array[5] = p_mtx.elements[1][1];
	p_array[6] = 0;
	p_array[7] = 0;
	p_array[8] = 0;
	p_array[9] = 0;
	p_array[10] = 1;
	p_array[11] = 0;
	p_array[12] = p_mtx.elements[2][0];
	p_array[13] = p_mtx.elements[2][1];
	p_array[14] = 0;
	p_array[15] = 1;
}

static _FORCE_INLINE_ void store_transform(const Transform &p_mtx, float *p_array) {
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

static _FORCE_INLINE_ void store_camera(const CameraMatrix &p_mtx, float *p_array) {

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {

			p_array[i * 4 + j] = p_mtx.matrix[i][j];
		}
	}
}

RID RasterizerCanvasGLES3::light_internal_create() {

	LightInternal *li = memnew(LightInternal);

	glGenBuffers(1, &li->ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, li->ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(LightInternal::UBOData), &state.canvas_item_ubo_data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	return light_internal_owner.make_rid(li);
}

void RasterizerCanvasGLES3::light_internal_update(RID p_rid, Light *p_light) {

	LightInternal *li = light_internal_owner.getornull(p_rid);
	ERR_FAIL_COND(!li);

	store_transform2d(p_light->light_shader_xform, li->ubo_data.light_matrix);
	store_transform2d(p_light->xform_cache.affine_inverse(), li->ubo_data.local_matrix);
	store_camera(p_light->shadow_matrix_cache, li->ubo_data.shadow_matrix);

	for (int i = 0; i < 4; i++) {

		li->ubo_data.color[i] = p_light->color[i] * p_light->energy;
		li->ubo_data.shadow_color[i] = p_light->shadow_color[i];
	}

	li->ubo_data.light_pos[0] = p_light->light_shader_pos.x;
	li->ubo_data.light_pos[1] = p_light->light_shader_pos.y;
	li->ubo_data.shadowpixel_size = (1.0 / p_light->shadow_buffer_size) * (1.0 + p_light->shadow_smooth);
	li->ubo_data.light_outside_alpha = p_light->mode == VS::CANVAS_LIGHT_MODE_MASK ? 1.0 : 0.0;
	li->ubo_data.light_height = p_light->height;
	if (p_light->radius_cache == 0)
		li->ubo_data.shadow_gradient = 0;
	else
		li->ubo_data.shadow_gradient = p_light->shadow_gradient_length / (p_light->radius_cache * 1.1);

	li->ubo_data.shadow_distance_mult = (p_light->radius_cache * 1.1);

	glBindBuffer(GL_UNIFORM_BUFFER, li->ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightInternal::UBOData), &li->ubo_data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void RasterizerCanvasGLES3::light_internal_free(RID p_rid) {

	LightInternal *li = light_internal_owner.getornull(p_rid);
	ERR_FAIL_COND(!li);

	glDeleteBuffers(1, &li->ubo);
	light_internal_owner.free(p_rid);
	memdelete(li);
}

void RasterizerCanvasGLES3::canvas_begin() {

	if (storage->frame.current_rt && storage->frame.clear_request) {
		// a clear request may be pending, so do it

		glClearColor(storage->frame.clear_request_color.r, storage->frame.clear_request_color.g, storage->frame.clear_request_color.b, storage->frame.clear_request_color.a);
		glClear(GL_COLOR_BUFFER_BIT);
		storage->frame.clear_request = false;
	}

	reset_canvas();

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_TEXTURE_RECT, true);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_LIGHTING, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SHADOWS, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_NEAREST, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF5, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF13, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_DISTANCE_FIELD, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_NINEPATCH, false);

	state.canvas_shader.set_custom_shader(0);
	state.canvas_shader.bind();
	state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, Color(1, 1, 1, 1));
	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, Transform2D());
	state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, Transform2D());
	if (storage->frame.current_rt) {
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, Vector2(1.0 / storage->frame.current_rt->width, 1.0 / storage->frame.current_rt->height));
	} else {
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, Vector2(1.0, 1.0));
	}

	//state.canvas_shader.set_uniform(CanvasShaderGLES3::PROJECTION_MATRIX,state.vp);
	//state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX,Transform());
	//state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX,Transform());

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, state.canvas_item_ubo);
	glBindVertexArray(data.canvas_quad_array);
	state.using_texture_rect = true;
	state.using_ninepatch = false;
}

void RasterizerCanvasGLES3::canvas_end() {

	glBindVertexArray(0);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);

	state.using_texture_rect = false;
	state.using_ninepatch = false;
}

RasterizerStorageGLES3::Texture *RasterizerCanvasGLES3::_bind_canvas_texture(const RID &p_texture, const RID &p_normal_map) {

	RasterizerStorageGLES3::Texture *tex_return = NULL;

	if (p_texture == state.current_tex) {
		tex_return = state.current_tex_ptr;
	} else if (p_texture.is_valid()) {

		RasterizerStorageGLES3::Texture *texture = storage->texture_owner.getornull(p_texture);

		if (!texture) {
			state.current_tex = RID();
			state.current_tex_ptr = NULL;
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

		} else {

			texture = texture->get_ptr();

			if (texture->render_target)
				texture->render_target->used_in_frame = true;

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture->tex_id);
			state.current_tex = p_texture;
			state.current_tex_ptr = texture;

			tex_return = texture;
		}

	} else {

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
		state.current_tex = RID();
		state.current_tex_ptr = NULL;
	}

	if (p_normal_map == state.current_normal) {
		//do none
		state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, state.current_normal.is_valid());

	} else if (p_normal_map.is_valid()) {

		RasterizerStorageGLES3::Texture *normal_map = storage->texture_owner.getornull(p_normal_map);

		if (!normal_map) {
			state.current_normal = RID();
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);
			state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, false);

		} else {

			normal_map = normal_map->get_ptr();
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, normal_map->tex_id);
			state.current_normal = p_normal_map;
			state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, true);
		}

	} else {

		state.current_normal = RID();
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::USE_DEFAULT_NORMAL, false);
	}

	return tex_return;
}

void RasterizerCanvasGLES3::_set_texture_rect_mode(bool p_enable, bool p_ninepatch) {

	if (state.using_texture_rect == p_enable && state.using_ninepatch == p_ninepatch)
		return;

	if (p_enable) {
		glBindVertexArray(data.canvas_quad_array);

	} else {
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_NINEPATCH, p_ninepatch && p_enable);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_TEXTURE_RECT, p_enable);
	state.canvas_shader.bind();
	state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.canvas_item_modulate);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.extra_matrix);
	if (storage->frame.current_rt) {
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, Vector2(1.0 / storage->frame.current_rt->width, 1.0 / storage->frame.current_rt->height));
	} else {
		state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, Vector2(1.0, 1.0));
	}
	state.using_texture_rect = p_enable;
	state.using_ninepatch = p_ninepatch;
}

void RasterizerCanvasGLES3::_draw_polygon(const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {

	glBindVertexArray(data.polygon_buffer_pointer_array);
	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	uint32_t buffer_ofs = 0;

	//vertex
	glBufferSubData(GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_vertices);
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(Vector2), ((uint8_t *)0) + buffer_ofs);
	buffer_ofs += sizeof(Vector2) * p_vertex_count;
	//color

	if (p_singlecolor) {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		Color m = *p_colors;
		glVertexAttrib4f(VS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
	} else if (!p_colors) {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
	} else {

		glBufferSubData(GL_ARRAY_BUFFER, buffer_ofs, sizeof(Color) * p_vertex_count, p_colors);
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), ((uint8_t *)0) + buffer_ofs);
		buffer_ofs += sizeof(Color) * p_vertex_count;
	}

	if (p_uvs) {

		glBufferSubData(GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_uvs);
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), ((uint8_t *)0) + buffer_ofs);
		buffer_ofs += sizeof(Vector2) * p_vertex_count;

	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	//bind the indices buffer.
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(int) * p_index_count, p_indices);

	//draw the triangles.
	glDrawElements(GL_TRIANGLES, p_index_count, GL_UNSIGNED_INT, 0);

	storage->frame.canvas_draw_commands++;

	glBindVertexArray(0);
}

void RasterizerCanvasGLES3::_draw_generic(GLuint p_primitive, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {

	glBindVertexArray(data.polygon_buffer_pointer_array);
	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	uint32_t buffer_ofs = 0;

	//vertex
	glBufferSubData(GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_vertices);
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(Vector2), ((uint8_t *)0) + buffer_ofs);
	buffer_ofs += sizeof(Vector2) * p_vertex_count;
	//color

	if (p_singlecolor) {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		Color m = *p_colors;
		glVertexAttrib4f(VS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
	} else if (!p_colors) {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
	} else {

		glBufferSubData(GL_ARRAY_BUFFER, buffer_ofs, sizeof(Color) * p_vertex_count, p_colors);
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), ((uint8_t *)0) + buffer_ofs);
		buffer_ofs += sizeof(Color) * p_vertex_count;
	}

	if (p_uvs) {

		glBufferSubData(GL_ARRAY_BUFFER, buffer_ofs, sizeof(Vector2) * p_vertex_count, p_uvs);
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), ((uint8_t *)0) + buffer_ofs);
		buffer_ofs += sizeof(Vector2) * p_vertex_count;

	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glDrawArrays(p_primitive, 0, p_vertex_count);

	storage->frame.canvas_draw_commands++;

	glBindVertexArray(0);
}

void RasterizerCanvasGLES3::_draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs) {

	static const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

	//#define GLES_USE_PRIMITIVE_BUFFER

	int version = 0;
	int color_ofs = 0;
	int uv_ofs = 0;
	int stride = 2;

	if (p_colors) { //color
		version |= 1;
		color_ofs = stride;
		stride += 4;
	}

	if (p_uvs) { //uv
		version |= 2;
		uv_ofs = stride;
		stride += 2;
	}

	float b[(2 + 2 + 4) * 4];

	for (int i = 0; i < p_points; i++) {
		b[stride * i + 0] = p_vertices[i].x;
		b[stride * i + 1] = p_vertices[i].y;
	}

	if (p_colors) {

		for (int i = 0; i < p_points; i++) {
			b[stride * i + color_ofs + 0] = p_colors[i].r;
			b[stride * i + color_ofs + 1] = p_colors[i].g;
			b[stride * i + color_ofs + 2] = p_colors[i].b;
			b[stride * i + color_ofs + 3] = p_colors[i].a;
		}
	}

	if (p_uvs) {

		for (int i = 0; i < p_points; i++) {
			b[stride * i + uv_ofs + 0] = p_uvs[i].x;
			b[stride * i + uv_ofs + 1] = p_uvs[i].y;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, p_points * stride * 4, &b[0]);
	glBindVertexArray(data.polygon_buffer_quad_arrays[version]);
	glDrawArrays(prim[p_points], 0, p_points);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	storage->frame.canvas_draw_commands++;
}

void RasterizerCanvasGLES3::_canvas_item_render_commands(Item *p_item, Item *current_clip, bool &reclip) {

	int cc = p_item->commands.size();
	Item::Command **commands = p_item->commands.ptrw();

	for (int i = 0; i < cc; i++) {

		Item::Command *c = commands[i];

		switch (c->type) {
			case Item::Command::TYPE_LINE: {

				Item::CommandLine *line = static_cast<Item::CommandLine *>(c);
				_set_texture_rect_mode(false);

				_bind_canvas_texture(RID(), RID());

				glVertexAttrib4f(VS::ARRAY_COLOR, line->color.r, line->color.g, line->color.b, line->color.a);

				if (line->width <= 1) {
					Vector2 verts[2] = {
						Vector2(line->from.x, line->from.y),
						Vector2(line->to.x, line->to.y)
					};

#ifdef GLES_OVER_GL
					if (line->antialiased)
						glEnable(GL_LINE_SMOOTH);
#endif
					//glLineWidth(line->width);
					_draw_gui_primitive(2, verts, NULL, NULL);

#ifdef GLES_OVER_GL
					if (line->antialiased)
						glDisable(GL_LINE_SMOOTH);
#endif
				} else {
					//thicker line

					Vector2 t = (line->from - line->to).normalized().tangent() * line->width * 0.5;

					Vector2 verts[4] = {
						line->from - t,
						line->from + t,
						line->to + t,
						line->to - t,
					};

					//glLineWidth(line->width);
					_draw_gui_primitive(4, verts, NULL, NULL);
#ifdef GLES_OVER_GL
					if (line->antialiased) {
						glEnable(GL_LINE_SMOOTH);
						for (int i = 0; i < 4; i++) {
							Vector2 vertsl[2] = {
								verts[i],
								verts[(i + 1) % 4],
							};
							_draw_gui_primitive(2, vertsl, NULL, NULL);
						}
						glDisable(GL_LINE_SMOOTH);
					}
#endif
				}

			} break;
			case Item::Command::TYPE_POLYLINE: {

				Item::CommandPolyLine *pline = static_cast<Item::CommandPolyLine *>(c);
				_set_texture_rect_mode(false);

				_bind_canvas_texture(RID(), RID());

				if (pline->triangles.size()) {

					_draw_generic(GL_TRIANGLE_STRIP, pline->triangles.size(), pline->triangles.ptr(), NULL, pline->triangle_colors.ptr(), pline->triangle_colors.size() == 1);
#ifdef GLES_OVER_GL
					glEnable(GL_LINE_SMOOTH);
					if (pline->multiline) {
						//needs to be different
					} else {
						_draw_generic(GL_LINE_LOOP, pline->lines.size(), pline->lines.ptr(), NULL, pline->line_colors.ptr(), pline->line_colors.size() == 1);
					}
					glDisable(GL_LINE_SMOOTH);
#endif
				} else {

#ifdef GLES_OVER_GL
					if (pline->antialiased)
						glEnable(GL_LINE_SMOOTH);
#endif

					if (pline->multiline) {
						int todo = pline->lines.size() / 2;
						int max_per_call = data.polygon_buffer_size / (sizeof(real_t) * 4);
						int offset = 0;

						while (todo) {
							int to_draw = MIN(max_per_call, todo);
							_draw_generic(GL_LINES, to_draw * 2, &pline->lines.ptr()[offset], NULL, pline->line_colors.size() == 1 ? pline->line_colors.ptr() : &pline->line_colors.ptr()[offset], pline->line_colors.size() == 1);
							todo -= to_draw;
							offset += to_draw * 2;
						}

					} else {

						_draw_generic(GL_LINES, pline->lines.size(), pline->lines.ptr(), NULL, pline->line_colors.ptr(), pline->line_colors.size() == 1);
					}

#ifdef GLES_OVER_GL
					if (pline->antialiased)
						glDisable(GL_LINE_SMOOTH);
#endif
				}

			} break;
			case Item::Command::TYPE_RECT: {

				Item::CommandRect *rect = static_cast<Item::CommandRect *>(c);

				_set_texture_rect_mode(true);

				//set color
				glVertexAttrib4f(VS::ARRAY_COLOR, rect->modulate.r, rect->modulate.g, rect->modulate.b, rect->modulate.a);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(rect->texture, rect->normal_map);

				if (texture) {

					bool untile = false;

					if (rect->flags & CANVAS_RECT_TILE && !(texture->flags & VS::TEXTURE_FLAG_REPEAT)) {
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
						untile = true;
					}

					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					Rect2 src_rect = (rect->flags & CANVAS_RECT_REGION) ? Rect2(rect->source.position * texpixel_size, rect->source.size * texpixel_size) : Rect2(0, 0, 1, 1);
					Rect2 dst_rect = Rect2(rect->rect.position, rect->rect.size);

					if (dst_rect.size.width < 0) {
						dst_rect.position.x += dst_rect.size.width;
						dst_rect.size.width *= -1;
					}
					if (dst_rect.size.height < 0) {
						dst_rect.position.y += dst_rect.size.height;
						dst_rect.size.height *= -1;
					}

					if (rect->flags & CANVAS_RECT_FLIP_H) {
						src_rect.size.x *= -1;
					}

					if (rect->flags & CANVAS_RECT_FLIP_V) {
						src_rect.size.y *= -1;
					}

					if (rect->flags & CANVAS_RECT_TRANSPOSE) {
						dst_rect.size.x *= -1; // Encoding in the dst_rect.z uniform
					}

					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);

					state.canvas_shader.set_uniform(CanvasShaderGLES3::DST_RECT, Color(dst_rect.position.x, dst_rect.position.y, dst_rect.size.x, dst_rect.size.y));
					state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(src_rect.position.x, src_rect.position.y, src_rect.size.x, src_rect.size.y));
					state.canvas_shader.set_uniform(CanvasShaderGLES3::CLIP_RECT_UV, (rect->flags & CANVAS_RECT_CLIP_UV) ? true : false);

					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

					if (untile) {
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
					}

				} else {
					Rect2 dst_rect = Rect2(rect->rect.position, rect->rect.size);

					if (dst_rect.size.width < 0) {
						dst_rect.position.x += dst_rect.size.width;
						dst_rect.size.width *= -1;
					}
					if (dst_rect.size.height < 0) {
						dst_rect.position.y += dst_rect.size.height;
						dst_rect.size.height *= -1;
					}

					state.canvas_shader.set_uniform(CanvasShaderGLES3::DST_RECT, Color(dst_rect.position.x, dst_rect.position.y, dst_rect.size.x, dst_rect.size.y));
					state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(0, 0, 1, 1));
					state.canvas_shader.set_uniform(CanvasShaderGLES3::CLIP_RECT_UV, false);
					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
				}

				storage->frame.canvas_draw_commands++;

			} break;

			case Item::Command::TYPE_NINEPATCH: {

				Item::CommandNinePatch *np = static_cast<Item::CommandNinePatch *>(c);

				_set_texture_rect_mode(true, true);

				glVertexAttrib4f(VS::ARRAY_COLOR, np->color.r, np->color.g, np->color.b, np->color.a);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(np->texture, np->normal_map);

				Size2 texpixel_size;

				if (!texture) {

					texpixel_size = Size2(1, 1);

					state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(0, 0, 1, 1));

				} else {

					if (np->source != Rect2()) {
						texpixel_size = Size2(1.0 / np->source.size.width, 1.0 / np->source.size.height);
						state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(np->source.position.x / texture->width, np->source.position.y / texture->height, np->source.size.x / texture->width, np->source.size.y / texture->height));
					} else {
						texpixel_size = Size2(1.0 / texture->width, 1.0 / texture->height);
						state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(0, 0, 1, 1));
					}
				}

				state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				state.canvas_shader.set_uniform(CanvasShaderGLES3::CLIP_RECT_UV, false);
				state.canvas_shader.set_uniform(CanvasShaderGLES3::NP_REPEAT_H, int(np->axis_x));
				state.canvas_shader.set_uniform(CanvasShaderGLES3::NP_REPEAT_V, int(np->axis_y));
				state.canvas_shader.set_uniform(CanvasShaderGLES3::NP_DRAW_CENTER, np->draw_center);
				state.canvas_shader.set_uniform(CanvasShaderGLES3::NP_MARGINS, Color(np->margin[MARGIN_LEFT], np->margin[MARGIN_TOP], np->margin[MARGIN_RIGHT], np->margin[MARGIN_BOTTOM]));
				state.canvas_shader.set_uniform(CanvasShaderGLES3::DST_RECT, Color(np->rect.position.x, np->rect.position.y, np->rect.size.x, np->rect.size.y));

				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				storage->frame.canvas_draw_commands++;
			} break;

			case Item::Command::TYPE_PRIMITIVE: {

				Item::CommandPrimitive *primitive = static_cast<Item::CommandPrimitive *>(c);
				_set_texture_rect_mode(false);

				ERR_CONTINUE(primitive->points.size() < 1);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(primitive->texture, primitive->normal_map);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				}
				if (primitive->colors.size() == 1 && primitive->points.size() > 1) {

					Color c = primitive->colors[0];
					glVertexAttrib4f(VS::ARRAY_COLOR, c.r, c.g, c.b, c.a);

				} else if (primitive->colors.empty()) {
					glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
				}

				_draw_gui_primitive(primitive->points.size(), primitive->points.ptr(), primitive->colors.ptr(), primitive->uvs.ptr());

			} break;
			case Item::Command::TYPE_POLYGON: {

				Item::CommandPolygon *polygon = static_cast<Item::CommandPolygon *>(c);
				_set_texture_rect_mode(false);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(polygon->texture, polygon->normal_map);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				}
				_draw_polygon(polygon->indices.ptr(), polygon->count, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1);
#ifdef GLES_OVER_GL
				if (polygon->antialiased) {
					glEnable(GL_LINE_SMOOTH);
					_draw_generic(GL_LINE_LOOP, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1);
					glDisable(GL_LINE_SMOOTH);
				}
#endif

			} break;
			case Item::Command::TYPE_PARTICLES: {

				Item::CommandParticles *particles_cmd = static_cast<Item::CommandParticles *>(c);

				RasterizerStorageGLES3::Particles *particles = storage->particles_owner.getornull(particles_cmd->particles);
				if (!particles)
					break;

				glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1); //not used, so keep white

				VisualServerRaster::redraw_request();

				storage->particles_request_process(particles_cmd->particles);
				//enable instancing

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, true);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_PARTICLES, true);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, true);
				//reset shader and force rebind
				state.using_texture_rect = true;
				_set_texture_rect_mode(false);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(particles_cmd->texture, particles_cmd->normal_map);

				if (texture) {
					Size2 texpixel_size(1.0 / (texture->width / particles_cmd->h_frames), 1.0 / (texture->height / particles_cmd->v_frames));
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				} else {
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, Vector2(1.0, 1.0));
				}

				if (!particles->use_local_coords) {

					Transform2D inv_xf;
					inv_xf.set_axis(0, Vector2(particles->emission_transform.basis.get_axis(0).x, particles->emission_transform.basis.get_axis(0).y));
					inv_xf.set_axis(1, Vector2(particles->emission_transform.basis.get_axis(1).x, particles->emission_transform.basis.get_axis(1).y));
					inv_xf.set_origin(Vector2(particles->emission_transform.get_origin().x, particles->emission_transform.get_origin().y));
					inv_xf.affine_invert();

					state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform * inv_xf);
				}

				state.canvas_shader.set_uniform(CanvasShaderGLES3::H_FRAMES, particles_cmd->h_frames);
				state.canvas_shader.set_uniform(CanvasShaderGLES3::V_FRAMES, particles_cmd->v_frames);

				glBindVertexArray(data.particle_quad_array); //use particle quad array
				glBindBuffer(GL_ARRAY_BUFFER, particles->particle_buffers[0]); //bind particle buffer

				int stride = sizeof(float) * 4 * 6;

				int amount = particles->amount;

				if (particles->draw_order != VS::PARTICLES_DRAW_ORDER_LIFETIME) {

					glEnableVertexAttribArray(8); //xform x
					glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 3);
					glVertexAttribDivisor(8, 1);
					glEnableVertexAttribArray(9); //xform y
					glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 4);
					glVertexAttribDivisor(9, 1);
					glEnableVertexAttribArray(10); //xform z
					glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 5);
					glVertexAttribDivisor(10, 1);
					glEnableVertexAttribArray(11); //color
					glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + 0);
					glVertexAttribDivisor(11, 1);
					glEnableVertexAttribArray(12); //custom
					glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 2);
					glVertexAttribDivisor(12, 1);

					glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, amount);
				} else {
					//split

					int stride = sizeof(float) * 4 * 6;
					int split = int(Math::ceil(particles->phase * particles->amount));

					if (amount - split > 0) {
						glEnableVertexAttribArray(8); //xform x
						glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + stride * split + sizeof(float) * 4 * 3);
						glVertexAttribDivisor(8, 1);
						glEnableVertexAttribArray(9); //xform y
						glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + stride * split + sizeof(float) * 4 * 4);
						glVertexAttribDivisor(9, 1);
						glEnableVertexAttribArray(10); //xform z
						glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + stride * split + sizeof(float) * 4 * 5);
						glVertexAttribDivisor(10, 1);
						glEnableVertexAttribArray(11); //color
						glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + stride * split + 0);
						glVertexAttribDivisor(11, 1);
						glEnableVertexAttribArray(12); //custom
						glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + stride * split + sizeof(float) * 4 * 2);
						glVertexAttribDivisor(12, 1);

						glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, amount - split);
					}

					if (split > 0) {
						glEnableVertexAttribArray(8); //xform x
						glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 3);
						glVertexAttribDivisor(8, 1);
						glEnableVertexAttribArray(9); //xform y
						glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 4);
						glVertexAttribDivisor(9, 1);
						glEnableVertexAttribArray(10); //xform z
						glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 5);
						glVertexAttribDivisor(10, 1);
						glEnableVertexAttribArray(11); //color
						glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + 0);
						glVertexAttribDivisor(11, 1);
						glEnableVertexAttribArray(12); //custom
						glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + sizeof(float) * 4 * 2);
						glVertexAttribDivisor(12, 1);

						glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, split);
					}
				}

				glBindVertexArray(0);

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_PARTICLES, false);
				state.using_texture_rect = true;
				_set_texture_rect_mode(false);

			} break;
			case Item::Command::TYPE_CIRCLE: {

				_set_texture_rect_mode(false);

				Item::CommandCircle *circle = static_cast<Item::CommandCircle *>(c);
				static const int numpoints = 32;
				Vector2 points[numpoints + 1];
				points[numpoints] = circle->pos;
				int indices[numpoints * 3];

				for (int i = 0; i < numpoints; i++) {

					points[i] = circle->pos + Vector2(Math::sin(i * Math_PI * 2.0 / numpoints), Math::cos(i * Math_PI * 2.0 / numpoints)) * circle->radius;
					indices[i * 3 + 0] = i;
					indices[i * 3 + 1] = (i + 1) % numpoints;
					indices[i * 3 + 2] = numpoints;
				}

				_bind_canvas_texture(RID(), RID());
				_draw_polygon(indices, numpoints * 3, numpoints + 1, points, NULL, &circle->color, true);

				//_draw_polygon(numpoints*3,indices,points,NULL,&circle->color,RID(),true);
				//canvas_draw_circle(circle->indices.size(),circle->indices.ptr(),circle->points.ptr(),circle->uvs.ptr(),circle->colors.ptr(),circle->texture,circle->colors.size()==1);
			} break;
			case Item::Command::TYPE_TRANSFORM: {

				Item::CommandTransform *transform = static_cast<Item::CommandTransform *>(c);
				state.extra_matrix = transform->xform;
				state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.extra_matrix);

			} break;
			case Item::Command::TYPE_CLIP_IGNORE: {

				Item::CommandClipIgnore *ci = static_cast<Item::CommandClipIgnore *>(c);
				if (current_clip) {

					if (ci->ignore != reclip) {
						if (ci->ignore) {

							glDisable(GL_SCISSOR_TEST);
							reclip = true;
						} else {

							glEnable(GL_SCISSOR_TEST);
							//glScissor(viewport.x+current_clip->final_clip_rect.pos.x,viewport.y+ (viewport.height-(current_clip->final_clip_rect.pos.y+current_clip->final_clip_rect.size.height)),
							//current_clip->final_clip_rect.size.width,current_clip->final_clip_rect.size.height);

							int x = current_clip->final_clip_rect.position.x;
							int y = storage->frame.current_rt->height - (current_clip->final_clip_rect.position.y + current_clip->final_clip_rect.size.y);
							int w = current_clip->final_clip_rect.size.x;
							int h = current_clip->final_clip_rect.size.y;

							glScissor(x, y, w, h);

							reclip = false;
						}
					}
				}

			} break;
		}
	}
}

void RasterizerCanvasGLES3::_copy_texscreen(const Rect2 &p_rect) {

	glDisable(GL_BLEND);

	state.canvas_texscreen_used = true;
	//blur diffuse into effect mipmaps using separatable convolution
	//storage->shaders.copy.set_conditional(CopyShaderGLES3::GAUSSIAN_HORIZONTAL,true);

	Vector2 wh(storage->frame.current_rt->width, storage->frame.current_rt->height);

	Color blur_section(p_rect.position.x / wh.x, p_rect.position.y / wh.y, p_rect.size.x / wh.x, p_rect.size.y / wh.y);

	if (p_rect != Rect2()) {

		scene_render->state.effect_blur_shader.set_conditional(EffectBlurShaderGLES3::USE_BLUR_SECTION, true);
		storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_COPY_SECTION, true);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->effects.mip_maps[0].sizes[0].fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->color);

	storage->shaders.copy.bind();
	storage->shaders.copy.set_uniform(CopyShaderGLES3::COPY_SECTION, blur_section);

	scene_render->_copy_screen();

	for (int i = 0; i < storage->frame.current_rt->effects.mip_maps[1].sizes.size(); i++) {

		int vp_w = storage->frame.current_rt->effects.mip_maps[1].sizes[i].width;
		int vp_h = storage->frame.current_rt->effects.mip_maps[1].sizes[i].height;
		glViewport(0, 0, vp_w, vp_h);
		//horizontal pass
		scene_render->state.effect_blur_shader.set_conditional(EffectBlurShaderGLES3::GAUSSIAN_HORIZONTAL, true);
		scene_render->state.effect_blur_shader.bind();
		scene_render->state.effect_blur_shader.set_uniform(EffectBlurShaderGLES3::PIXEL_SIZE, Vector2(1.0 / vp_w, 1.0 / vp_h));
		scene_render->state.effect_blur_shader.set_uniform(EffectBlurShaderGLES3::LOD, float(i));
		scene_render->state.effect_blur_shader.set_uniform(EffectBlurShaderGLES3::BLUR_SECTION, blur_section);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->effects.mip_maps[0].color); //previous level, since mipmaps[0] starts one level bigger
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->effects.mip_maps[1].sizes[i].fbo);

		scene_render->_copy_screen();

		scene_render->state.effect_blur_shader.set_conditional(EffectBlurShaderGLES3::GAUSSIAN_HORIZONTAL, false);

		//vertical pass
		scene_render->state.effect_blur_shader.set_conditional(EffectBlurShaderGLES3::GAUSSIAN_VERTICAL, true);
		scene_render->state.effect_blur_shader.bind();
		scene_render->state.effect_blur_shader.set_uniform(EffectBlurShaderGLES3::PIXEL_SIZE, Vector2(1.0 / vp_w, 1.0 / vp_h));
		scene_render->state.effect_blur_shader.set_uniform(EffectBlurShaderGLES3::LOD, float(i));
		scene_render->state.effect_blur_shader.set_uniform(EffectBlurShaderGLES3::BLUR_SECTION, blur_section);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->effects.mip_maps[1].color);
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->effects.mip_maps[0].sizes[i + 1].fbo); //next level, since mipmaps[0] starts one level bigger

		scene_render->_copy_screen();

		scene_render->state.effect_blur_shader.set_conditional(EffectBlurShaderGLES3::GAUSSIAN_VERTICAL, false);
	}

	scene_render->state.effect_blur_shader.set_conditional(EffectBlurShaderGLES3::USE_BLUR_SECTION, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_COPY_SECTION, false);

	glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo); //back to front
	glViewport(0, 0, storage->frame.current_rt->width, storage->frame.current_rt->height);

	state.canvas_shader.bind(); //back to canvas
	_bind_canvas_texture(state.current_tex, state.current_normal);

	if (state.using_texture_rect) {
		state.using_texture_rect = false;
		_set_texture_rect_mode(state.using_texture_rect, state.using_ninepatch);
	}

	glEnable(GL_BLEND);
}

void RasterizerCanvasGLES3::canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light) {

	Item *current_clip = NULL;
	RasterizerStorageGLES3::Shader *shader_cache = NULL;

	bool rebind_shader = true;

	Size2 rt_size = Size2(storage->frame.current_rt->width, storage->frame.current_rt->height);

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_DISTANCE_FIELD, false);

	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_item_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(CanvasItemUBO), &state.canvas_item_ubo_data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.current_tex = RID();
	state.current_tex_ptr = NULL;
	state.current_normal = RID();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	int last_blend_mode = -1;

	RID canvas_last_material;

	bool prev_distance_field = false;

	while (p_item_list) {

		Item *ci = p_item_list;

		if (prev_distance_field != ci->distance_field) {

			state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_DISTANCE_FIELD, ci->distance_field);
			prev_distance_field = ci->distance_field;
			rebind_shader = true;
		}

		if (current_clip != ci->final_clip_owner) {

			current_clip = ci->final_clip_owner;

			//setup clip
			if (current_clip) {

				glEnable(GL_SCISSOR_TEST);
				glScissor(current_clip->final_clip_rect.position.x, (rt_size.height - (current_clip->final_clip_rect.position.y + current_clip->final_clip_rect.size.height)), current_clip->final_clip_rect.size.width, current_clip->final_clip_rect.size.height);

			} else {

				glDisable(GL_SCISSOR_TEST);
			}
		}

		if (ci->copy_back_buffer) {

			if (ci->copy_back_buffer->full) {

				_copy_texscreen(Rect2());
			} else {
				_copy_texscreen(ci->copy_back_buffer->rect);
			}
		}

		//begin rect
		Item *material_owner = ci->material_owner ? ci->material_owner : ci;

		RID material = material_owner->material;

		if (material != canvas_last_material || rebind_shader) {

			RasterizerStorageGLES3::Material *material_ptr = storage->material_owner.getornull(material);
			RasterizerStorageGLES3::Shader *shader_ptr = NULL;

			if (material_ptr) {

				shader_ptr = material_ptr->shader;

				if (shader_ptr && shader_ptr->mode != VS::SHADER_CANVAS_ITEM) {
					shader_ptr = NULL; //do not use non canvasitem shader
				}
			}

			if (shader_ptr && shader_ptr != shader_cache) {

				if (shader_ptr->canvas_item.uses_screen_texture && !state.canvas_texscreen_used) {
					//copy if not copied before
					_copy_texscreen(Rect2());
				}

				if (shader_ptr->canvas_item.uses_time) {
					VisualServerRaster::redraw_request();
				}

				state.canvas_shader.set_custom_shader(shader_ptr->custom_code_id);
				state.canvas_shader.bind();

				if (material_ptr->ubo_id) {
					glBindBufferBase(GL_UNIFORM_BUFFER, 2, material_ptr->ubo_id);
				}

				int tc = material_ptr->textures.size();
				RID *textures = material_ptr->textures.ptrw();
				ShaderLanguage::ShaderNode::Uniform::Hint *texture_hints = shader_ptr->texture_hints.ptrw();

				for (int i = 0; i < tc; i++) {

					glActiveTexture(GL_TEXTURE2 + i);

					RasterizerStorageGLES3::Texture *t = storage->texture_owner.getornull(textures[i]);
					if (!t) {

						switch (texture_hints[i]) {
							case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO:
							case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK: {
								glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);
							} break;
							case ShaderLanguage::ShaderNode::Uniform::HINT_ANISO: {
								glBindTexture(GL_TEXTURE_2D, storage->resources.aniso_tex);
							} break;
							case ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL: {
								glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);
							} break;
							default: {
								glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
							} break;
						}

						//check hints

						continue;
					}

					t = t->get_ptr();

					if (storage->config.srgb_decode_supported && t->using_srgb) {
						//no srgb in 2D
						glTexParameteri(t->target, _TEXTURE_SRGB_DECODE_EXT, _SKIP_DECODE_EXT);
						t->using_srgb = false;
					}

					glBindTexture(t->target, t->tex_id);
				}

			} else if (!shader_ptr) {
				state.canvas_shader.set_custom_shader(0);
				state.canvas_shader.bind();
			}

			shader_cache = shader_ptr;

			canvas_last_material = material;
			rebind_shader = false;
		}

		int blend_mode = shader_cache ? shader_cache->canvas_item.blend_mode : RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX;
		bool unshaded = shader_cache && (shader_cache->canvas_item.light_mode == RasterizerStorageGLES3::Shader::CanvasItem::LIGHT_MODE_UNSHADED || blend_mode != RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX);
		bool reclip = false;

		if (last_blend_mode != blend_mode) {

			switch (blend_mode) {

				case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX: {
					glBlendEquation(GL_FUNC_ADD);
					if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
					} else {
						glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					}

				} break;
				case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_ADD: {

					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE);

				} break;
				case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_SUB: {

					glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				} break;
				case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MUL: {
					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(GL_DST_COLOR, GL_ZERO);
				} break;
				case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_PMALPHA: {
					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} break;
			}

			last_blend_mode = blend_mode;
		}

		state.canvas_item_modulate = unshaded ? ci->final_modulate : Color(ci->final_modulate.r * p_modulate.r, ci->final_modulate.g * p_modulate.g, ci->final_modulate.b * p_modulate.b, ci->final_modulate.a * p_modulate.a);

		state.final_transform = ci->final_transform;
		state.extra_matrix = Transform2D();

		state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.canvas_item_modulate);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.extra_matrix);
		if (storage->frame.current_rt) {
			state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, Vector2(1.0 / storage->frame.current_rt->width, 1.0 / storage->frame.current_rt->height));
		} else {
			state.canvas_shader.set_uniform(CanvasShaderGLES3::SCREEN_PIXEL_SIZE, Vector2(1.0, 1.0));
		}
		if (unshaded || (state.canvas_item_modulate.a > 0.001 && (!shader_cache || shader_cache->canvas_item.light_mode != RasterizerStorageGLES3::Shader::CanvasItem::LIGHT_MODE_LIGHT_ONLY) && !ci->light_masked))
			_canvas_item_render_commands(ci, current_clip, reclip);

		if ((blend_mode == RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX || blend_mode == RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_PMALPHA) && p_light && !unshaded) {

			Light *light = p_light;
			bool light_used = false;
			VS::CanvasLightMode mode = VS::CANVAS_LIGHT_MODE_ADD;
			state.canvas_item_modulate = ci->final_modulate; // remove the canvas modulate

			while (light) {

				if (ci->light_mask & light->item_mask && p_z >= light->z_min && p_z <= light->z_max && ci->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {

					//intersects this light

					if (!light_used || mode != light->mode) {

						mode = light->mode;

						switch (mode) {

							case VS::CANVAS_LIGHT_MODE_ADD: {
								glBlendEquation(GL_FUNC_ADD);
								glBlendFunc(GL_SRC_ALPHA, GL_ONE);

							} break;
							case VS::CANVAS_LIGHT_MODE_SUB: {
								glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
								glBlendFunc(GL_SRC_ALPHA, GL_ONE);
							} break;
							case VS::CANVAS_LIGHT_MODE_MIX:
							case VS::CANVAS_LIGHT_MODE_MASK: {
								glBlendEquation(GL_FUNC_ADD);
								glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

							} break;
						}
					}

					if (!light_used) {

						state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_LIGHTING, true);
						light_used = true;
					}

					bool has_shadow = light->shadow_buffer.is_valid() && ci->light_mask & light->item_shadow_mask;

					state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SHADOWS, has_shadow);
					if (has_shadow) {
						state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_USE_GRADIENT, light->shadow_gradient_length > 0);
						switch (light->shadow_filter) {

							case VS::CANVAS_LIGHT_FILTER_NONE: state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_NEAREST, true); break;
							case VS::CANVAS_LIGHT_FILTER_PCF3: state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF3, true); break;
							case VS::CANVAS_LIGHT_FILTER_PCF5: state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF5, true); break;
							case VS::CANVAS_LIGHT_FILTER_PCF7: state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF7, true); break;
							case VS::CANVAS_LIGHT_FILTER_PCF9: state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF9, true); break;
							case VS::CANVAS_LIGHT_FILTER_PCF13: state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF13, true); break;
						}
					}

					bool light_rebind = state.canvas_shader.bind();

					if (light_rebind) {

						state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.canvas_item_modulate);
						state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);
						state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, Transform2D());
					}

					glBindBufferBase(GL_UNIFORM_BUFFER, 1, static_cast<LightInternal *>(light->light_internal.get_data())->ubo);

					if (has_shadow) {

						RasterizerStorageGLES3::CanvasLightShadow *cls = storage->canvas_light_shadow_owner.get(light->shadow_buffer);
						glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 2);
						glBindTexture(GL_TEXTURE_2D, cls->distance);

						/*canvas_shader.set_uniform(CanvasShaderGLES3::SHADOW_MATRIX,light->shadow_matrix_cache);
						canvas_shader.set_uniform(CanvasShaderGLES3::SHADOW_ESM_MULTIPLIER,light->shadow_esm_mult);
						canvas_shader.set_uniform(CanvasShaderGLES3::LIGHT_SHADOW_COLOR,light->shadow_color);*/
					}

					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 1);
					RasterizerStorageGLES3::Texture *t = storage->texture_owner.getornull(light->texture);
					if (!t) {
						glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
					} else {

						glBindTexture(t->target, t->tex_id);
					}

					glActiveTexture(GL_TEXTURE0);
					_canvas_item_render_commands(ci, current_clip, reclip); //redraw using light
				}

				light = light->next_ptr;
			}

			if (light_used) {

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_LIGHTING, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SHADOWS, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_NEAREST, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF3, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF5, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF7, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF9, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF13, false);

				state.canvas_shader.bind();

				last_blend_mode = -1;

				/*
				//this is set again, so it should not be needed anyway?
				state.canvas_item_modulate = unshaded ? ci->final_modulate : Color(
							ci->final_modulate.r * p_modulate.r,
							ci->final_modulate.g * p_modulate.g,
							ci->final_modulate.b * p_modulate.b,
							ci->final_modulate.a * p_modulate.a );


				state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX,state.final_transform);
				state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX,Transform2D());
				state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE,state.canvas_item_modulate);

				glBlendEquation(GL_FUNC_ADD);

				if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				}

				//@TODO RESET canvas_blend_mode
				*/
			}
		}

		if (reclip) {

			glEnable(GL_SCISSOR_TEST);
			glScissor(current_clip->final_clip_rect.position.x, (rt_size.height - (current_clip->final_clip_rect.position.y + current_clip->final_clip_rect.size.height)), current_clip->final_clip_rect.size.width, current_clip->final_clip_rect.size.height);
		}

		p_item_list = p_item_list->next;
	}

	if (current_clip) {
		glDisable(GL_SCISSOR_TEST);
	}
}

void RasterizerCanvasGLES3::canvas_debug_viewport_shadows(Light *p_lights_with_shadow) {

	Light *light = p_lights_with_shadow;

	canvas_begin(); //reset
	glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
	int h = 10;
	int w = storage->frame.current_rt->width;
	int ofs = h;
	glDisable(GL_BLEND);

	//print_line(" debug lights ");
	while (light) {

		//print_line("debug light");
		if (light->shadow_buffer.is_valid()) {

			//print_line("sb is valid");
			RasterizerStorageGLES3::CanvasLightShadow *sb = storage->canvas_light_shadow_owner.get(light->shadow_buffer);
			if (sb) {
				glBindTexture(GL_TEXTURE_2D, sb->distance);
				//glBindTexture(GL_TEXTURE_2D,storage->resources.white_tex);
				draw_generic_textured_rect(Rect2(h, ofs, w - h * 2, h), Rect2(0, 0, 1, 1));
				ofs += h * 2;
			}
		}

		light = light->shadows_next_ptr;
	}
}

void RasterizerCanvasGLES3::canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {

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

	state.canvas_shadow_shader.bind();

	glViewport(0, 0, cls->size, cls->height);
	glClearDepth(1.0f);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	VS::CanvasOccluderPolygonCullMode cull = VS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;

	for (int i = 0; i < 4; i++) {

		//make sure it remains orthogonal, makes easy to read angle later

		Transform light;
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
		projection = projection * CameraMatrix(Transform().looking_at(cam_target, Vector3(0, 0, -1)).affine_inverse());

		state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::PROJECTION_MATRIX, projection);
		state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::LIGHT_MATRIX, light);
		state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::DISTANCE_NORM, 1.0 / p_far);

		if (i == 0)
			*p_xform_cache = projection;

		glViewport(0, (cls->height / 4) * i, cls->size, cls->height / 4);

		LightOccluderInstance *instance = p_occluders;

		while (instance) {

			RasterizerStorageGLES3::CanvasOccluder *cc = storage->canvas_occluder_owner.get(instance->polygon_buffer);
			if (!cc || cc->len == 0 || !(p_light_mask & instance->light_mask)) {

				instance = instance->next;
				continue;
			}

			state.canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES3::WORLD_MATRIX, instance->xform_cache);
			if (cull != instance->cull_cache) {

				cull = instance->cull_cache;
				switch (cull) {
					case VS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED: {

						glDisable(GL_CULL_FACE);

					} break;
					case VS::CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE: {

						glEnable(GL_CULL_FACE);
						glCullFace(GL_FRONT);
					} break;
					case VS::CANVAS_OCCLUDER_POLYGON_CULL_COUNTER_CLOCKWISE: {

						glEnable(GL_CULL_FACE);
						glCullFace(GL_BACK);

					} break;
				}
			}
			/*
			if (i==0) {
				for(int i=0;i<cc->lines.size();i++) {
					Vector2 p = instance->xform_cache.xform(cc->lines.get(i));
					Plane pp(Vector3(p.x,p.y,0),1);
					pp.normal = light.xform(pp.normal);
					pp = projection.xform4(pp);
					print_line(itos(i)+": "+pp.normal/pp.d);
					//pp=light_mat.xform4(pp);
					//print_line(itos(i)+": "+pp.normal/pp.d);
				}
			}
*/
			glBindVertexArray(cc->array_id);
			glDrawElements(GL_TRIANGLES, cc->len * 3, GL_UNSIGNED_SHORT, 0);

			instance = instance->next;
		}
	}

	glBindVertexArray(0);
}
void RasterizerCanvasGLES3::reset_canvas() {

	if (storage->frame.current_rt) {
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);
		glColorMask(1, 1, 1, 1); //don't touch alpha
	}

	glBindVertexArray(0);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	//glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	//glLineWidth(1.0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//use for reading from screen
	if (storage->frame.current_rt && !storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_NO_SAMPLING]) {
		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 3);
		glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->effects.mip_maps[0].color);
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);

	Transform canvas_transform;

	if (storage->frame.current_rt) {

		float csy = 1.0;
		if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP]) {
			csy = -1.0;
		}
		canvas_transform.translate(-(storage->frame.current_rt->width / 2.0f), -(storage->frame.current_rt->height / 2.0f), 0.0f);
		canvas_transform.scale(Vector3(2.0f / storage->frame.current_rt->width, csy * -2.0f / storage->frame.current_rt->height, 1.0f));
	} else {
		Vector2 ssize = OS::get_singleton()->get_window_size();
		canvas_transform.translate(-(ssize.width / 2.0f), -(ssize.height / 2.0f), 0.0f);
		canvas_transform.scale(Vector3(2.0f / ssize.width, -2.0f / ssize.height, 1.0f));
	}

	state.vp = canvas_transform;

	store_transform(canvas_transform, state.canvas_item_ubo_data.projection_matrix);
	state.canvas_item_ubo_data.time = storage->frame.time[0];

	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_item_ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CanvasItemUBO), &state.canvas_item_ubo_data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.canvas_texscreen_used = false;
}

void RasterizerCanvasGLES3::draw_generic_textured_rect(const Rect2 &p_rect, const Rect2 &p_src) {

	state.canvas_shader.set_uniform(CanvasShaderGLES3::DST_RECT, Color(p_rect.position.x, p_rect.position.y, p_rect.size.x, p_rect.size.y));
	state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(p_src.position.x, p_src.position.y, p_src.size.x, p_src.size.y));
	state.canvas_shader.set_uniform(CanvasShaderGLES3::CLIP_RECT_UV, false);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

void RasterizerCanvasGLES3::draw_window_margins(int *black_margin, RID *black_image) {

	Vector2 window_size = OS::get_singleton()->get_window_size();
	int window_h = window_size.height;
	int window_w = window_size.width;

	glBindFramebuffer(GL_FRAMEBUFFER, storage->system_fbo);
	glViewport(0, 0, window_size.width, window_size.height);
	canvas_begin();

	if (black_image[MARGIN_LEFT].is_valid()) {
		_bind_canvas_texture(black_image[MARGIN_LEFT], RID());
		Size2 sz(storage->texture_get_width(black_image[MARGIN_LEFT]), storage->texture_get_height(black_image[MARGIN_LEFT]));
		draw_generic_textured_rect(Rect2(0, 0, black_margin[MARGIN_LEFT], window_h), Rect2(0, 0, sz.x, sz.y));
	} else if (black_margin[MARGIN_LEFT]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(0, 0, black_margin[MARGIN_LEFT], window_h), Rect2(0, 0, 1, 1));
	}

	if (black_image[MARGIN_RIGHT].is_valid()) {
		_bind_canvas_texture(black_image[MARGIN_RIGHT], RID());
		Size2 sz(storage->texture_get_width(black_image[MARGIN_RIGHT]), storage->texture_get_height(black_image[MARGIN_RIGHT]));
		draw_generic_textured_rect(Rect2(window_w - black_margin[MARGIN_RIGHT], 0, black_margin[MARGIN_RIGHT], window_h), Rect2(0, 0, sz.x, sz.y));
	} else if (black_margin[MARGIN_RIGHT]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(window_w - black_margin[MARGIN_RIGHT], 0, black_margin[MARGIN_RIGHT], window_h), Rect2(0, 0, 1, 1));
	}

	if (black_image[MARGIN_TOP].is_valid()) {
		_bind_canvas_texture(black_image[MARGIN_TOP], RID());

		Size2 sz(storage->texture_get_width(black_image[MARGIN_TOP]), storage->texture_get_height(black_image[MARGIN_TOP]));
		draw_generic_textured_rect(Rect2(0, 0, window_w, black_margin[MARGIN_TOP]), Rect2(0, 0, sz.x, sz.y));

	} else if (black_margin[MARGIN_TOP]) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(0, 0, window_w, black_margin[MARGIN_TOP]), Rect2(0, 0, 1, 1));
	}

	if (black_image[MARGIN_BOTTOM].is_valid()) {

		_bind_canvas_texture(black_image[MARGIN_BOTTOM], RID());

		Size2 sz(storage->texture_get_width(black_image[MARGIN_BOTTOM]), storage->texture_get_height(black_image[MARGIN_BOTTOM]));
		draw_generic_textured_rect(Rect2(0, window_h - black_margin[MARGIN_BOTTOM], window_w, black_margin[MARGIN_BOTTOM]), Rect2(0, 0, sz.x, sz.y));

	} else if (black_margin[MARGIN_BOTTOM]) {

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);

		draw_generic_textured_rect(Rect2(0, window_h - black_margin[MARGIN_BOTTOM], window_w, black_margin[MARGIN_BOTTOM]), Rect2(0, 0, 1, 1));
	}
}

void RasterizerCanvasGLES3::initialize() {

	{
		//quad buffers

		glGenBuffers(1, &data.canvas_quad_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, data.canvas_quad_vertices);
		{
			const float qv[8] = {
				0, 0,
				0, 1,
				1, 1,
				1, 0
			};

			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8, qv, GL_STATIC_DRAW);
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		glGenVertexArrays(1, &data.canvas_quad_array);
		glBindVertexArray(data.canvas_quad_array);
		glBindBuffer(GL_ARRAY_BUFFER, data.canvas_quad_vertices);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
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
		glEnableVertexAttribArray(VS::ARRAY_VERTEX);
		glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (float *)0 + 2);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}
	{

		uint32_t poly_size = GLOBAL_DEF("rendering/limits/buffers/canvas_polygon_buffer_size_kb", 128);
		poly_size *= 1024; //kb
		poly_size = MAX(poly_size, (2 + 2 + 4) * 4 * sizeof(float));
		glGenBuffers(1, &data.polygon_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
		glBufferData(GL_ARRAY_BUFFER, poly_size, NULL, GL_DYNAMIC_DRAW); //allocate max size
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		data.polygon_buffer_size = poly_size;

		//quad arrays
		for (int i = 0; i < 4; i++) {
			glGenVertexArrays(1, &data.polygon_buffer_quad_arrays[i]);
			glBindVertexArray(data.polygon_buffer_quad_arrays[i]);
			glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

			int uv_ofs = 0;
			int color_ofs = 0;
			int stride = 2 * 4;

			if (i & 1) { //color
				color_ofs = stride;
				stride += 4 * 4;
			}

			if (i & 2) { //uv
				uv_ofs = stride;
				stride += 2 * 4;
			}

			glEnableVertexAttribArray(VS::ARRAY_VERTEX);
			glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + 0);

			if (i & 1) {
				glEnableVertexAttribArray(VS::ARRAY_COLOR);
				glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + color_ofs);
			}

			if (i & 2) {
				glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
				glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)NULL) + uv_ofs);
			}

			glBindVertexArray(0);
		}

		glGenVertexArrays(1, &data.polygon_buffer_pointer_array);

		uint32_t index_size = GLOBAL_DEF("rendering/limits/buffers/canvas_polygon_index_buffer_size_kb", 128);
		index_size *= 1024; //kb
		glGenBuffers(1, &data.polygon_index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, NULL, GL_DYNAMIC_DRAW); //allocate max size
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	store_transform(Transform(), state.canvas_item_ubo_data.projection_matrix);

	glGenBuffers(1, &state.canvas_item_ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_item_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(CanvasItemUBO), &state.canvas_item_ubo_data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.canvas_shader.init();
	state.canvas_shader.set_base_material_tex_index(2);
	state.canvas_shadow_shader.init();

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);
	state.canvas_shadow_shader.set_conditional(CanvasShadowShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_PIXEL_SNAP, GLOBAL_DEF("rendering/quality/2d/use_pixel_snap", false));
}

void RasterizerCanvasGLES3::finalize() {

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);

	glDeleteVertexArrays(1, &data.polygon_buffer_pointer_array);
}

RasterizerCanvasGLES3::RasterizerCanvasGLES3() {
}

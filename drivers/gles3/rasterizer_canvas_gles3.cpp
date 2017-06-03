/*************************************************************************/
/*  rasterizer_canvas_gles3.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
	li->ubo_data.shadowpixel_size = 1.0 / p_light->shadow_buffer_size;
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

	/*canvas_shader.unbind();
	canvas_shader.set_custom_shader(0);
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_MODULATE,false);
	canvas_shader.bind();
	canvas_shader.set_uniform(CanvasShaderGLES2::TEXTURE, 0);
	canvas_use_modulate=false;*/

	reset_canvas();

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_TEXTURE_RECT, true);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_LIGHTING, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SHADOWS, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_NEAREST, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF5, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF13, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_DISTANCE_FIELD, false);

	state.canvas_shader.set_custom_shader(0);
	state.canvas_shader.bind();
	state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, Color(1, 1, 1, 1));
	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, Transform2D());
	state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, Transform2D());

	//state.canvas_shader.set_uniform(CanvasShaderGLES3::PROJECTION_MATRIX,state.vp);
	//state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX,Transform());
	//state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX,Transform());

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, state.canvas_item_ubo);
	glBindVertexArray(data.canvas_quad_array);
	state.using_texture_rect = true;
}

void RasterizerCanvasGLES3::canvas_end() {

	glBindVertexArray(0);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);

	state.using_texture_rect = false;
}

RasterizerStorageGLES3::Texture *RasterizerCanvasGLES3::_bind_canvas_texture(const RID &p_texture) {

	if (p_texture == state.current_tex) {
		return state.current_tex_ptr;
	}

	if (p_texture.is_valid()) {

		RasterizerStorageGLES3::Texture *texture = storage->texture_owner.getornull(p_texture);

		if (!texture) {
			state.current_tex = RID();
			state.current_tex_ptr = NULL;
			glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
			return NULL;
		}

		if (texture->render_target)
			texture->render_target->used_in_frame = true;

		glBindTexture(GL_TEXTURE_2D, texture->tex_id);
		state.current_tex = p_texture;
		state.current_tex_ptr = texture;

		return texture;

	} else {

		glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
		state.current_tex = RID();
		state.current_tex_ptr = NULL;
	}

	return NULL;
}

void RasterizerCanvasGLES3::_set_texture_rect_mode(bool p_enable) {

	if (state.using_texture_rect == p_enable)
		return;

	if (p_enable) {
		glBindVertexArray(data.canvas_quad_array);

	} else {
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_TEXTURE_RECT, p_enable);
	state.canvas_shader.bind();
	state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.canvas_item_modulate);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.extra_matrix);

	state.using_texture_rect = p_enable;
}

void RasterizerCanvasGLES3::_draw_polygon(int p_vertex_count, const int *p_indices, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, const RID &p_texture, bool p_singlecolor) {

	bool do_colors = false;
	Color m;
	if (p_singlecolor) {
		m = *p_colors;
		glVertexAttrib4f(VS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
	} else if (!p_colors) {

		glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
	} else
		do_colors = true;

	RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(p_texture);

#ifndef GLES_NO_CLIENT_ARRAYS
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(Vector2), p_vertices);
	if (do_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), p_colors);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (texture && p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), p_uvs);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	if (p_indices) {
		glDrawElements(GL_TRIANGLES, p_vertex_count, GL_UNSIGNED_INT, p_indices);
	} else {
		glDrawArrays(GL_TRIANGLES, 0, p_vertex_count);
	}

#else //WebGL specific impl.
	glBindBuffer(GL_ARRAY_BUFFER, gui_quad_buffer);
	float *b = GlobalVertexBuffer;
	int ofs = 0;
	if (p_vertex_count > MAX_POLYGON_VERTICES) {
		print_line("Too many vertices to render");
		return;
	}
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(float) * 2, ((float *)0) + ofs);
	for (int i = 0; i < p_vertex_count; i++) {
		b[ofs++] = p_vertices[i].x;
		b[ofs++] = p_vertices[i].y;
	}

	if (p_colors && do_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(float) * 4, ((float *)0) + ofs);
		for (int i = 0; i < p_vertex_count; i++) {
			b[ofs++] = p_colors[i].r;
			b[ofs++] = p_colors[i].g;
			b[ofs++] = p_colors[i].b;
			b[ofs++] = p_colors[i].a;
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(float) * 2, ((float *)0) + ofs);
		for (int i = 0; i < p_vertex_count; i++) {
			b[ofs++] = p_uvs[i].x;
			b[ofs++] = p_uvs[i].y;
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glBufferSubData(GL_ARRAY_BUFFER, 0, ofs * 4, &b[0]);

	//bind the indices buffer.
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices_buffer);

	static const int _max_draw_poly_indices = 16 * 1024; // change this size if needed!!!
	ERR_FAIL_COND(p_vertex_count > _max_draw_poly_indices);
	static uint16_t _draw_poly_indices[_max_draw_poly_indices];
	for (int i = 0; i < p_vertex_count; i++) {
		_draw_poly_indices[i] = p_indices[i];
		//OS::get_singleton()->print("ind: %d ", p_indices[i]);
	};

	//copy the data to GPU.
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, p_vertex_count * sizeof(uint16_t), &_draw_poly_indices[0]);

	//draw the triangles.
	glDrawElements(GL_TRIANGLES, p_vertex_count, GL_UNSIGNED_SHORT, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif

	storage->frame.canvas_draw_commands++;
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

	glBindBuffer(GL_ARRAY_BUFFER, data.primitive_quad_buffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, p_points * stride * 4, &b[0]);
	glBindVertexArray(data.primitive_quad_buffer_arrays[version]);
	glDrawArrays(prim[p_points], 0, p_points);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	storage->frame.canvas_draw_commands++;
}

void RasterizerCanvasGLES3::_canvas_item_render_commands(Item *p_item, Item *current_clip, bool &reclip) {

	int cc = p_item->commands.size();
	Item::Command **commands = p_item->commands.ptr();

	for (int i = 0; i < cc; i++) {

		Item::Command *c = commands[i];

		switch (c->type) {
			case Item::Command::TYPE_LINE: {

				Item::CommandLine *line = static_cast<Item::CommandLine *>(c);
				_set_texture_rect_mode(false);

				_bind_canvas_texture(RID());

				glVertexAttrib4f(VS::ARRAY_COLOR, line->color.r, line->color.g, line->color.b, line->color.a);

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

			} break;
			case Item::Command::TYPE_RECT: {

				Item::CommandRect *rect = static_cast<Item::CommandRect *>(c);

				_set_texture_rect_mode(true);

				//set color
				glVertexAttrib4f(VS::ARRAY_COLOR, rect->modulate.r, rect->modulate.g, rect->modulate.b, rect->modulate.a);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(rect->texture);

				if (texture) {

					bool untile = false;

					if (rect->flags & CANVAS_RECT_TILE && !(texture->flags & VS::TEXTURE_FLAG_REPEAT)) {
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
						untile = true;
					}

					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					Rect2 src_rect = (rect->flags & CANVAS_RECT_REGION) ? Rect2(rect->source.pos * texpixel_size, rect->source.size * texpixel_size) : Rect2(0, 0, 1, 1);

					if (rect->flags & CANVAS_RECT_FLIP_H) {
						src_rect.size.x *= -1;
					}

					if (rect->flags & CANVAS_RECT_FLIP_V) {
						src_rect.size.y *= -1;
					}

					if (rect->flags & CANVAS_RECT_TRANSPOSE) {
						//err..
					}

					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);

					glVertexAttrib4f(1, rect->rect.pos.x, rect->rect.pos.y, rect->rect.size.x, rect->rect.size.y);
					glVertexAttrib4f(2, src_rect.pos.x, src_rect.pos.y, src_rect.size.x, src_rect.size.y);
					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

					if (untile) {
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
						glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
					}

				} else {

					glVertexAttrib4f(1, rect->rect.pos.x, rect->rect.pos.y, rect->rect.size.x, rect->rect.size.y);
					glVertexAttrib4f(2, 0, 0, 1, 1);
					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
				}

				storage->frame.canvas_draw_commands++;

			} break;

			case Item::Command::TYPE_NINEPATCH: {

				Item::CommandNinePatch *np = static_cast<Item::CommandNinePatch *>(c);

				_set_texture_rect_mode(true);

				glVertexAttrib4f(VS::ARRAY_COLOR, np->color.r, np->color.g, np->color.b, np->color.a);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(np->texture);

				if (!texture) {

					glVertexAttrib4f(1, np->rect.pos.x, np->rect.pos.y, np->rect.size.x, np->rect.size.y);
					glVertexAttrib4f(2, 0, 0, 1, 1);
					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
					continue;
				}

				Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);

				state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);

#define DSTRECT(m_x, m_y, m_w, m_h) glVertexAttrib4f(1, m_x, m_y, m_w, m_h)
#define SRCRECT(m_x, m_y, m_w, m_h) glVertexAttrib4f(2, (m_x)*texpixel_size.x, (m_y)*texpixel_size.y, (m_w)*texpixel_size.x, (m_h)*texpixel_size.y)

				//top left
				DSTRECT(np->rect.pos.x, np->rect.pos.y, np->margin[MARGIN_LEFT], np->margin[MARGIN_TOP]);
				SRCRECT(np->source.pos.x, np->source.pos.y, np->margin[MARGIN_LEFT], np->margin[MARGIN_TOP]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//top right
				DSTRECT(np->rect.pos.x + np->rect.size.width - np->margin[MARGIN_RIGHT], np->rect.pos.y, np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP]);
				SRCRECT(np->source.pos.x + np->source.size.width - np->margin[MARGIN_RIGHT], np->source.pos.y, np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//bottom right
				DSTRECT(np->rect.pos.x + np->rect.size.width - np->margin[MARGIN_RIGHT], np->rect.pos.y + np->rect.size.height - np->margin[MARGIN_BOTTOM], np->margin[MARGIN_RIGHT], np->margin[MARGIN_BOTTOM]);
				SRCRECT(np->source.pos.x + np->source.size.width - np->margin[MARGIN_RIGHT], np->source.pos.y + np->source.size.height - np->margin[MARGIN_BOTTOM], np->margin[MARGIN_RIGHT], np->margin[MARGIN_BOTTOM]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//bottom left
				DSTRECT(np->rect.pos.x, np->rect.pos.y + np->rect.size.height - np->margin[MARGIN_BOTTOM], np->margin[MARGIN_LEFT], np->margin[MARGIN_BOTTOM]);
				SRCRECT(np->source.pos.x, np->source.pos.y + np->source.size.height - np->margin[MARGIN_BOTTOM], np->margin[MARGIN_LEFT], np->margin[MARGIN_BOTTOM]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//top
				DSTRECT(np->rect.pos.x + np->margin[MARGIN_LEFT], np->rect.pos.y, np->rect.size.width - np->margin[MARGIN_LEFT] - np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP]);
				SRCRECT(np->source.pos.x + np->margin[MARGIN_LEFT], np->source.pos.y, np->source.size.width - np->margin[MARGIN_LEFT] - np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//bottom
				DSTRECT(np->rect.pos.x + np->margin[MARGIN_LEFT], np->rect.pos.y + np->rect.size.height - np->margin[MARGIN_BOTTOM], np->rect.size.width - np->margin[MARGIN_LEFT] - np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP]);
				SRCRECT(np->source.pos.x + np->margin[MARGIN_LEFT], np->source.pos.y + np->source.size.height - np->margin[MARGIN_BOTTOM], np->source.size.width - np->margin[MARGIN_LEFT] - np->margin[MARGIN_LEFT], np->margin[MARGIN_TOP]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//left
				DSTRECT(np->rect.pos.x, np->rect.pos.y + np->margin[MARGIN_TOP], np->margin[MARGIN_LEFT], np->rect.size.height - np->margin[MARGIN_TOP] - np->margin[MARGIN_BOTTOM]);
				SRCRECT(np->source.pos.x, np->source.pos.y + np->margin[MARGIN_TOP], np->margin[MARGIN_LEFT], np->source.size.height - np->margin[MARGIN_TOP] - np->margin[MARGIN_BOTTOM]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				//right
				DSTRECT(np->rect.pos.x + np->rect.size.width - np->margin[MARGIN_RIGHT], np->rect.pos.y + np->margin[MARGIN_TOP], np->margin[MARGIN_RIGHT], np->rect.size.height - np->margin[MARGIN_TOP] - np->margin[MARGIN_BOTTOM]);
				SRCRECT(np->source.pos.x + np->source.size.width - np->margin[MARGIN_RIGHT], np->source.pos.y + np->margin[MARGIN_TOP], np->margin[MARGIN_RIGHT], np->source.size.height - np->margin[MARGIN_TOP] - np->margin[MARGIN_BOTTOM]);
				glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

				if (np->draw_center) {

					//center
					DSTRECT(np->rect.pos.x + np->margin[MARGIN_LEFT], np->rect.pos.y + np->margin[MARGIN_TOP], np->rect.size.x - np->margin[MARGIN_LEFT] - np->margin[MARGIN_RIGHT], np->rect.size.height - np->margin[MARGIN_TOP] - np->margin[MARGIN_BOTTOM]);
					SRCRECT(np->source.pos.x + np->margin[MARGIN_LEFT], np->source.pos.y + np->margin[MARGIN_TOP], np->source.size.x - np->margin[MARGIN_LEFT] - np->margin[MARGIN_RIGHT], np->source.size.height - np->margin[MARGIN_TOP] - np->margin[MARGIN_BOTTOM]);
					glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
				}

#undef SRCRECT
#undef DSTRECT

				storage->frame.canvas_draw_commands++;
			} break;

			case Item::Command::TYPE_PRIMITIVE: {

				Item::CommandPrimitive *primitive = static_cast<Item::CommandPrimitive *>(c);
				_set_texture_rect_mode(false);

				ERR_CONTINUE(primitive->points.size() < 1);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(primitive->texture);

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

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(polygon->texture);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				}
				//_draw_polygon(polygon->count,polygon->indices.ptr(),polygon->points.ptr(),polygon->uvs.ptr(),polygon->colors.ptr(),polygon->texture,polygon->colors.size()==1);

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

							int x = current_clip->final_clip_rect.pos.x;
							int y = storage->frame.current_rt->height - (current_clip->final_clip_rect.pos.y + current_clip->final_clip_rect.size.y);
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

#if 0
void RasterizerGLES2::_canvas_item_setup_shader_params(ShaderMaterial *material,Shader* shader) {

	if (canvas_shader.bind())
		rebind_texpixel_size=true;

	if (material->shader_version!=shader->version) {
		//todo optimize uniforms
		material->shader_version=shader->version;
	}

	if (shader->has_texscreen && framebuffer.active) {

		int x = viewport.x;
		int y = window_size.height-(viewport.height+viewport.y);

		canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_SCREEN_MULT,Vector2(float(viewport.width)/framebuffer.width,float(viewport.height)/framebuffer.height));
		canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_SCREEN_CLAMP,Color(float(x)/framebuffer.width,float(y)/framebuffer.height,float(x+viewport.width)/framebuffer.width,float(y+viewport.height)/framebuffer.height));
		canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_TEX,max_texture_units-1);
		glActiveTexture(GL_TEXTURE0+max_texture_units-1);
		glBindTexture(GL_TEXTURE_2D,framebuffer.sample_color);
		if (framebuffer.scale==1 && !canvas_texscreen_used) {
#ifdef GLEW_ENABLED
			if (current_rt) {
				glReadBuffer(GL_COLOR_ATTACHMENT0);
			} else {
				glReadBuffer(GL_BACK);
			}
#endif
			if (current_rt) {
				glCopyTexSubImage2D(GL_TEXTURE_2D,0,viewport.x,viewport.y,viewport.x,viewport.y,viewport.width,viewport.height);
				canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_SCREEN_CLAMP,Color(float(x)/framebuffer.width,float(viewport.y)/framebuffer.height,float(x+viewport.width)/framebuffer.width,float(y+viewport.height)/framebuffer.height));
				//window_size.height-(viewport.height+viewport.y)
			} else {
				glCopyTexSubImage2D(GL_TEXTURE_2D,0,x,y,x,y,viewport.width,viewport.height);
			}

			canvas_texscreen_used=true;
		}

		glActiveTexture(GL_TEXTURE0);

	}

	if (shader->has_screen_uv) {
		canvas_shader.set_uniform(CanvasShaderGLES2::SCREEN_UV_MULT,Vector2(1.0/viewport.width,1.0/viewport.height));
	}


	uses_texpixel_size=shader->uses_texpixel_size;

}

#endif

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
				glScissor(current_clip->final_clip_rect.pos.x, (rt_size.height - (current_clip->final_clip_rect.pos.y + current_clip->final_clip_rect.size.height)), current_clip->final_clip_rect.size.width, current_clip->final_clip_rect.size.height);

			} else {

				glDisable(GL_SCISSOR_TEST);
			}
		}
#if 0
		if (ci->copy_back_buffer && framebuffer.active && framebuffer.scale==1) {

			Rect2 rect;
			int x,y;

			if (ci->copy_back_buffer->full) {

				x = viewport.x;
				y = window_size.height-(viewport.height+viewport.y);
			} else {
				x = viewport.x+ci->copy_back_buffer->screen_rect.pos.x;
				y = window_size.height-(viewport.y+ci->copy_back_buffer->screen_rect.pos.y+ci->copy_back_buffer->screen_rect.size.y);
			}
			glActiveTexture(GL_TEXTURE0+max_texture_units-1);
			glBindTexture(GL_TEXTURE_2D,framebuffer.sample_color);

#ifdef GLEW_ENABLED
			if (current_rt) {
				glReadBuffer(GL_COLOR_ATTACHMENT0);
			} else {
				glReadBuffer(GL_BACK);
			}
#endif
			if (current_rt) {
				glCopyTexSubImage2D(GL_TEXTURE_2D,0,viewport.x,viewport.y,viewport.x,viewport.y,viewport.width,viewport.height);
				//window_size.height-(viewport.height+viewport.y)
			} else {
				glCopyTexSubImage2D(GL_TEXTURE_2D,0,x,y,x,y,viewport.width,viewport.height);
			}

			canvas_texscreen_used=true;
			glActiveTexture(GL_TEXTURE0);

		}

#endif

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

				state.canvas_shader.set_custom_shader(shader_ptr->custom_code_id);
				state.canvas_shader.bind();

				if (material_ptr->ubo_id) {
					glBindBufferBase(GL_UNIFORM_BUFFER, 2, material_ptr->ubo_id);
				}

				int tc = material_ptr->textures.size();
				RID *textures = material_ptr->textures.ptr();
				ShaderLanguage::ShaderNode::Uniform::Hint *texture_hints = shader_ptr->texture_hints.ptr();

				for (int i = 0; i < tc; i++) {

					glActiveTexture(GL_TEXTURE1 + i);

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

		state.canvas_item_modulate = unshaded ? ci->final_modulate : Color(
																			 ci->final_modulate.r * p_modulate.r,
																			 ci->final_modulate.g * p_modulate.g,
																			 ci->final_modulate.b * p_modulate.b,
																			 ci->final_modulate.a * p_modulate.a);

		state.final_transform = ci->final_transform;
		state.extra_matrix = Transform2D();

		state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.canvas_item_modulate);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);
		state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.extra_matrix);

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
			glScissor(current_clip->final_clip_rect.pos.x, (rt_size.height - (current_clip->final_clip_rect.pos.y + current_clip->final_clip_rect.size.height)), current_clip->final_clip_rect.size.width, current_clip->final_clip_rect.size.height);
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

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
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
			glBindBuffer(GL_ARRAY_BUFFER, cc->vertex_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cc->index_id);
			glVertexAttribPointer(VS::ARRAY_VERTEX, 3, GL_FLOAT, false, 0, 0);
			glDrawElements(GL_TRIANGLES, cc->len * 3, GL_UNSIGNED_SHORT, 0);

			instance = instance->next;
		}
	}

	glDisableVertexAttribArray(VS::ARRAY_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
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
	for (int i = 0; i < VS::ARRAY_MAX; i++) {
		glDisableVertexAttribArray(i);
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
	for (int i = 0; i < 4; i++) {
		state.canvas_item_ubo_data.time[i] = storage->frame.time[i];
	}

	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_item_ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CanvasItemUBO), &state.canvas_item_ubo_data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.canvas_texscreen_used = false;
}

void RasterizerCanvasGLES3::draw_generic_textured_rect(const Rect2 &p_rect, const Rect2 &p_src) {

	glVertexAttrib4f(1, p_rect.pos.x, p_rect.pos.y, p_rect.size.x, p_rect.size.y);
	glVertexAttrib4f(2, p_src.pos.x, p_src.pos.y, p_src.size.x, p_src.size.y);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
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

		glGenBuffers(1, &data.primitive_quad_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, data.primitive_quad_buffer);
		glBufferData(GL_ARRAY_BUFFER, (2 + 2 + 4) * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW); //allocate max size
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		for (int i = 0; i < 4; i++) {
			glGenVertexArrays(1, &data.primitive_quad_buffer_arrays[i]);
			glBindVertexArray(data.primitive_quad_buffer_arrays[i]);
			glBindBuffer(GL_ARRAY_BUFFER, data.primitive_quad_buffer);

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
	}

	store_transform(Transform(), state.canvas_item_ubo_data.projection_matrix);

	glGenBuffers(1, &state.canvas_item_ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_item_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(CanvasItemUBO), &state.canvas_item_ubo_data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.canvas_shader.init();
	state.canvas_shader.set_base_material_tex_index(1);
	state.canvas_shadow_shader.init();

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);
	state.canvas_shadow_shader.set_conditional(CanvasShadowShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);
}

void RasterizerCanvasGLES3::finalize() {

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);
}

RasterizerCanvasGLES3::RasterizerCanvasGLES3() {
}

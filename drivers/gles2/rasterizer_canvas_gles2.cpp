/*************************************************************************/
/*  rasterizer_canvas_gles2.cpp                                          */
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
#include "rasterizer_canvas_gles2.h"
#include "os/os.h"
#include "project_settings.h"
#include "rasterizer_scene_gles2.h"
#include "servers/visual/visual_server_raster.h"
#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif

RID RasterizerCanvasGLES2::light_internal_create() {

	return RID();
}

void RasterizerCanvasGLES2::light_internal_update(RID p_rid, Light *p_light) {
}

void RasterizerCanvasGLES2::light_internal_free(RID p_rid) {
}

void RasterizerCanvasGLES2::_set_uniforms() {

	state.canvas_shader.set_uniform(CanvasShaderGLES2::PROJECTION_MATRIX, state.uniforms.projection_matrix);
	state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);
	state.canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, state.uniforms.extra_matrix);

	state.canvas_shader.set_uniform(CanvasShaderGLES2::FINAL_MODULATE, state.uniforms.final_modulate);
}

void RasterizerCanvasGLES2::canvas_begin() {

	if (storage->frame.clear_request) {
		glClearColor(storage->frame.clear_request_color.r,
				storage->frame.clear_request_color.g,
				storage->frame.clear_request_color.b,
				storage->frame.clear_request_color.a);
		glClear(GL_COLOR_BUFFER_BIT);
		storage->frame.clear_request = false;
	}

	if (storage->frame.current_rt) {
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);
		glColorMask(1, 1, 1, 1);
	}

	reset_canvas();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
	glDisableVertexAttribArray(VS::ARRAY_COLOR);

	// set up default uniforms

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

	state.uniforms.projection_matrix = canvas_transform;

	state.uniforms.final_modulate = Color(1, 1, 1, 1);

	state.uniforms.modelview_matrix = Transform2D();
	state.uniforms.extra_matrix = Transform2D();

	_set_uniforms();
	_bind_quad_buffer();
}

void RasterizerCanvasGLES2::canvas_end() {

	_flush();

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	for (int i = 0; i < VS::ARRAY_MAX; i++) {
		glDisableVertexAttribArray(i);
	}

	state.using_texture_rect = false;
	state.using_ninepatch = false;
}

RasterizerStorageGLES2::Texture *RasterizerCanvasGLES2::_bind_canvas_texture(const RID &p_texture, const RID &p_normal_map) {

	RasterizerStorageGLES2::Texture *tex_return = NULL;

	if (p_texture.is_valid()) {

		RasterizerStorageGLES2::Texture *texture = storage->texture_owner.getornull(p_texture);

		if (!texture) {
			state.current_tex = RID();
			state.current_tex_ptr = NULL;

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

		} else {

			texture = texture->get_ptr();

			if (texture->redraw_if_visible) {
				VisualServerRaster::redraw_request();
			}

			if (texture->render_target) {
				texture->render_target->used_in_frame = true;
			}

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture->tex_id);

			state.current_tex = p_texture;
			state.current_tex_ptr = texture;

			tex_return = texture;
		}
	} else {
		state.current_tex = RID();
		state.current_tex_ptr = NULL;

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
	}

	return tex_return;
}

void RasterizerCanvasGLES2::_set_texture_rect_mode(bool p_enable, bool p_ninepatch) {
}

bool RasterizerCanvasGLES2::_prepare_ninepatch(RasterizerCanvas::Item::CommandNinePatch *np) {
	float buffer[16 * 2 + 16 * 2];

	if (data.polygon_buffer_size - data.polygon_mem_offset < sizeof(buffer))
		return false;

	RasterizerStorageGLES2::Texture *tex;

	if (np->texture.is_valid())
		tex = storage->texture_owner.getornull(np->texture);
	else
		tex = NULL;

	if (!tex) {
		print_line("TODO: ninepatch without texture");
		return true;
	}

	RenderItem::NinepatchCommand *cmd = render_commands.allocate<RenderItem::NinepatchCommand>();
	if (!cmd)
		return false;

	Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);

	{

		// first row

		buffer[(0 * 4 * 4) + 0] = np->rect.position.x;
		buffer[(0 * 4 * 4) + 1] = np->rect.position.y;

		buffer[(0 * 4 * 4) + 2] = np->source.position.x * texpixel_size.x;
		buffer[(0 * 4 * 4) + 3] = np->source.position.y * texpixel_size.y;

		buffer[(0 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT];
		buffer[(0 * 4 * 4) + 5] = np->rect.position.y;

		buffer[(0 * 4 * 4) + 6] = (np->source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
		buffer[(0 * 4 * 4) + 7] = np->source.position.y * texpixel_size.y;

		buffer[(0 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT];
		buffer[(0 * 4 * 4) + 9] = np->rect.position.y;

		buffer[(0 * 4 * 4) + 10] = (np->source.position.x + np->source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
		buffer[(0 * 4 * 4) + 11] = np->source.position.y * texpixel_size.y;

		buffer[(0 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
		buffer[(0 * 4 * 4) + 13] = np->rect.position.y;

		buffer[(0 * 4 * 4) + 14] = (np->source.position.x + np->source.size.x) * texpixel_size.x;
		buffer[(0 * 4 * 4) + 15] = np->source.position.y * texpixel_size.y;

		// second row

		buffer[(1 * 4 * 4) + 0] = np->rect.position.x;
		buffer[(1 * 4 * 4) + 1] = np->rect.position.y + np->margin[MARGIN_TOP];

		buffer[(1 * 4 * 4) + 2] = np->source.position.x * texpixel_size.x;
		buffer[(1 * 4 * 4) + 3] = (np->source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

		buffer[(1 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT];
		buffer[(1 * 4 * 4) + 5] = np->rect.position.y + np->margin[MARGIN_TOP];

		buffer[(1 * 4 * 4) + 6] = (np->source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
		buffer[(1 * 4 * 4) + 7] = (np->source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

		buffer[(1 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT];
		buffer[(1 * 4 * 4) + 9] = np->rect.position.y + np->margin[MARGIN_TOP];

		buffer[(1 * 4 * 4) + 10] = (np->source.position.x + np->source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
		buffer[(1 * 4 * 4) + 11] = (np->source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

		buffer[(1 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
		buffer[(1 * 4 * 4) + 13] = np->rect.position.y + np->margin[MARGIN_TOP];

		buffer[(1 * 4 * 4) + 14] = (np->source.position.x + np->source.size.x) * texpixel_size.x;
		buffer[(1 * 4 * 4) + 15] = (np->source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

		// thrid row

		buffer[(2 * 4 * 4) + 0] = np->rect.position.x;
		buffer[(2 * 4 * 4) + 1] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM];

		buffer[(2 * 4 * 4) + 2] = np->source.position.x * texpixel_size.x;
		buffer[(2 * 4 * 4) + 3] = (np->source.position.y + np->source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

		buffer[(2 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT];
		buffer[(2 * 4 * 4) + 5] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM];

		buffer[(2 * 4 * 4) + 6] = (np->source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
		buffer[(2 * 4 * 4) + 7] = (np->source.position.y + np->source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

		buffer[(2 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT];
		buffer[(2 * 4 * 4) + 9] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM];

		buffer[(2 * 4 * 4) + 10] = (np->source.position.x + np->source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
		buffer[(2 * 4 * 4) + 11] = (np->source.position.y + np->source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

		buffer[(2 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
		buffer[(2 * 4 * 4) + 13] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM];

		buffer[(2 * 4 * 4) + 14] = (np->source.position.x + np->source.size.x) * texpixel_size.x;
		buffer[(2 * 4 * 4) + 15] = (np->source.position.y + np->source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

		// fourth row

		buffer[(3 * 4 * 4) + 0] = np->rect.position.x;
		buffer[(3 * 4 * 4) + 1] = np->rect.position.y + np->rect.size.y;

		buffer[(3 * 4 * 4) + 2] = np->source.position.x * texpixel_size.x;
		buffer[(3 * 4 * 4) + 3] = (np->source.position.y + np->source.size.y) * texpixel_size.y;

		buffer[(3 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT];
		buffer[(3 * 4 * 4) + 5] = np->rect.position.y + np->rect.size.y;

		buffer[(3 * 4 * 4) + 6] = (np->source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
		buffer[(3 * 4 * 4) + 7] = (np->source.position.y + np->source.size.y) * texpixel_size.y;

		buffer[(3 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT];
		buffer[(3 * 4 * 4) + 9] = np->rect.position.y + np->rect.size.y;

		buffer[(3 * 4 * 4) + 10] = (np->source.position.x + np->source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
		buffer[(3 * 4 * 4) + 11] = (np->source.position.y + np->source.size.y) * texpixel_size.y;

		buffer[(3 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
		buffer[(3 * 4 * 4) + 13] = np->rect.position.y + np->rect.size.y;

		buffer[(3 * 4 * 4) + 14] = (np->source.position.x + np->source.size.x) * texpixel_size.x;
		buffer[(3 * 4 * 4) + 15] = (np->source.position.y + np->source.size.y) * texpixel_size.y;

		// print_line(String::num((np->source.position.y + np->source.size.y) * texpixel_size.y));
	}

	cmd->command = np;
	cmd->vertices_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, buffer, sizeof(buffer));
	data.polygon_mem_offset += sizeof(buffer);

	return true;
}

bool RasterizerCanvasGLES2::_prepare_polygon(RasterizerCanvas::Item::Command *command, const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {
	const uint32_t indices_size = sizeof(int) * p_index_count;
	if (indices_size > data.polygon_index_buffer_size - data.polygon_index_mem_offset)
		return false;

	const uint32_t vertices_size = sizeof(Vector2) * p_vertex_count;

	uint32_t colors_size;

	const bool use_colors = !p_singlecolor && p_colors;

	if (use_colors)
		colors_size = sizeof(Color) * p_vertex_count;
	else
		colors_size = 0;

	uint32_t uvs_size;

	if (p_uvs)
		uvs_size = sizeof(Vector2) * p_vertex_count;
	else
		uvs_size = 0;

	if (vertices_size + colors_size + uvs_size > data.polygon_buffer_size - data.polygon_mem_offset)
		return false;

	RenderItem::PolygonCommand *cmd = render_commands.allocate<RenderItem::PolygonCommand>();
	if (!cmd)
		return false;

	cmd->command = command;

	cmd->count = p_index_count;
	cmd->color = p_colors ? *p_colors : Color(1, 1, 1, 1);

	cmd->vertices_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, p_vertices, vertices_size);
	data.polygon_mem_offset += vertices_size;

	cmd->use_colors = use_colors;
	cmd->colors_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, p_colors, colors_size);
	data.polygon_mem_offset += colors_size;

	cmd->use_uvs = p_uvs;
	cmd->uvs_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, p_uvs, uvs_size);
	data.polygon_mem_offset += uvs_size;

	cmd->indices_offset = data.polygon_index_mem_offset;

	memcpy(data.polygon_index_mem_buffer + data.polygon_index_mem_offset, p_indices, indices_size);
	data.polygon_index_mem_offset += indices_size;

	return true;
}

void RasterizerCanvasGLES2::_flush_rects(const RenderItem::RectsCommand *rect) {
	if (rect->use_texture) {
		RasterizerStorageGLES2::Texture *tex = _bind_canvas_texture(rect->texture, RID());

		if (tex) {
			Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);
			state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);
		}
	}
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, true);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, false);

	if (state.canvas_shader.bind())
		_set_uniforms();

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	glEnableVertexAttribArray(VS::ARRAY_COLOR);
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
	glEnableVertexAttribArray(VS::ARRAY_TEX_UV2);

	const uint32_t qv_size = sizeof(float) * 2;
	const uint32_t rect_size = sizeof(float) * 4;
	const uint32_t colors_size = sizeof(float) * 4;
	const uint32_t stride = qv_size + colors_size + rect_size * 2;

	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, stride, (uint8_t *)0 + rect->vertices_offset);
	glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, stride, ((uint8_t *)0) + rect->vertices_offset + qv_size);
	glVertexAttribPointer(VS::ARRAY_TEX_UV, 4, GL_FLOAT, GL_FALSE, stride, (uint8_t *)0 + rect->vertices_offset + qv_size + colors_size);
	glVertexAttribPointer(VS::ARRAY_TEX_UV2, 4, GL_FLOAT, GL_FALSE, stride, (uint8_t *)0 + rect->vertices_offset + qv_size + colors_size + rect_size);

	if (rect->untile) {
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}

	glDrawArrays(GL_TRIANGLES, 0, rect->count);
	++storage->info.render.draw_call_count;

	if (rect->untile) {
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES2::_flush_polygon(const RenderItem::PolygonCommand *cmd) {
	switch (cmd->command->type) {
		case Item::Command::TYPE_POLYGON: {
			Item::CommandPolygon *polygon = static_cast<Item::CommandPolygon *>(cmd->command);

			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, true);

			if (state.canvas_shader.bind())
				_set_uniforms();

			RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(polygon->texture, polygon->normal_map);

			if (texture) {
				Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
				state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);
			}
		} break;
		case Item::Command::TYPE_CIRCLE: {
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, false);

			if (state.canvas_shader.bind())
				_set_uniforms();

			_bind_canvas_texture(RID(), RID());
		} break;
	}

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), ((uint8_t *)0) + cmd->vertices_offset);

	if (cmd->use_colors) {
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Color), ((uint8_t *)0) + cmd->colors_offset);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttrib4f(VS::ARRAY_COLOR, cmd->color.r, cmd->color.g, cmd->color.b, cmd->color.a);
	}

	if (cmd->use_uvs) {
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), ((uint8_t *)0) + cmd->uvs_offset);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);

	glDrawElements(GL_TRIANGLES, cmd->count, GL_UNSIGNED_INT, ((uint8_t *)0) + cmd->indices_offset);
	++storage->info.render.draw_call_count;

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

bool RasterizerCanvasGLES2::_prepare_generic(GLuint p_primitive, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {
	const uint32_t vertices_size = sizeof(Vector2) * p_vertex_count;

	uint32_t colors_size;

	const bool use_colors = !p_singlecolor && p_colors;

	if (use_colors)
		colors_size = sizeof(Color) * p_vertex_count;
	else
		colors_size = 0;

	uint32_t uvs_size;

	if (p_uvs)
		uvs_size = sizeof(Vector2) * p_vertex_count;
	else
		uvs_size = 0;

	if (vertices_size + colors_size + uvs_size > data.polygon_buffer_size - data.polygon_mem_offset)
		return false;

	RenderItem::GenericCommand *cmd = render_commands.allocate<RenderItem::GenericCommand>();
	if (!cmd)
		return false;

	cmd->primitive = p_primitive;
	cmd->count = p_vertex_count;

	cmd->color = p_colors ? *p_colors : Color(1, 1, 1, 1);

	cmd->vertices_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, p_vertices, vertices_size);
	data.polygon_mem_offset += vertices_size;

	cmd->use_colors = use_colors;
	cmd->colors_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, p_colors, colors_size);
	data.polygon_mem_offset += colors_size;

	cmd->use_uvs = p_uvs;
	cmd->uvs_offset = data.polygon_mem_offset;

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, p_uvs, uvs_size);
	data.polygon_mem_offset += uvs_size;

	return true;
}

void RasterizerCanvasGLES2::_flush_generic(const RenderItem::GenericCommand *cmd) {

	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, false);

	if (state.canvas_shader.bind())
		_set_uniforms();

	_bind_canvas_texture(RID(), RID());

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), ((uint8_t *)0) + cmd->vertices_offset);

	if (cmd->use_colors) {
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Color), ((uint8_t *)0) + cmd->colors_offset);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttrib4f(VS::ARRAY_COLOR, cmd->color.r, cmd->color.g, cmd->color.b, cmd->color.a);
	}

	if (cmd->use_uvs) {
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), ((uint8_t *)0) + cmd->uvs_offset);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glDrawArrays(cmd->primitive, 0, cmd->count);
	++storage->info.render.draw_call_count;

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES2::_flush_ninepatch(const RenderItem::NinepatchCommand *cmd) {
	Item::CommandNinePatch *np = cmd->command;

	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, true);

	if (state.canvas_shader.bind())
		_set_uniforms();

	glDisableVertexAttribArray(VS::ARRAY_COLOR);
	glVertexAttrib4fv(VS::ARRAY_COLOR, np->color.components);

	RasterizerStorageGLES2::Texture *tex = _bind_canvas_texture(np->texture, np->normal_map);

	if (!tex) {
		print_line("TODO: ninepatch without texture");
		return;
	}

	Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);

	// state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);
	state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.ninepatch_elements);

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glEnableVertexAttribArray(VS::ARRAY_TEX_UV);

	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (uint8_t *)0 + cmd->vertices_offset);
	glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (uint8_t *)0 + cmd->vertices_offset + sizeof(float) * 2);

	glDrawElements(GL_TRIANGLES, 18 * 3 - (np->draw_center ? 0 : 6), GL_UNSIGNED_BYTE, NULL);
	++storage->info.render.draw_call_count;

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

bool RasterizerCanvasGLES2::_prepare_gui_primitive(RasterizerCanvas::Item::Command *command, int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs) {
	int color_offset = 0;
	int uv_offset = 0;
	int stride = 2;

	if (p_colors) {
		color_offset = stride;
		stride += 4;
	}

	if (p_uvs) {
		uv_offset = stride;
		stride += 2;
	}

	const uint32_t size = p_points * stride * 4 * sizeof(float);

	if (size > data.polygon_buffer_size - data.polygon_mem_offset)
		return false;

	RenderItem::GuiPrimitiveCommand *cmd = render_commands.allocate<RenderItem::GuiPrimitiveCommand>();
	if (!cmd)
		return false;

	cmd->command = command;

	cmd->count = p_points;

	cmd->offset = data.polygon_mem_offset;

	cmd->use_colors = p_colors;
	cmd->color_offset = data.polygon_mem_offset + color_offset * sizeof(float);

	cmd->use_uvs = p_uvs;
	cmd->uvs_offset = data.polygon_mem_offset + uv_offset * sizeof(float);
	cmd->stride = stride * sizeof(float);

	float buffer_data[(2 + 2 + 4) * 4];

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

	memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, buffer_data, size);
	data.polygon_mem_offset += size;

	return true;
}

void RasterizerCanvasGLES2::_flush_gui_primitive(const RenderItem::GuiPrimitiveCommand *cmd) {

	static const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

	switch (cmd->command->type) {
		case Item::Command::TYPE_LINE: {
			Item::CommandLine *line = static_cast<Item::CommandLine *>(cmd->command);

			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, false);
			state.canvas_shader.bind();

			_set_uniforms();

			_bind_canvas_texture(RID(), RID());

			glDisableVertexAttribArray(VS::ARRAY_COLOR);
			glVertexAttrib4fv(VS::ARRAY_COLOR, line->color.components);

			state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);
		} break;
		case Item::Command::TYPE_PRIMITIVE: {
			Item::CommandPrimitive *primitive = static_cast<Item::CommandPrimitive *>(cmd->command);

			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_UV_ATTRIBUTE, true);

			if (state.canvas_shader.bind())
				_set_uniforms();

			ERR_BREAK(primitive->points.size() < 1);

			RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(primitive->texture, primitive->normal_map);

			if (texture) {
				Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
				state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);
			}

			if (primitive->colors.size() == 1 && primitive->points.size() > 1) {
				Color c = primitive->colors[0];
				glVertexAttrib4f(VS::ARRAY_COLOR, c.r, c.g, c.b, c.a);
			} else if (primitive->colors.empty()) {
				glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
			}
		} break;
	}

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);

	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, cmd->stride, (uint8_t *)0 + cmd->offset);

	if (cmd->use_colors) {
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, cmd->stride, (uint8_t *)0 + cmd->color_offset);
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (cmd->use_uvs) {
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, cmd->stride, (uint8_t *)0 + cmd->uvs_offset);
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glDrawArrays(prim[cmd->count], 0, cmd->count);
	++storage->info.render.draw_call_count;

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES2::_canvas_render_command(Item::Command *command, Item *current_clip, bool &reclip) {
	switch (command->type) {
		case Item::Command::TYPE_TRANSFORM: {
			Item::CommandTransform *transform = static_cast<Item::CommandTransform *>(command);
			state.uniforms.extra_matrix = transform->xform;
			state.canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, state.uniforms.extra_matrix);
		} break;

		case Item::Command::TYPE_PARTICLES: {

		} break;

		case Item::Command::TYPE_CLIP_IGNORE: {

			Item::CommandClipIgnore *ci = static_cast<Item::CommandClipIgnore *>(command);
			if (current_clip) {
				if (ci->ignore != reclip) {
					if (ci->ignore) {
						glDisable(GL_SCISSOR_TEST);
						reclip = true;
					} else {
						glEnable(GL_SCISSOR_TEST);

						int x = current_clip->final_clip_rect.position.x;
						int y = storage->frame.current_rt->height - (current_clip->final_clip_rect.position.y + current_clip->final_clip_rect.size.y);
						int w = current_clip->final_clip_rect.size.x;
						int h = current_clip->final_clip_rect.size.y;

						if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
							y = current_clip->final_clip_rect.position.y;

						glScissor(x, y, w, h);

						reclip = false;
					}
				}
			}

		} break;

		default: {
			print_line("other");
		} break;
	}
}

void RasterizerCanvasGLES2::_copy_texscreen(const Rect2 &p_rect) {

	// This isn't really working yet, so disabling for now.

	/*
	glDisable(GL_BLEND);

	state.canvas_texscreen_used = true;

	Vector2 wh(storage->frame.current_rt->width, storage->frame.current_rt->height);
	Color copy_section(p_rect.position.x / wh.x, p_rect.position.y / wh.y, p_rect.size.x / wh.x, p_rect.size.y / wh.y);

	if (p_rect != Rect2()) {
		// only use section

		storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_COPY_SECTION, true);
	}


	storage->shaders.copy.bind();
	storage->shaders.copy.set_uniform(CopyShaderGLES2::COPY_SECTION, copy_section);

	_bind_quad_buffer();

	glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->copy_screen_effect.fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->color);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDisableVertexAttribArray(VS::ARRAY_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);

	state.canvas_shader.bind();
	_bind_canvas_texture(state.current_tex, state.current_normal);

	glEnable(GL_BLEND);
	*/
}

void RasterizerCanvasGLES2::_canvas_item_process(Item *p_item) {

	Item::Command **commands = p_item->commands.ptrw();
	bool prepared = true;

	int i = 0;
	const int size = p_item->commands.size();

	while (i < size) {
		Item::Command *command = commands[i];

		switch (command->type) {
			case Item::Command::TYPE_RECT: {
				Item::CommandRect *r = static_cast<Item::CommandRect *>(command);

				RasterizerStorageGLES2::Texture *texture = storage->texture_owner.getornull(r->texture);

				Size2 texpixel_size;
				if (texture)
					texpixel_size = Size2(1.0 / texture->width, 1.0 / texture->height);

				RenderItem::RectsCommand *rcmd = push_command<RenderItem::RectsCommand>();

				rcmd->vertices_offset = data.polygon_mem_offset;
				rcmd->count = 0;
				rcmd->untile = r->flags & CANVAS_RECT_TILE && !(texture->flags & VS::TEXTURE_FLAG_REPEAT);
				rcmd->texture = r->texture;
				rcmd->use_texture = true;

				while (true) {
					Rect2 dst_rect;
					Rect2 src_rect;

					if (!texture) {
						dst_rect = Rect2(r->rect.position, r->rect.size);

						if (dst_rect.size.width < 0) {
							dst_rect.position.x += dst_rect.size.width;
							dst_rect.size.width *= -1;
						}
						if (dst_rect.size.height < 0) {
							dst_rect.position.y += dst_rect.size.height;
							dst_rect.size.height *= -1;
						}

						src_rect = Rect2(0, 0, 1, 1);
					} else {
						src_rect = (r->flags & CANVAS_RECT_REGION) ? Rect2(r->source.position * texpixel_size, r->source.size * texpixel_size) : Rect2(0, 0, 1, 1);
						dst_rect = Rect2(r->rect.position, r->rect.size);

						if (dst_rect.size.width < 0) {
							dst_rect.position.x += dst_rect.size.width;
							dst_rect.size.width *= -1;
						}
						if (dst_rect.size.height < 0) {
							dst_rect.position.y += dst_rect.size.height;
							dst_rect.size.height *= -1;
						}

						if (r->flags & CANVAS_RECT_FLIP_H) {
							src_rect.size.x *= -1;
						}

						if (r->flags & CANVAS_RECT_FLIP_V) {
							src_rect.size.y *= -1;
						}

						if (r->flags & CANVAS_RECT_TRANSPOSE) {
							dst_rect.size.x *= -1; // Encoding in the dst_rect.z uniform
						}
					}

					struct Vertex {
						Vector2 position;
						Color color;
						Rect2 src;
						Rect2 dest;
					};

					const uint8_t count = 6;

					Vertex vertices[count] = {
						{ Vector2(0, 0), r->modulate, src_rect, dst_rect },
						{ Vector2(0, 1), r->modulate, src_rect, dst_rect },
						{ Vector2(1, 1), r->modulate, src_rect, dst_rect },
						{ Vector2(1, 1), r->modulate, src_rect, dst_rect },
						{ Vector2(1, 0), r->modulate, src_rect, dst_rect },
						{ Vector2(0, 0), r->modulate, src_rect, dst_rect },
					};

					if (data.polygon_buffer_size - data.polygon_mem_offset < sizeof(vertices)) {
						RenderItem::RectsCommand cmddata = *rcmd;
						_flush();

						rcmd = push_command<RenderItem::RectsCommand>();
						*rcmd = cmddata;

						rcmd->vertices_offset = data.polygon_mem_offset;
						rcmd->count = 0;
						rcmd->use_texture = false;
					}

					memcpy(data.polygon_mem_buffer + data.polygon_mem_offset, vertices, sizeof(vertices));
					data.polygon_mem_offset += sizeof(vertices);

					rcmd->count += count;

					if (++i >= size)
						break;

					command = commands[i];
					if (command->type != Item::Command::TYPE_RECT)
						break;

					Item::CommandRect *next_r = static_cast<Item::CommandRect *>(command);

					if (next_r->flags != r->flags)
						break;

					if (next_r->texture != rcmd->texture)
						break;

					r = static_cast<Item::CommandRect *>(command);
				}

				continue;
			} break;
			case Item::Command::TYPE_CIRCLE: {

				Item::CommandCircle *circle = static_cast<Item::CommandCircle *>(command);

				static const int num_points = 32;

				Vector2 points[num_points + 1];
				points[num_points] = circle->pos;

				int indices[num_points * 3];

				for (int i = 0; i < num_points; i++) {
					points[i] = circle->pos + Vector2(Math::sin(i * Math_PI * 2.0 / num_points), Math::cos(i * Math_PI * 2.0 / num_points)) * circle->radius;
					indices[i * 3 + 0] = i;
					indices[i * 3 + 1] = (i + 1) % num_points;
					indices[i * 3 + 2] = num_points;
				}

				prepared = _prepare_polygon(command, indices, num_points * 3, num_points + 1, points, NULL, &circle->color, true);
			} break;
			case Item::Command::TYPE_POLYGON: {
				Item::CommandPolygon *polygon = static_cast<Item::CommandPolygon *>(command);
				prepared = _prepare_polygon(command, polygon->indices.ptr(), polygon->count, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1);
			} break;
			case Item::Command::TYPE_NINEPATCH: {
				Item::CommandNinePatch *ninepatch = static_cast<Item::CommandNinePatch *>(command);
				prepared = _prepare_ninepatch(ninepatch);
			} break;
			case Item::Command::TYPE_POLYLINE: {
				Item::CommandPolyLine *pline = static_cast<Item::CommandPolyLine *>(command);

				if (pline->triangles.size()) {
					prepared = _prepare_generic(GL_TRIANGLE_STRIP, pline->triangles.size(), pline->triangles.ptr(), NULL, pline->triangle_colors.ptr(), pline->triangle_colors.size() == 1);
				} else {
					if (pline->multiline) {
						int todo = pline->lines.size() / 2;
						int offset = 0;

						while (todo) {
							const int max_current = (data.polygon_buffer_size - data.polygon_mem_offset) / (sizeof(real_t) * 4);
							const int to_draw = MIN(max_current, todo);

							prepared = _prepare_generic(GL_LINES, to_draw * 2, &pline->lines.ptr()[offset], NULL, pline->line_colors.size() == 1 ? pline->line_colors.ptr() : &pline->line_colors.ptr()[offset], pline->line_colors.size() == 1);

							if (!prepared) {
								_flush();
								continue;
							}

							todo -= to_draw;
							offset += to_draw * 2;
						}
					} else
						prepared = _prepare_generic(GL_LINES, pline->lines.size(), pline->lines.ptr(), NULL, pline->line_colors.ptr(), pline->line_colors.size() == 1);
				}
			} break;
			case Item::Command::TYPE_LINE: {

				Item::CommandLine *line = static_cast<Item::CommandLine *>(command);

				if (line->width <= 1) {
					Vector2 verts[2] = {
						Vector2(line->from.x, line->from.y),
						Vector2(line->to.x, line->to.y)
					};

					prepared = _prepare_gui_primitive(command, 2, verts, NULL, NULL);
				} else {
					Vector2 t = (line->from - line->to).normalized().tangent() * line->width * 0.5;

					Vector2 verts[4] = {
						line->from - t,
						line->from + t,
						line->to + t,
						line->to - t
					};

					prepared = _prepare_gui_primitive(command, 4, verts, NULL, NULL);
				}
			} break;
			case Item::Command::TYPE_PRIMITIVE: {
				Item::CommandPrimitive *primitive = static_cast<Item::CommandPrimitive *>(command);

				ERR_CONTINUE(primitive->points.size() < 1);

				prepared = _prepare_gui_primitive(command, primitive->points.size(), primitive->points.ptr(), primitive->colors.ptr(), primitive->uvs.ptr());
			} break;
			default: {
				RenderItem::PassThroughCommand *cmd = render_commands.allocate<RenderItem::PassThroughCommand>();
				prepared = cmd;

				if (prepared)
					cmd->command = command;
			} break;
		}

		if (!prepared)
			_flush();
		else
			++i;
	}
}

void RasterizerCanvasGLES2::_flush() {
	if (render_commands.empty())
		return;

	glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);

	glBufferSubData(GL_ARRAY_BUFFER, 0, data.polygon_mem_offset, data.polygon_mem_buffer);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, data.polygon_index_mem_offset, data.polygon_index_mem_buffer);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	data.polygon_mem_offset = 0;
	data.polygon_index_mem_offset = 0;

	RenderItem::CommandBase *cmd;
	for (RenderCommands::Ptr cmds = render_commands.ptr(); cmd = cmds.current(); cmds.next()) {
		switch (cmd->type) {
			case RenderItem::TYPE_BEGINLIST: {
				RenderItem::BeginListCommand *begin = static_cast<RenderItem::BeginListCommand *>(cmd);

				state.current_clip = NULL;
				state.shader_cache = NULL;
				state.rebind_shader = true;
				state.rt_size = Size2(storage->frame.current_rt->width, storage->frame.current_rt->height);
				state.current_tex = RID();
				state.current_tex_ptr = NULL;
				state.current_normal = RID();
				state.last_blend_mode = -1;
				state.canvas_last_material = RID();
				state.p_modulate = begin->p_modulate;
				state.reclip = false;

				data.polygon_mem_offset = 0;
				data.polygon_index_mem_offset = 0;

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
			} break;
			case RenderItem::TYPE_ENDLIST: {
				if (state.current_clip) {
					glDisable(GL_SCISSOR_TEST);
				}
			} break;
			case RenderItem::TYPE_PASSTHROUGH: {
				RenderItem::PassThroughCommand *passthrough = static_cast<RenderItem::PassThroughCommand *>(cmd);
				_canvas_render_command(passthrough->command, NULL, state.reclip);
			} break;
			case RenderItem::TYPE_RECTS: {
				RenderItem::RectsCommand *rects = static_cast<RenderItem::RectsCommand *>(cmd);
				_flush_rects(rects);
			} break;
			case RenderItem::TYPE_POLYGON: {
				RenderItem::PolygonCommand *polygon = static_cast<RenderItem::PolygonCommand *>(cmd);
				_flush_polygon(polygon);
			} break;
			case RenderItem::TYPE_NINEPATCH: {
				RenderItem::NinepatchCommand *ninepatch = static_cast<RenderItem::NinepatchCommand *>(cmd);
				_flush_ninepatch(ninepatch);
			} break;
			case RenderItem::TYPE_GENERIC: {
				RenderItem::GenericCommand *generic = static_cast<RenderItem::GenericCommand *>(cmd);
				_flush_generic(generic);
			} break;
			case RenderItem::TYPE_GUI_PRIMITIVE: {
				RenderItem::GuiPrimitiveCommand *gui = static_cast<RenderItem::GuiPrimitiveCommand *>(cmd);
				_flush_gui_primitive(gui);
			} break;
			case RenderItem::TYPE_CHANGEITEM: {
				RenderItem::ChangeItemCommand *change = static_cast<RenderItem::ChangeItemCommand *>(cmd);
				Item *ci = change->item;

				if (state.reclip) {
					glEnable(GL_SCISSOR_TEST);
					int y = storage->frame.current_rt->height - (state.current_clip->final_clip_rect.position.y + state.current_clip->final_clip_rect.size.y);
					if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
						y = state.current_clip->final_clip_rect.position.y;
					glScissor(state.current_clip->final_clip_rect.position.x, y, state.current_clip->final_clip_rect.size.width, state.current_clip->final_clip_rect.size.height);
				}

				state.rebind_shader = true; // hacked in for now.

				if (state.current_clip != ci->final_clip_owner) {

					state.current_clip = ci->final_clip_owner;

					if (state.current_clip) {
						glEnable(GL_SCISSOR_TEST);
						int y = storage->frame.current_rt->height - (state.current_clip->final_clip_rect.position.y + state.current_clip->final_clip_rect.size.y);
						if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
							y = state.current_clip->final_clip_rect.position.y;
						glScissor(state.current_clip->final_clip_rect.position.x, y, state.current_clip->final_clip_rect.size.width, state.current_clip->final_clip_rect.size.height);
					} else {
						glDisable(GL_SCISSOR_TEST);
					}
				}

				// TODO: copy back buffer

				if (ci->copy_back_buffer) {
					if (ci->copy_back_buffer->full) {
						_copy_texscreen(Rect2());
					} else {
						_copy_texscreen(ci->copy_back_buffer->rect);
					}
				}

				Item *material_owner = ci->material_owner ? ci->material_owner : ci;

				RID material = material_owner->material;

				if (material != state.canvas_last_material || state.rebind_shader) {

					RasterizerStorageGLES2::Material *material_ptr = storage->material_owner.getornull(material);
					RasterizerStorageGLES2::Shader *shader_ptr = NULL;

					if (material_ptr) {
						shader_ptr = material_ptr->shader;

						if (shader_ptr && shader_ptr->mode != VS::SHADER_CANVAS_ITEM) {
							shader_ptr = NULL; // not a canvas item shader, don't use.
						}
					}

					if (shader_ptr) {
						if (shader_ptr->canvas_item.uses_screen_texture) {
							_copy_texscreen(Rect2());
						}

						if (shader_ptr != state.shader_cache) {

							if (shader_ptr->canvas_item.uses_time) {
								VisualServerRaster::redraw_request();
							}

							state.canvas_shader.set_custom_shader(shader_ptr->custom_code_id);
							state.canvas_shader.bind();
						}

						int tc = material_ptr->textures.size();
						RID *textures = material_ptr->textures.ptrw();

						ShaderLanguage::ShaderNode::Uniform::Hint *texture_hints = shader_ptr->texture_hints.ptrw();

						for (int i = 0; i < tc; i++) {

							glActiveTexture(GL_TEXTURE2 + i);

							RasterizerStorageGLES2::Texture *t = storage->texture_owner.getornull(textures[i]);

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

								continue;
							}

							t = t->get_ptr();

							if (t->redraw_if_visible) {
								VisualServerRaster::redraw_request();
							}

							glBindTexture(t->target, t->tex_id);
						}
					} else {
						state.canvas_shader.set_custom_shader(0);
						state.canvas_shader.bind();
					}

					state.shader_cache = shader_ptr;

					state.canvas_last_material = material;

					state.rebind_shader = false;
				}

				int blend_mode = state.shader_cache ? state.shader_cache->canvas_item.blend_mode : RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX;
				bool unshaded = true || (state.shader_cache && blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX);
				state.reclip = false;

				if (state.last_blend_mode != blend_mode) {

					switch (blend_mode) {

						case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX: {
							glBlendEquation(GL_FUNC_ADD);
							if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
								glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
							} else {
								glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
							}

						} break;
						case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_ADD: {

							glBlendEquation(GL_FUNC_ADD);
							glBlendFunc(GL_SRC_ALPHA, GL_ONE);

						} break;
						case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_SUB: {

							glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
							glBlendFunc(GL_SRC_ALPHA, GL_ONE);
						} break;
						case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MUL: {
							glBlendEquation(GL_FUNC_ADD);
							glBlendFunc(GL_DST_COLOR, GL_ZERO);
						} break;
						case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA: {
							glBlendEquation(GL_FUNC_ADD);
							glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
						} break;
					}
				}

				state.uniforms.final_modulate = unshaded ? ci->final_modulate : Color(ci->final_modulate.r * state.p_modulate.r, ci->final_modulate.g * state.p_modulate.g, ci->final_modulate.b * state.p_modulate.b, ci->final_modulate.a * state.p_modulate.a);

				state.uniforms.modelview_matrix = ci->final_transform;
				state.uniforms.extra_matrix = Transform2D();

				_set_uniforms();
			} break;
		}
	}

	render_commands.clear();
}

void RasterizerCanvasGLES2::canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {

	RenderItem::BeginListCommand *begin = push_command<RenderItem::BeginListCommand>();
	begin->p_modulate = p_modulate;

	while (p_item_list) {
		RenderItem::ChangeItemCommand *cmd = push_command<RenderItem::ChangeItemCommand>();
		cmd->item = p_item_list;

		_canvas_item_process(p_item_list);
		p_item_list = p_item_list->next;
	}

	push_command<RenderItem::EndListCommand>();
}

void RasterizerCanvasGLES2::canvas_debug_viewport_shadows(Light *p_lights_with_shadow) {
}

void RasterizerCanvasGLES2::canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {
}

void RasterizerCanvasGLES2::reset_canvas() {

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glEnable(GL_BLEND);

	if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	// bind the back buffer to a texture so shaders can use it.
	// It should probably use texture unit -3 (as GLES3 does as well) but currently that's buggy.
	// keeping this for now as there's nothing else that uses texture unit 2
	// TODO ^
	if (storage->frame.current_rt) {
		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->copy_screen_effect.color);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES2::_bind_quad_buffer() {
	glBindBuffer(GL_ARRAY_BUFFER, data.canvas_quad_vertices);
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, 0, NULL);
}
void RasterizerCanvasGLES2::draw_generic_textured_rect(const Rect2 &p_rect, const Rect2 &p_src) {

	state.canvas_shader.set_uniform(CanvasShaderGLES2::DST_RECT, Color(p_rect.position.x, p_rect.position.y, p_rect.size.x, p_rect.size.y));
	state.canvas_shader.set_uniform(CanvasShaderGLES2::SRC_RECT, Color(p_src.position.x, p_src.position.y, p_src.size.x, p_src.size.y));

	glDrawArrays(GL_TRIANGLES, 0, 6);
	++storage->info.render.draw_call_count;
}

void RasterizerCanvasGLES2::draw_window_margins(int *black_margin, RID *black_image) {
}

void RasterizerCanvasGLES2::initialize() {

	// quad buffer
	{
		glGenBuffers(1, &data.canvas_quad_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, data.canvas_quad_vertices);

		const float qv[12] = {
			0, 0,
			0, 1,
			1, 1,
			1, 1,
			1, 0,
			0, 0
		};

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, qv, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// polygon buffer
	{
		uint32_t poly_size = GLOBAL_DEF("rendering/limits/buffers/canvas_polygon_buffer_size_kb", 128);
		poly_size *= 1024;
		poly_size = MAX(poly_size, (2 + 2 + 4) * 4 * sizeof(float));
		glGenBuffers(1, &data.polygon_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, data.polygon_buffer);
		glBufferData(GL_ARRAY_BUFFER, poly_size, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		data.polygon_buffer_size = poly_size;

		data.polygon_mem_buffer = (uint8_t *)memalloc(poly_size);
		ERR_FAIL_COND(!data.polygon_buffer);
		zeromem(data.polygon_mem_buffer, poly_size);
		data.polygon_mem_offset = 0;

		uint32_t index_size = GLOBAL_DEF("rendering/limits/buffers/canvas_polygon_index_size_kb", 128);
		index_size *= 1024; // kb
		glGenBuffers(1, &data.polygon_index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.polygon_index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		data.polygon_index_buffer_size = index_size;

		data.polygon_index_mem_buffer = (uint8_t *)memalloc(index_size);
		ERR_FAIL_COND(!data.polygon_index_mem_buffer);
		zeromem(data.polygon_index_mem_buffer, index_size);
		data.polygon_index_mem_offset = 0;
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
		;
#undef _EIDX

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elems), elems, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	state.canvas_shader.init();

	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, true);
	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT_ATTRIB, false);

	state.canvas_shader.bind();

	uint32_t render_commands_buffer_size = GLOBAL_DEF("rendering/limits/buffers/canvas_render_commands_buffer_size_kb", 128);
	render_commands_buffer_size *= 1024;

	render_commands.init(render_commands_buffer_size);
}

void RasterizerCanvasGLES2::finalize() {
	glDeleteBuffers(1, &data.polygon_buffer);
	glDeleteBuffers(1, &data.polygon_index_buffer);

	memfree(data.polygon_mem_buffer);
	memfree(data.polygon_index_mem_buffer);
}

RasterizerCanvasGLES2::RasterizerCanvasGLES2() {
}

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

	state.canvas_shader.set_uniform(CanvasShaderGLES2::TIME, storage->frame.time[0]);

	if (storage->frame.current_rt) {
		Vector2 screen_pixel_size;
		screen_pixel_size.x = 1.0 / storage->frame.current_rt->width;
		screen_pixel_size.y = 1.0 / storage->frame.current_rt->height;

		state.canvas_shader.set_uniform(CanvasShaderGLES2::SCREEN_PIXEL_SIZE, screen_pixel_size);
	}

	state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, state.uniforms.texpixel_size);
}

void RasterizerCanvasGLES2::canvas_begin() {
	data.primitive = GL_TRIANGLES;
	data.texture = GL_NONE;

	state.canvas_shader.bind();
	if (storage->frame.current_rt) {
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);
		glColorMask(1, 1, 1, 1);
	}

	if (storage->frame.clear_request) {
		glColorMask(true, true, true, true);
		glClearColor(storage->frame.clear_request_color.r,
				storage->frame.clear_request_color.g,
				storage->frame.clear_request_color.b,
				storage->frame.clear_request_color.a);
		glClear(GL_COLOR_BUFFER_BIT);
		storage->frame.clear_request = false;
	}

	/*
	if (storage->frame.current_rt) {
		glBindFramebuffer(GL_FRAMEBUFFER, storage->frame.current_rt->fbo);
		glColorMask(1, 1, 1, 1);
	}
	*/

	reset_canvas();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
	data.texture = storage->resources.white_tex;

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
	state.prev_uniforms = state.uniforms;
}

void RasterizerCanvasGLES2::canvas_end() {

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	for (int i = 0; i < VS::ARRAY_MAX; i++) {
		glDisableVertexAttribArray(i);
	}

	state.using_texture_rect = false;
	state.using_ninepatch = false;
}

RasterizerStorageGLES2::Texture *RasterizerCanvasGLES2::_bind_canvas_texture(const RID &p_texture, const RID &p_normal_map) {

	RasterizerStorageGLES2::Texture *tex_return = NULL;
	GLuint newtexid;

	if (p_texture.is_valid()) {

		RasterizerStorageGLES2::Texture *texture = storage->texture_owner.getornull(p_texture);

		if (!texture) {
			state.current_tex = RID();
			state.current_tex_ptr = NULL;

			newtexid = storage->resources.white_tex;

		} else {

			texture = texture->get_ptr();

			if (texture->redraw_if_visible) {
				VisualServerRaster::redraw_request();
			}

			if (texture->render_target) {
				texture->render_target->used_in_frame = true;
			}

			newtexid = texture->tex_id;

			state.current_tex = p_texture;
			state.current_tex_ptr = texture;

			tex_return = texture;
		}
	} else {
		state.current_tex = RID();
		state.current_tex_ptr = NULL;

		newtexid = storage->resources.white_tex;
	}

	if (data.texture != newtexid) {
		_flush();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, newtexid);
		data.texture = newtexid;
	}

	return tex_return;
}

void RasterizerCanvasGLES2::_set_texture_rect_mode(bool p_enable, bool p_ninepatch) {
}

void RasterizerCanvasGLES2::_draw_polygon(const int *p_indices, int p_index_count, int p_vertex_count, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, bool p_singlecolor) {
	_begin(GL_TRIANGLES);
	_prepare(p_vertex_count, p_index_count);

	Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

	bool single;
	Color color;

	if (p_singlecolor) {
		single = true;
		color = *p_colors;
	} else if (!p_colors) {
		single = true;
		color = Color(1, 1, 1, 1);
	} else {
		single = false;
	}

	const bool use_single_color = single;
	const Color single_color = color;

	for (int i = 0; i < p_vertex_count; ++i) {
		v->v = p_vertices[i];

		if (use_single_color)
			v->c = single_color;
		else
			v->c = p_colors[i];

		if (p_uvs)
			v->uv = p_uvs[i];
		else
			v->uv = Vector2();

		++v;
	}

	memcpy(data.mem_index_buffer + data.mem_index_buffer_offset, p_indices, p_index_count * sizeof(int));

	_commit(p_vertex_count, p_index_count);
}

void RasterizerCanvasGLES2::_canvas_item_render_commands(Item *p_item, Item *current_clip, bool &reclip, RasterizerStorageGLES2::Material *p_material) {

	int command_count = p_item->commands.size();
	Item::Command **commands = p_item->commands.ptrw();

	for (int i = 0; i < command_count; i++) {

		Item::Command *command = commands[i];

		if (command->type != Item::Command::TYPE_RECT && state.tiled) {
			_flush();
			_untile();
		}

		switch (command->type) {

			case Item::Command::TYPE_LINE: {
				const Item::CommandLine *line = static_cast<Item::CommandLine *>(command);

				if (line->width <= 1) {
					const int p_vertex_count = 2;
					const int p_index_count = 2;

					_begin(GL_LINES);
					_prepare(p_vertex_count, p_index_count);

					_bind_shader(p_material);
					_bind_canvas_texture(RID(), RID());

					Vertex vertices[p_vertex_count];

					vertices[0].v = Vector2(line->from.x, line->from.y);
					vertices[0].c = line->color;
					vertices[0].uv = Vector2();

					vertices[1].v = Vector2(line->to.x, line->to.y);
					vertices[1].c = line->color;
					vertices[1].uv = Vector2();

					memcpy(data.mem_vertex_buffer + data.mem_vertex_buffer_offset, vertices, sizeof(vertices));

					const int indices[p_index_count] = { 0, 1 };

					memcpy(data.mem_index_buffer + data.mem_index_buffer_offset, indices, sizeof(indices));

					_commit(p_vertex_count, p_index_count);
				} else {
					const int p_vertex_count = 4;
					const int p_index_count = 6;

					_begin(GL_TRIANGLES);
					_prepare(p_vertex_count, p_index_count);

					_bind_shader(p_material);
					_bind_canvas_texture(RID(), RID());

					Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

					Vector2 t = (line->from - line->to).normalized().tangent() * line->width * 0.5;

					v[0].v = line->from - t;
					v[0].c = line->color;
					v[0].uv = Vector2();

					v[1].v = line->from + t;
					v[1].c = line->color;
					v[1].uv = Vector2();

					v[2].v = line->to + t;
					v[2].c = line->color;
					v[2].uv = Vector2();

					v[3].v = line->to - t;
					v[3].c = line->color;
					v[3].uv = Vector2();

					const int indices[p_index_count] = {
						0, 1, 2,
						2, 3, 0
					};

					memcpy(data.mem_index_buffer + data.mem_index_buffer_offset, indices, sizeof(indices));

					_commit(p_vertex_count, p_index_count);
				}

			} break;

			case Item::Command::TYPE_RECT: {
				const int p_vertex_count = 4;
				const int p_index_count = 6;

				_begin(GL_TRIANGLES);
				_prepare(p_vertex_count, p_index_count);

				Item::CommandRect *r = static_cast<Item::CommandRect *>(command);

				_bind_shader(p_material);

				Rect2 src_rect;
				Rect2 dst_rect;

				RasterizerStorageGLES2::Texture *tex = _bind_canvas_texture(r->texture, r->normal_map);

				if (!tex) {
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

					const bool tiled = r->flags & CANVAS_RECT_TILE && !(tex->flags & VS::TEXTURE_FLAG_REPEAT);

					if (tiled != state.tiled) {
						_flush();

						if (tiled) {
							glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
							glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
							state.tiled = true;
						} else {
							_untile();
						}
					}

					Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);

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
						src_rect.position.x -= src_rect.size.width;
					}
					if (r->flags & CANVAS_RECT_FLIP_V) {
						src_rect.size.y *= -1;
						src_rect.position.y -= src_rect.size.height;
					}
					if (r->flags & CANVAS_RECT_TRANSPOSE) {
						dst_rect.size.x *= -1; // Encoding in the dst_rect.z uniform
					}

					state.uniforms.texpixel_size = texpixel_size;
				}

				Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

				// 0,0
				v[0].v = dst_rect.position;
				v[0].c = r->modulate;
				v[0].uv = src_rect.position;

				// 0,1
				v[1].v = Vector2(dst_rect.position.x, dst_rect.position.y + dst_rect.size.y);
				v[1].c = r->modulate;
				v[1].uv = Vector2(src_rect.position.x, src_rect.position.y + src_rect.size.y);

				// 1,1
				v[2].v = Vector2(dst_rect.position.x + dst_rect.size.x, dst_rect.position.y + dst_rect.size.y);
				v[2].c = r->modulate;
				v[2].uv = Vector2(src_rect.position.x + src_rect.size.x, src_rect.position.y + src_rect.size.y);

				// 1,0
				v[3].v = Vector2(dst_rect.position.x + dst_rect.size.x, dst_rect.position.y);
				v[3].c = r->modulate;
				v[3].uv = Vector2(src_rect.position.x + src_rect.size.x, src_rect.position.y);

				const int indices[p_index_count] = {
					0, 1, 2,
					2, 3, 0
				};

				memcpy(data.mem_index_buffer + data.mem_index_buffer_offset, indices, sizeof(int) * p_index_count);

				_commit(p_vertex_count, p_index_count);
			} break;

			case Item::Command::TYPE_NINEPATCH: {
				const int p_vertex_count = 16;
				const int p_index_count = 54;

				_begin(GL_TRIANGLES);
				_prepare(p_vertex_count, p_index_count);

				Item::CommandNinePatch *np = static_cast<Item::CommandNinePatch *>(command);

				_bind_shader(p_material);
				RasterizerStorageGLES2::Texture *tex = _bind_canvas_texture(np->texture, np->normal_map);

				if (!tex) {
					print_line("TODO: ninepatch without texture");
					continue;
				}

				Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);

				state.uniforms.texpixel_size = texpixel_size;

				Rect2 source = np->source;
				if (source.size.x == 0 && source.size.y == 0) {
					source.size.x = tex->width;
					source.size.y = tex->height;
				}

				// prepare vertex buffer

				// this buffer contains [ POS POS UV UV ] *

				Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

				v[0].v = np->rect.position;
				v[0].c = np->color;
				v[0].uv = source.position * texpixel_size;

				v[1].v = np->rect.position + Vector2(np->margin[MARGIN_LEFT], 0);
				v[1].c = np->color;
				v[1].uv = (source.position + Vector2(np->margin[MARGIN_LEFT], 0)) * texpixel_size;

				v[2].v = np->rect.position + Vector2(np->rect.size.x - np->margin[MARGIN_RIGHT], 0);
				v[2].c = np->color;
				v[2].uv = (source.position + Vector2(source.size.x - np->margin[MARGIN_RIGHT], 0)) * texpixel_size;

				v[3].v = np->rect.position + Vector2(np->rect.size.x, 0);
				v[3].c = np->color;
				v[3].uv = (source.position + Vector2(source.size.x, 0)) * texpixel_size;

				v[4].v = np->rect.position + Vector2(0, np->margin[MARGIN_TOP]);
				v[4].c = np->color;
				v[4].uv = (source.position + Vector2(0, np->margin[MARGIN_TOP])) * texpixel_size;

				v[5].v = np->rect.position + Vector2(np->margin[MARGIN_LEFT], np->margin[MARGIN_TOP]);
				v[5].c = np->color;
				v[5].uv = (source.position + Vector2(np->margin[MARGIN_LEFT], np->margin[MARGIN_TOP])) * texpixel_size;

				v[6].v = np->rect.position + Vector2(np->rect.size.x - np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP]);
				v[6].c = np->color;
				v[6].uv = (source.position + Vector2(source.size.x - np->margin[MARGIN_RIGHT], np->margin[MARGIN_TOP])) * texpixel_size;

				v[7].v = np->rect.position + Vector2(np->rect.size.x, np->margin[MARGIN_TOP]);
				v[7].c = np->color;
				v[7].uv = (source.position + Vector2(source.size.x, np->margin[MARGIN_TOP])) * texpixel_size;

				v[8].v = np->rect.position + Vector2(0, np->rect.size.y - np->margin[MARGIN_BOTTOM]);
				v[8].c = np->color;
				v[8].uv = (source.position + Vector2(0, source.size.y - np->margin[MARGIN_BOTTOM])) * texpixel_size;

				v[9].v = np->rect.position + Vector2(np->margin[MARGIN_LEFT], np->rect.size.y - np->margin[MARGIN_BOTTOM]);
				v[9].c = np->color;
				v[9].uv = (source.position + Vector2(np->margin[MARGIN_LEFT], source.size.y - np->margin[MARGIN_BOTTOM])) * texpixel_size;

				v[10].v = np->rect.position + np->rect.size - Vector2(np->margin[MARGIN_RIGHT], np->margin[MARGIN_BOTTOM]);
				v[10].c = np->color;
				v[10].uv = (source.position + source.size - Vector2(np->margin[MARGIN_RIGHT], np->margin[MARGIN_BOTTOM])) * texpixel_size;

				v[11].v = np->rect.position + np->rect.size - Vector2(0, np->margin[MARGIN_BOTTOM]);
				v[11].c = np->color;
				v[11].uv = (source.position + source.size - Vector2(0, np->margin[MARGIN_BOTTOM])) * texpixel_size;

				v[12].v = np->rect.position + Vector2(0, np->rect.size.y);
				v[12].c = np->color;
				v[12].uv = (source.position + Vector2(0, source.size.y)) * texpixel_size;

				v[13].v = np->rect.position + Vector2(np->margin[MARGIN_LEFT], np->rect.size.y);
				v[13].c = np->color;
				v[13].uv = (source.position + Vector2(np->margin[MARGIN_LEFT], source.size.y)) * texpixel_size;

				v[14].v = np->rect.position + np->rect.size - Vector2(np->margin[MARGIN_RIGHT], 0);
				v[14].c = np->color;
				v[14].uv = (source.position + source.size - Vector2(np->margin[MARGIN_RIGHT], 0)) * texpixel_size;

				v[15].v = np->rect.position + np->rect.size;
				v[15].c = np->color;
				v[15].uv = (source.position + source.size) * texpixel_size;

				memcpy(data.mem_index_buffer + data.mem_index_buffer_offset, data.ninepatch_elements, sizeof(data.ninepatch_elements));

				_commit(p_vertex_count, p_index_count - (np->draw_center ? 0 : 6));
			} break;

			case Item::Command::TYPE_CIRCLE: {
				Item::CommandCircle *circle = static_cast<Item::CommandCircle *>(command);

				_bind_shader(p_material);

				const int num_points = 32;

				Vector2 points[num_points + 1];
				points[num_points] = circle->pos;

				int indices[num_points * 3];

				for (int i = 0; i < num_points; i++) {
					points[i] = circle->pos + Vector2(Math::sin(i * Math_PI * 2.0 / num_points), Math::cos(i * Math_PI * 2.0 / num_points)) * circle->radius;
					indices[i * 3 + 0] = i;
					indices[i * 3 + 1] = (i + 1) % num_points;
					indices[i * 3 + 2] = num_points;
				}

				_bind_canvas_texture(RID(), RID());

				_draw_polygon(indices, num_points * 3, num_points + 1, points, NULL, &circle->color, true);
			} break;

			case Item::Command::TYPE_POLYGON: {
				Item::CommandPolygon *polygon = static_cast<Item::CommandPolygon *>(command);

				const int *indices = polygon->indices.ptr();
				if (!indices) // self-intersecting polygon
					break;

				_bind_shader(p_material);
				RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(polygon->texture, polygon->normal_map);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.uniforms.texpixel_size = texpixel_size;
				}

				_draw_polygon(indices, polygon->count, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1);
			} break;

			case Item::Command::TYPE_POLYLINE: {
				Item::CommandPolyLine *pline = static_cast<Item::CommandPolyLine *>(command);

				if (pline->triangles.size()) {
					const int p_vertex_count = pline->triangles.size();
					const int p_triangle_count = p_vertex_count - 2;
					const int p_index_count = p_triangle_count * 3;

					_begin(GL_TRIANGLES);
					_prepare(p_vertex_count, p_index_count);

					_bind_shader(p_material);
					_bind_canvas_texture(RID(), RID());

					const Vector2 *t = pline->triangles.ptr();
					Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

					const bool p_singlecolor = pline->triangle_colors.size() == 1;
					const Color *p_colors = pline->triangle_colors.ptr();

					bool single;
					Color color;

					if (pline->triangle_colors.size() == 1) {
						single = true;
						color = *p_colors;
					} else if (!p_colors) {
						single = true;
						color = Color(1, 1, 1, 1);
					} else {
						single = false;
					}

					const bool use_single_color = single;
					const Color single_color = color;

					for (int i = 0; i < p_vertex_count; ++i) {
						if (use_single_color)
							v->c = single_color;
						else
							v->c = p_colors[i];

						v->uv = Vector2();
						v->v = t[i];

						++v;
					}

					for (int i = 0; i < p_triangle_count; ++i) {
						const int indices[3] = {
							i, i + 1, i + 2
						};

						memcpy(data.mem_index_buffer + data.mem_index_buffer_offset + i * 3, indices, sizeof(indices));
					}

					_commit(p_vertex_count, p_index_count);
				} else {
					_begin(GL_LINES);

					_bind_shader(p_material);
					_bind_canvas_texture(RID(), RID());

					const Color *p_colors = pline->line_colors.ptr();

					bool single;
					Color color;

					if (pline->line_colors.size() == 1) {
						single = true;
						color = *p_colors;
					} else if (!p_colors) {
						single = true;
						color = Color(1, 1, 1, 1);
					} else {
						single = false;
					}

					const bool use_single_color = single;
					const Color single_color = color;

					const Vector2 *p_lines = pline->lines.ptr();

					if (pline->multiline) {
						const int p_lines_count = pline->lines.size() / 2;

						for (int i = 0; i < p_lines_count; ++i) {
							const int p_vertex_count = 2;
							const int p_index_count = 2;

							_prepare(p_vertex_count, p_index_count);

							Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

							for (int j = 0; j < 2; ++j) {
								if (use_single_color)
									v->c = single_color;
								else
									v->c = p_colors[i];

								v->uv = Vector2();
								v->v = p_lines[i * 2 + j];

								++v;
							}

							const int indices[p_index_count] = { 0, 1 };

							memcpy(data.mem_index_buffer + data.mem_index_buffer_offset, indices, sizeof(indices));

							_commit(p_vertex_count, p_index_count);
						}
					} else {
						const int p_vertex_count = pline->lines.size();
						const int p_lines_count = p_vertex_count - 1;
						const int p_index_count = p_lines_count * 2;

						_prepare(p_vertex_count, p_index_count);

						_bind_shader(p_material);
						_bind_canvas_texture(RID(), RID());

						Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;

						for (int i = 0; i < p_vertex_count; ++i) {
							if (use_single_color)
								v->c = single_color;
							else
								v->c = p_colors[i];

							v->uv = Vector2();
							v->v = p_lines[i];

							++v;
						}

						for (int i = 0; i < p_lines_count; ++i) {
							const int indices[2] = { i, i + 1 };

							memcpy(data.mem_index_buffer + data.mem_index_buffer_offset + i * 2, indices, sizeof(indices));
						}

						_commit(p_vertex_count, p_index_count);
					}
				}
			} break;

			case Item::Command::TYPE_PRIMITIVE: {
				Item::CommandPrimitive *primitive = static_cast<Item::CommandPrimitive *>(command);

				const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

				ERR_CONTINUE(primitive->points.size() < 1);

				_bind_shader(p_material);
				RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(primitive->texture, primitive->normal_map);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.uniforms.texpixel_size = texpixel_size;
				}

				const int p_vertex_count = primitive->points.size();
				const int p_index_count = p_vertex_count;

				_begin(prim[p_vertex_count]);
				_prepare(p_vertex_count, p_index_count);

				Vertex *v = data.mem_vertex_buffer + data.mem_vertex_buffer_offset;
				int *index = data.mem_index_buffer + data.mem_index_buffer_offset;

				Color c;
				bool p_single_color;

				const Color *p_colors = primitive->colors.ptr();
				const Vector2 *p_uvs = primitive->uvs.ptr();
				const Vector2 *p_points = primitive->points.ptr();

				if (primitive->colors.size() == 1 && primitive->points.size() > 1) {
					p_single_color = true;
					c = primitive->colors[0];
				} else if (primitive->colors.empty()) {
					p_single_color = true;
					c = Color(1, 1, 1, 1);
				} else {
					p_single_color = false;
				}

				const bool use_single_color = p_single_color;
				const Color single_color = c;

				for (int i = 0; i < p_vertex_count; ++i) {
					if (use_single_color)
						v->c = single_color;
					else
						v->c = p_colors[i];

					if (p_uvs)
						v->uv = p_uvs[i];
					else
						v->uv = Vector2();

					v->v = p_points[i];

					index[i] = i;

					++v;
				}

				_commit(p_vertex_count, p_index_count);
			} break;

			case Item::Command::TYPE_TRANSFORM: {
				Item::CommandTransform *transform = static_cast<Item::CommandTransform *>(command);
				state.uniforms.extra_matrix = transform->xform;
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
}

void RasterizerCanvasGLES2::_copy_texscreen(const Rect2 &p_rect) {

	// This isn't really working yet, so disabling for now.
}

void RasterizerCanvasGLES2::canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {

	Item *current_clip = NULL;

	RasterizerStorageGLES2::Shader *shader_cache = NULL;

	bool rebind_shader = true;

	Size2 rt_size = Size2(storage->frame.current_rt->width, storage->frame.current_rt->height);

	state.current_tex = RID();
	state.current_tex_ptr = NULL;
	state.current_normal = RID();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
	data.texture = storage->resources.white_tex;

	int last_blend_mode = -1;

	RID canvas_last_material = RID();

	while (p_item_list) {

		Item *ci = p_item_list;

		Item *material_owner = ci->material_owner ? ci->material_owner : ci;

		if (current_clip != ci->final_clip_owner) {
			current_clip = ci->final_clip_owner;

			if (current_clip) {
				glEnable(GL_SCISSOR_TEST);
				int y = storage->frame.current_rt->height - (current_clip->final_clip_rect.position.y + current_clip->final_clip_rect.size.y);
				if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
					y = current_clip->final_clip_rect.position.y;
				glScissor(current_clip->final_clip_rect.position.x, y, current_clip->final_clip_rect.size.width, current_clip->final_clip_rect.size.height);
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

		RID material = material_owner->material;
		RasterizerStorageGLES2::Material *material_ptr = storage->material_owner.getornull(material);

		if (material != canvas_last_material || rebind_shader) {

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

				if (shader_ptr != shader_cache) {

					if (shader_ptr->canvas_item.uses_time) {
						VisualServerRaster::redraw_request();
					}

					state.canvas_shader.set_custom_shader(shader_ptr->custom_code_id);
					state.canvas_shader.bind();
				}

				int tc = material_ptr->textures.size();
				Pair<StringName, RID> *textures = material_ptr->textures.ptrw();

				ShaderLanguage::ShaderNode::Uniform::Hint *texture_hints = shader_ptr->texture_hints.ptrw();

				for (int i = 0; i < tc; i++) {

					glActiveTexture(GL_TEXTURE2 + i);

					RasterizerStorageGLES2::Texture *t = storage->texture_owner.getornull(textures[i].second);

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
			state.canvas_shader.use_material((void *)material_ptr);

			shader_cache = shader_ptr;

			canvas_last_material = material;

			rebind_shader = false;
		}

		int blend_mode = shader_cache ? shader_cache->canvas_item.blend_mode : RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX;
		bool unshaded = true || (shader_cache && blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX);
		bool reclip = false;

		if (last_blend_mode != blend_mode) {

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

		state.uniforms.final_modulate = unshaded ? ci->final_modulate : Color(ci->final_modulate.r * p_modulate.r, ci->final_modulate.g * p_modulate.g, ci->final_modulate.b * p_modulate.b, ci->final_modulate.a * p_modulate.a);

		state.uniforms.modelview_matrix = ci->final_transform;
		state.uniforms.extra_matrix = Transform2D();

		_set_uniforms();
		_canvas_item_render_commands(p_item_list, NULL, reclip, material_ptr);

		// TODO: figure out when to _flush to get better batching results
		_flush();

		rebind_shader = true; // hacked in for now.

		if (reclip) {
			glEnable(GL_SCISSOR_TEST);
			int y = storage->frame.current_rt->height - (current_clip->final_clip_rect.position.y + current_clip->final_clip_rect.size.y);
			if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
				y = current_clip->final_clip_rect.position.y;
			glScissor(current_clip->final_clip_rect.position.x, y, current_clip->final_clip_rect.size.width, current_clip->final_clip_rect.size.height);
		}

		p_item_list = p_item_list->next;
	}

	_flush();

	if (current_clip) {
		glDisable(GL_SCISSOR_TEST);
	}
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

void RasterizerCanvasGLES2::draw_generic_textured_rect(const Rect2 &dst_rect, const Rect2 &src_rect) {

	const int p_index_count = 6;
	const int p_vertex_count = 4;

	Vertex v[p_vertex_count];
	Color c(1, 1, 1, 1);

	// 0,0
	v[0].v = dst_rect.position;
	v[0].c = c;
	v[0].uv = src_rect.position;

	// 0,1
	v[1].v = Vector2(dst_rect.position.x, dst_rect.position.y + dst_rect.size.y);
	v[1].c = c;
	v[1].uv = Vector2(src_rect.position.x, src_rect.position.y + src_rect.size.y);

	// 1,1
	v[2].v = Vector2(dst_rect.position.x + dst_rect.size.x, dst_rect.position.y + dst_rect.size.y);
	v[2].c = c;
	v[2].uv = Vector2(src_rect.position.x + src_rect.size.x, src_rect.position.y + src_rect.size.y);

	// 1,0
	v[3].v = Vector2(dst_rect.position.x + dst_rect.size.x, dst_rect.position.y);
	v[3].c = c;
	v[3].uv = Vector2(src_rect.position.x + src_rect.size.x, src_rect.position.y);

	const int indices[p_index_count] = {
		0, 1, 2,
		2, 3, 0
	};

	_draw(GL_TRIANGLES, p_vertex_count, v, p_index_count, indices);
}

void RasterizerCanvasGLES2::draw_window_margins(int *black_margin, RID *black_image) {
}

void RasterizerCanvasGLES2::initialize() {

	// polygon buffer
	{
		uint32_t poly_size = GLOBAL_DEF("rendering/limits/buffers/canvas_polygon_buffer_size_kb", 128);
		poly_size *= 1024;
		poly_size = MAX(poly_size, (2 + 2 + 4) * 4 * sizeof(float));
		glGenBuffers(1, &data.vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, data.vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, poly_size, NULL, GL_DYNAMIC_DRAW);

		data.vertex_buffer_size = poly_size;

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		uint32_t index_size = GLOBAL_DEF("rendering/limits/buffers/canvas_polygon_index_size_kb", 128);
		index_size *= 1024; // kb
		glGenBuffers(1, &data.index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, NULL, GL_DYNAMIC_DRAW);

		data.index_buffer_size = index_size;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	// ninepatch buffers
	{
		// array buffer

#define _EIDX(y, x) (y * 4 + x)
		const int elems[3 * 2 * 9] = {

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

		memcpy(data.ninepatch_elements, elems, sizeof(elems));
	}

	{
		const uint32_t size = data.vertex_buffer_size / sizeof(Vertex);
		data.mem_vertex_buffer = (Vertex *)memalloc(sizeof(Vertex) * size);
		data.mem_vertex_buffer_offset = 0;
		data.mem_vertex_buffer_size = size;
	}

	{
		const uint32_t size = data.index_buffer_size / sizeof(int);
		data.mem_index_buffer = (int *)memalloc(sizeof(int) * size);
		data.mem_index_buffer_offset = 0;
		data.mem_index_buffer_size = size;
	}

	state.canvas_shader.init();

	state.canvas_shader.bind();
}

void RasterizerCanvasGLES2::finalize() {
}

RasterizerCanvasGLES2::RasterizerCanvasGLES2() {
}

void RasterizerCanvasGLES2::_begin(const GLuint p_primitive) {
	if (data.primitive != p_primitive) {
		_flush();
		data.primitive = p_primitive;
	}
}

void RasterizerCanvasGLES2::_prepare(const int p_vertex_count, const int p_index_count) {
	if (data.mem_vertex_buffer_size - data.mem_vertex_buffer_offset < p_vertex_count ||
			data.mem_index_buffer_size - data.mem_index_buffer_offset < p_index_count) {
		_flush();
	}
}

void RasterizerCanvasGLES2::_draw(const GLuint p_primitive, const int p_vertex_count, const Vertex *p_vertices, const int p_index_count, const int *p_indices) {
	glBindBuffer(GL_ARRAY_BUFFER, data.vertex_buffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertex) * p_vertex_count, p_vertices);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.index_buffer);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(int) * p_index_count, p_indices);

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), NULL);

	glEnableVertexAttribArray(VS::ARRAY_COLOR);
	glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), ((uint8_t *)0) + sizeof(Vector2));

	glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
	glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), ((uint8_t *)0) + sizeof(Vector2) + sizeof(Color));

	glDrawElements(p_primitive, p_index_count, GL_UNSIGNED_INT, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES2::_flush() {
	if (data.mem_vertex_buffer_offset) {
		_draw(data.primitive, data.mem_vertex_buffer_offset, data.mem_vertex_buffer, data.mem_index_buffer_offset, data.mem_index_buffer);
	}

	data.mem_vertex_buffer_offset = 0;
	data.mem_index_buffer_offset = 0;
}

void RasterizerCanvasGLES2::_commit(const int p_vertex_count, const int p_index_count) {
	ERR_FAIL_COND(!p_vertex_count);
	ERR_FAIL_COND(!p_index_count);

	if (state.uniforms.extra_matrix != state.prev_uniforms.extra_matrix ||
			state.uniforms.final_modulate != state.prev_uniforms.final_modulate ||
			state.uniforms.modelview_matrix != state.prev_uniforms.modelview_matrix ||
			state.uniforms.projection_matrix != state.prev_uniforms.projection_matrix ||
			state.uniforms.texpixel_size != state.prev_uniforms.texpixel_size ||
			state.uniforms.time != state.prev_uniforms.time) {

		_set_uniforms();
		state.prev_uniforms = state.uniforms;
		_flush();
	}

	const int new_index_offset = data.mem_index_buffer_offset + p_index_count;

	for (int i = data.mem_index_buffer_offset; i < new_index_offset; ++i)
		data.mem_index_buffer[i] += data.mem_vertex_buffer_offset;

	data.mem_vertex_buffer_offset += p_vertex_count;
	data.mem_index_buffer_offset = new_index_offset;
}

void RasterizerCanvasGLES2::_untile() {
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	state.tiled = false;
}

void RasterizerCanvasGLES2::_bind_shader(RasterizerStorageGLES2::Material *p_material) {
	if (!state.canvas_shader.is_dirty()) {
		return;
	}

	_flush();

	if (state.canvas_shader.bind()) {
		state.canvas_shader.use_material((void *)p_material);
	}
}

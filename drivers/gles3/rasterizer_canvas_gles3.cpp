/*************************************************************************/
/*  rasterizer_canvas_gles3.cpp                                          */
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

#include "rasterizer_canvas_gles3.h"
#include "drivers/gles3/rasterizer_platforms.h"
#ifdef GLES3_BACKEND_ENABLED

#include "core/os/os.h"
#include "drivers/gles3/rasterizer_asserts.h"
#include "rasterizer_scene_gles3.h"
#include "rasterizer_storage_gles3.h"

#include "core/config/project_settings.h"
#include "servers/rendering/rendering_server_default.h"

//static const GLenum gl_primitive[] = {
//	GL_POINTS,
//	GL_LINES,
//	GL_LINE_STRIP,
//	GL_LINE_LOOP,
//	GL_TRIANGLES,
//	GL_TRIANGLE_STRIP,
//	GL_TRIANGLE_FAN
//};

#if 0
void RasterizerCanvasGLES3::_batch_upload_buffers() {
	// noop?
	if (!bdata.vertices.size())
		return;

	glBindBuffer(GL_ARRAY_BUFFER, bdata.gl_vertex_buffer);

	// usage flag is a project setting
	GLenum buffer_usage_flag = GL_DYNAMIC_DRAW;
	if (bdata.buffer_mode_batch_upload_flag_stream) {
		buffer_usage_flag = GL_STREAM_DRAW;
	}

	// orphan the old (for now)
	if (bdata.buffer_mode_batch_upload_send_null) {
		glBufferData(GL_ARRAY_BUFFER, 0, 0, buffer_usage_flag); // GL_DYNAMIC_DRAW);
	}

	switch (bdata.fvf) {
		case RasterizerStorageCommon::FVF_UNBATCHED: // should not happen
			break;
		case RasterizerStorageCommon::FVF_REGULAR: // no change
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertex) * bdata.vertices.size(), bdata.vertices.get_data(), buffer_usage_flag);
			break;
		case RasterizerStorageCommon::FVF_COLOR:
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertexColored) * bdata.unit_vertices.size(), bdata.unit_vertices.get_unit(0), buffer_usage_flag);
			break;
		case RasterizerStorageCommon::FVF_LIGHT_ANGLE:
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertexLightAngled) * bdata.unit_vertices.size(), bdata.unit_vertices.get_unit(0), buffer_usage_flag);
			break;
		case RasterizerStorageCommon::FVF_MODULATED:
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertexModulated) * bdata.unit_vertices.size(), bdata.unit_vertices.get_unit(0), buffer_usage_flag);
			break;
		case RasterizerStorageCommon::FVF_LARGE:
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertexLarge) * bdata.unit_vertices.size(), bdata.unit_vertices.get_unit(0), buffer_usage_flag);
			break;
	}

	// might not be necessary
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES3::_batch_render_lines(const Batch &p_batch, RasterizerStorageGLES3::Material *p_material, bool p_anti_alias) {
	_set_texture_rect_mode(false);

	if (state.canvas_shader.bind()) {
		_set_uniforms();
		state.canvas_shader.use_material((void *)p_material);
	}

	_bind_canvas_texture(RID(), RID());

	glDisableVertexAttribArray(RS::ARRAY_COLOR);
	glVertexAttrib4fv(RS::ARRAY_COLOR, (float *)&p_batch.color);

#ifdef GLES_OVER_GL
	if (p_anti_alias)
		glEnable(GL_LINE_SMOOTH);
#endif

	int sizeof_vert = sizeof(BatchVertex);

	// bind the index and vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, bdata.gl_vertex_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bdata.gl_index_buffer);

	uint64_t pointer = 0;
	glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof_vert, (const void *)pointer);

	glDisableVertexAttribArray(RS::ARRAY_TEX_UV);

	int64_t offset = p_batch.first_vert; // 6 inds per quad at 2 bytes each

	int num_elements = p_batch.num_commands * 2;
	glDrawArrays(GL_LINES, offset, num_elements);

	storage->info.render._2d_draw_call_count++;

	// may not be necessary .. state change optimization still TODO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

#ifdef GLES_OVER_GL
	if (p_anti_alias)
		glDisable(GL_LINE_SMOOTH);
#endif
}

void RasterizerCanvasGLES3::_batch_render_generic(const Batch &p_batch, RasterizerStorageGLES3::Material *p_material) {
	ERR_FAIL_COND(p_batch.num_commands <= 0);

	const bool &use_light_angles = bdata.use_light_angles;
	const bool &use_modulate = bdata.use_modulate;
	const bool &use_large_verts = bdata.use_large_verts;
	const bool &colored_verts = bdata.use_colored_vertices | use_light_angles | use_modulate | use_large_verts;

	int sizeof_vert;

	switch (bdata.fvf) {
		default:
			sizeof_vert = 0; // prevent compiler warning - this should never happen
			break;
		case RasterizerStorageCommon::FVF_UNBATCHED: {
			sizeof_vert = 0; // prevent compiler warning - this should never happen
			return;
		} break;
		case RasterizerStorageCommon::FVF_REGULAR: // no change
			sizeof_vert = sizeof(BatchVertex);
			break;
		case RasterizerStorageCommon::FVF_COLOR:
			sizeof_vert = sizeof(BatchVertexColored);
			break;
		case RasterizerStorageCommon::FVF_LIGHT_ANGLE:
			sizeof_vert = sizeof(BatchVertexLightAngled);
			break;
		case RasterizerStorageCommon::FVF_MODULATED:
			sizeof_vert = sizeof(BatchVertexModulated);
			break;
		case RasterizerStorageCommon::FVF_LARGE:
			sizeof_vert = sizeof(BatchVertexLarge);
			break;
	}

	// make sure to set all conditionals BEFORE binding the shader
	_set_texture_rect_mode(false, use_light_angles, use_modulate, use_large_verts);

	// batch tex
	const BatchTex &tex = bdata.batch_textures[p_batch.batch_texture_id];
	//VSG::rasterizer->gl_check_for_error();

	// force repeat is set if non power of 2 texture, and repeat is needed if hardware doesn't support npot
	if (tex.tile_mode == BatchTex::TILE_FORCE_REPEAT) {
		state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_FORCE_REPEAT, true);
	}

	if (state.canvas_shader.bind()) {
		_set_uniforms();
		state.canvas_shader.use_material((void *)p_material);
	}

	_bind_canvas_texture(tex.RID_texture, tex.RID_normal);

	// bind the index and vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, bdata.gl_vertex_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bdata.gl_index_buffer);

	uint64_t pointer = 0;
	glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof_vert, (const void *)pointer);

	// always send UVs, even within a texture specified because a shader can still use UVs
	glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
	glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (2 * 4)));

	// color
	if (!colored_verts) {
		glDisableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttrib4fv(RS::ARRAY_COLOR, p_batch.color.get_data());
	} else {
		glEnableVertexAttribArray(RS::ARRAY_COLOR);
		glVertexAttribPointer(RS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (4 * 4)));
	}

	if (use_light_angles) {
		glEnableVertexAttribArray(RS::ARRAY_TANGENT);
		glVertexAttribPointer(RS::ARRAY_TANGENT, 1, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (8 * 4)));
	}

	if (use_modulate) {
		glEnableVertexAttribArray(RS::ARRAY_TEX_UV2);
		glVertexAttribPointer(RS::ARRAY_TEX_UV2, 4, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (9 * 4)));
	}

	if (use_large_verts) {
		glEnableVertexAttribArray(RS::ARRAY_BONES);
		glVertexAttribPointer(RS::ARRAY_BONES, 2, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (13 * 4)));
		glEnableVertexAttribArray(RS::ARRAY_WEIGHTS);
		glVertexAttribPointer(RS::ARRAY_WEIGHTS, 4, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (15 * 4)));
	}

	// We only want to set the GL wrapping mode if the texture is not already tiled (i.e. set in Import).
	// This  is an optimization left over from the legacy renderer.
	// If we DID set tiling in the API, and reverted to clamped, then the next draw using this texture
	// may use clamped mode incorrectly.
	bool tex_is_already_tiled = tex.flags & RasterizerStorageGLES3::TEXTURE_FLAG_REPEAT;

	if (tex.tile_mode == BatchTex::TILE_NORMAL) {
		// if the texture is imported as tiled, no need to set GL state, as it will already be bound with repeat
		if (!tex_is_already_tiled) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		}
	}

	// we need to convert explicitly from pod Vec2 to Vector2 ...
	// could use a cast but this might be unsafe in future
	Vector2 tps;
	tex.tex_pixel_size.to(tps);
	state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, tps);

	switch (p_batch.type) {
		default: {
			// prevent compiler warning
		} break;
		case RasterizerStorageCommon::BT_RECT: {
			int64_t offset = p_batch.first_vert * 3;

			int num_elements = p_batch.num_commands * 6;
			glDrawElements(GL_TRIANGLES, num_elements, GL_UNSIGNED_SHORT, (void *)offset);
		} break;
		case RasterizerStorageCommon::BT_POLY: {
			int64_t offset = p_batch.first_vert;

			int num_elements = p_batch.num_commands;
			glDrawArrays(GL_TRIANGLES, offset, num_elements);
		} break;
	}

	storage->info.render._2d_draw_call_count++;

	switch (tex.tile_mode) {
		case BatchTex::TILE_FORCE_REPEAT: {
			state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_FORCE_REPEAT, false);
		} break;
		case BatchTex::TILE_NORMAL: {
			// if the texture is imported as tiled, no need to revert GL state
			if (!tex_is_already_tiled) {
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
		} break;
		default: {
		} break;
	}

	// could these have ifs?
	glDisableVertexAttribArray(RS::ARRAY_TEX_UV);
	glDisableVertexAttribArray(RS::ARRAY_COLOR);
	glDisableVertexAttribArray(RS::ARRAY_TANGENT);
	glDisableVertexAttribArray(RS::ARRAY_TEX_UV2);
	glDisableVertexAttribArray(RS::ARRAY_BONES);
	glDisableVertexAttribArray(RS::ARRAY_WEIGHTS);

	// may not be necessary .. state change optimization still TODO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
#endif
void RasterizerCanvasGLES3::render_batches(Item::Command *const *p_commands, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES3::Material *p_material) {
	int num_batches = bdata.batches.size();

	for (int batch_num = 0; batch_num < num_batches; batch_num++) {
		const Batch &batch = bdata.batches[batch_num];

		switch (batch.type) {
			case RasterizerStorageCommon::BT_RECT: {
				//_batch_render_generic(batch, p_material);
			} break;
			case RasterizerStorageCommon::BT_POLY: {
				//_batch_render_generic(batch, p_material);
			} break;
			case RasterizerStorageCommon::BT_LINE: {
				//_batch_render_lines(batch, p_material, false);
			} break;
			case RasterizerStorageCommon::BT_LINE_AA: {
				//_batch_render_lines(batch, p_material, true);
			} break;
			default: {
				int end_command = batch.first_command + batch.num_commands;

				for (int i = batch.first_command; i < end_command; i++) {
					Item::Command *command = p_commands[i];

					switch (command->type) {
#if 0
						case Item::Command::TYPE_LINE: {
							Item::CommandLine *line = static_cast<Item::CommandLine *>(command);

							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							_bind_canvas_texture(RID(), RID());

							glDisableVertexAttribArray(RS::ARRAY_COLOR);
							glVertexAttrib4fv(RS::ARRAY_COLOR, line->color.components);

							state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);

							if (line->width <= 1) {
								Vector2 verts[2] = {
									Vector2(line->from.x, line->from.y),
									Vector2(line->to.x, line->to.y)
								};

#ifdef GLES_OVER_GL
								if (line->antialiased)
									glEnable(GL_LINE_SMOOTH);
#endif
								_draw_gui_primitive(2, verts, NULL, NULL);

#ifdef GLES_OVER_GL
								if (line->antialiased)
									glDisable(GL_LINE_SMOOTH);
#endif
							} else {
								Vector2 t = (line->from - line->to).normalized().tangent() * line->width * 0.5;

								Vector2 verts[4] = {
									line->from - t,
									line->from + t,
									line->to + t,
									line->to - t
								};

								_draw_gui_primitive(4, verts, NULL, NULL);
#ifdef GLES_OVER_GL
								if (line->antialiased) {
									glEnable(GL_LINE_SMOOTH);
									for (int j = 0; j < 4; j++) {
										Vector2 vertsl[2] = {
											verts[j],
											verts[(j + 1) % 4],
										};
										_draw_gui_primitive(2, vertsl, NULL, NULL);
									}
									glDisable(GL_LINE_SMOOTH);
								}
#endif
							}
						} break;
#endif
						case Item::Command::TYPE_PRIMITIVE: {
							Item::CommandPrimitive *pr = static_cast<Item::CommandPrimitive *>(command);

							switch (pr->point_count) {
								case 2: {
									_legacy_draw_line(pr, p_material);
								} break;
								default: {
									_legacy_draw_primitive(pr, p_material);
								} break;
							}

						} break;

						case Item::Command::TYPE_RECT: {
							Item::CommandRect *r = static_cast<Item::CommandRect *>(command);
							_bind_quad_buffer();
							glDisableVertexAttribArray(RS::ARRAY_COLOR);
							glVertexAttrib4fv(RS::ARRAY_COLOR, r->modulate.components);

							bool can_tile = true;

							// we will take account of render target textures which need to be drawn upside down
							// quirk of opengl
							bool upside_down = r->flags & CANVAS_RECT_FLIP_V;

							// very inefficient, improve this
							if (r->texture.is_valid()) {
								RasterizerStorageGLES3::Texture *texture = storage->texture_owner.get_or_null(r->texture);

								if (texture) {
									if (texture->is_upside_down())
										upside_down = true;
								}
							}

							if (r->texture.is_valid() && r->flags & CANVAS_RECT_TILE && !storage->config.support_npot_repeat_mipmap) {
								// workaround for when setting tiling does not work due to hardware limitation

								RasterizerStorageGLES3::Texture *texture = storage->texture_owner.get_or_null(r->texture);

								if (texture) {
									texture = texture->get_ptr();

									if (next_power_of_2(texture->alloc_width) != (unsigned int)texture->alloc_width && next_power_of_2(texture->alloc_height) != (unsigned int)texture->alloc_height) {
										state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_FORCE_REPEAT, true);
										can_tile = false;
									}
								}
							}

							// On some widespread Nvidia cards, the normal draw method can produce some
							// flickering in draw_rect and especially TileMap rendering (tiles randomly flicker).
							// See GH-9913.
							// To work it around, we use a simpler draw method which does not flicker, but gives
							// a non negligible performance hit, so it's opt-in (GH-24466).
							if (use_nvidia_rect_workaround) {
								// are we using normal maps, if so we want to use light angle
								bool send_light_angles = false;

								// only need to use light angles when normal mapping
								// otherwise we can use the default shader
								if (state.current_normal != RID()) {
									send_light_angles = true;
								}

								_set_texture_rect_mode(false, send_light_angles);

								if (state.canvas_shader.bind()) {
									_set_uniforms();
									state.canvas_shader.use_material((void *)p_material);
								}

								Vector2 points[4] = {
									r->rect.position,
									r->rect.position + Vector2(r->rect.size.x, 0.0),
									r->rect.position + r->rect.size,
									r->rect.position + Vector2(0.0, r->rect.size.y),
								};

								if (r->rect.size.x < 0) {
									SWAP(points[0], points[1]);
									SWAP(points[2], points[3]);
								}
								if (r->rect.size.y < 0) {
									SWAP(points[0], points[3]);
									SWAP(points[1], points[2]);
								}

								// FTODO
								//RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(r->texture, r->normal_map);
								RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(r->texture, RID());

								if (texture) {
									Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);

									Rect2 src_rect = (r->flags & CANVAS_RECT_REGION) ? Rect2(r->source.position * texpixel_size, r->source.size * texpixel_size) : Rect2(0, 0, 1, 1);

									Vector2 uvs[4] = {
										src_rect.position,
										src_rect.position + Vector2(src_rect.size.x, 0.0),
										src_rect.position + src_rect.size,
										src_rect.position + Vector2(0.0, src_rect.size.y),
									};

									// for encoding in light angle
									bool flip_h = false;
									bool flip_v = false;

									if (r->flags & CANVAS_RECT_TRANSPOSE) {
										SWAP(uvs[1], uvs[3]);
									}

									if (r->flags & CANVAS_RECT_FLIP_H) {
										SWAP(uvs[0], uvs[1]);
										SWAP(uvs[2], uvs[3]);
										flip_h = true;
										flip_v = !flip_v;
									}
									if (upside_down) {
										SWAP(uvs[0], uvs[3]);
										SWAP(uvs[1], uvs[2]);
										flip_v = !flip_v;
									}

									state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);

									bool untile = false;

									if (can_tile && r->flags & CANVAS_RECT_TILE && !(texture->flags & RasterizerStorageGLES3::TEXTURE_FLAG_REPEAT)) {
										texture->GLSetRepeat(RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
										untile = true;
									}

									if (send_light_angles) {
										// for single rects, there is no need to fully utilize the light angle,
										// we only need it to encode flips (horz and vert). But the shader can be reused with
										// batching in which case the angle encodes the transform as well as
										// the flips.
										// Note transpose is NYI. I don't think it worked either with the non-nvidia method.

										// if horizontal flip, angle is 180
										float angle = 0.0f;
										if (flip_h)
											angle = Math_PI;

										// add 1 (to take care of zero floating point error with sign)
										angle += 1.0f;

										// flip if necessary
										if (flip_v)
											angle *= -1.0f;

										// light angle must be sent for each vert, instead as a single uniform in the uniform draw method
										// this has the benefit of enabling batching with light angles.
										float light_angles[4] = { angle, angle, angle, angle };

										_draw_gui_primitive(4, points, NULL, uvs, light_angles);
									} else {
										_draw_gui_primitive(4, points, NULL, uvs);
									}

									if (untile) {
										texture->GLSetRepeat(RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
									}
								} else {
									static const Vector2 uvs[4] = {
										Vector2(0.0, 0.0),
										Vector2(0.0, 1.0),
										Vector2(1.0, 1.0),
										Vector2(1.0, 0.0),
									};

									state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, Vector2());
									_draw_gui_primitive(4, points, NULL, uvs);
								}

							} else {
								// This branch is better for performance, but can produce flicker on Nvidia, see above comment.

								_set_texture_rect_mode(true);

								if (state.canvas_shader.bind()) {
									_set_uniforms();
									state.canvas_shader.use_material((void *)p_material);
								}
								_bind_quad_buffer();

								// FTODO
								//RasterizerStorageGLES3::Texture *tex = _bind_canvas_texture(r->texture, r->normal_map);
								RasterizerStorageGLES3::Texture *tex = _bind_canvas_texture(r->texture, RID());

								if (!tex) {
									Rect2 dst_rect = Rect2(r->rect.position, r->rect.size);

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

									glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
									storage->info.render._2d_draw_call_count++;
								} else {
									bool untile = false;

									if (can_tile && r->flags & CANVAS_RECT_TILE && !(tex->flags & RasterizerStorageGLES3::TEXTURE_FLAG_REPEAT)) {
										tex->GLSetRepeat(RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
										untile = true;
									}

									Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);
									Rect2 src_rect = (r->flags & CANVAS_RECT_REGION) ? Rect2(r->source.position * texpixel_size, r->source.size * texpixel_size) : Rect2(0, 0, 1, 1);

									Rect2 dst_rect = Rect2(r->rect.position, r->rect.size);

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

									if (upside_down) {
										src_rect.size.y *= -1;
									}

									if (r->flags & CANVAS_RECT_TRANSPOSE) {
										dst_rect.size.x *= -1; // Encoding in the dst_rect.z uniform
									}

									state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);

									state.canvas_shader.set_uniform(CanvasShaderGLES3::DST_RECT, Color(dst_rect.position.x, dst_rect.position.y, dst_rect.size.x, dst_rect.size.y));
									state.canvas_shader.set_uniform(CanvasShaderGLES3::SRC_RECT, Color(src_rect.position.x, src_rect.position.y, src_rect.size.x, src_rect.size.y));

									glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
									storage->info.render._2d_draw_call_count++;

									if (untile) {
										tex->GLSetRepeat(RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
									}
								}

								glBindBuffer(GL_ARRAY_BUFFER, 0);
								glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
							}

							state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_FORCE_REPEAT, false);

						} break;
						case Item::Command::TYPE_NINEPATCH: {
							Item::CommandNinePatch *np = static_cast<Item::CommandNinePatch *>(command);

							_set_texture_rect_mode(false);
							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}
							_bind_quad_buffer();

							glDisableVertexAttribArray(RS::ARRAY_COLOR);
							glVertexAttrib4fv(RS::ARRAY_COLOR, np->color.components);

							// FTODO
							//RasterizerStorageGLES3::Texture *tex = _bind_canvas_texture(np->texture, np->normal_map);
							RasterizerStorageGLES3::Texture *tex = _bind_canvas_texture(np->texture, RID());

							if (!tex) {
								// FIXME: Handle textureless ninepatch gracefully
								WARN_PRINT("NinePatch without texture not supported yet in OpenGL backend, skipping.");
								continue;
							}
							if (tex->width == 0 || tex->height == 0) {
								WARN_PRINT("Cannot set empty texture to NinePatch.");
								continue;
							}

							Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);

							// state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);
							state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);

							Rect2 source = np->source;
							if (source.size.x == 0 && source.size.y == 0) {
								source.size.x = tex->width;
								source.size.y = tex->height;
							}

							float screen_scale = 1.0;

							if ((bdata.settings_ninepatch_mode == 1) && (source.size.x != 0) && (source.size.y != 0)) {
								screen_scale = MIN(np->rect.size.x / source.size.x, np->rect.size.y / source.size.y);
								screen_scale = MIN(1.0, screen_scale);
							}

							// prepare vertex buffer

							// this buffer contains [ POS POS UV UV ] *

							float buffer[16 * 2 + 16 * 2];

							{
								// first row

								buffer[(0 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(0 * 4 * 4) + 1] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(0 * 4 * 4) + 3] = source.position.y * texpixel_size.y;

								buffer[(0 * 4 * 4) + 4] = np->rect.position.x + np->margin[SIDE_LEFT] * screen_scale;
								buffer[(0 * 4 * 4) + 5] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 6] = (source.position.x + np->margin[SIDE_LEFT]) * texpixel_size.x;
								buffer[(0 * 4 * 4) + 7] = source.position.y * texpixel_size.y;

								buffer[(0 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[SIDE_RIGHT] * screen_scale;
								buffer[(0 * 4 * 4) + 9] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[SIDE_RIGHT]) * texpixel_size.x;
								buffer[(0 * 4 * 4) + 11] = source.position.y * texpixel_size.y;

								buffer[(0 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(0 * 4 * 4) + 13] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(0 * 4 * 4) + 15] = source.position.y * texpixel_size.y;

								// second row

								buffer[(1 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(1 * 4 * 4) + 1] = np->rect.position.y + np->margin[SIDE_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(1 * 4 * 4) + 3] = (source.position.y + np->margin[SIDE_TOP]) * texpixel_size.y;

								buffer[(1 * 4 * 4) + 4] = np->rect.position.x + np->margin[SIDE_LEFT] * screen_scale;
								buffer[(1 * 4 * 4) + 5] = np->rect.position.y + np->margin[SIDE_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 6] = (source.position.x + np->margin[SIDE_LEFT]) * texpixel_size.x;
								buffer[(1 * 4 * 4) + 7] = (source.position.y + np->margin[SIDE_TOP]) * texpixel_size.y;

								buffer[(1 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[SIDE_RIGHT] * screen_scale;
								buffer[(1 * 4 * 4) + 9] = np->rect.position.y + np->margin[SIDE_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[SIDE_RIGHT]) * texpixel_size.x;
								buffer[(1 * 4 * 4) + 11] = (source.position.y + np->margin[SIDE_TOP]) * texpixel_size.y;

								buffer[(1 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(1 * 4 * 4) + 13] = np->rect.position.y + np->margin[SIDE_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(1 * 4 * 4) + 15] = (source.position.y + np->margin[SIDE_TOP]) * texpixel_size.y;

								// third row

								buffer[(2 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(2 * 4 * 4) + 1] = np->rect.position.y + np->rect.size.y - np->margin[SIDE_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(2 * 4 * 4) + 3] = (source.position.y + source.size.y - np->margin[SIDE_BOTTOM]) * texpixel_size.y;

								buffer[(2 * 4 * 4) + 4] = np->rect.position.x + np->margin[SIDE_LEFT] * screen_scale;
								buffer[(2 * 4 * 4) + 5] = np->rect.position.y + np->rect.size.y - np->margin[SIDE_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 6] = (source.position.x + np->margin[SIDE_LEFT]) * texpixel_size.x;
								buffer[(2 * 4 * 4) + 7] = (source.position.y + source.size.y - np->margin[SIDE_BOTTOM]) * texpixel_size.y;

								buffer[(2 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[SIDE_RIGHT] * screen_scale;
								buffer[(2 * 4 * 4) + 9] = np->rect.position.y + np->rect.size.y - np->margin[SIDE_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[SIDE_RIGHT]) * texpixel_size.x;
								buffer[(2 * 4 * 4) + 11] = (source.position.y + source.size.y - np->margin[SIDE_BOTTOM]) * texpixel_size.y;

								buffer[(2 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(2 * 4 * 4) + 13] = np->rect.position.y + np->rect.size.y - np->margin[SIDE_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(2 * 4 * 4) + 15] = (source.position.y + source.size.y - np->margin[SIDE_BOTTOM]) * texpixel_size.y;

								// fourth row

								buffer[(3 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(3 * 4 * 4) + 1] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(3 * 4 * 4) + 3] = (source.position.y + source.size.y) * texpixel_size.y;

								buffer[(3 * 4 * 4) + 4] = np->rect.position.x + np->margin[SIDE_LEFT] * screen_scale;
								buffer[(3 * 4 * 4) + 5] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 6] = (source.position.x + np->margin[SIDE_LEFT]) * texpixel_size.x;
								buffer[(3 * 4 * 4) + 7] = (source.position.y + source.size.y) * texpixel_size.y;

								buffer[(3 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[SIDE_RIGHT] * screen_scale;
								buffer[(3 * 4 * 4) + 9] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[SIDE_RIGHT]) * texpixel_size.x;
								buffer[(3 * 4 * 4) + 11] = (source.position.y + source.size.y) * texpixel_size.y;

								buffer[(3 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(3 * 4 * 4) + 13] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(3 * 4 * 4) + 15] = (source.position.y + source.size.y) * texpixel_size.y;
							}

							glBindBuffer(GL_ARRAY_BUFFER, data.ninepatch_vertices);
							glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (16 + 16) * 2, buffer, _buffer_upload_usage_flag);

							glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.ninepatch_elements);

							//glEnableVertexAttribArray(RS::ARRAY_VERTEX);
							glEnableVertexAttribArray(RS::ARRAY_TEX_UV);

							//glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
							glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), CAST_INT_TO_UCHAR_PTR((sizeof(float) * 2)));

							//glDrawElements(GL_TRIANGLES, 18 * 3 - (np->draw_center ? 0 : 6), GL_UNSIGNED_SHORT, NULL);

							glBindBuffer(GL_ARRAY_BUFFER, 0);
							glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
							storage->info.render._2d_draw_call_count++;

						} break;
#if 0
						case Item::Command::TYPE_CIRCLE: {
							Item::CommandCircle *circle = static_cast<Item::CommandCircle *>(command);

							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							static const int num_points = 32;

							Vector2 points[num_points + 1];
							points[num_points] = circle->pos;

							int indices[num_points * 3];

							for (int j = 0; j < num_points; j++) {
								points[j] = circle->pos + Vector2(Math::sin(j * Math_PI * 2.0 / num_points), Math::cos(j * Math_PI * 2.0 / num_points)) * circle->radius;
								indices[j * 3 + 0] = j;
								indices[j * 3 + 1] = (j + 1) % num_points;
								indices[j * 3 + 2] = num_points;
							}

							_bind_canvas_texture(RID(), RID());

							_draw_polygon(indices, num_points * 3, num_points + 1, points, NULL, &circle->color, true);
						} break;
#endif
						case Item::Command::TYPE_POLYGON: {
							Item::CommandPolygon *polygon = static_cast<Item::CommandPolygon *>(command);
							//const PolyData &pd = _polydata[polygon->polygon.polygon_id];

							switch (polygon->primitive) {
								case RS::PRIMITIVE_TRIANGLES: {
									_legacy_draw_poly_triangles(polygon, p_material);
								} break;
								default:
									break;
							}

						} break;
#if 0
						case Item::Command::TYPE_MESH: {
							Item::CommandMesh *mesh = static_cast<Item::CommandMesh *>(command);

							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(mesh->texture, mesh->normal_map);

							if (texture) {
								Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
								state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
							}

							RasterizerStorageGLES3::Mesh *mesh_data = storage->mesh_owner.get_or_null(mesh->mesh);
							if (mesh_data) {
								for (int j = 0; j < mesh_data->surfaces.size(); j++) {
									RasterizerStorageGLES3::Surface *s = mesh_data->surfaces[j];
									// materials are ignored in 2D meshes, could be added but many things (ie, lighting mode, reading from screen, etc) would break as they are not meant be set up at this point of drawing

									glBindBuffer(GL_ARRAY_BUFFER, s->vertex_id);

									if (s->index_array_len > 0) {
										glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
									}

									for (int k = 0; k < RS::ARRAY_MAX - 1; k++) {
										if (s->attribs[k].enabled) {
											glEnableVertexAttribArray(k);
											glVertexAttribPointer(s->attribs[k].index, s->attribs[k].size, s->attribs[k].type, s->attribs[k].normalized, s->attribs[k].stride, CAST_INT_TO_UCHAR_PTR(s->attribs[k].offset));
										} else {
											glDisableVertexAttribArray(k);
											switch (k) {
												case RS::ARRAY_NORMAL: {
													glVertexAttrib4f(RS::ARRAY_NORMAL, 0.0, 0.0, 1, 1);
												} break;
												case RS::ARRAY_COLOR: {
													glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);

												} break;
												default: {
												}
											}
										}
									}

									if (s->index_array_len > 0) {
										glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
									} else {
										glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
									}
								}

								for (int j = 1; j < RS::ARRAY_MAX - 1; j++) {
									glDisableVertexAttribArray(j);
								}
							}

							storage->info.render._2d_draw_call_count++;
						} break;
						case Item::Command::TYPE_MULTIMESH: {
							Item::CommandMultiMesh *mmesh = static_cast<Item::CommandMultiMesh *>(command);

							RasterizerStorageGLES3::MultiMesh *multi_mesh = storage->multimesh_owner.get_or_null(mmesh->multimesh);

							if (!multi_mesh)
								break;

							RasterizerStorageGLES3::Mesh *mesh_data = storage->mesh_owner.get_or_null(multi_mesh->mesh);

							if (!mesh_data)
								break;

							state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, multi_mesh->custom_data_format != RS::MULTIMESH_CUSTOM_DATA_NONE);
							state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, true);
							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(mmesh->texture, mmesh->normal_map);

							if (texture) {
								Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
								state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
							}

							//reset shader and force rebind

							int amount = MIN(multi_mesh->size, multi_mesh->visible_instances);

							if (amount == -1) {
								amount = multi_mesh->size;
							}

							int stride = multi_mesh->color_floats + multi_mesh->custom_data_floats + multi_mesh->xform_floats;

							int color_ofs = multi_mesh->xform_floats;
							int custom_data_ofs = color_ofs + multi_mesh->color_floats;

							// drawing

							const float *base_buffer = multi_mesh->data.ptr();

							for (int j = 0; j < mesh_data->surfaces.size(); j++) {
								RasterizerStorageGLES3::Surface *s = mesh_data->surfaces[j];
								// materials are ignored in 2D meshes, could be added but many things (ie, lighting mode, reading from screen, etc) would break as they are not meant be set up at this point of drawing

								//bind buffers for mesh surface
								glBindBuffer(GL_ARRAY_BUFFER, s->vertex_id);

								if (s->index_array_len > 0) {
									glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
								}

								for (int k = 0; k < RS::ARRAY_MAX - 1; k++) {
									if (s->attribs[k].enabled) {
										glEnableVertexAttribArray(k);
										glVertexAttribPointer(s->attribs[k].index, s->attribs[k].size, s->attribs[k].type, s->attribs[k].normalized, s->attribs[k].stride, CAST_INT_TO_UCHAR_PTR(s->attribs[k].offset));
									} else {
										glDisableVertexAttribArray(k);
										switch (k) {
											case RS::ARRAY_NORMAL: {
												glVertexAttrib4f(RS::ARRAY_NORMAL, 0.0, 0.0, 1, 1);
											} break;
											case RS::ARRAY_COLOR: {
												glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);

											} break;
											default: {
											}
										}
									}
								}

								for (int k = 0; k < amount; k++) {
									const float *buffer = base_buffer + k * stride;

									{
										glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 0, &buffer[0]);
										glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 1, &buffer[4]);
										if (multi_mesh->transform_format == RS::MULTIMESH_TRANSFORM_3D) {
											glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 2, &buffer[8]);
										} else {
											glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 2, 0.0, 0.0, 1.0, 0.0);
										}
									}

									if (multi_mesh->color_floats) {
										if (multi_mesh->color_format == RS::MULTIMESH_COLOR_8BIT) {
											uint8_t *color_data = (uint8_t *)(buffer + color_ofs);
											glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 3, color_data[0] / 255.0, color_data[1] / 255.0, color_data[2] / 255.0, color_data[3] / 255.0);
										} else {
											glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 3, buffer + color_ofs);
										}
									} else {
										glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 3, 1.0, 1.0, 1.0, 1.0);
									}

									if (multi_mesh->custom_data_floats) {
										if (multi_mesh->custom_data_format == RS::MULTIMESH_CUSTOM_DATA_8BIT) {
											uint8_t *custom_data = (uint8_t *)(buffer + custom_data_ofs);
											glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 4, custom_data[0] / 255.0, custom_data[1] / 255.0, custom_data[2] / 255.0, custom_data[3] / 255.0);
										} else {
											glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 4, buffer + custom_data_ofs);
										}
									}

									if (s->index_array_len > 0) {
										glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
									} else {
										glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
									}
								}
							}

							// LIGHT ANGLE PR replaced USE_INSTANCE_CUSTOM line with below .. think it was a typo,
							// but just in case, made this note.
							//_set_texture_rect_mode(false);
							state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, false);
							state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, false);

							storage->info.render._2d_draw_call_count++;
						} break;
						case Item::Command::TYPE_POLYLINE: {
							Item::CommandPolyLine *pline = static_cast<Item::CommandPolyLine *>(command);

							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

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
									_draw_generic(GL_LINE_STRIP, pline->lines.size(), pline->lines.ptr(), NULL, pline->line_colors.ptr(), pline->line_colors.size() == 1);
								}

#ifdef GLES_OVER_GL
								if (pline->antialiased)
									glDisable(GL_LINE_SMOOTH);
#endif
							}
						} break;

						case Item::Command::TYPE_PRIMITIVE: {
							Item::CommandPrimitive *primitive = static_cast<Item::CommandPrimitive *>(command);
							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							ERR_CONTINUE(primitive->points.size() < 1);

							RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(primitive->texture, primitive->normal_map);

							if (texture) {
								Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
								state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
							}

							// we need a temporary because this must be nulled out
							// if only a single color specified
							const Color *colors = primitive->colors.ptr();
							if (primitive->colors.size() == 1 && primitive->points.size() > 1) {
								Color c = primitive->colors[0];
								glVertexAttrib4f(RS::ARRAY_COLOR, c.r, c.g, c.b, c.a);
								colors = nullptr;
							} else if (primitive->colors.empty()) {
								glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1);
							}
#ifdef RASTERIZER_EXTRA_CHECKS
							else {
								RAST_DEV_DEBUG_ASSERT(primitive->colors.size() == primitive->points.size());
							}

							if (primitive->uvs.ptr()) {
								RAST_DEV_DEBUG_ASSERT(primitive->uvs.size() == primitive->points.size());
							}
#endif

							_draw_gui_primitive(primitive->points.size(), primitive->points.ptr(), colors, primitive->uvs.ptr());
						} break;
#endif
						case Item::Command::TYPE_TRANSFORM: {
							Item::CommandTransform *transform = static_cast<Item::CommandTransform *>(command);
							state.uniforms.extra_matrix = transform->xform;
							state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, state.uniforms.extra_matrix);
						} break;

						case Item::Command::TYPE_PARTICLES: {
						} break;
						case Item::Command::TYPE_CLIP_IGNORE: {
							Item::CommandClipIgnore *ci = static_cast<Item::CommandClipIgnore *>(command);
							if (p_current_clip) {
								if (ci->ignore != r_reclip) {
									if (ci->ignore) {
										glDisable(GL_SCISSOR_TEST);
										r_reclip = true;
									} else {
										glEnable(GL_SCISSOR_TEST);

										int x = p_current_clip->final_clip_rect.position.x;
										int y = storage->frame.current_rt->height - (p_current_clip->final_clip_rect.position.y + p_current_clip->final_clip_rect.size.y);
										int w = p_current_clip->final_clip_rect.size.x;
										int h = p_current_clip->final_clip_rect.size.y;

										// FTODO
										//										if (storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_VFLIP])
										//											y = p_current_clip->final_clip_rect.position.y;

										glScissor(x, y, w, h);

										r_reclip = false;
									}
								}
							}

						} break;
						default: {
							// FIXME: Proper error handling if relevant
							//print_line("other");
						} break;
					}
				}

			} // default
			break;
		}
	}
}

void RasterizerCanvasGLES3::canvas_end() {
	batch_canvas_end();
	RasterizerCanvasBaseGLES3::canvas_end();
}

void RasterizerCanvasGLES3::canvas_begin() {
	batch_canvas_begin();
	RasterizerCanvasBaseGLES3::canvas_begin();
}

void RasterizerCanvasGLES3::canvas_render_items_begin(const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {
	batch_canvas_render_items_begin(p_modulate, p_light, p_base_transform);
}

void RasterizerCanvasGLES3::canvas_render_items_end() {
	batch_canvas_render_items_end();
}

void RasterizerCanvasGLES3::canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) {
	storage->frame.current_rt = nullptr;

	// first set the current render target
	storage->_set_current_render_target(p_to_render_target);

	// binds the render target (framebuffer)
	canvas_begin();

	canvas_render_items_begin(p_modulate, p_light_list, p_canvas_transform);
	canvas_render_items_internal(p_item_list, 0, p_modulate, p_light_list, p_canvas_transform);
	canvas_render_items_end();

	canvas_end();

	// not sure why these are needed to get frame to render?
	storage->_set_current_render_target(RID());
	//		storage->frame.current_rt = nullptr;
	//		canvas_begin();
	//		canvas_end();
}

void RasterizerCanvasGLES3::canvas_render_items_internal(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {
	batch_canvas_render_items(p_item_list, p_z, p_modulate, p_light, p_base_transform);

	//glClearColor(Math::randf(), 0, 1, 1);
}

void RasterizerCanvasGLES3::canvas_render_items_implementation(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {
	// parameters are easier to pass around in a structure
	RenderItemState ris;
	ris.item_group_z = p_z;
	ris.item_group_modulate = p_modulate;
	ris.item_group_light = p_light;
	ris.item_group_base_transform = p_base_transform;

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SKELETON, false);

	state.current_tex = RID();
	state.current_tex_ptr = NULL;
	state.current_normal = RID();
	state.canvas_texscreen_used = false;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	while (p_item_list) {
		Item *ci = p_item_list;
		_legacy_canvas_render_item(ci, ris);
		p_item_list = p_item_list->next;
	}

	if (ris.current_clip) {
		glDisable(GL_SCISSOR_TEST);
	}

	state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SKELETON, false);
}

// Legacy non-batched implementation for regression testing.
// Should be removed after testing phase to avoid duplicate codepaths.
void RasterizerCanvasGLES3::_legacy_canvas_render_item(Item *p_ci, RenderItemState &r_ris) {
	storage->info.render._2d_item_count++;

	// defaults
	state.current_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
	state.current_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;

	if (p_ci->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT) {
		state.current_filter = p_ci->texture_filter;
	}

	if (p_ci->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) {
		state.current_repeat = p_ci->texture_repeat;
	}

	if (r_ris.current_clip != p_ci->final_clip_owner) {
		r_ris.current_clip = p_ci->final_clip_owner;

		if (r_ris.current_clip) {
			glEnable(GL_SCISSOR_TEST);
			int y = storage->_dims.rt_height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
			//			int y = storage->frame.current_rt->height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
			// FTODO
			//			if (storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_VFLIP])
			//			y = r_ris.current_clip->final_clip_rect.position.y;
			glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);

			// debug VFLIP
			//			if ((r_ris.current_clip->final_clip_rect.position.x == 223)
			//					&& (y == 54)
			//					&& (r_ris.current_clip->final_clip_rect.size.width == 1383))
			//			{
			//				glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);
			//			}

		} else {
			glDisable(GL_SCISSOR_TEST);
		}
	}

	// TODO: copy back buffer

	if (p_ci->copy_back_buffer) {
		if (p_ci->copy_back_buffer->full) {
			_copy_texscreen(Rect2());
		} else {
			_copy_texscreen(p_ci->copy_back_buffer->rect);
		}
	}

#if 0
	RasterizerStorageGLES3::Skeleton *skeleton = NULL;

	{
		//skeleton handling
		if (p_ci->skeleton.is_valid() && storage->skeleton_owner.owns(p_ci->skeleton)) {
			skeleton = storage->skeleton_owner.get(p_ci->skeleton);
			if (!skeleton->use_2d) {
				skeleton = NULL;
			} else {
				state.skeleton_transform = r_ris.item_group_base_transform * skeleton->base_transform_2d;
				state.skeleton_transform_inverse = state.skeleton_transform.affine_inverse();
				state.skeleton_texture_size = Vector2(skeleton->size * 2, 0);
			}
		}

		bool use_skeleton = skeleton != NULL;
		if (r_ris.prev_use_skeleton != use_skeleton) {
			r_ris.rebind_shader = true;
			state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SKELETON, use_skeleton);
			r_ris.prev_use_skeleton = use_skeleton;
		}

		if (skeleton) {
			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 3);
			glBindTexture(GL_TEXTURE_2D, skeleton->tex_id);
			state.using_skeleton = true;
		} else {
			state.using_skeleton = false;
		}
	}
#endif

	Item *material_owner = p_ci->material_owner ? p_ci->material_owner : p_ci;

	RID material = material_owner->material;
	RasterizerStorageGLES3::Material *material_ptr = storage->material_owner.get_or_null(material);

	if (material != r_ris.canvas_last_material || r_ris.rebind_shader) {
		RasterizerStorageGLES3::Shader *shader_ptr = NULL;

		if (material_ptr) {
			shader_ptr = material_ptr->shader;

			if (shader_ptr && shader_ptr->mode != RS::SHADER_CANVAS_ITEM) {
				shader_ptr = NULL; // not a canvas item shader, don't use.
			}
		}

		if (shader_ptr) {
			if (shader_ptr->canvas_item.uses_screen_texture) {
				if (!state.canvas_texscreen_used) {
					//copy if not copied before
					_copy_texscreen(Rect2());

					// blend mode will have been enabled so make sure we disable it again later on
					//last_blend_mode = last_blend_mode != RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_DISABLED ? last_blend_mode : -1;
				}

				if (storage->frame.current_rt->copy_screen_effect.color) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
					glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->copy_screen_effect.color);
				}
			}

			if (shader_ptr != r_ris.shader_cache) {
				if (shader_ptr->canvas_item.uses_time) {
					RenderingServerDefault::redraw_request();
				}

				state.canvas_shader.set_custom_shader(shader_ptr->custom_code_id);
				state.canvas_shader.bind();
			}

			int tc = material_ptr->textures.size();
			Pair<StringName, RID> *textures = material_ptr->textures.ptrw();

			ShaderLanguage::ShaderNode::Uniform::Hint *texture_hints = shader_ptr->texture_hints.ptrw();

			for (int i = 0; i < tc; i++) {
				glActiveTexture(GL_TEXTURE0 + i);

				RasterizerStorageGLES3::Texture *t = storage->texture_owner.get_or_null(textures[i].second);

				if (!t) {
					switch (texture_hints[i]) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO:
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK: {
							glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_ANISOTROPY: {
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

				if (t->redraw_if_visible) {
					RenderingServerDefault::redraw_request();
				}

				t = t->get_ptr();

#ifdef TOOLS_ENABLED
				if (t->detect_normal && texture_hints[i] == ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL) {
					t->detect_normal(t->detect_normal_ud);
				}
#endif
				if (t->render_target)
					t->render_target->used_in_frame = true;

				glBindTexture(t->target, t->tex_id);
			}

		} else {
			state.canvas_shader.set_custom_shader(0);
			state.canvas_shader.bind();
		}
		state.canvas_shader.use_material((void *)material_ptr);

		r_ris.shader_cache = shader_ptr;

		r_ris.canvas_last_material = material;

		r_ris.rebind_shader = false;
	}

	int blend_mode = r_ris.shader_cache ? r_ris.shader_cache->canvas_item.blend_mode : RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX;
	bool unshaded = r_ris.shader_cache && (r_ris.shader_cache->canvas_item.light_mode == RasterizerStorageGLES3::Shader::CanvasItem::LIGHT_MODE_UNSHADED || (blend_mode != RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX && blend_mode != RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_PMALPHA));
	bool reclip = false;

	if (r_ris.last_blend_mode != blend_mode) {
		switch (blend_mode) {
			case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
				}

			} break;
			case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_ADD: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
				}

			} break;
			case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_SUB: {
				glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
				}
			} break;
			case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MUL: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
				} else {
					glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
				}
			} break;
			case RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_PMALPHA: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
				}
			} break;
		}
	}

	state.uniforms.final_modulate = unshaded ? p_ci->final_modulate : Color(p_ci->final_modulate.r * r_ris.item_group_modulate.r, p_ci->final_modulate.g * r_ris.item_group_modulate.g, p_ci->final_modulate.b * r_ris.item_group_modulate.b, p_ci->final_modulate.a * r_ris.item_group_modulate.a);

	state.uniforms.modelview_matrix = p_ci->final_transform;
	state.uniforms.extra_matrix = Transform2D();

	_set_uniforms();

	if (unshaded || (state.uniforms.final_modulate.a > 0.001 && (!r_ris.shader_cache || r_ris.shader_cache->canvas_item.light_mode != RasterizerStorageGLES3::Shader::CanvasItem::LIGHT_MODE_LIGHT_ONLY) && !p_ci->light_masked))
		_legacy_canvas_item_render_commands(p_ci, NULL, reclip, material_ptr);

	r_ris.rebind_shader = true; // hacked in for now.

	if ((blend_mode == RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_MIX || blend_mode == RasterizerStorageGLES3::Shader::CanvasItem::BLEND_MODE_PMALPHA) && r_ris.item_group_light && !unshaded) {
		Light *light = r_ris.item_group_light;
		bool light_used = false;
		RS::CanvasLightBlendMode bmode = RS::CANVAS_LIGHT_BLEND_MODE_ADD;
		state.uniforms.final_modulate = p_ci->final_modulate; // remove the canvas modulate

		while (light) {
			if (p_ci->light_mask & light->item_mask && r_ris.item_group_z >= light->z_min && r_ris.item_group_z <= light->z_max && p_ci->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {
				//intersects this light

				if (!light_used || bmode != light->blend_mode) {
					bmode = light->blend_mode;

					switch (bmode) {
						case RS::CANVAS_LIGHT_BLEND_MODE_ADD: {
							glBlendEquation(GL_FUNC_ADD);
							glBlendFunc(GL_SRC_ALPHA, GL_ONE);

						} break;
						case RS::CANVAS_LIGHT_BLEND_MODE_SUB: {
							glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
							glBlendFunc(GL_SRC_ALPHA, GL_ONE);
						} break;
						case RS::CANVAS_LIGHT_BLEND_MODE_MIX: {
							//						case RS::CANVAS_LIGHT_MODE_MASK: {
							glBlendEquation(GL_FUNC_ADD);
							glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

						} break;
					}
				}

				if (!light_used) {
					state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_LIGHTING, true);
					light_used = true;
				}

				// FTODO
				//bool has_shadow = light->shadow_buffer.is_valid() && p_ci->light_mask & light->item_shadow_mask;
				bool has_shadow = light->use_shadow && p_ci->light_mask & light->item_shadow_mask;

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_SHADOWS, has_shadow);
				if (has_shadow) {
					// FTODO
					//state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_USE_GRADIENT, light->shadow_gradient_length > 0);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_USE_GRADIENT, false);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_NEAREST, light->shadow_filter == RS::CANVAS_LIGHT_FILTER_NONE);
					//state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF3, light->shadow_filter == RS::CANVAS_LIGHT_FILTER_PCF3);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF3, false);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF5, light->shadow_filter == RS::CANVAS_LIGHT_FILTER_PCF5);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF7, false);
					//state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF7, light->shadow_filter == RS::CANVAS_LIGHT_FILTER_PCF7);
					//state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF9, light->shadow_filter == RS::CANVAS_LIGHT_FILTER_PCF9);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF9, false);
					state.canvas_shader.set_conditional(CanvasShaderGLES3::SHADOW_FILTER_PCF13, light->shadow_filter == RS::CANVAS_LIGHT_FILTER_PCF13);
				}

				state.canvas_shader.bind();
				state.using_light = light;
				state.using_shadow = has_shadow;

				//always re-set uniforms, since light parameters changed
				_set_uniforms();
				state.canvas_shader.use_material((void *)material_ptr);

				glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 6);
				RasterizerStorageGLES3::Texture *t = storage->texture_owner.get_or_null(light->texture);
				if (!t) {
					glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
				} else {
					t = t->get_ptr();

					glBindTexture(t->target, t->tex_id);
				}

				glActiveTexture(GL_TEXTURE0);
				_legacy_canvas_item_render_commands(p_ci, NULL, reclip, material_ptr); //redraw using light

				state.using_light = NULL;
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

			r_ris.last_blend_mode = -1;

#if 0
			//this is set again, so it should not be needed anyway?
			state.canvas_item_modulate = unshaded ? ci->final_modulate : Color(ci->final_modulate.r * p_modulate.r, ci->final_modulate.g * p_modulate.g, ci->final_modulate.b * p_modulate.b, ci->final_modulate.a * p_modulate.a);

			state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);
			state.canvas_shader.set_uniform(CanvasShaderGLES3::EXTRA_MATRIX, Transform2D());
			state.canvas_shader.set_uniform(CanvasShaderGLES3::FINAL_MODULATE, state.canvas_item_modulate);

			glBlendEquation(GL_FUNC_ADD);

			if (storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
				glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			} else {
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			}

			//@TODO RESET canvas_blend_mode
#endif
		}
	}

	if (reclip) {
		glEnable(GL_SCISSOR_TEST);
		int y = storage->frame.current_rt->height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
		// FTODO
		//		if (storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_VFLIP])
		//		y = r_ris.current_clip->final_clip_rect.position.y;
		glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);
	}
}

void RasterizerCanvasGLES3::gl_enable_scissor(int p_x, int p_y, int p_width, int p_height) const {
	glEnable(GL_SCISSOR_TEST);
	glScissor(p_x, p_y, p_width, p_height);
}

void RasterizerCanvasGLES3::gl_disable_scissor() const {
	glDisable(GL_SCISSOR_TEST);
}

void RasterizerCanvasGLES3::initialize() {
	RasterizerCanvasBaseGLES3::initialize();

	batch_initialize();

	// just reserve some space (may not be needed as we are orphaning, but hey ho)
	glGenBuffers(1, &bdata.gl_vertex_buffer);

	if (bdata.vertex_buffer_size_bytes) {
		glBindBuffer(GL_ARRAY_BUFFER, bdata.gl_vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, bdata.vertex_buffer_size_bytes, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// pre fill index buffer, the indices never need to change so can be static
		glGenBuffers(1, &bdata.gl_index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bdata.gl_index_buffer);

		Vector<uint16_t> indices;
		indices.resize(bdata.index_buffer_size_units);

		for (unsigned int q = 0; q < bdata.max_quads; q++) {
			int i_pos = q * 6; //  6 inds per quad
			int q_pos = q * 4; // 4 verts per quad
			indices.set(i_pos, q_pos);
			indices.set(i_pos + 1, q_pos + 1);
			indices.set(i_pos + 2, q_pos + 2);
			indices.set(i_pos + 3, q_pos);
			indices.set(i_pos + 4, q_pos + 2);
			indices.set(i_pos + 5, q_pos + 3);

			// we can only use 16 bit indices in OpenGL!
#ifdef DEBUG_ENABLED
			CRASH_COND((q_pos + 3) > 65535);
#endif
		}

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, bdata.index_buffer_size_bytes, &indices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	} // only if there is a vertex buffer (batching is on)
}

RasterizerCanvasGLES3::RasterizerCanvasGLES3() {
	batch_constructor();
}

#endif // GLES3_BACKEND_ENABLED

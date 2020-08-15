/*************************************************************************/
/*  rasterizer_canvas_gles2.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/os.h"
#include "core/project_settings.h"
#include "rasterizer_scene_gles2.h"
#include "servers/visual/visual_server_raster.h"

static const GLenum gl_primitive[] = {
	GL_POINTS,
	GL_LINES,
	GL_LINE_STRIP,
	GL_LINE_LOOP,
	GL_TRIANGLES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN
};

RasterizerCanvasGLES2::BatchData::BatchData() {
	reset_flush();
	gl_vertex_buffer = 0;
	gl_index_buffer = 0;
	max_quads = 0;
	vertex_buffer_size_units = 0;
	vertex_buffer_size_bytes = 0;
	index_buffer_size_units = 0;
	index_buffer_size_bytes = 0;
	use_colored_vertices = false;
	use_light_angles = false;
	settings_use_batching = false;
	settings_max_join_item_commands = 0;
	settings_colored_vertex_format_threshold = 0.0f;
	settings_batch_buffer_num_verts = 0;
	scissor_threshold_area = 0.0f;
	joined_item_batch_flags = 0;
	diagnose_frame = false;
	next_diagnose_tick = 10000;
	diagnose_frame_number = 9999999999; // some high number
	join_across_z_indices = true;
	settings_item_reordering_lookahead = 0;

	settings_use_batching_original_choice = false;
	settings_flash_batching = false;
	settings_diagnose_frame = false;
	settings_scissor_lights = false;
	settings_scissor_threshold = -1.0f;
	settings_use_single_rect_fallback = false;
	settings_light_max_join_items = 16;

	settings_uv_contract = false;
	settings_uv_contract_amount = 0.0f;

	stats_items_sorted = 0;
	stats_light_items_joined = 0;
}

void RasterizerCanvasGLES2::RenderItemState::reset() {
	current_clip = nullptr;
	shader_cache = nullptr;
	rebind_shader = true;
	prev_use_skeleton = false;
	last_blend_mode = -1;
	canvas_last_material = RID();
	item_group_z = 0;
	item_group_light = nullptr;
	final_modulate = Color(-1.0, -1.0, -1.0, -1.0); // just something unlikely

	joined_item = nullptr;
}

// just translate the color into something easily readable and not too verbose
String RasterizerCanvasGLES2::BatchColor::to_string() const {
	String sz = "{";
	const float *data = get_data();
	for (int c = 0; c < 4; c++) {
		float f = data[c];
		int val = ((f * 255.0f) + 0.5f);
		sz += String(Variant(val)) + " ";
	}
	sz += "}";
	return sz;
}

RasterizerStorageGLES2::Texture *RasterizerCanvasGLES2::_get_canvas_texture(const RID &p_texture) const {
	if (p_texture.is_valid()) {

		RasterizerStorageGLES2::Texture *texture = storage->texture_owner.getornull(p_texture);

		if (texture) {
			return texture->get_ptr();
		}
	}

	return 0;
}

int RasterizerCanvasGLES2::_batch_find_or_create_tex(const RID &p_texture, const RID &p_normal, bool p_tile, int p_previous_match) {

	// optimization .. in 99% cases the last matched value will be the same, so no need to traverse the list
	if (p_previous_match > 0) // if it is zero, it will get hit first in the linear search anyway
	{
		const BatchTex &batch_texture = bdata.batch_textures[p_previous_match];

		// note for future reference, if RID implementation changes, this could become more expensive
		if ((batch_texture.RID_texture == p_texture) && (batch_texture.RID_normal == p_normal)) {
			// tiling mode must also match
			bool tiles = batch_texture.tile_mode != BatchTex::TILE_OFF;

			if (tiles == p_tile)
				// match!
				return p_previous_match;
		}
	}

	// not the previous match .. we will do a linear search ... slower, but should happen
	// not very often except with non-batchable runs, which are going to be slow anyway
	// n.b. could possibly be replaced later by a fast hash table
	for (int n = 0; n < bdata.batch_textures.size(); n++) {
		const BatchTex &batch_texture = bdata.batch_textures[n];
		if ((batch_texture.RID_texture == p_texture) && (batch_texture.RID_normal == p_normal)) {

			// tiling mode must also match
			bool tiles = batch_texture.tile_mode != BatchTex::TILE_OFF;

			if (tiles == p_tile)
				// match!
				return n;
		}
	}

	// pushing back from local variable .. not ideal but has to use a Vector because non pod
	// due to RIDs
	BatchTex new_batch_tex;
	new_batch_tex.RID_texture = p_texture;
	new_batch_tex.RID_normal = p_normal;

	// get the texture
	RasterizerStorageGLES2::Texture *texture = _get_canvas_texture(p_texture);

	if (texture) {
		new_batch_tex.tex_pixel_size.x = 1.0 / texture->width;
		new_batch_tex.tex_pixel_size.y = 1.0 / texture->height;
		new_batch_tex.flags = texture->flags;
	} else {
		// maybe doesn't need doing...
		new_batch_tex.tex_pixel_size.x = 1.0;
		new_batch_tex.tex_pixel_size.y = 1.0;
		new_batch_tex.flags = 0;
	}

	if (p_tile) {
		if (texture) {
			// default
			new_batch_tex.tile_mode = BatchTex::TILE_NORMAL;

			// no hardware support for non power of 2 tiling
			if (!storage->config.support_npot_repeat_mipmap) {
				if (next_power_of_2(texture->alloc_width) != (unsigned int)texture->alloc_width && next_power_of_2(texture->alloc_height) != (unsigned int)texture->alloc_height) {
					new_batch_tex.tile_mode = BatchTex::TILE_FORCE_REPEAT;
				}
			}
		} else {
			// this should not happen?
			new_batch_tex.tile_mode = BatchTex::TILE_OFF;
		}
	} else {
		new_batch_tex.tile_mode = BatchTex::TILE_OFF;
	}

	// push back
	bdata.batch_textures.push_back(new_batch_tex);

	return bdata.batch_textures.size() - 1;
}

void RasterizerCanvasGLES2::_batch_upload_buffers() {

	// noop?
	if (!bdata.vertices.size())
		return;

	glBindBuffer(GL_ARRAY_BUFFER, bdata.gl_vertex_buffer);

	// orphan the old (for now)
	glBufferData(GL_ARRAY_BUFFER, 0, 0, GL_DYNAMIC_DRAW);

	if (!bdata.use_light_angles) {
		if (!bdata.use_colored_vertices) {
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertex) * bdata.vertices.size(), bdata.vertices.get_data(), GL_DYNAMIC_DRAW);
		} else {
			glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertexColored) * bdata.unit_vertices.size(), bdata.unit_vertices.get_unit(0), GL_DYNAMIC_DRAW);
		}
	} else {
		glBufferData(GL_ARRAY_BUFFER, sizeof(BatchVertexLightAngled) * bdata.unit_vertices.size(), bdata.unit_vertices.get_unit(0), GL_DYNAMIC_DRAW);
	}

	// might not be necessary
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

RasterizerCanvasGLES2::Batch *RasterizerCanvasGLES2::_batch_request_new(bool p_blank) {
	Batch *batch = bdata.batches.request();
	if (!batch) {
		// grow the batches
		bdata.batches.grow();

		// and the temporary batches (used for color verts)
		bdata.batches_temp.reset();
		bdata.batches_temp.grow();

		// this should always succeed after growing
		batch = bdata.batches.request();
#ifdef DEBUG_ENABLED
		CRASH_COND(!batch);
#endif
	}

	if (p_blank)
		memset(batch, 0, sizeof(Batch));

	return batch;
}

// This function may be called MULTIPLE TIMES for each item, so needs to record how far it has got
bool RasterizerCanvasGLES2::prefill_joined_item(FillState &r_fill_state, int &r_command_start, Item *p_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material) {
	// we will prefill batches and vertices ready for sending in one go to the vertex buffer
	int command_count = p_item->commands.size();
	Item::Command *const *commands = p_item->commands.ptr();

	// checking the color for not being white makes it 92/90 times faster in the case where it is white
	bool multiply_final_modulate = false;
	if (!r_fill_state.use_hardware_transform && (r_fill_state.final_modulate != Color(1, 1, 1, 1))) {
		multiply_final_modulate = true;
	}

	// start batch is a dummy batch (tex id -1) .. could be made more efficient
	if (!r_fill_state.curr_batch) {
		r_fill_state.curr_batch = _batch_request_new();
		r_fill_state.curr_batch->type = Batch::BT_DEFAULT;
		r_fill_state.curr_batch->first_command = r_command_start;
		// should tex_id be set to -1? check this
	}

	// we need to return which command we got up to, so
	// store this outside the loop
	int command_num;

	// do as many commands as possible until the vertex buffer will be full up
	for (command_num = r_command_start; command_num < command_count; command_num++) {

		Item::Command *command = commands[command_num];

		switch (command->type) {

			default: {
				_prefill_default_batch(r_fill_state, command_num, *p_item);
			} break;
			case Item::Command::TYPE_TRANSFORM: {
				// if the extra matrix has been sent already,
				// break this extra matrix software path (as we don't want to unset it on the GPU etc)
				if (r_fill_state.extra_matrix_sent) {
					_prefill_default_batch(r_fill_state, command_num, *p_item);
				} else {
					// Extra matrix fast path.
					// Instead of sending the command immediately, we store the modified transform (in combined)
					// for software transform, and only flush this transform command if we NEED to (i.e. we want to
					// render some default commands)
					Item::CommandTransform *transform = static_cast<Item::CommandTransform *>(command);
					const Transform2D &extra_matrix = transform->xform;

					if (r_fill_state.use_hardware_transform) {
						// if we are using hardware transform mode, we have already sent the final transform,
						// so we only want to software transform the extra matrix
						r_fill_state.transform_combined = extra_matrix;
					} else {
						r_fill_state.transform_combined = p_item->final_transform * extra_matrix;
					}
					// after a transform command, always use some form of software transform (either the combined final + extra, or just the extra)
					// until we flush this dirty extra matrix because we need to render default commands.
					r_fill_state.transform_mode = _find_transform_mode(r_fill_state.transform_combined);

					// make a note of which command the dirty extra matrix is store in, so we can send it later
					// if necessary
					r_fill_state.transform_extra_command_number_p1 = command_num + 1; // plus 1 so we can test against zero
				}
			} break;
			case Item::Command::TYPE_RECT: {

				Item::CommandRect *rect = static_cast<Item::CommandRect *>(command);

				// unoptimized - could this be done once per batch / batch texture?
				bool send_light_angles = rect->normal_map != RID();

				bool buffer_full = false;

				// the template params must be explicit for compilation,
				// this forces building the multiple versions of the function.
				if (send_light_angles) {
					buffer_full = prefill_rect<true>(rect, r_fill_state, r_command_start, command_num, command_count, commands, p_item, multiply_final_modulate);
				} else {
					buffer_full = prefill_rect<false>(rect, r_fill_state, r_command_start, command_num, command_count, commands, p_item, multiply_final_modulate);
				}

				if (buffer_full)
					return true;

			} break;
		}
	}

	// VERY IMPORTANT to return where we got to, because this func may be called multiple
	// times per item.
	// Don't miss out on this step by calling return earlier in the function without setting r_command_start.
	r_command_start = command_num;

	return false;
}

void RasterizerCanvasGLES2::_batch_render_rects(const Batch &p_batch, RasterizerStorageGLES2::Material *p_material) {

	ERR_FAIL_COND(p_batch.num_commands <= 0);

	const bool &colored_verts = bdata.use_colored_vertices;
	const bool &use_light_angles = bdata.use_light_angles;

	int sizeof_vert;
	if (!use_light_angles) {
		if (!colored_verts) {
			sizeof_vert = sizeof(BatchVertex);
		} else {
			sizeof_vert = sizeof(BatchVertexColored);
		}
	} else {
		sizeof_vert = sizeof(BatchVertexLightAngled);
	}

	// batch tex
	const BatchTex &tex = bdata.batch_textures[p_batch.batch_texture_id];

	// make sure to set all conditionals BEFORE binding the shader
	_set_texture_rect_mode(false, use_light_angles);

	// force repeat is set if non power of 2 texture, and repeat is needed if hardware doesn't support npot
	if (tex.tile_mode == BatchTex::TILE_FORCE_REPEAT) {
		state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_FORCE_REPEAT, true);
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
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof_vert, (const void *)pointer);

	// always send UVs, even within a texture specified because a shader can still use UVs
	glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (2 * 4)));
	glEnableVertexAttribArray(VS::ARRAY_TEX_UV);

	// color
	if (!colored_verts) {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttrib4fv(VS::ARRAY_COLOR, p_batch.color.get_data());
	} else {
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (4 * 4)));
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (use_light_angles) {
		glVertexAttribPointer(VS::ARRAY_TANGENT, 1, GL_FLOAT, GL_FALSE, sizeof_vert, CAST_INT_TO_UCHAR_PTR(pointer + (8 * 4)));
		glEnableVertexAttribArray(VS::ARRAY_TANGENT);
	}

	// We only want to set the GL wrapping mode if the texture is not already tiled (i.e. set in Import).
	// This  is an optimization left over from the legacy renderer.
	// If we DID set tiling in the API, and reverted to clamped, then the next draw using this texture
	// may use clamped mode incorrectly.
	bool tex_is_already_tiled = tex.flags & VS::TEXTURE_FLAG_REPEAT;

	if (tex.tile_mode == BatchTex::TILE_NORMAL) {
		// if the texture is imported as tiled, no need to set GL state, as it will already be bound with repeat
		if (!tex_is_already_tiled) {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		}
	}

	// we need to convert explicitly from pod Vec2 to Vector2 ...
	// could use a cast but this might be unsafe in future
	Vector2 tps;
	tex.tex_pixel_size.to(tps);
	state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, tps);

	int64_t offset = p_batch.first_quad * 6 * 2; // 6 inds per quad at 2 bytes each

	int num_elements = p_batch.num_commands * 6;
	glDrawElements(GL_TRIANGLES, num_elements, GL_UNSIGNED_SHORT, (void *)offset);

	storage->info.render._2d_draw_call_count++;

	switch (tex.tile_mode) {
		case BatchTex::TILE_FORCE_REPEAT: {
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_FORCE_REPEAT, false);
		} break;
		case BatchTex::TILE_NORMAL: {
			// if the texture is imported as tiled, no need to revert GL state
			if (!tex_is_already_tiled) {
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
		} break;
		default: {
		} break;
	}

	// could these have ifs?
	glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	glDisableVertexAttribArray(VS::ARRAY_COLOR);
	glDisableVertexAttribArray(VS::ARRAY_TANGENT);

	// may not be necessary .. state change optimization still TODO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

#ifdef DEBUG_ENABLED
String RasterizerCanvasGLES2::get_command_type_string(const Item::Command &p_command) const {
	String sz = "";

	switch (p_command.type) {
		default:
			break;
		case Item::Command::TYPE_LINE: {
			sz = "l";
		} break;
		case Item::Command::TYPE_POLYLINE: {
			sz = "PL";
		} break;
		case Item::Command::TYPE_RECT: {
			sz = "r";
		} break;
		case Item::Command::TYPE_NINEPATCH: {
			sz = "n";
		} break;
		case Item::Command::TYPE_PRIMITIVE: {
			sz = "PR";
		} break;
		case Item::Command::TYPE_POLYGON: {
			sz = "p";
		} break;
		case Item::Command::TYPE_MESH: {
			sz = "m";
		} break;
		case Item::Command::TYPE_MULTIMESH: {
			sz = "MM";
		} break;
		case Item::Command::TYPE_PARTICLES: {
			sz = "PA";
		} break;
		case Item::Command::TYPE_CIRCLE: {
			sz = "c";
		} break;
		case Item::Command::TYPE_TRANSFORM: {
			sz = "t";

			// add a bit more info in debug build
			const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(&p_command);
			const Transform2D &mat = transform->xform;

			sz += " ";
			sz += String(Variant(mat.elements[2]));
			sz += " ";
		} break;
		case Item::Command::TYPE_CLIP_IGNORE: {
			sz = "CI";
		} break;
	} // switch

	return sz;
}

void RasterizerCanvasGLES2::diagnose_batches(Item::Command *const *p_commands) {
	int num_batches = bdata.batches.size();

	BatchColor curr_color;
	curr_color.set(Color(-1, -1, -1, -1));
	bool first_color_change = true;

	for (int batch_num = 0; batch_num < num_batches; batch_num++) {
		const Batch &batch = bdata.batches[batch_num];
		bdata.frame_string += "\t\t\tbatch ";

		switch (batch.type) {
			case Batch::BT_RECT: {
				bdata.frame_string += "R ";
				bdata.frame_string += itos(batch.first_command) + "-";
				bdata.frame_string += itos(batch.num_commands);

				int tex_id = (int)bdata.batch_textures[batch.batch_texture_id].RID_texture.get_id();
				bdata.frame_string += " [" + itos(batch.batch_texture_id) + " - " + itos(tex_id) + "]";

				bdata.frame_string += " " + batch.color.to_string();

				if (batch.num_commands > 1) {
					bdata.frame_string += " MULTI";
				}
				if (curr_color != batch.color) {
					curr_color = batch.color;
					if (!first_color_change) {
						bdata.frame_string += " color";
					} else {
						first_color_change = false;
					}
				}
				bdata.frame_string += "\n";
			} break;
			default: {
				bdata.frame_string += "D ";
				bdata.frame_string += itos(batch.first_command) + "-";
				bdata.frame_string += itos(batch.num_commands) + " ";

				int num_show = MIN(batch.num_commands, 16);
				for (int n = 0; n < num_show; n++) {
					const Item::Command &comm = *p_commands[batch.first_command + n];
					bdata.frame_string += get_command_type_string(comm) + " ";
				}

				bdata.frame_string += "\n";
			} break;
		}
	}
}
#endif

void RasterizerCanvasGLES2::render_batches(Item::Command *const *p_commands, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material) {

	int num_batches = bdata.batches.size();

	for (int batch_num = 0; batch_num < num_batches; batch_num++) {
		const Batch &batch = bdata.batches[batch_num];

		switch (batch.type) {
			case Batch::BT_RECT: {
				_batch_render_rects(batch, p_material);
			} break;
			default: {
				int end_command = batch.first_command + batch.num_commands;

				for (int i = batch.first_command; i < end_command; i++) {

					Item::Command *command = p_commands[i];

					switch (command->type) {

						case Item::Command::TYPE_LINE: {

							Item::CommandLine *line = static_cast<Item::CommandLine *>(command);

							_set_texture_rect_mode(false);
							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							_bind_canvas_texture(RID(), RID());

							glDisableVertexAttribArray(VS::ARRAY_COLOR);
							glVertexAttrib4fv(VS::ARRAY_COLOR, line->color.components);

							state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);

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

						case Item::Command::TYPE_RECT: {

							Item::CommandRect *r = static_cast<Item::CommandRect *>(command);

							glDisableVertexAttribArray(VS::ARRAY_COLOR);
							glVertexAttrib4fv(VS::ARRAY_COLOR, r->modulate.components);

							bool can_tile = true;
							if (r->texture.is_valid() && r->flags & CANVAS_RECT_TILE && !storage->config.support_npot_repeat_mipmap) {
								// workaround for when setting tiling does not work due to hardware limitation

								RasterizerStorageGLES2::Texture *texture = storage->texture_owner.getornull(r->texture);

								if (texture) {

									texture = texture->get_ptr();

									if (next_power_of_2(texture->alloc_width) != (unsigned int)texture->alloc_width && next_power_of_2(texture->alloc_height) != (unsigned int)texture->alloc_height) {
										state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_FORCE_REPEAT, true);
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

								RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(r->texture, r->normal_map);

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
									if (r->flags & CANVAS_RECT_FLIP_V) {
										SWAP(uvs[0], uvs[3]);
										SWAP(uvs[1], uvs[2]);
										flip_v = !flip_v;
									}

									state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);

									bool untile = false;

									if (can_tile && r->flags & CANVAS_RECT_TILE && !(texture->flags & VS::TEXTURE_FLAG_REPEAT)) {
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
									}
								} else {
									static const Vector2 uvs[4] = {
										Vector2(0.0, 0.0),
										Vector2(0.0, 1.0),
										Vector2(1.0, 1.0),
										Vector2(1.0, 0.0),
									};

									state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, Vector2());
									_draw_gui_primitive(4, points, NULL, uvs);
								}

							} else {
								// This branch is better for performance, but can produce flicker on Nvidia, see above comment.
								_bind_quad_buffer();

								_set_texture_rect_mode(true);

								if (state.canvas_shader.bind()) {
									_set_uniforms();
									state.canvas_shader.use_material((void *)p_material);
								}

								RasterizerStorageGLES2::Texture *tex = _bind_canvas_texture(r->texture, r->normal_map);

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

									state.canvas_shader.set_uniform(CanvasShaderGLES2::DST_RECT, Color(dst_rect.position.x, dst_rect.position.y, dst_rect.size.x, dst_rect.size.y));
									state.canvas_shader.set_uniform(CanvasShaderGLES2::SRC_RECT, Color(0, 0, 1, 1));

									glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
									storage->info.render._2d_draw_call_count++;
								} else {

									bool untile = false;

									if (can_tile && r->flags & CANVAS_RECT_TILE && !(tex->flags & VS::TEXTURE_FLAG_REPEAT)) {
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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

									if (r->flags & CANVAS_RECT_FLIP_V) {
										src_rect.size.y *= -1;
									}

									if (r->flags & CANVAS_RECT_TRANSPOSE) {
										dst_rect.size.x *= -1; // Encoding in the dst_rect.z uniform
									}

									state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);

									state.canvas_shader.set_uniform(CanvasShaderGLES2::DST_RECT, Color(dst_rect.position.x, dst_rect.position.y, dst_rect.size.x, dst_rect.size.y));
									state.canvas_shader.set_uniform(CanvasShaderGLES2::SRC_RECT, Color(src_rect.position.x, src_rect.position.y, src_rect.size.x, src_rect.size.y));

									glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
									storage->info.render._2d_draw_call_count++;

									if (untile) {
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
										glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
									}
								}

								glBindBuffer(GL_ARRAY_BUFFER, 0);
								glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
							}

							state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_FORCE_REPEAT, false);

						} break;

						case Item::Command::TYPE_NINEPATCH: {

							Item::CommandNinePatch *np = static_cast<Item::CommandNinePatch *>(command);

							_set_texture_rect_mode(false);
							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							glDisableVertexAttribArray(VS::ARRAY_COLOR);
							glVertexAttrib4fv(VS::ARRAY_COLOR, np->color.components);

							RasterizerStorageGLES2::Texture *tex = _bind_canvas_texture(np->texture, np->normal_map);

							if (!tex) {
								// FIXME: Handle textureless ninepatch gracefully
								WARN_PRINT("NinePatch without texture not supported yet in GLES2 backend, skipping.");
								continue;
							}
							if (tex->width == 0 || tex->height == 0) {
								WARN_PRINT("Cannot set empty texture to NinePatch.");
								continue;
							}

							Size2 texpixel_size(1.0 / tex->width, 1.0 / tex->height);

							// state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, state.uniforms.modelview_matrix);
							state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);

							Rect2 source = np->source;
							if (source.size.x == 0 && source.size.y == 0) {
								source.size.x = tex->width;
								source.size.y = tex->height;
							}

							float screen_scale = 1.0;

							if (source.size.x != 0 && source.size.y != 0) {

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

								buffer[(0 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT] * screen_scale;
								buffer[(0 * 4 * 4) + 5] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 6] = (source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
								buffer[(0 * 4 * 4) + 7] = source.position.y * texpixel_size.y;

								buffer[(0 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT] * screen_scale;
								buffer[(0 * 4 * 4) + 9] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
								buffer[(0 * 4 * 4) + 11] = source.position.y * texpixel_size.y;

								buffer[(0 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(0 * 4 * 4) + 13] = np->rect.position.y;

								buffer[(0 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(0 * 4 * 4) + 15] = source.position.y * texpixel_size.y;

								// second row

								buffer[(1 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(1 * 4 * 4) + 1] = np->rect.position.y + np->margin[MARGIN_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(1 * 4 * 4) + 3] = (source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

								buffer[(1 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT] * screen_scale;
								buffer[(1 * 4 * 4) + 5] = np->rect.position.y + np->margin[MARGIN_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 6] = (source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
								buffer[(1 * 4 * 4) + 7] = (source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

								buffer[(1 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT] * screen_scale;
								buffer[(1 * 4 * 4) + 9] = np->rect.position.y + np->margin[MARGIN_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
								buffer[(1 * 4 * 4) + 11] = (source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

								buffer[(1 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(1 * 4 * 4) + 13] = np->rect.position.y + np->margin[MARGIN_TOP] * screen_scale;

								buffer[(1 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(1 * 4 * 4) + 15] = (source.position.y + np->margin[MARGIN_TOP]) * texpixel_size.y;

								// third row

								buffer[(2 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(2 * 4 * 4) + 1] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(2 * 4 * 4) + 3] = (source.position.y + source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

								buffer[(2 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT] * screen_scale;
								buffer[(2 * 4 * 4) + 5] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 6] = (source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
								buffer[(2 * 4 * 4) + 7] = (source.position.y + source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

								buffer[(2 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT] * screen_scale;
								buffer[(2 * 4 * 4) + 9] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
								buffer[(2 * 4 * 4) + 11] = (source.position.y + source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

								buffer[(2 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(2 * 4 * 4) + 13] = np->rect.position.y + np->rect.size.y - np->margin[MARGIN_BOTTOM] * screen_scale;

								buffer[(2 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(2 * 4 * 4) + 15] = (source.position.y + source.size.y - np->margin[MARGIN_BOTTOM]) * texpixel_size.y;

								// fourth row

								buffer[(3 * 4 * 4) + 0] = np->rect.position.x;
								buffer[(3 * 4 * 4) + 1] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 2] = source.position.x * texpixel_size.x;
								buffer[(3 * 4 * 4) + 3] = (source.position.y + source.size.y) * texpixel_size.y;

								buffer[(3 * 4 * 4) + 4] = np->rect.position.x + np->margin[MARGIN_LEFT] * screen_scale;
								buffer[(3 * 4 * 4) + 5] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 6] = (source.position.x + np->margin[MARGIN_LEFT]) * texpixel_size.x;
								buffer[(3 * 4 * 4) + 7] = (source.position.y + source.size.y) * texpixel_size.y;

								buffer[(3 * 4 * 4) + 8] = np->rect.position.x + np->rect.size.x - np->margin[MARGIN_RIGHT] * screen_scale;
								buffer[(3 * 4 * 4) + 9] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 10] = (source.position.x + source.size.x - np->margin[MARGIN_RIGHT]) * texpixel_size.x;
								buffer[(3 * 4 * 4) + 11] = (source.position.y + source.size.y) * texpixel_size.y;

								buffer[(3 * 4 * 4) + 12] = np->rect.position.x + np->rect.size.x;
								buffer[(3 * 4 * 4) + 13] = np->rect.position.y + np->rect.size.y;

								buffer[(3 * 4 * 4) + 14] = (source.position.x + source.size.x) * texpixel_size.x;
								buffer[(3 * 4 * 4) + 15] = (source.position.y + source.size.y) * texpixel_size.y;
							}

							glBindBuffer(GL_ARRAY_BUFFER, data.ninepatch_vertices);
							glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (16 + 16) * 2, buffer, GL_DYNAMIC_DRAW);

							glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data.ninepatch_elements);

							glEnableVertexAttribArray(VS::ARRAY_VERTEX);
							glEnableVertexAttribArray(VS::ARRAY_TEX_UV);

							glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
							glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), CAST_INT_TO_UCHAR_PTR((sizeof(float) * 2)));

							glDrawElements(GL_TRIANGLES, 18 * 3 - (np->draw_center ? 0 : 6), GL_UNSIGNED_BYTE, NULL);

							glBindBuffer(GL_ARRAY_BUFFER, 0);
							glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
							storage->info.render._2d_draw_call_count++;

						} break;

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

						case Item::Command::TYPE_POLYGON: {

							Item::CommandPolygon *polygon = static_cast<Item::CommandPolygon *>(command);

							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(polygon->texture, polygon->normal_map);

							if (texture) {
								Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
								state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);
							}

							_draw_polygon(polygon->indices.ptr(), polygon->count, polygon->points.size(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->colors.size() == 1, polygon->weights.ptr(), polygon->bones.ptr());
#ifdef GLES_OVER_GL
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
						} break;
						case Item::Command::TYPE_MESH: {

							Item::CommandMesh *mesh = static_cast<Item::CommandMesh *>(command);
							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(mesh->texture, mesh->normal_map);

							if (texture) {
								Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
								state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);
							}

							RasterizerStorageGLES2::Mesh *mesh_data = storage->mesh_owner.getornull(mesh->mesh);
							if (mesh_data) {

								for (int j = 0; j < mesh_data->surfaces.size(); j++) {
									RasterizerStorageGLES2::Surface *s = mesh_data->surfaces[j];
									// materials are ignored in 2D meshes, could be added but many things (ie, lighting mode, reading from screen, etc) would break as they are not meant be set up at this point of drawing

									glBindBuffer(GL_ARRAY_BUFFER, s->vertex_id);

									if (s->index_array_len > 0) {
										glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
									}

									for (int k = 0; k < VS::ARRAY_MAX - 1; k++) {
										if (s->attribs[k].enabled) {
											glEnableVertexAttribArray(k);
											glVertexAttribPointer(s->attribs[k].index, s->attribs[k].size, s->attribs[k].type, s->attribs[k].normalized, s->attribs[k].stride, CAST_INT_TO_UCHAR_PTR(s->attribs[k].offset));
										} else {
											glDisableVertexAttribArray(k);
											switch (k) {
												case VS::ARRAY_NORMAL: {
													glVertexAttrib4f(VS::ARRAY_NORMAL, 0.0, 0.0, 1, 1);
												} break;
												case VS::ARRAY_COLOR: {
													glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);

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

								for (int j = 1; j < VS::ARRAY_MAX - 1; j++) {
									glDisableVertexAttribArray(j);
								}
							}

							storage->info.render._2d_draw_call_count++;
						} break;
						case Item::Command::TYPE_MULTIMESH: {
							Item::CommandMultiMesh *mmesh = static_cast<Item::CommandMultiMesh *>(command);

							RasterizerStorageGLES2::MultiMesh *multi_mesh = storage->multimesh_owner.getornull(mmesh->multimesh);

							if (!multi_mesh)
								break;

							RasterizerStorageGLES2::Mesh *mesh_data = storage->mesh_owner.getornull(multi_mesh->mesh);

							if (!mesh_data)
								break;

							state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_INSTANCE_CUSTOM, multi_mesh->custom_data_format != VS::MULTIMESH_CUSTOM_DATA_NONE);
							state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_INSTANCING, true);
							_set_texture_rect_mode(false);

							if (state.canvas_shader.bind()) {
								_set_uniforms();
								state.canvas_shader.use_material((void *)p_material);
							}

							RasterizerStorageGLES2::Texture *texture = _bind_canvas_texture(mmesh->texture, mmesh->normal_map);

							if (texture) {
								Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
								state.canvas_shader.set_uniform(CanvasShaderGLES2::COLOR_TEXPIXEL_SIZE, texpixel_size);
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
								RasterizerStorageGLES2::Surface *s = mesh_data->surfaces[j];
								// materials are ignored in 2D meshes, could be added but many things (ie, lighting mode, reading from screen, etc) would break as they are not meant be set up at this point of drawing

								//bind buffers for mesh surface
								glBindBuffer(GL_ARRAY_BUFFER, s->vertex_id);

								if (s->index_array_len > 0) {
									glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
								}

								for (int k = 0; k < VS::ARRAY_MAX - 1; k++) {
									if (s->attribs[k].enabled) {
										glEnableVertexAttribArray(k);
										glVertexAttribPointer(s->attribs[k].index, s->attribs[k].size, s->attribs[k].type, s->attribs[k].normalized, s->attribs[k].stride, CAST_INT_TO_UCHAR_PTR(s->attribs[k].offset));
									} else {
										glDisableVertexAttribArray(k);
										switch (k) {
											case VS::ARRAY_NORMAL: {
												glVertexAttrib4f(VS::ARRAY_NORMAL, 0.0, 0.0, 1, 1);
											} break;
											case VS::ARRAY_COLOR: {
												glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);

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
										if (multi_mesh->transform_format == VS::MULTIMESH_TRANSFORM_3D) {
											glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 2, &buffer[8]);
										} else {
											glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 2, 0.0, 0.0, 1.0, 0.0);
										}
									}

									if (multi_mesh->color_floats) {
										if (multi_mesh->color_format == VS::MULTIMESH_COLOR_8BIT) {
											uint8_t *color_data = (uint8_t *)(buffer + color_ofs);
											glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 3, color_data[0] / 255.0, color_data[1] / 255.0, color_data[2] / 255.0, color_data[3] / 255.0);
										} else {
											glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 3, buffer + color_ofs);
										}
									} else {
										glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 3, 1.0, 1.0, 1.0, 1.0);
									}

									if (multi_mesh->custom_data_floats) {
										if (multi_mesh->custom_data_format == VS::MULTIMESH_CUSTOM_DATA_8BIT) {
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

							_set_texture_rect_mode(false);
							state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_INSTANCING, false);

							storage->info.render._2d_draw_call_count++;
						} break;
						case Item::Command::TYPE_POLYLINE: {
							Item::CommandPolyLine *pline = static_cast<Item::CommandPolyLine *>(command);

							state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_TEXTURE_RECT, false);

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

							_draw_gui_primitive(primitive->points.size(), primitive->points.ptr(), primitive->colors.ptr(), primitive->uvs.ptr());
						} break;

						case Item::Command::TYPE_TRANSFORM: {
							Item::CommandTransform *transform = static_cast<Item::CommandTransform *>(command);
							state.uniforms.extra_matrix = transform->xform;
							state.canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, state.uniforms.extra_matrix);
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

										if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
											y = p_current_clip->final_clip_rect.position.y;

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

	// zero all the batch data ready for a new run
	bdata.reset_flush();
}

void RasterizerCanvasGLES2::render_joined_item_commands(const BItemJoined &p_bij, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material, bool p_lit) {

	Item *item = 0;
	Item *first_item = bdata.item_refs[p_bij.first_item_ref].item;

	FillState fill_state;
	fill_state.reset();
	fill_state.use_hardware_transform = p_bij.use_hardware_transform();
	fill_state.extra_matrix_sent = false;

	// in the special case of custom shaders that read from VERTEX (i.e. vertex position)
	// we want to disable software transform of extra matrix
	if (bdata.joined_item_batch_flags & RasterizerStorageGLES2::Shader::CanvasItem::PREVENT_VERTEX_BAKING) {
		fill_state.extra_matrix_sent = true;
	}

	for (unsigned int i = 0; i < p_bij.num_item_refs; i++) {
		const BItemRef &ref = bdata.item_refs[p_bij.first_item_ref + i];
		item = ref.item;

		if (!p_lit) {
			// if not lit we use the complex calculated final modulate
			fill_state.final_modulate = ref.final_modulate;
		} else {
			// if lit we ignore canvas modulate and just use the item modulate
			fill_state.final_modulate = item->final_modulate;
		}

		int command_count = item->commands.size();
		int command_start = 0;

		// ONCE OFF fill state setup, that will be retained over multiple calls to
		// prefill_joined_item()
		fill_state.transform_combined = item->final_transform;

		// decide the initial transform mode, and make a backup
		// in orig_transform_mode in case we need to switch back
		if (!fill_state.use_hardware_transform) {
			fill_state.transform_mode = _find_transform_mode(fill_state.transform_combined);
		} else {
			fill_state.transform_mode = TM_NONE;
		}
		fill_state.orig_transform_mode = fill_state.transform_mode;

		// keep track of when we added an extra matrix
		// so we can defer sending until we see a default command
		fill_state.transform_extra_command_number_p1 = 0;

		while (command_start < command_count) {
			// fill as many batches as possible (until all done, or the vertex buffer is full)
			bool bFull = prefill_joined_item(fill_state, command_start, item, p_current_clip, r_reclip, p_material);

			if (bFull) {
				// always pass first item (commands for default are always first item)
				flush_render_batches(first_item, p_current_clip, r_reclip, p_material);
				fill_state.reset();
			}
		}
	}

	// flush if any left
	flush_render_batches(first_item, p_current_clip, r_reclip, p_material);
}

void RasterizerCanvasGLES2::flush_render_batches(Item *p_first_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material) {

	// some heuristic to decide whether to use colored verts.
	// feel free to tweak this.
	// this could use hysteresis, to prevent jumping between methods
	// .. however probably not necessary
	bdata.use_colored_vertices = false;

	if (bdata.use_light_angles) {
		_translate_batches_to_larger_FVF<BatchVertexLightAngled, true>();
	} else {
		// only check whether to convert if there are quads (prevent divide by zero)
		// and we haven't decided to prevent color baking (due to e.g. MODULATE
		// being used in a shader)
		if (bdata.total_quads && !(bdata.joined_item_batch_flags & RasterizerStorageGLES2::Shader::CanvasItem::PREVENT_COLOR_BAKING)) {
			// minus 1 to prevent single primitives (ratio 1.0) always being converted to colored..
			// in that case it is slightly cheaper to just have the color as part of the batch
			float ratio = (float)(bdata.total_color_changes - 1) / (float)bdata.total_quads;

			// use bigger than or equal so that 0.0 threshold can force always using colored verts
			if (ratio >= bdata.settings_colored_vertex_format_threshold) {
				bdata.use_colored_vertices = true;

				// small perf cost versus going straight to colored verts (maybe around 10%)
				// however more straightforward
				_translate_batches_to_larger_FVF<BatchVertexColored, false>();
				//_batch_translate_to_colored();
			}
		}
	} // if not using light angles

	// send buffers to opengl
	_batch_upload_buffers();

	Item::Command *const *commands = p_first_item->commands.ptr();

#ifdef DEBUG_ENABLED
	if (bdata.diagnose_frame) {
		diagnose_batches(commands);
	}
#endif

	render_batches(commands, p_current_clip, r_reclip, p_material);
}

void RasterizerCanvasGLES2::_canvas_item_render_commands(Item *p_item, Item *p_current_clip, bool &r_reclip, RasterizerStorageGLES2::Material *p_material) {

	int command_count = p_item->commands.size();

	Item::Command *const *commands = p_item->commands.ptr();

	// legacy .. just create one massive batch and render everything as before
	bdata.batches.reset();
	Batch *batch = _batch_request_new();
	batch->type = Batch::BT_DEFAULT;
	batch->num_commands = command_count;

	render_batches(commands, p_current_clip, r_reclip, p_material);
}

void RasterizerCanvasGLES2::record_items(Item *p_item_list, int p_z) {
	while (p_item_list) {
		BSortItem *s = bdata.sort_items.request_with_grow();

		s->item = p_item_list;
		s->z_index = p_z;

		p_item_list = p_item_list->next;
	}
}

void RasterizerCanvasGLES2::sort_items() {
	// turned off?
	if (!bdata.settings_item_reordering_lookahead) {
		return;
	}

	for (int s = 0; s < bdata.sort_items.size() - 2; s++) {
		if (sort_items_from(s)) {
#ifdef DEBUG_ENABLED
			bdata.stats_items_sorted++;
#endif
		}
	}
}

bool RasterizerCanvasGLES2::sort_items_from(int p_start) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V((p_start + 1) >= bdata.sort_items.size(), false)
#endif

	const BSortItem &start = bdata.sort_items[p_start];
	int start_z = start.z_index;

	// check start is the right type for sorting
	if (start.item->commands.size() != 1) {
		return false;
	}
	const Item::Command &command_start = *start.item->commands[0];
	if (command_start.type != Item::Command::TYPE_RECT) {
		return false;
	}

	BSortItem &second = bdata.sort_items[p_start + 1];
	if (second.z_index != start_z) {
		// no sorting across z indices (for now)
		return false;
	}

	// if the neighbours are already a good match
	if (_sort_items_match(start, second)) // order is crucial, start first
	{
		return false;
	}

	// local cached aabb
	Rect2 second_AABB = second.item->global_rect_cache;

	// if the start and 2nd items overlap, can do no more
	if (start.item->global_rect_cache.intersects(second_AABB)) {
		return false;
	}

	// which neighbour to test
	int test_last = 2 + bdata.settings_item_reordering_lookahead;
	for (int test = 2; test < test_last; test++) {
		int test_sort_item_id = p_start + test;

		// if we've got to the end of the list, can't sort any more, give up
		if (test_sort_item_id >= bdata.sort_items.size()) {
			return false;
		}

		BSortItem *test_sort_item = &bdata.sort_items[test_sort_item_id];

		// across z indices?
		if (test_sort_item->z_index != start_z) {
			return false;
		}

		Item *test_item = test_sort_item->item;

		// if the test item overlaps the second item, we can't swap, AT ALL
		// because swapping an item OVER this one would cause artefacts
		if (second_AABB.intersects(test_item->global_rect_cache)) {
			return false;
		}

		// do they match?
		if (!_sort_items_match(start, *test_sort_item)) // order is crucial, start first
		{
			continue;
		}

		// we can only swap if there are no AABB overlaps with sandwiched neighbours
		bool ok = true;

		// start from 2, no need to check 1 as the second has already been checked against this item
		// in the intersection test above
		for (int sn = 2; sn < test; sn++) {
			BSortItem *sandwich_neighbour = &bdata.sort_items[p_start + sn];
			if (test_item->global_rect_cache.intersects(sandwich_neighbour->item->global_rect_cache)) {
				ok = false;
				break;
			}
		}
		if (!ok) {
			continue;
		}

		// it is ok to exchange them!
		BSortItem temp;
		temp.assign(second);
		second.assign(*test_sort_item);
		test_sort_item->assign(temp);

		return true;
	} // for test

	return false;
}

void RasterizerCanvasGLES2::join_sorted_items() {
	sort_items();

	int z = VS::CANVAS_ITEM_Z_MIN;
	_render_item_state.item_group_z = z;

	for (int s = 0; s < bdata.sort_items.size(); s++) {
		const BSortItem &si = bdata.sort_items[s];
		Item *ci = si.item;

		// change z?
		if (si.z_index != z) {
			z = si.z_index;

			// may not be required
			_render_item_state.item_group_z = z;

			// if z ranged lights are present, sometimes we have to disable joining over z_indices.
			// we do this here.
			// Note this restriction may be able to be relaxed with light bitfields, investigate!
			if (!bdata.join_across_z_indices) {
				_render_item_state.join_batch_break = true;
			}
		}

		bool join;

		if (_render_item_state.join_batch_break) {
			// always start a new batch for this item
			join = false;

			// could be another batch break (i.e. prevent NEXT item from joining this)
			// so we still need to run try_join_item
			// even though we know join is false.
			// also we need to run try_join_item for every item because it keeps the state up to date,
			// if we didn't run it the state would be out of date.
			try_join_item(ci, _render_item_state, _render_item_state.join_batch_break);
		} else {
			join = try_join_item(ci, _render_item_state, _render_item_state.join_batch_break);
		}

		// assume the first item will always return no join
		if (!join) {
			_render_item_state.joined_item = bdata.items_joined.request_with_grow();
			_render_item_state.joined_item->first_item_ref = bdata.item_refs.size();
			_render_item_state.joined_item->num_item_refs = 1;
			_render_item_state.joined_item->bounding_rect = ci->global_rect_cache;
			_render_item_state.joined_item->z_index = z;
			_render_item_state.joined_item->flags = bdata.joined_item_batch_flags;

			// add the reference
			BItemRef *r = bdata.item_refs.request_with_grow();
			r->item = ci;
			// we are storing final_modulate in advance per item reference
			// for baking into vertex colors.
			// this may not be ideal... as we are increasing the size of item reference,
			// but it is stupidly complex to calculate later, which would probably be slower.
			r->final_modulate = _render_item_state.final_modulate;
		} else {
			CRASH_COND(_render_item_state.joined_item == 0);
			_render_item_state.joined_item->num_item_refs += 1;
			_render_item_state.joined_item->bounding_rect = _render_item_state.joined_item->bounding_rect.merge(ci->global_rect_cache);

			BItemRef *r = bdata.item_refs.request_with_grow();
			r->item = ci;
			r->final_modulate = _render_item_state.final_modulate;
		}

	} // for s through sort items
}

void RasterizerCanvasGLES2::join_items(Item *p_item_list, int p_z) {

	_render_item_state.item_group_z = p_z;

	// join is whether to join to the previous batch.
	// batch_break is whether to PREVENT the next batch from joining with us
	// batch_break must be preserved over z_indices,
	// so is stored in _render_item_state.join_batch_break

	// if z ranged lights are present, sometimes we have to disable joining over z_indices.
	// we do this here
	if (!bdata.join_across_z_indices) {
		_render_item_state.join_batch_break = true;
	}

	while (p_item_list) {

		Item *ci = p_item_list;

		bool join;

		if (_render_item_state.join_batch_break) {
			// always start a new batch for this item
			join = false;

			// could be another batch break (i.e. prevent NEXT item from joining this)
			// so we still need to run try_join_item
			// even though we know join is false.
			// also we need to run try_join_item for every item because it keeps the state up to date,
			// if we didn't run it the state would be out of date.
			try_join_item(ci, _render_item_state, _render_item_state.join_batch_break);
		} else {
			join = try_join_item(ci, _render_item_state, _render_item_state.join_batch_break);
		}

		// assume the first item will always return no join
		if (!join) {
			_render_item_state.joined_item = bdata.items_joined.request_with_grow();
			_render_item_state.joined_item->first_item_ref = bdata.item_refs.size();
			_render_item_state.joined_item->num_item_refs = 1;
			_render_item_state.joined_item->bounding_rect = ci->global_rect_cache;
			_render_item_state.joined_item->z_index = p_z;

			// add the reference
			BItemRef *r = bdata.item_refs.request_with_grow();
			r->item = ci;
			// we are storing final_modulate in advance per item reference
			// for baking into vertex colors.
			// this may not be ideal... as we are increasing the size of item reference,
			// but it is stupidly complex to calculate later, which would probably be slower.
			r->final_modulate = _render_item_state.final_modulate;
		} else {
			CRASH_COND(_render_item_state.joined_item == 0);
			_render_item_state.joined_item->num_item_refs += 1;
			_render_item_state.joined_item->bounding_rect = _render_item_state.joined_item->bounding_rect.merge(ci->global_rect_cache);

			BItemRef *r = bdata.item_refs.request_with_grow();
			r->item = ci;
			r->final_modulate = _render_item_state.final_modulate;
		}

		p_item_list = p_item_list->next;
	}
}

void RasterizerCanvasGLES2::canvas_end() {
#ifdef DEBUG_ENABLED
	if (bdata.diagnose_frame) {
		bdata.frame_string += "canvas_end\n";
		if (bdata.stats_items_sorted) {
			bdata.frame_string += "\titems reordered: " + itos(bdata.stats_items_sorted) + "\n";
		}
		if (bdata.stats_light_items_joined) {
			bdata.frame_string += "\tlight items joined: " + itos(bdata.stats_light_items_joined) + "\n";
		}

		print_line(bdata.frame_string);
	}
#endif

	RasterizerCanvasBaseGLES2::canvas_end();
}

void RasterizerCanvasGLES2::canvas_begin() {
	// diagnose_frame?
	bdata.frame_string = ""; // just in case, always set this as we don't want a string leak in release...
#ifdef DEBUG_ENABLED
	if (bdata.settings_diagnose_frame) {
		bdata.diagnose_frame = false;

		uint32_t tick = OS::get_singleton()->get_ticks_msec();
		uint64_t frame = Engine::get_singleton()->get_frames_drawn();

		if (tick >= bdata.next_diagnose_tick) {
			bdata.next_diagnose_tick = tick + 10000;

			// the plus one is prevent starting diagnosis half way through frame
			bdata.diagnose_frame_number = frame + 1;
		}

		if (frame == bdata.diagnose_frame_number) {
			bdata.diagnose_frame = true;
			bdata.reset_stats();
		}

		if (bdata.diagnose_frame) {
			bdata.frame_string = "canvas_begin FRAME " + itos(frame) + "\n";
		}
	}
#endif

	RasterizerCanvasBaseGLES2::canvas_begin();
}

void RasterizerCanvasGLES2::canvas_render_items_begin(const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {
	// if we are debugging, flash each frame between batching renderer and old version to compare for regressions
	if (bdata.settings_flash_batching) {
		if ((Engine::get_singleton()->get_frames_drawn() % 2) == 0)
			bdata.settings_use_batching = true;
		else
			bdata.settings_use_batching = false;
	}

	if (!bdata.settings_use_batching) {
		return;
	}

	// this only needs to be done when screen size changes, but this should be
	// infrequent enough
	_calculate_scissor_threshold_area();

	// set up render item state for all the z_indexes (this is common to all z_indexes)
	_render_item_state.reset();
	_render_item_state.item_group_modulate = p_modulate;
	_render_item_state.item_group_light = p_light;
	_render_item_state.item_group_base_transform = p_base_transform;
	_render_item_state.light_region.reset();

	// batch break must be preserved over the different z indices,
	// to prevent joining to an item on a previous index if not allowed
	_render_item_state.join_batch_break = false;

	// whether to join across z indices depends on whether there are z ranged lights.
	// joined z_index items can be wrongly classified with z ranged lights.
	bdata.join_across_z_indices = true;

	int light_count = 0;
	while (p_light) {
		light_count++;

		if ((p_light->z_min != VS::CANVAS_ITEM_Z_MIN) || (p_light->z_max != VS::CANVAS_ITEM_Z_MAX)) {
			// prevent joining across z indices. This would have caused visual regressions
			bdata.join_across_z_indices = false;
		}

		p_light = p_light->next_ptr;
	}

	// can't use the light region bitfield if there are too many lights
	// hopefully most games won't blow this limit..
	// if they do they will work but it won't batch join items just in case
	if (light_count > 64) {
		_render_item_state.light_region.too_many_lights = true;
	}
}

void RasterizerCanvasGLES2::canvas_render_items_end() {
	if (!bdata.settings_use_batching) {
		return;
	}

	join_sorted_items();

#ifdef DEBUG_ENABLED
	if (bdata.diagnose_frame) {
		bdata.frame_string += "items\n";
	}
#endif

	// batching render is deferred until after going through all the z_indices, joining all the items
	canvas_render_items_implementation(0, 0, _render_item_state.item_group_modulate,
			_render_item_state.item_group_light,
			_render_item_state.item_group_base_transform);

	bdata.items_joined.reset();
	bdata.item_refs.reset();
	bdata.sort_items.reset();
}

void RasterizerCanvasGLES2::canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {
	// stage 1 : join similar items, so that their state changes are not repeated,
	// and commands from joined items can be batched together
	if (bdata.settings_use_batching) {
		record_items(p_item_list, p_z);
		return;
	}

	// only legacy renders at this stage, batched renderer doesn't render until canvas_render_items_end()
	canvas_render_items_implementation(p_item_list, p_z, p_modulate, p_light, p_base_transform);
}

void RasterizerCanvasGLES2::canvas_render_items_implementation(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light, const Transform2D &p_base_transform) {

	// parameters are easier to pass around in a structure
	RenderItemState ris;
	ris.item_group_z = p_z;
	ris.item_group_modulate = p_modulate;
	ris.item_group_light = p_light;
	ris.item_group_base_transform = p_base_transform;

	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SKELETON, false);

	state.current_tex = RID();
	state.current_tex_ptr = NULL;
	state.current_normal = RID();
	state.canvas_texscreen_used = false;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	if (bdata.settings_use_batching) {
		for (int j = 0; j < bdata.items_joined.size(); j++) {
			render_joined_item(bdata.items_joined[j], ris);
		}
	} else {
		while (p_item_list) {

			Item *ci = p_item_list;
			_canvas_render_item(ci, ris);
			p_item_list = p_item_list->next;
		}
	}

	if (ris.current_clip) {
		glDisable(GL_SCISSOR_TEST);
	}

	state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SKELETON, false);
}

// This function is a dry run of the state changes when drawing the item.
// It should duplicate the logic in _canvas_render_item,
// to decide whether items are similar enough to join
// i.e. no state differences between the 2 items.
bool RasterizerCanvasGLES2::try_join_item(Item *p_ci, RenderItemState &r_ris, bool &r_batch_break) {

	// if we set max join items to zero we can effectively prevent any joining, so
	// none of the other logic needs to run. Good for testing regression bugs, and
	// could conceivably be faster in some games.
	if (!bdata.settings_max_join_item_commands) {
		return false;
	}

	// if there are any state changes we change join to false
	// we also set r_batch_break to true if we don't want this item joined to the next
	// (e.g. an item that must not be joined at all)
	r_batch_break = false;
	bool join = true;

	// light_masked may possibly need state checking here. Check for regressions!

	// we will now allow joining even if final modulate is different
	// we will instead bake the final modulate into the vertex colors
	//	if (p_ci->final_modulate != r_ris.final_modulate) {
	//		join = false;
	//		r_ris.final_modulate = p_ci->final_modulate;
	//	}

	if (r_ris.current_clip != p_ci->final_clip_owner) {
		r_ris.current_clip = p_ci->final_clip_owner;
		join = false;
	}

	// TODO: copy back buffer

	if (p_ci->copy_back_buffer) {
		join = false;
	}

	RasterizerStorageGLES2::Skeleton *skeleton = NULL;

	{
		//skeleton handling
		if (p_ci->skeleton.is_valid() && storage->skeleton_owner.owns(p_ci->skeleton)) {
			skeleton = storage->skeleton_owner.get(p_ci->skeleton);
			if (!skeleton->use_2d) {
				skeleton = NULL;
			}
		}

		bool use_skeleton = skeleton != NULL;
		if (r_ris.prev_use_skeleton != use_skeleton) {
			r_ris.rebind_shader = true;
			r_ris.prev_use_skeleton = use_skeleton;
			join = false;
		}

		if (skeleton) {
			join = false;
			state.using_skeleton = true;
		} else {
			state.using_skeleton = false;
		}
	}

	Item *material_owner = p_ci->material_owner ? p_ci->material_owner : p_ci;

	RID material = material_owner->material;
	RasterizerStorageGLES2::Material *material_ptr = storage->material_owner.getornull(material);

	if (material != r_ris.canvas_last_material || r_ris.rebind_shader) {

		join = false;
		RasterizerStorageGLES2::Shader *shader_ptr = NULL;

		if (material_ptr) {
			shader_ptr = material_ptr->shader;

			if (shader_ptr && shader_ptr->mode != VS::SHADER_CANVAS_ITEM) {
				shader_ptr = NULL; // not a canvas item shader, don't use.
			}
		}

		if (shader_ptr) {
			if (shader_ptr->canvas_item.uses_screen_texture) {
				if (!state.canvas_texscreen_used) {
					join = false;
				}
			}
		}

		r_ris.shader_cache = shader_ptr;

		r_ris.canvas_last_material = material;

		r_ris.rebind_shader = false;
	}

	int blend_mode = r_ris.shader_cache ? r_ris.shader_cache->canvas_item.blend_mode : RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX;
	bool unshaded = r_ris.shader_cache && (r_ris.shader_cache->canvas_item.light_mode == RasterizerStorageGLES2::Shader::CanvasItem::LIGHT_MODE_UNSHADED || (blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX && blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA));
	bool reclip = false;

	// we are precalculating the final_modulate ahead of time because we need this for baking of final modulate into vertex colors
	// (only in software transform mode)
	// This maybe inefficient storing it...
	r_ris.final_modulate = unshaded ? p_ci->final_modulate : (p_ci->final_modulate * r_ris.item_group_modulate);

	if (r_ris.last_blend_mode != blend_mode) {
		join = false;
		r_ris.last_blend_mode = blend_mode;
	}

	// does the shader contain BUILTINs which should break the batching?
	bdata.joined_item_batch_flags = 0;
	if (r_ris.shader_cache) {

		unsigned int and_flags = r_ris.shader_cache->canvas_item.batch_flags & (RasterizerStorageGLES2::Shader::CanvasItem::PREVENT_COLOR_BAKING | RasterizerStorageGLES2::Shader::CanvasItem::PREVENT_VERTEX_BAKING);
		if (and_flags) {

			bool break_batching = true;

			if (and_flags == RasterizerStorageGLES2::Shader::CanvasItem::PREVENT_COLOR_BAKING) {
				// in some circumstances, if the modulate is identity, we still allow baking because reading modulate / color
				// will still be okay to do in the shader with no ill effects
				if (r_ris.final_modulate == Color(1, 1, 1, 1)) {
					break_batching = false;
				}
			}

			if (break_batching) {
				join = false;
				r_batch_break = true;

				// save the flags so that they don't need to be recalculated in the 2nd pass
				bdata.joined_item_batch_flags |= r_ris.shader_cache->canvas_item.batch_flags;
			}
		}
	}

	if ((blend_mode == RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX || blend_mode == RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA) && r_ris.item_group_light && !unshaded) {

		// we cannot join lit items easily.
		// it is possible, but not if they overlap, because
		// a + light_blend + b + light_blend IS NOT THE SAME AS
		// a + b + light_blend

		bool light_allow_join = true;

		// this is a quick getout if we have turned off light joining
		if ((bdata.settings_light_max_join_items == 0) || r_ris.light_region.too_many_lights) {
			light_allow_join = false;
		} else {
			// do light joining...

			// first calculate the light bitfield
			uint64_t light_bitfield = 0;
			uint64_t shadow_bitfield = 0;
			Light *light = r_ris.item_group_light;

			int light_count = -1;
			while (light) {
				light_count++;
				uint64_t light_bit = 1ULL << light_count;

				// note that as a cost of batching, the light culling will be less effective
				if (p_ci->light_mask & light->item_mask && r_ris.item_group_z >= light->z_min && r_ris.item_group_z <= light->z_max) {

					// Note that with the above test, it is possible to also include a bound check.
					// Tests so far have indicated better performance without it, but there may be reason to change this at a later stage,
					// so I leave the line here for reference:
					// && p_ci->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {

					light_bitfield |= light_bit;

					bool has_shadow = light->shadow_buffer.is_valid() && p_ci->light_mask & light->item_shadow_mask;

					if (has_shadow) {
						shadow_bitfield |= light_bit;
					}
				}

				light = light->next_ptr;
			}

			// now compare to previous
			if ((r_ris.light_region.light_bitfield != light_bitfield) || (r_ris.light_region.shadow_bitfield != shadow_bitfield)) {
				light_allow_join = false;

				r_ris.light_region.light_bitfield = light_bitfield;
				r_ris.light_region.shadow_bitfield = shadow_bitfield;
			} else {
				// only do these checks if necessary
				if (join && (!r_batch_break)) {

					// we still can't join, even if the lights are exactly the same, if there is overlap between the previous and this item
					if (r_ris.joined_item && light_bitfield) {
						if ((int)r_ris.joined_item->num_item_refs <= bdata.settings_light_max_join_items) {
							for (uint32_t r = 0; r < r_ris.joined_item->num_item_refs; r++) {
								Item *pRefItem = bdata.item_refs[r_ris.joined_item->first_item_ref + r].item;
								if (p_ci->global_rect_cache.intersects(pRefItem->global_rect_cache)) {
									light_allow_join = false;
									break;
								}
							}

#ifdef DEBUG_ENABLED
							if (light_allow_join) {
								bdata.stats_light_items_joined++;
							}
#endif

						} // if below max join items
						else {
							// just don't allow joining if above overlap check max items
							light_allow_join = false;
						}
					}

				} // if not batch broken already (no point in doing expensive overlap tests if not needed)
			} // if bitfields don't match
		} // if do light joining

		if (!light_allow_join) {
			// can't join
			join = false;
		}

	} else {
		// if the last item had lights, we should not join it to this one (which has no lights)
		if (r_ris.light_region.light_bitfield || r_ris.light_region.shadow_bitfield) {
			join = false;

			// setting these to zero ensures that any following item with lights will, by definition,
			// be affected by a different set of lights, and thus prevent a join
			r_ris.light_region.light_bitfield = 0;
			r_ris.light_region.shadow_bitfield = 0;
		}
	}

	if (reclip) {
		join = false;
	}

	// non rects will break the batching anyway, we don't want to record item changes, detect this
	if (!r_batch_break && _detect_batch_break(p_ci)) {
		join = false;
		r_batch_break = true;
	}

	return join;
}

bool RasterizerCanvasGLES2::_detect_batch_break(Item *p_ci) {
	int command_count = p_ci->commands.size();

	// Any item that contains commands that are default
	// (i.e. not handled by software transform and the batching renderer) should not be joined.

	// In order to work this out, it does a lookahead through the commands,
	// which could potentially be very expensive. As such it makes sense to put a limit on this
	// to some small number, which will catch nearly all cases which need joining,
	// but not be overly expensive in the case of items with large numbers of commands.

	// It is hard to know what this number should be, empirically,
	// and this has not been fully investigated. It works to join single sprite items when set to 1 or above.
	// Note that there is a cost to increasing this because it has to look in advance through
	// the commands.
	// On the other hand joining items where possible will usually be better up to a certain
	// number where the cost of software transform is higher than separate drawcalls with hardware
	// transform.

	// if there are more than this number of commands in the item, we
	// don't allow joining (separate state changes, and hardware transform)
	// This is set to quite a conservative (low) number until investigated properly.
	// const int MAX_JOIN_ITEM_COMMANDS = 16;

	if (command_count > bdata.settings_max_join_item_commands) {
		return true;
	} else {
		Item::Command *const *commands = p_ci->commands.ptr();

		// do as many commands as possible until the vertex buffer will be full up
		for (int command_num = 0; command_num < command_count; command_num++) {

			Item::Command *command = commands[command_num];
			CRASH_COND(!command);

			switch (command->type) {

				default: {
					return true;
				} break;
				case Item::Command::TYPE_RECT:
				case Item::Command::TYPE_TRANSFORM: {
				} break;
			} // switch

		} // for through commands

	} // else

	return false;
}

// Legacy non-batched implementation for regression testing.
// Should be removed after testing phase to avoid duplicate codepaths.
void RasterizerCanvasGLES2::_canvas_render_item(Item *p_ci, RenderItemState &r_ris) {
	storage->info.render._2d_item_count++;

	if (r_ris.current_clip != p_ci->final_clip_owner) {

		r_ris.current_clip = p_ci->final_clip_owner;

		if (r_ris.current_clip) {
			glEnable(GL_SCISSOR_TEST);
			int y = storage->frame.current_rt->height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
			if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
				y = r_ris.current_clip->final_clip_rect.position.y;
			glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);
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

	RasterizerStorageGLES2::Skeleton *skeleton = NULL;

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
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SKELETON, use_skeleton);
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

	Item *material_owner = p_ci->material_owner ? p_ci->material_owner : p_ci;

	RID material = material_owner->material;
	RasterizerStorageGLES2::Material *material_ptr = storage->material_owner.getornull(material);

	if (material != r_ris.canvas_last_material || r_ris.rebind_shader) {

		RasterizerStorageGLES2::Shader *shader_ptr = NULL;

		if (material_ptr) {
			shader_ptr = material_ptr->shader;

			if (shader_ptr && shader_ptr->mode != VS::SHADER_CANVAS_ITEM) {
				shader_ptr = NULL; // not a canvas item shader, don't use.
			}
		}

		if (shader_ptr) {
			if (shader_ptr->canvas_item.uses_screen_texture) {
				if (!state.canvas_texscreen_used) {
					//copy if not copied before
					_copy_texscreen(Rect2());

					// blend mode will have been enabled so make sure we disable it again later on
					//last_blend_mode = last_blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_DISABLED ? last_blend_mode : -1;
				}

				if (storage->frame.current_rt->copy_screen_effect.color) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
					glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->copy_screen_effect.color);
				}
			}

			if (shader_ptr != r_ris.shader_cache) {

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

				glActiveTexture(GL_TEXTURE0 + i);

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

				if (t->redraw_if_visible) {
					VisualServerRaster::redraw_request();
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

	int blend_mode = r_ris.shader_cache ? r_ris.shader_cache->canvas_item.blend_mode : RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX;
	bool unshaded = r_ris.shader_cache && (r_ris.shader_cache->canvas_item.light_mode == RasterizerStorageGLES2::Shader::CanvasItem::LIGHT_MODE_UNSHADED || (blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX && blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA));
	bool reclip = false;

	if (r_ris.last_blend_mode != blend_mode) {

		switch (blend_mode) {

			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
				}

			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_ADD: {

				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
				}

			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_SUB: {

				glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
				}
			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MUL: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
				} else {
					glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
				}
			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
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

	if (unshaded || (state.uniforms.final_modulate.a > 0.001 && (!r_ris.shader_cache || r_ris.shader_cache->canvas_item.light_mode != RasterizerStorageGLES2::Shader::CanvasItem::LIGHT_MODE_LIGHT_ONLY) && !p_ci->light_masked))
		_canvas_item_render_commands(p_ci, NULL, reclip, material_ptr);

	r_ris.rebind_shader = true; // hacked in for now.

	if ((blend_mode == RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX || blend_mode == RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA) && r_ris.item_group_light && !unshaded) {

		Light *light = r_ris.item_group_light;
		bool light_used = false;
		VS::CanvasLightMode mode = VS::CANVAS_LIGHT_MODE_ADD;
		state.uniforms.final_modulate = p_ci->final_modulate; // remove the canvas modulate

		while (light) {

			if (p_ci->light_mask & light->item_mask && r_ris.item_group_z >= light->z_min && r_ris.item_group_z <= light->z_max && p_ci->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {

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

					state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_LIGHTING, true);
					light_used = true;
				}

				bool has_shadow = light->shadow_buffer.is_valid() && p_ci->light_mask & light->item_shadow_mask;

				state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SHADOWS, has_shadow);
				if (has_shadow) {
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_USE_GRADIENT, light->shadow_gradient_length > 0);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_NEAREST, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_NONE);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF3, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF3);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF5, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF5);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF7, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF7);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF9, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF9);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF13, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF13);
				}

				state.canvas_shader.bind();
				state.using_light = light;
				state.using_shadow = has_shadow;

				//always re-set uniforms, since light parameters changed
				_set_uniforms();
				state.canvas_shader.use_material((void *)material_ptr);

				glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
				RasterizerStorageGLES2::Texture *t = storage->texture_owner.getornull(light->texture);
				if (!t) {
					glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
				} else {
					t = t->get_ptr();

					glBindTexture(t->target, t->tex_id);
				}

				glActiveTexture(GL_TEXTURE0);
				_canvas_item_render_commands(p_ci, NULL, reclip, material_ptr); //redraw using light

				state.using_light = NULL;
			}

			light = light->next_ptr;
		}

		if (light_used) {

			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_LIGHTING, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SHADOWS, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_NEAREST, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF3, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF5, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF7, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF9, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF13, false);

			state.canvas_shader.bind();

			r_ris.last_blend_mode = -1;

			/*
			//this is set again, so it should not be needed anyway?
			state.canvas_item_modulate = unshaded ? ci->final_modulate : Color(
						ci->final_modulate.r * p_modulate.r,
						ci->final_modulate.g * p_modulate.g,
						ci->final_modulate.b * p_modulate.b,
						ci->final_modulate.a * p_modulate.a );

			state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX,state.final_transform);
			state.canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX,Transform2D());
			state.canvas_shader.set_uniform(CanvasShaderGLES2::FINAL_MODULATE,state.canvas_item_modulate);

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
		int y = storage->frame.current_rt->height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
		if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
			y = r_ris.current_clip->final_clip_rect.position.y;
		glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);
	}
}

void RasterizerCanvasGLES2::render_joined_item(const BItemJoined &p_bij, RenderItemState &r_ris) {

	storage->info.render._2d_item_count++;

#ifdef DEBUG_ENABLED
	if (bdata.diagnose_frame) {
		bdata.frame_string += "\tjoined_item " + itos(p_bij.num_item_refs) + " refs\n";
		if (p_bij.z_index != 0) {
			bdata.frame_string += "\t\t(z " + itos(p_bij.z_index) + ")\n";
		}
	}
#endif

	// this must be reset for each joined item,
	// it only exists to prevent capturing the screen more than once per item
	state.canvas_texscreen_used = false;

	// all the joined items will share the same state with the first item
	Item *ci = bdata.item_refs[p_bij.first_item_ref].item;

	if (r_ris.current_clip != ci->final_clip_owner) {

		r_ris.current_clip = ci->final_clip_owner;

		if (r_ris.current_clip) {
			glEnable(GL_SCISSOR_TEST);
			int y = storage->frame.current_rt->height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
			if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
				y = r_ris.current_clip->final_clip_rect.position.y;
			glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);
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

	RasterizerStorageGLES2::Skeleton *skeleton = NULL;

	{
		//skeleton handling
		if (ci->skeleton.is_valid() && storage->skeleton_owner.owns(ci->skeleton)) {
			skeleton = storage->skeleton_owner.get(ci->skeleton);
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
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SKELETON, use_skeleton);
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

	Item *material_owner = ci->material_owner ? ci->material_owner : ci;

	RID material = material_owner->material;
	RasterizerStorageGLES2::Material *material_ptr = storage->material_owner.getornull(material);

	if (material != r_ris.canvas_last_material || r_ris.rebind_shader) {

		RasterizerStorageGLES2::Shader *shader_ptr = NULL;

		if (material_ptr) {
			shader_ptr = material_ptr->shader;

			if (shader_ptr && shader_ptr->mode != VS::SHADER_CANVAS_ITEM) {
				shader_ptr = NULL; // not a canvas item shader, don't use.
			}
		}

		if (shader_ptr) {
			if (shader_ptr->canvas_item.uses_screen_texture) {
				if (!state.canvas_texscreen_used) {
					//copy if not copied before
					_copy_texscreen(Rect2());

					// blend mode will have been enabled so make sure we disable it again later on
					//last_blend_mode = last_blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_DISABLED ? last_blend_mode : -1;
				}

				if (storage->frame.current_rt->copy_screen_effect.color) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
					glBindTexture(GL_TEXTURE_2D, storage->frame.current_rt->copy_screen_effect.color);
				}
			}

			if (shader_ptr != r_ris.shader_cache) {

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

				glActiveTexture(GL_TEXTURE0 + i);

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

				if (t->redraw_if_visible) {
					VisualServerRaster::redraw_request();
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

	int blend_mode = r_ris.shader_cache ? r_ris.shader_cache->canvas_item.blend_mode : RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX;
	bool unshaded = r_ris.shader_cache && (r_ris.shader_cache->canvas_item.light_mode == RasterizerStorageGLES2::Shader::CanvasItem::LIGHT_MODE_UNSHADED || (blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX && blend_mode != RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA));
	bool reclip = false;

	if (r_ris.last_blend_mode != blend_mode) {

		switch (blend_mode) {

			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
				}

			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_ADD: {

				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
				}

			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_SUB: {

				glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
				} else {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
				}
			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MUL: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
				} else {
					glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
				}
			} break;
			case RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
				}
			} break;
		}
	}

	// using software transform
	if (!p_bij.use_hardware_transform()) {
		state.uniforms.modelview_matrix = Transform2D();
		// final_modulate will be baked per item ref so the final_modulate can be an identity color
		state.uniforms.final_modulate = Color(1, 1, 1, 1);
	} else {
		state.uniforms.modelview_matrix = ci->final_transform;
		// could use the stored version of final_modulate in item ref? Test which is faster NYI
		state.uniforms.final_modulate = unshaded ? ci->final_modulate : (ci->final_modulate * r_ris.item_group_modulate);
	}
	state.uniforms.extra_matrix = Transform2D();

	_set_uniforms();

	if (unshaded || (state.uniforms.final_modulate.a > 0.001 && (!r_ris.shader_cache || r_ris.shader_cache->canvas_item.light_mode != RasterizerStorageGLES2::Shader::CanvasItem::LIGHT_MODE_LIGHT_ONLY) && !ci->light_masked))
		render_joined_item_commands(p_bij, NULL, reclip, material_ptr, false);

	r_ris.rebind_shader = true; // hacked in for now.

	if ((blend_mode == RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_MIX || blend_mode == RasterizerStorageGLES2::Shader::CanvasItem::BLEND_MODE_PMALPHA) && r_ris.item_group_light && !unshaded) {

		Light *light = r_ris.item_group_light;
		bool light_used = false;
		VS::CanvasLightMode mode = VS::CANVAS_LIGHT_MODE_ADD;

		// we leave this set to 1, 1, 1, 1 if using software because the colors are baked into the vertices
		if (p_bij.use_hardware_transform()) {
			state.uniforms.final_modulate = ci->final_modulate; // remove the canvas modulate
		}

		while (light) {

			// use the bounding rect of the joined items, NOT only the bounding rect of the first item.
			// note this is a cost of batching, the light culling will be less effective

			// note that the r_ris.item_group_z will be out of date because we are using deferred rendering till canvas_render_items_end()
			// so we have to test z against the stored value in the joined item
			if (ci->light_mask & light->item_mask && p_bij.z_index >= light->z_min && p_bij.z_index <= light->z_max && p_bij.bounding_rect.intersects_transformed(light->xform_cache, light->rect_cache)) {

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

					state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_LIGHTING, true);
					light_used = true;
				}

				bool has_shadow = light->shadow_buffer.is_valid() && ci->light_mask & light->item_shadow_mask;

				state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SHADOWS, has_shadow);
				if (has_shadow) {
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_USE_GRADIENT, light->shadow_gradient_length > 0);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_NEAREST, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_NONE);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF3, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF3);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF5, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF5);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF7, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF7);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF9, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF9);
					state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF13, light->shadow_filter == VS::CANVAS_LIGHT_FILTER_PCF13);
				}

				state.canvas_shader.bind();
				state.using_light = light;
				state.using_shadow = has_shadow;

				//always re-set uniforms, since light parameters changed
				_set_uniforms();
				state.canvas_shader.use_material((void *)material_ptr);

				glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
				RasterizerStorageGLES2::Texture *t = storage->texture_owner.getornull(light->texture);
				if (!t) {
					glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
				} else {
					t = t->get_ptr();

					glBindTexture(t->target, t->tex_id);
				}

				glActiveTexture(GL_TEXTURE0);

				// redraw using light.
				// if there is no clip item, we can consider scissoring to the intersection area between the light and the item
				// this can greatly reduce fill rate ..
				// at the cost of glScissor commands, so is optional
				if (!bdata.settings_scissor_lights || r_ris.current_clip) {
					render_joined_item_commands(p_bij, NULL, reclip, material_ptr, true);
				} else {
					bool scissor = _light_scissor_begin(p_bij.bounding_rect, light->xform_cache, light->rect_cache);
					render_joined_item_commands(p_bij, NULL, reclip, material_ptr, true);
					if (scissor) {
						glDisable(GL_SCISSOR_TEST);
					}
				}

				state.using_light = NULL;
			}

			light = light->next_ptr;
		}

		if (light_used) {

			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_LIGHTING, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::USE_SHADOWS, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_NEAREST, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF3, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF5, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF7, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF9, false);
			state.canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_FILTER_PCF13, false);

			state.canvas_shader.bind();

			r_ris.last_blend_mode = -1;

			/*
			//this is set again, so it should not be needed anyway?
			state.canvas_item_modulate = unshaded ? ci->final_modulate : Color(
						ci->final_modulate.r * p_modulate.r,
						ci->final_modulate.g * p_modulate.g,
						ci->final_modulate.b * p_modulate.b,
						ci->final_modulate.a * p_modulate.a );

			state.canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX,state.final_transform);
			state.canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX,Transform2D());
			state.canvas_shader.set_uniform(CanvasShaderGLES2::FINAL_MODULATE,state.canvas_item_modulate);

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
		int y = storage->frame.current_rt->height - (r_ris.current_clip->final_clip_rect.position.y + r_ris.current_clip->final_clip_rect.size.y);
		if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
			y = r_ris.current_clip->final_clip_rect.position.y;
		glScissor(r_ris.current_clip->final_clip_rect.position.x, y, r_ris.current_clip->final_clip_rect.size.width, r_ris.current_clip->final_clip_rect.size.height);
	}
}

bool RasterizerCanvasGLES2::_light_find_intersection(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect, Rect2 &r_cliprect) const {
	// transform light to world space (note this is done in the earlier intersection test, so could
	// be made more efficient)
	Vector2 pts[4] = {
		p_light_xform.xform(p_light_rect.position),
		p_light_xform.xform(Vector2(p_light_rect.position.x + p_light_rect.size.x, p_light_rect.position.y)),
		p_light_xform.xform(Vector2(p_light_rect.position.x, p_light_rect.position.y + p_light_rect.size.y)),
		p_light_xform.xform(Vector2(p_light_rect.position.x + p_light_rect.size.x, p_light_rect.position.y + p_light_rect.size.y)),
	};

	// calculate the light bound rect in world space
	Rect2 lrect(pts[0].x, pts[0].y, 0, 0);
	for (int n = 1; n < 4; n++) {
		lrect.expand_to(pts[n]);
	}

	// intersection between the 2 rects
	// they should probably always intersect, because of earlier check, but just in case...
	if (!p_item_rect.intersects(lrect))
		return false;

	// note this does almost the same as Rect2.clip but slightly more efficient for our use case
	r_cliprect.position.x = MAX(p_item_rect.position.x, lrect.position.x);
	r_cliprect.position.y = MAX(p_item_rect.position.y, lrect.position.y);

	Point2 item_rect_end = p_item_rect.position + p_item_rect.size;
	Point2 lrect_end = lrect.position + lrect.size;

	r_cliprect.size.x = MIN(item_rect_end.x, lrect_end.x) - r_cliprect.position.x;
	r_cliprect.size.y = MIN(item_rect_end.y, lrect_end.y) - r_cliprect.position.y;

	return true;
}

bool RasterizerCanvasGLES2::_light_scissor_begin(const Rect2 &p_item_rect, const Transform2D &p_light_xform, const Rect2 &p_light_rect) const {

	float area_item = p_item_rect.size.x * p_item_rect.size.y; // double check these are always positive

	// quick reject .. the area of pixels saved can never be more than the area of the item
	if (area_item < bdata.scissor_threshold_area) {
		return false;
	}

	Rect2 cliprect;
	if (!_light_find_intersection(p_item_rect, p_light_xform, p_light_rect, cliprect)) {
		// should not really occur .. but just in case
		cliprect = Rect2(0, 0, 0, 0);
	} else {
		// some conditions not to scissor
		// determine the area (fill rate) that will be saved
		float area_cliprect = cliprect.size.x * cliprect.size.y;
		float area_saved = area_item - area_cliprect;

		// if area saved is too small, don't scissor
		if (area_saved < bdata.scissor_threshold_area) {
			return false;
		}
	}

	glEnable(GL_SCISSOR_TEST);
	int y = storage->frame.current_rt->height - (cliprect.position.y + cliprect.size.y);
	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP])
		y = cliprect.position.y;
	glScissor(cliprect.position.x, y, cliprect.size.width, cliprect.size.height);

	return true;
}

void RasterizerCanvasGLES2::_calculate_scissor_threshold_area() {
	if (!bdata.settings_scissor_lights) {
		return;
	}

	// scissor area threshold is 0.0 to 1.0 in the settings for ease of use.
	// we need to translate to an absolute area to determine quickly whether
	// to scissor.
	if (bdata.settings_scissor_threshold < 0.0001f) {
		bdata.scissor_threshold_area = -1.0f; // will always pass
	} else {
		// in pixels
		int w = storage->frame.current_rt->width;
		int h = storage->frame.current_rt->height;

		int screen_area = w * h;

		bdata.scissor_threshold_area = bdata.settings_scissor_threshold * screen_area;
	}
}

void RasterizerCanvasGLES2::initialize() {
	RasterizerCanvasBaseGLES2::initialize();

	bdata.settings_use_batching = GLOBAL_GET("rendering/batching/options/use_batching");
	bdata.settings_max_join_item_commands = GLOBAL_GET("rendering/batching/parameters/max_join_item_commands");
	bdata.settings_colored_vertex_format_threshold = GLOBAL_GET("rendering/batching/parameters/colored_vertex_format_threshold");
	bdata.settings_item_reordering_lookahead = GLOBAL_GET("rendering/batching/parameters/item_reordering_lookahead");
	bdata.settings_light_max_join_items = GLOBAL_GET("rendering/batching/lights/max_join_items");
	bdata.settings_use_single_rect_fallback = GLOBAL_GET("rendering/batching/options/single_rect_fallback");

	// alternatively only enable uv contract if pixel snap in use,
	// but with this enable bool, it should not be necessary
	bdata.settings_uv_contract = GLOBAL_GET("rendering/batching/precision/uv_contract");
	bdata.settings_uv_contract_amount = (float)GLOBAL_GET("rendering/batching/precision/uv_contract_amount") / 1000000.0f;

	// we can use the threshold to determine whether to turn scissoring off or on
	bdata.settings_scissor_threshold = GLOBAL_GET("rendering/batching/lights/scissor_area_threshold");
	if (bdata.settings_scissor_threshold > 0.999f) {
		bdata.settings_scissor_lights = false;
	} else {
		bdata.settings_scissor_lights = true;

		// apply power of 4 relationship for the area, as most of the important changes
		// will be happening at low values of scissor threshold
		bdata.settings_scissor_threshold *= bdata.settings_scissor_threshold;
		bdata.settings_scissor_threshold *= bdata.settings_scissor_threshold;
	}

	// The sweet spot on my desktop for cache is actually smaller than the max, and this
	// is the default. This saves memory too so we will use it for now, needs testing to see whether this varies according
	// to device / platform.
	bdata.settings_batch_buffer_num_verts = GLOBAL_GET("rendering/batching/parameters/batch_buffer_size");

	// override the use_batching setting in the editor
	// (note that if the editor can't start, you can't change the use_batching project setting!)
	if (Engine::get_singleton()->is_editor_hint()) {
		bool use_in_editor = GLOBAL_GET("rendering/batching/options/use_batching_in_editor");
		bdata.settings_use_batching = use_in_editor;

		// fix some settings in the editor, as the performance not worth the risk
		bdata.settings_use_single_rect_fallback = false;
	}

	// if we are using batching, we will purposefully disable the nvidia workaround.
	// This is because the only reason to use the single rect fallback is the approx 2x speed
	// of the uniform drawing technique. If we used nvidia workaround, speed would be
	// approx equal to the batcher drawing technique (indexed primitive + VB).
	if (bdata.settings_use_batching) {
		use_nvidia_rect_workaround = false;
	}

	// For debugging, if flash is set in project settings, it will flash on alternate frames
	// between the non-batched renderer and the batched renderer,
	// in order to find regressions.
	// This should not be used except during development.
	// make a note of the original choice in case we are flashing on and off the batching
	bdata.settings_use_batching_original_choice = bdata.settings_use_batching;
	bdata.settings_flash_batching = GLOBAL_GET("rendering/batching/debug/flash_batching");
	if (!bdata.settings_use_batching) {
		// no flash when batching turned off
		bdata.settings_flash_batching = false;
	}

	// frame diagnosis. print out the batches every nth frame
	bdata.settings_diagnose_frame = false;
	if (!Engine::get_singleton()->is_editor_hint() && bdata.settings_use_batching) {
		bdata.settings_diagnose_frame = GLOBAL_GET("rendering/batching/debug/diagnose_frame");
	}

	// the maximum num quads in a batch is limited by GLES2. We can have only 16 bit indices,
	// which means we can address a vertex buffer of max size 65535. 4 vertices are needed per quad.

	// Note this determines the memory use by the vertex buffer vector. max quads (65536/4)-1
	// but can be reduced to save memory if really required (will result in more batches though)
	const int max_possible_quads = (65536 / 4) - 1;
	const int min_possible_quads = 8; // some reasonable small value

	// value from project settings
	int max_quads = bdata.settings_batch_buffer_num_verts / 4;

	// sanity checks
	max_quads = CLAMP(max_quads, min_possible_quads, max_possible_quads);
	bdata.settings_max_join_item_commands = CLAMP(bdata.settings_max_join_item_commands, 0, 65535);
	bdata.settings_colored_vertex_format_threshold = CLAMP(bdata.settings_colored_vertex_format_threshold, 0.0f, 1.0f);
	bdata.settings_scissor_threshold = CLAMP(bdata.settings_scissor_threshold, 0.0f, 1.0f);
	bdata.settings_light_max_join_items = CLAMP(bdata.settings_light_max_join_items, 0, 65535);
	bdata.settings_item_reordering_lookahead = CLAMP(bdata.settings_item_reordering_lookahead, 0, 65535);

	// for debug purposes, output a string with the batching options
	String batching_options_string = "OpenGL ES 2.0 Batching: ";
	if (bdata.settings_use_batching) {
		batching_options_string += "ON";

		if (OS::get_singleton()->is_stdout_verbose()) {
			batching_options_string += "\n\tOPTIONS\n";
			batching_options_string += "\tmax_join_item_commands " + itos(bdata.settings_max_join_item_commands) + "\n";
			batching_options_string += "\tcolored_vertex_format_threshold " + String(Variant(bdata.settings_colored_vertex_format_threshold)) + "\n";
			batching_options_string += "\tbatch_buffer_size " + itos(bdata.settings_batch_buffer_num_verts) + "\n";
			batching_options_string += "\tlight_scissor_area_threshold " + String(Variant(bdata.settings_scissor_threshold)) + "\n";

			batching_options_string += "\titem_reordering_lookahead " + itos(bdata.settings_item_reordering_lookahead) + "\n";
			batching_options_string += "\tlight_max_join_items " + itos(bdata.settings_light_max_join_items) + "\n";
			batching_options_string += "\tsingle_rect_fallback " + String(Variant(bdata.settings_use_single_rect_fallback)) + "\n";

			batching_options_string += "\tdebug_flash " + String(Variant(bdata.settings_flash_batching)) + "\n";
			batching_options_string += "\tdiagnose_frame " + String(Variant(bdata.settings_diagnose_frame));
		}

		print_line(batching_options_string);
	}

	// special case, for colored vertex format threshold.
	// as the comparison is >=, we want to be able to totally turn on or off
	// conversion to colored vertex format at the extremes, so we will force
	// 1.0 to be just above 1.0
	if (bdata.settings_colored_vertex_format_threshold > 0.995f) {
		bdata.settings_colored_vertex_format_threshold = 1.01f;
	}

	// save memory when batching off
	if (!bdata.settings_use_batching) {
		max_quads = 0;
	}

	uint32_t sizeof_batch_vert = sizeof(BatchVertex);

	bdata.max_quads = max_quads;

	// 4 verts per quad
	bdata.vertex_buffer_size_units = max_quads * 4;

	// the index buffer can be longer than 65535, only the indices need to be within this range
	bdata.index_buffer_size_units = max_quads * 6;

	// this comes out at approx 64K for non-colored vertex buffer, and 128K for colored vertex buffer
	bdata.vertex_buffer_size_bytes = bdata.vertex_buffer_size_units * sizeof_batch_vert;
	bdata.index_buffer_size_bytes = bdata.index_buffer_size_units * 2; // 16 bit inds

	// create equal number of normal and (max) unit sized verts (as the normal may need to be translated to a larger FVF)
	bdata.vertices.create(bdata.vertex_buffer_size_units); // 512k
	bdata.unit_vertices.create(bdata.vertices.max_size(), sizeof(BatchVertexLightAngled));

	// extra data per vert needed for larger FVFs
	bdata.light_angles.create(bdata.vertices.max_size());

	// num batches will be auto increased dynamically if required
	bdata.batches.create(1024);
	bdata.batches_temp.create(bdata.batches.max_size());

	// batch textures can also be increased dynamically
	bdata.batch_textures.create(32);

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

		for (int q = 0; q < max_quads; q++) {
			int i_pos = q * 6; //  6 inds per quad
			int q_pos = q * 4; // 4 verts per quad
			indices.set(i_pos, q_pos);
			indices.set(i_pos + 1, q_pos + 1);
			indices.set(i_pos + 2, q_pos + 2);
			indices.set(i_pos + 3, q_pos);
			indices.set(i_pos + 4, q_pos + 2);
			indices.set(i_pos + 5, q_pos + 3);

			// we can only use 16 bit indices in GLES2!
#ifdef DEBUG_ENABLED
			CRASH_COND((q_pos + 3) > 65535);
#endif
		}

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, bdata.index_buffer_size_bytes, &indices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	} // only if there is a vertex buffer (batching is on)
}

RasterizerCanvasGLES2::RasterizerCanvasGLES2() {

	bdata.settings_use_batching = false;
}

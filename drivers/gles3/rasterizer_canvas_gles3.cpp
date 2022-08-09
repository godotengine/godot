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

#ifdef GLES3_ENABLED

#include "core/os/os.h"
#include "rasterizer_scene_gles3.h"

#include "core/config/project_settings.h"
#include "servers/rendering/rendering_server_default.h"
#include "storage/config.h"
#include "storage/material_storage.h"
#include "storage/mesh_storage.h"
#include "storage/texture_storage.h"

#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif

//static const GLenum gl_primitive[] = {
//	GL_POINTS,
//	GL_LINES,
//	GL_LINE_STRIP,
//	GL_LINE_LOOP,
//	GL_TRIANGLES,
//	GL_TRIANGLE_STRIP,
//	GL_TRIANGLE_FAN
//};

void RasterizerCanvasGLES3::_update_transform_2d_to_mat4(const Transform2D &p_transform, float *p_mat4) {
	p_mat4[0] = p_transform.columns[0][0];
	p_mat4[1] = p_transform.columns[0][1];
	p_mat4[2] = 0;
	p_mat4[3] = 0;
	p_mat4[4] = p_transform.columns[1][0];
	p_mat4[5] = p_transform.columns[1][1];
	p_mat4[6] = 0;
	p_mat4[7] = 0;
	p_mat4[8] = 0;
	p_mat4[9] = 0;
	p_mat4[10] = 1;
	p_mat4[11] = 0;
	p_mat4[12] = p_transform.columns[2][0];
	p_mat4[13] = p_transform.columns[2][1];
	p_mat4[14] = 0;
	p_mat4[15] = 1;
}

void RasterizerCanvasGLES3::_update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4) {
	p_mat2x4[0] = p_transform.columns[0][0];
	p_mat2x4[1] = p_transform.columns[1][0];
	p_mat2x4[2] = 0;
	p_mat2x4[3] = p_transform.columns[2][0];

	p_mat2x4[4] = p_transform.columns[0][1];
	p_mat2x4[5] = p_transform.columns[1][1];
	p_mat2x4[6] = 0;
	p_mat2x4[7] = p_transform.columns[2][1];
}

void RasterizerCanvasGLES3::_update_transform_2d_to_mat2x3(const Transform2D &p_transform, float *p_mat2x3) {
	p_mat2x3[0] = p_transform.columns[0][0];
	p_mat2x3[1] = p_transform.columns[0][1];
	p_mat2x3[2] = p_transform.columns[1][0];
	p_mat2x3[3] = p_transform.columns[1][1];
	p_mat2x3[4] = p_transform.columns[2][0];
	p_mat2x3[5] = p_transform.columns[2][1];
}

void RasterizerCanvasGLES3::_update_transform_to_mat4(const Transform3D &p_transform, float *p_mat4) {
	p_mat4[0] = p_transform.basis.rows[0][0];
	p_mat4[1] = p_transform.basis.rows[1][0];
	p_mat4[2] = p_transform.basis.rows[2][0];
	p_mat4[3] = 0;
	p_mat4[4] = p_transform.basis.rows[0][1];
	p_mat4[5] = p_transform.basis.rows[1][1];
	p_mat4[6] = p_transform.basis.rows[2][1];
	p_mat4[7] = 0;
	p_mat4[8] = p_transform.basis.rows[0][2];
	p_mat4[9] = p_transform.basis.rows[1][2];
	p_mat4[10] = p_transform.basis.rows[2][2];
	p_mat4[11] = 0;
	p_mat4[12] = p_transform.origin.x;
	p_mat4[13] = p_transform.origin.y;
	p_mat4[14] = p_transform.origin.z;
	p_mat4[15] = 1;
}

void RasterizerCanvasGLES3::canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();

	Transform2D canvas_transform_inverse = p_canvas_transform.affine_inverse();

	// Clear out any state that may have been left from the 3D pass.
	reset_canvas();

	// TODO: Setup Directional Lights

	// TODO: Setup lights

	{
		//update canvas state uniform buffer
		StateBuffer state_buffer;

		Size2i ssize = texture_storage->render_target_get_size(p_to_render_target);

		Transform3D screen_transform;
		screen_transform.translate_local(-(ssize.width / 2.0f), -(ssize.height / 2.0f), 0.0f);
		screen_transform.scale(Vector3(2.0f / ssize.width, 2.0f / ssize.height, 1.0f));
		_update_transform_to_mat4(screen_transform, state_buffer.screen_transform);
		_update_transform_2d_to_mat4(p_canvas_transform, state_buffer.canvas_transform);

		Transform2D normal_transform = p_canvas_transform;
		normal_transform.columns[0].normalize();
		normal_transform.columns[1].normalize();
		normal_transform.columns[2] = Vector2();
		_update_transform_2d_to_mat4(normal_transform, state_buffer.canvas_normal_transform);

		state_buffer.canvas_modulate[0] = p_modulate.r;
		state_buffer.canvas_modulate[1] = p_modulate.g;
		state_buffer.canvas_modulate[2] = p_modulate.b;
		state_buffer.canvas_modulate[3] = p_modulate.a;

		Size2 render_target_size = texture_storage->render_target_get_size(p_to_render_target);
		state_buffer.screen_pixel_size[0] = 1.0 / render_target_size.x;
		state_buffer.screen_pixel_size[1] = 1.0 / render_target_size.y;

		// TODO: temporary, this should be set at the top of this function
		glViewport(0, 0, render_target_size.x, render_target_size.y);

		state_buffer.time = state.time;
		state_buffer.use_pixel_snap = p_snap_2d_vertices_to_pixel;

		state_buffer.directional_light_count = 0; //directional_light_count;

		Vector2 canvas_scale = p_canvas_transform.get_scale();

		state_buffer.sdf_to_screen[0] = render_target_size.width / canvas_scale.x;
		state_buffer.sdf_to_screen[1] = render_target_size.height / canvas_scale.y;

		state_buffer.screen_to_sdf[0] = 1.0 / state_buffer.sdf_to_screen[0];
		state_buffer.screen_to_sdf[1] = 1.0 / state_buffer.sdf_to_screen[1];

		Rect2 sdf_rect = texture_storage->render_target_get_sdf_rect(p_to_render_target);
		Rect2 sdf_tex_rect(sdf_rect.position / canvas_scale, sdf_rect.size / canvas_scale);

		state_buffer.sdf_to_tex[0] = 1.0 / sdf_tex_rect.size.width;
		state_buffer.sdf_to_tex[1] = 1.0 / sdf_tex_rect.size.height;
		state_buffer.sdf_to_tex[2] = -sdf_tex_rect.position.x / sdf_tex_rect.size.width;
		state_buffer.sdf_to_tex[3] = -sdf_tex_rect.position.y / sdf_tex_rect.size.height;

		state_buffer.tex_to_sdf = 1.0 / ((canvas_scale.x + canvas_scale.y) * 0.5);
		glBindBufferBase(GL_UNIFORM_BUFFER, BASE_UNIFORM_LOCATION, state.canvas_state_buffer);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(StateBuffer), &state_buffer, GL_STREAM_DRAW);

		GLuint global_buffer = material_storage->global_shader_uniforms_get_uniform_buffer();

		glBindBufferBase(GL_UNIFORM_BUFFER, GLOBAL_UNIFORM_LOCATION, global_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	{
		state.default_filter = p_default_filter;
		state.default_repeat = p_default_repeat;
	}

	r_sdf_used = false;
	int item_count = 0;
	bool backbuffer_cleared = false;
	bool time_used = false;
	bool material_screen_texture_found = false;
	Rect2 back_buffer_rect;
	bool backbuffer_copy = false;

	Item *ci = p_item_list;
	Item *canvas_group_owner = nullptr;

	while (ci) {
		if (ci->copy_back_buffer && canvas_group_owner == nullptr) {
			backbuffer_copy = true;

			if (ci->copy_back_buffer->full) {
				back_buffer_rect = Rect2();
			} else {
				back_buffer_rect = ci->copy_back_buffer->rect;
			}
		}

		// Check material for something that may change flow of rendering, but do not bind for now.
		RID material = ci->material_owner == nullptr ? ci->material : ci->material_owner->material;
		if (material.is_valid()) {
			GLES3::CanvasMaterialData *md = static_cast<GLES3::CanvasMaterialData *>(material_storage->material_get_data(material, RS::SHADER_CANVAS_ITEM));
			if (md && md->shader_data->valid) {
				if (md->shader_data->uses_screen_texture && canvas_group_owner == nullptr) {
					if (!material_screen_texture_found) {
						backbuffer_copy = true;
						back_buffer_rect = Rect2();
					}
				}

				if (md->shader_data->uses_sdf) {
					r_sdf_used = true;
				}
				if (md->shader_data->uses_time) {
					time_used = true;
				}
			}
		}

		if (ci->canvas_group_owner != nullptr) {
			if (canvas_group_owner == nullptr) {
				// Canvas group begins here, render until before this item

				_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
				item_count = 0;

				Rect2i group_rect = ci->canvas_group_owner->global_rect_cache;

				if (ci->canvas_group_owner->canvas_group->mode == RS::CANVAS_GROUP_MODE_OPAQUE) {
					texture_storage->render_target_copy_to_back_buffer(p_to_render_target, group_rect, false);
				} else if (!backbuffer_cleared) {
					texture_storage->render_target_clear_back_buffer(p_to_render_target, Rect2i(), Color(0, 0, 0, 0));
					backbuffer_cleared = true;
				}

				backbuffer_copy = false;
				canvas_group_owner = ci->canvas_group_owner; //continue until owner found
			}

			ci->canvas_group_owner = nullptr; //must be cleared
		}

		if (!backbuffer_cleared && canvas_group_owner == nullptr && ci->canvas_group != nullptr && !backbuffer_copy) {
			texture_storage->render_target_clear_back_buffer(p_to_render_target, Rect2i(), Color(0, 0, 0, 0));
			backbuffer_cleared = true;
		}

		if (ci == canvas_group_owner) {
			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list, true);
			item_count = 0;

			if (ci->canvas_group->blur_mipmaps) {
				texture_storage->render_target_gen_back_buffer_mipmaps(p_to_render_target, ci->global_rect_cache);
			}

			canvas_group_owner = nullptr;
		}

		if (backbuffer_copy) {
			//render anything pending, including clearing if no items

			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
			item_count = 0;

			texture_storage->render_target_copy_to_back_buffer(p_to_render_target, back_buffer_rect, true);

			backbuffer_copy = false;
			material_screen_texture_found = true; //after a backbuffer copy, screen texture makes no further copies
		}

		// just add all items for now
		items[item_count++] = ci;

		if (!ci->next || item_count == MAX_RENDER_ITEMS - 1) {
			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
			//then reset
			item_count = 0;
		}

		ci = ci->next;
	}

	if (time_used) {
		RenderingServerDefault::redraw_request();
	}

	// Clear out state used in 2D pass
	reset_canvas();
}

void RasterizerCanvasGLES3::_render_items(RID p_to_render_target, int p_item_count, const Transform2D &p_canvas_transform_inverse, Light *p_lights, bool p_to_backbuffer) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	Item *current_clip = nullptr;

	Transform2D canvas_transform_inverse = p_canvas_transform_inverse;

	canvas_begin(p_to_render_target, p_to_backbuffer);

	RID prev_material;
	uint32_t index = 0;
	GLES3::CanvasShaderData::BlendMode last_blend_mode = GLES3::CanvasShaderData::BLEND_MODE_MIX;
	GLES3::CanvasShaderData *shader_data_cache = nullptr;

	state.current_tex = texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_WHITE);
	state.current_tex_ptr = nullptr;
	state.current_normal = RID();
	state.current_specular = RID();
	state.canvas_texscreen_used = false;
	state.current_shader_version = state.canvas_shader_default_version;

	for (int i = 0; i < p_item_count; i++) {
		Item *ci = items[i];

		if (current_clip != ci->final_clip_owner) {
			_render_batch(index);

			current_clip = ci->final_clip_owner;
			//setup clip
			if (current_clip) {
				glEnable(GL_SCISSOR_TEST);
				glScissor(current_clip->final_clip_rect.position.x, current_clip->final_clip_rect.position.y, current_clip->final_clip_rect.size.x, current_clip->final_clip_rect.size.y);
			} else {
				glDisable(GL_SCISSOR_TEST);
			}
		}

		RID material = ci->material_owner == nullptr ? ci->material : ci->material_owner->material;

		if (material.is_null() && ci->canvas_group != nullptr) {
			material = default_canvas_group_material;
		}

		if (material != prev_material) {
			_render_batch(index);
			GLES3::CanvasMaterialData *material_data = nullptr;
			if (material.is_valid()) {
				material_data = static_cast<GLES3::CanvasMaterialData *>(material_storage->material_get_data(material, RS::SHADER_CANVAS_ITEM));
			}
			if (material_data) {
				if (material_data->shader_data->version.is_valid() && material_data->shader_data->valid) {
					// Bind uniform buffer and textures
					material_data->bind_uniforms();
					state.current_shader_version = material_data->shader_data->version;
					shader_data_cache = material_data->shader_data;
				} else {
					state.current_shader_version = state.canvas_shader_default_version;
					shader_data_cache = nullptr;
				}
			} else {
				state.current_shader_version = state.canvas_shader_default_version;
				shader_data_cache = nullptr;
			}
			prev_material = material;
		}

		GLES3::CanvasShaderData::BlendMode blend_mode = shader_data_cache ? shader_data_cache->blend_mode : GLES3::CanvasShaderData::BLEND_MODE_MIX;

		if (last_blend_mode != blend_mode) {
			if (last_blend_mode == GLES3::CanvasShaderData::BLEND_MODE_DISABLED) {
				// re-enable it
				glEnable(GL_BLEND);
			} else if (blend_mode == GLES3::CanvasShaderData::BLEND_MODE_DISABLED) {
				// disable it
				glDisable(GL_BLEND);
			}

			switch (blend_mode) {
				case GLES3::CanvasShaderData::BLEND_MODE_DISABLED: {
					// Nothing to do here.

				} break;
				case GLES3::CanvasShaderData::BLEND_MODE_MIX: {
					glBlendEquation(GL_FUNC_ADD);
					if (state.transparent_render_target) {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
					} else {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
					}

				} break;
				case GLES3::CanvasShaderData::BLEND_MODE_ADD: {
					glBlendEquation(GL_FUNC_ADD);
					if (state.transparent_render_target) {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
					} else {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
					}

				} break;
				case GLES3::CanvasShaderData::BLEND_MODE_SUB: {
					glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
					if (state.transparent_render_target) {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE);
					} else {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
					}
				} break;
				case GLES3::CanvasShaderData::BLEND_MODE_MUL: {
					glBlendEquation(GL_FUNC_ADD);
					if (state.transparent_render_target) {
						glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
					} else {
						glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
					}

				} break;
				case GLES3::CanvasShaderData::BLEND_MODE_PMALPHA: {
					glBlendEquation(GL_FUNC_ADD);
					if (state.transparent_render_target) {
						glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
					} else {
						glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
					}

				} break;
			}
			last_blend_mode = blend_mode;
		}

		_render_item(p_to_render_target, ci, canvas_transform_inverse, current_clip, p_lights, index);
	}
	// Render last command
	_render_batch(index);
}

void RasterizerCanvasGLES3::_render_item(RID p_render_target, const Item *p_item, const Transform2D &p_canvas_transform_inverse, Item *&current_clip, Light *p_lights, uint32_t &r_index) {
	// Used by Polygon and Mesh.
	static const GLenum prim[5] = { GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_TRIANGLES, GL_TRIANGLE_STRIP };

	RS::CanvasItemTextureFilter current_filter = state.default_filter;
	RS::CanvasItemTextureRepeat current_repeat = state.default_repeat;

	if (p_item->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT) {
		current_filter = p_item->texture_filter;
	}

	if (p_item->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) {
		current_repeat = p_item->texture_repeat;
	}

	Transform2D base_transform = p_canvas_transform_inverse * p_item->final_transform;
	Transform2D draw_transform; // Used by transform command

	Color base_color = p_item->final_modulate;

	uint32_t base_flags = 0;

	bool reclip = false;

	bool skipping = false;

	const Item::Command *c = p_item->commands;
	while (c) {
		if (skipping && c->type != Item::Command::TYPE_ANIMATION_SLICE) {
			c = c->next;
			continue;
		}

		if (c->type != Item::Command::TYPE_MESH) {
			// For Meshes, this gets updated below.
			_update_transform_2d_to_mat2x3(base_transform * draw_transform, state.instance_data_array[r_index].world);
		}

		for (int i = 0; i < 4; i++) {
			state.instance_data_array[r_index].modulation[i] = 0.0;
			state.instance_data_array[r_index].ninepatch_margins[i] = 0.0;
			state.instance_data_array[r_index].src_rect[i] = 0.0;
			state.instance_data_array[r_index].dst_rect[i] = 0.0;
			state.instance_data_array[r_index].lights[i] = uint32_t(0);
		}
		state.instance_data_array[r_index].color_texture_pixel_size[0] = 0.0;
		state.instance_data_array[r_index].color_texture_pixel_size[1] = 0.0;

		state.instance_data_array[r_index].pad[0] = 0.0;
		state.instance_data_array[r_index].pad[1] = 0.0;

		state.instance_data_array[r_index].flags = base_flags | (state.instance_data_array[r_index == 0 ? 0 : r_index - 1].flags & (FLAGS_DEFAULT_NORMAL_MAP_USED | FLAGS_DEFAULT_SPECULAR_MAP_USED)); //reset on each command for sanity, keep canvastexture binding config

		switch (c->type) {
			case Item::Command::TYPE_RECT: {
				const Item::CommandRect *rect = static_cast<const Item::CommandRect *>(c);

				if (rect->flags & CANVAS_RECT_TILE) {
					current_repeat = RenderingServer::CanvasItemTextureRepeat::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED;
				}

				if (rect->texture != state.current_tex || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_RECT) {
					_render_batch(r_index);

					state.current_primitive_points = 0;
					state.current_command = Item::Command::TYPE_RECT;
				}
				_bind_canvas_texture(rect->texture, current_filter, current_repeat, r_index);
				GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_QUAD);

				Rect2 src_rect;
				Rect2 dst_rect;

				if (rect->texture != RID()) {
					src_rect = (rect->flags & CANVAS_RECT_REGION) ? Rect2(rect->source.position * state.current_pixel_size, rect->source.size * state.current_pixel_size) : Rect2(0, 0, 1, 1);
					dst_rect = Rect2(rect->rect.position, rect->rect.size);

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

					if (rect->flags & CANVAS_RECT_CLIP_UV) {
						state.instance_data_array[r_index].flags |= FLAGS_CLIP_RECT_UV;
					}

				} else {
					dst_rect = Rect2(rect->rect.position, rect->rect.size);

					if (dst_rect.size.width < 0) {
						dst_rect.position.x += dst_rect.size.width;
						dst_rect.size.width *= -1;
					}
					if (dst_rect.size.height < 0) {
						dst_rect.position.y += dst_rect.size.height;
						dst_rect.size.height *= -1;
					}

					src_rect = Rect2(0, 0, 1, 1);
				}

				if (rect->flags & CANVAS_RECT_MSDF) {
					state.instance_data_array[r_index].flags |= FLAGS_USE_MSDF;
					state.instance_data_array[r_index].msdf[0] = rect->px_range; // Pixel range.
					state.instance_data_array[r_index].msdf[1] = rect->outline; // Outline size.
					state.instance_data_array[r_index].msdf[2] = 0.f; // Reserved.
					state.instance_data_array[r_index].msdf[3] = 0.f; // Reserved.
				}

				state.instance_data_array[r_index].modulation[0] = rect->modulate.r * base_color.r;
				state.instance_data_array[r_index].modulation[1] = rect->modulate.g * base_color.g;
				state.instance_data_array[r_index].modulation[2] = rect->modulate.b * base_color.b;
				state.instance_data_array[r_index].modulation[3] = rect->modulate.a * base_color.a;

				state.instance_data_array[r_index].src_rect[0] = src_rect.position.x;
				state.instance_data_array[r_index].src_rect[1] = src_rect.position.y;
				state.instance_data_array[r_index].src_rect[2] = src_rect.size.width;
				state.instance_data_array[r_index].src_rect[3] = src_rect.size.height;

				state.instance_data_array[r_index].dst_rect[0] = dst_rect.position.x;
				state.instance_data_array[r_index].dst_rect[1] = dst_rect.position.y;
				state.instance_data_array[r_index].dst_rect[2] = dst_rect.size.width;
				state.instance_data_array[r_index].dst_rect[3] = dst_rect.size.height;

				r_index++;
				if (r_index >= state.max_instances_per_batch - 1) {
					_render_batch(r_index);
				}
			} break;

			case Item::Command::TYPE_NINEPATCH: {
				const Item::CommandNinePatch *np = static_cast<const Item::CommandNinePatch *>(c);

				if (np->texture != state.current_tex || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_NINEPATCH) {
					_render_batch(r_index);

					state.current_primitive_points = 0;
					state.current_command = Item::Command::TYPE_NINEPATCH;
				}

				//bind textures
				_bind_canvas_texture(np->texture, current_filter, current_repeat, r_index);
				GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_NINEPATCH);

				Rect2 src_rect;
				Rect2 dst_rect(np->rect.position.x, np->rect.position.y, np->rect.size.x, np->rect.size.y);

				if (np->texture == RID()) {
					state.current_pixel_size = Size2(1, 1);
					src_rect = Rect2(0, 0, 1, 1);

				} else {
					if (np->source != Rect2()) {
						src_rect = Rect2(np->source.position.x * state.current_pixel_size.width, np->source.position.y * state.current_pixel_size.height, np->source.size.x * state.current_pixel_size.width, np->source.size.y * state.current_pixel_size.height);
						state.instance_data_array[r_index].color_texture_pixel_size[0] = 1.0 / np->source.size.width;
						state.instance_data_array[r_index].color_texture_pixel_size[1] = 1.0 / np->source.size.height;

					} else {
						src_rect = Rect2(0, 0, 1, 1);
					}
				}

				state.instance_data_array[r_index].modulation[0] = np->color.r * base_color.r;
				state.instance_data_array[r_index].modulation[1] = np->color.g * base_color.g;
				state.instance_data_array[r_index].modulation[2] = np->color.b * base_color.b;
				state.instance_data_array[r_index].modulation[3] = np->color.a * base_color.a;

				state.instance_data_array[r_index].src_rect[0] = src_rect.position.x;
				state.instance_data_array[r_index].src_rect[1] = src_rect.position.y;
				state.instance_data_array[r_index].src_rect[2] = src_rect.size.width;
				state.instance_data_array[r_index].src_rect[3] = src_rect.size.height;

				state.instance_data_array[r_index].dst_rect[0] = dst_rect.position.x;
				state.instance_data_array[r_index].dst_rect[1] = dst_rect.position.y;
				state.instance_data_array[r_index].dst_rect[2] = dst_rect.size.width;
				state.instance_data_array[r_index].dst_rect[3] = dst_rect.size.height;

				state.instance_data_array[r_index].flags |= int(np->axis_x) << FLAGS_NINEPATCH_H_MODE_SHIFT;
				state.instance_data_array[r_index].flags |= int(np->axis_y) << FLAGS_NINEPATCH_V_MODE_SHIFT;

				if (np->draw_center) {
					state.instance_data_array[r_index].flags |= FLAGS_NINEPACH_DRAW_CENTER;
				}

				state.instance_data_array[r_index].ninepatch_margins[0] = np->margin[SIDE_LEFT];
				state.instance_data_array[r_index].ninepatch_margins[1] = np->margin[SIDE_TOP];
				state.instance_data_array[r_index].ninepatch_margins[2] = np->margin[SIDE_RIGHT];
				state.instance_data_array[r_index].ninepatch_margins[3] = np->margin[SIDE_BOTTOM];

				r_index++;
				if (r_index >= state.max_instances_per_batch - 1) {
					_render_batch(r_index);
				}

				// Restore if overridden.
				state.instance_data_array[r_index].color_texture_pixel_size[0] = state.current_pixel_size.x;
				state.instance_data_array[r_index].color_texture_pixel_size[1] = state.current_pixel_size.y;
			} break;

			case Item::Command::TYPE_POLYGON: {
				const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(c);

				PolygonBuffers *pb = polygon_buffers.polygons.getptr(polygon->polygon.polygon_id);
				ERR_CONTINUE(!pb);

				if (polygon->texture != state.current_tex || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_POLYGON) {
					_render_batch(r_index);

					state.current_primitive_points = 0;
					state.current_command = Item::Command::TYPE_POLYGON;
				}
				_bind_canvas_texture(polygon->texture, current_filter, current_repeat, r_index);
				GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_ATTRIBUTES);

				state.current_primitive = polygon->primitive;
				state.instance_data_array[r_index].modulation[0] = base_color.r;
				state.instance_data_array[r_index].modulation[1] = base_color.g;
				state.instance_data_array[r_index].modulation[2] = base_color.b;
				state.instance_data_array[r_index].modulation[3] = base_color.a;

				for (int j = 0; j < 4; j++) {
					state.instance_data_array[r_index].src_rect[j] = 0;
					state.instance_data_array[r_index].dst_rect[j] = 0;
					state.instance_data_array[r_index].ninepatch_margins[j] = 0;
				}

				_bind_instance_data_buffer(1);
				glBindVertexArray(pb->vertex_array);

				if (pb->color_disabled) {
					glVertexAttrib4f(RS::ARRAY_COLOR, pb->color.r, pb->color.g, pb->color.b, pb->color.a);
				}

				if (pb->index_buffer != 0) {
					glDrawElements(prim[polygon->primitive], pb->count, GL_UNSIGNED_INT, nullptr);
				} else {
					glDrawArrays(prim[polygon->primitive], 0, pb->count);
				}
				glBindVertexArray(0);
				state.fences[state.current_buffer] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

				state.current_buffer = (state.current_buffer + 1) % state.canvas_instance_data_buffers.size();

				if (pb->color_disabled) {
					// Reset so this doesn't pollute other draw calls.
					glVertexAttrib4f(RS::ARRAY_COLOR, 1.0, 1.0, 1.0, 1.0);
				}
			} break;

			case Item::Command::TYPE_PRIMITIVE: {
				const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(c);

				if (state.current_primitive_points != primitive->point_count || state.current_command != Item::Command::TYPE_PRIMITIVE) {
					_render_batch(r_index);
					state.current_primitive_points = primitive->point_count;
					state.current_command = Item::Command::TYPE_PRIMITIVE;
				}
				_bind_canvas_texture(RID(), current_filter, current_repeat, r_index);
				GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_PRIMITIVE);

				for (uint32_t j = 0; j < MIN(3u, primitive->point_count); j++) {
					state.instance_data_array[r_index].points[j * 2 + 0] = primitive->points[j].x;
					state.instance_data_array[r_index].points[j * 2 + 1] = primitive->points[j].y;
					state.instance_data_array[r_index].uvs[j * 2 + 0] = primitive->uvs[j].x;
					state.instance_data_array[r_index].uvs[j * 2 + 1] = primitive->uvs[j].y;
					Color col = primitive->colors[j] * base_color;
					state.instance_data_array[r_index].colors[j * 2 + 0] = (uint32_t(Math::make_half_float(col.g)) << 16) | Math::make_half_float(col.r);
					state.instance_data_array[r_index].colors[j * 2 + 1] = (uint32_t(Math::make_half_float(col.a)) << 16) | Math::make_half_float(col.b);
				}
				r_index++;
				if (primitive->point_count == 4) {
					// Reset base data
					_update_transform_2d_to_mat2x3(base_transform * draw_transform, state.instance_data_array[r_index].world);
					state.instance_data_array[r_index].color_texture_pixel_size[0] = 0.0;
					state.instance_data_array[r_index].color_texture_pixel_size[1] = 0.0;

					state.instance_data_array[r_index].flags = base_flags | (state.instance_data_array[r_index == 0 ? 0 : r_index - 1].flags & (FLAGS_DEFAULT_NORMAL_MAP_USED | FLAGS_DEFAULT_SPECULAR_MAP_USED)); //reset on each command for sanity, keep canvastexture binding config

					for (uint32_t j = 0; j < 3; j++) {
						//second half of triangle
						state.instance_data_array[r_index].points[j * 2 + 0] = primitive->points[j + 1].x;
						state.instance_data_array[r_index].points[j * 2 + 1] = primitive->points[j + 1].y;
						state.instance_data_array[r_index].uvs[j * 2 + 0] = primitive->uvs[j + 1].x;
						state.instance_data_array[r_index].uvs[j * 2 + 1] = primitive->uvs[j + 1].y;
						Color col = primitive->colors[j + 1] * base_color;
						state.instance_data_array[r_index].colors[j * 2 + 0] = (uint32_t(Math::make_half_float(col.g)) << 16) | Math::make_half_float(col.r);
						state.instance_data_array[r_index].colors[j * 2 + 1] = (uint32_t(Math::make_half_float(col.a)) << 16) | Math::make_half_float(col.b);
					}
					r_index++;
				}
				if (r_index >= state.max_instances_per_batch - 1) {
					_render_batch(r_index);
				}
			} break;

			case Item::Command::TYPE_MESH:
			case Item::Command::TYPE_MULTIMESH:
			case Item::Command::TYPE_PARTICLES: {
				GLES3::MeshStorage *mesh_storage = GLES3::MeshStorage::get_singleton();
				RID mesh;
				RID mesh_instance;
				RID texture;
				Color modulate(1, 1, 1, 1);
				uint32_t instance_count = 1;
				GLuint multimesh_buffer = 0;
				uint32_t multimesh_stride = 0;
				uint32_t multimesh_color_offset = 0;
				bool multimesh_uses_color = false;
				bool multimesh_uses_custom_data = false;

				if (c->type == Item::Command::TYPE_MESH) {
					const Item::CommandMesh *m = static_cast<const Item::CommandMesh *>(c);
					mesh = m->mesh;
					mesh_instance = m->mesh_instance;
					texture = m->texture;
					modulate = m->modulate;
					_update_transform_2d_to_mat2x3(base_transform * draw_transform * m->transform, state.instance_data_array[r_index].world);
				} else if (c->type == Item::Command::TYPE_MULTIMESH) {
					const Item::CommandMultiMesh *mm = static_cast<const Item::CommandMultiMesh *>(c);
					RID multimesh = mm->multimesh;
					mesh = mesh_storage->multimesh_get_mesh(multimesh);
					texture = mm->texture;

					if (mesh_storage->multimesh_get_transform_format(multimesh) != RS::MULTIMESH_TRANSFORM_2D) {
						break;
					}

					instance_count = mesh_storage->multimesh_get_instances_to_draw(multimesh);

					if (instance_count == 0) {
						break;
					}

					multimesh_buffer = mesh_storage->multimesh_get_gl_buffer(multimesh);
					multimesh_stride = mesh_storage->multimesh_get_stride(multimesh);
					multimesh_color_offset = mesh_storage->multimesh_get_color_offset(multimesh);
					multimesh_uses_color = mesh_storage->multimesh_uses_colors(multimesh);
					multimesh_uses_custom_data = mesh_storage->multimesh_uses_custom_data(multimesh);
				}

				// TODO: implement particles here

				if (mesh.is_null()) {
					break;
				}

				if (texture != state.current_tex || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_PRIMITIVE) {
					_render_batch(r_index);
					state.current_primitive_points = 0;
					state.current_command = c->type;
				}

				_bind_canvas_texture(texture, current_filter, current_repeat, r_index);
				if (instance_count == 1) {
					GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_ATTRIBUTES);
				} else {
					GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_INSTANCED);
				}

				uint32_t surf_count = mesh_storage->mesh_get_surface_count(mesh);

				state.instance_data_array[r_index].modulation[0] = base_color.r * modulate.r;
				state.instance_data_array[r_index].modulation[1] = base_color.g * modulate.g;
				state.instance_data_array[r_index].modulation[2] = base_color.b * modulate.b;
				state.instance_data_array[r_index].modulation[3] = base_color.a * modulate.a;

				for (int j = 0; j < 4; j++) {
					state.instance_data_array[r_index].src_rect[j] = 0;
					state.instance_data_array[r_index].dst_rect[j] = 0;
					state.instance_data_array[r_index].ninepatch_margins[j] = 0;
				}
				_bind_instance_data_buffer(1);
				for (uint32_t j = 0; j < surf_count; j++) {
					void *surface = mesh_storage->mesh_get_surface(mesh, j);

					RS::PrimitiveType primitive = mesh_storage->mesh_surface_get_primitive(surface);
					ERR_CONTINUE(primitive < 0 || primitive >= RS::PRIMITIVE_MAX);

					GLuint vertex_array_gl = 0;
					GLuint index_array_gl = 0;

					uint32_t input_mask = 0; // 2D meshes always use the same vertex format
					if (mesh_instance.is_valid()) {
						mesh_storage->mesh_instance_surface_get_vertex_arrays_and_format(mesh_instance, j, input_mask, vertex_array_gl);
					} else {
						mesh_storage->mesh_surface_get_vertex_arrays_and_format(surface, input_mask, vertex_array_gl);
					}

					index_array_gl = mesh_storage->mesh_surface_get_index_buffer(surface, 0);
					bool use_index_buffer = false;
					glBindVertexArray(vertex_array_gl);
					if (index_array_gl != 0) {
						glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_array_gl);
						use_index_buffer = true;
					}

					if (instance_count > 1) {
						// Bind instance buffers.
						glBindBuffer(GL_ARRAY_BUFFER, multimesh_buffer);
						glEnableVertexAttribArray(1);
						glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, multimesh_stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(0));
						glVertexAttribDivisor(1, 1);
						glEnableVertexAttribArray(2);
						glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, multimesh_stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(4 * 4));
						glVertexAttribDivisor(2, 1);

						if (multimesh_uses_color || multimesh_uses_custom_data) {
							glEnableVertexAttribArray(5);
							glVertexAttribIPointer(5, 4, GL_UNSIGNED_INT, multimesh_stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(multimesh_color_offset * sizeof(float)));
							glVertexAttribDivisor(5, 1);
						}
					}

					GLenum primitive_gl = prim[int(primitive)];
					if (instance_count == 1) {
						if (use_index_buffer) {
							glDrawElements(primitive_gl, mesh_storage->mesh_surface_get_vertices_drawn_count(surface), mesh_storage->mesh_surface_get_index_type(surface), 0);
						} else {
							glDrawArrays(primitive_gl, 0, mesh_storage->mesh_surface_get_vertices_drawn_count(surface));
						}
					} else {
						if (use_index_buffer) {
							glDrawElementsInstanced(primitive_gl, mesh_storage->mesh_surface_get_vertices_drawn_count(surface), mesh_storage->mesh_surface_get_index_type(surface), 0, instance_count);
						} else {
							glDrawArraysInstanced(primitive_gl, 0, mesh_storage->mesh_surface_get_vertices_drawn_count(surface), instance_count);
						}
					}

					state.fences[state.current_buffer] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

					state.current_buffer = (state.current_buffer + 1) % state.canvas_instance_data_buffers.size();
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
					if (instance_count > 1) {
						glDisableVertexAttribArray(5);
						glDisableVertexAttribArray(6);
						glDisableVertexAttribArray(7);
						glDisableVertexAttribArray(8);
					}
				}
			} break;
			case Item::Command::TYPE_TRANSFORM: {
				const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
				draw_transform = transform->xform;
			} break;

			case Item::Command::TYPE_CLIP_IGNORE: {
				const Item::CommandClipIgnore *ci = static_cast<const Item::CommandClipIgnore *>(c);
				if (current_clip) {
					if (ci->ignore != reclip) {
						if (ci->ignore) {
							glDisable(GL_SCISSOR_TEST);
							reclip = true;
						} else {
							// Scissor area is already set
							glEnable(GL_SCISSOR_TEST);
							reclip = false;
						}
					}
				}
			} break;
			case Item::Command::TYPE_ANIMATION_SLICE: {
				/*
				const Item::CommandAnimationSlice *as = static_cast<const Item::CommandAnimationSlice *>(c);
				double current_time = RendererCompositorRD::singleton->get_total_time();
				double local_time = Math::fposmod(current_time - as->offset, as->animation_length);
				skipping = !(local_time >= as->slice_begin && local_time < as->slice_end);

				RenderingServerDefault::redraw_request(); // animation visible means redraw request
				*/
			} break;
		}

		c = c->next;
	}

	if (current_clip && reclip) {
		//will make it re-enable clipping if needed afterwards
		current_clip = nullptr;
	}
}

void RasterizerCanvasGLES3::_render_batch(uint32_t &r_index) {
	if (r_index > 0) {
		_bind_instance_data_buffer(r_index);
		glBindVertexArray(data.canvas_quad_array);
		if (state.current_primitive_points == 0) {
			glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, r_index);
		} else {
			static const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLES };
			glDrawArraysInstanced(prim[state.current_primitive_points], 0, state.current_primitive_points, r_index);
		}
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		state.fences[state.current_buffer] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		state.current_buffer = (state.current_buffer + 1) % state.canvas_instance_data_buffers.size();
		//copy the new data into the base of the batch
		for (int i = 0; i < 4; i++) {
			state.instance_data_array[0].modulation[i] = state.instance_data_array[r_index].modulation[i];
			state.instance_data_array[0].ninepatch_margins[i] = state.instance_data_array[r_index].ninepatch_margins[i];
			state.instance_data_array[0].src_rect[i] = state.instance_data_array[r_index].src_rect[i];
			state.instance_data_array[0].dst_rect[i] = state.instance_data_array[r_index].dst_rect[i];
			state.instance_data_array[0].lights[i] = state.instance_data_array[r_index].lights[i];
		}
		state.instance_data_array[0].flags = state.instance_data_array[r_index].flags;
		state.instance_data_array[0].color_texture_pixel_size[0] = state.instance_data_array[r_index].color_texture_pixel_size[0];
		state.instance_data_array[0].color_texture_pixel_size[1] = state.instance_data_array[r_index].color_texture_pixel_size[1];

		state.instance_data_array[0].pad[0] = state.instance_data_array[r_index].pad[0];
		state.instance_data_array[0].pad[1] = state.instance_data_array[r_index].pad[1];
		for (int i = 0; i < 6; i++) {
			state.instance_data_array[0].world[i] = state.instance_data_array[r_index].world[i];
		}

		r_index = 0;
	}
}

void RasterizerCanvasGLES3::_bind_instance_data_buffer(uint32_t p_max_index) {
	if (p_max_index == 0) {
		return;
	}
	// If the previous operation is not done yet, allocate a new buffer
	if (state.fences[state.current_buffer] != GLsync()) {
		GLint syncStatus;
		glGetSynciv(state.fences[state.current_buffer], GL_SYNC_STATUS, sizeof(GLint), nullptr, &syncStatus);
		if (syncStatus == GL_UNSIGNALED) {
			_allocate_instance_data_buffer();
		} else {
			glDeleteSync(state.fences[state.current_buffer]);
		}
	}

	glBindBufferBase(GL_UNIFORM_BUFFER, INSTANCE_UNIFORM_LOCATION, state.canvas_instance_data_buffers[state.current_buffer]);
#ifdef JAVASCRIPT_ENABLED
	//WebGL 2.0 does not support mapping buffers, so use slow glBufferData instead
	glBufferData(GL_UNIFORM_BUFFER, sizeof(InstanceData) * p_max_index, state.instance_data_array, GL_DYNAMIC_DRAW);
#else
	void *ubo = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(InstanceData) * p_max_index, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
	memcpy(ubo, state.instance_data_array, sizeof(InstanceData) * p_max_index);
	glUnmapBuffer(GL_UNIFORM_BUFFER);
#endif
}

RID RasterizerCanvasGLES3::light_create() {
	return RID();
}

void RasterizerCanvasGLES3::light_set_texture(RID p_rid, RID p_texture) {
}

void RasterizerCanvasGLES3::light_set_use_shadow(RID p_rid, bool p_enable) {
}

void RasterizerCanvasGLES3::light_update_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) {
}

void RasterizerCanvasGLES3::light_update_directional_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_cull_distance, const Rect2 &p_clip_rect, LightOccluderInstance *p_occluders) {
}

void RasterizerCanvasGLES3::render_sdf(RID p_render_target, LightOccluderInstance *p_occluders) {
}

RID RasterizerCanvasGLES3::occluder_polygon_create() {
	return RID();
}

void RasterizerCanvasGLES3::occluder_polygon_set_shape(RID p_occluder, const Vector<Vector2> &p_points, bool p_closed) {
}

void RasterizerCanvasGLES3::occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) {
}

void RasterizerCanvasGLES3::set_shadow_texture_size(int p_size) {
}

bool RasterizerCanvasGLES3::free(RID p_rid) {
	return true;
}

void RasterizerCanvasGLES3::update() {
}

void RasterizerCanvasGLES3::canvas_begin(RID p_to_render_target, bool p_to_backbuffer) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

	GLES3::RenderTarget *render_target = texture_storage->get_render_target(p_to_render_target);

	if (p_to_backbuffer) {
		glBindFramebuffer(GL_FRAMEBUFFER, render_target->backbuffer_fbo);
		glActiveTexture(GL_TEXTURE0 + config->max_texture_image_units - 4);
		GLES3::Texture *tex = texture_storage->get_texture(texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_WHITE));
		glBindTexture(GL_TEXTURE_2D, tex->tex_id);
	} else {
		glBindFramebuffer(GL_FRAMEBUFFER, render_target->fbo);
		glActiveTexture(GL_TEXTURE0 + config->max_texture_image_units - 4);
		glBindTexture(GL_TEXTURE_2D, render_target->backbuffer);
	}

	if (render_target->is_transparent) {
		state.transparent_render_target = true;
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		state.transparent_render_target = false;
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	if (render_target && render_target->clear_requested) {
		const Color &col = render_target->clear_color;
		glClearColor(col.r, col.g, col.b, col.a);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		render_target->clear_requested = false;
	}

	glActiveTexture(GL_TEXTURE0);
	GLES3::Texture *tex = texture_storage->get_texture(texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_WHITE));
	glBindTexture(GL_TEXTURE_2D, tex->tex_id);
}

void RasterizerCanvasGLES3::_bind_canvas_texture(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, uint32_t &r_index) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

	if (p_texture == RID()) {
		p_texture = texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_WHITE);
	}

	if (state.current_tex == p_texture) {
		return; //nothing to do, its the same
	}
	state.current_tex = p_texture;

	GLES3::CanvasTexture *ct = nullptr;

	GLES3::Texture *t = texture_storage->get_texture(p_texture);

	if (t) {
		//regular texture
		if (!t->canvas_texture) {
			t->canvas_texture = memnew(GLES3::CanvasTexture);
			t->canvas_texture->diffuse = p_texture;
		}

		ct = t->canvas_texture;
	} else {
		ct = texture_storage->get_canvas_texture(p_texture);
	}

	if (!ct) {
		// Invalid Texture RID.
		_bind_canvas_texture(default_canvas_texture, p_base_filter, p_base_repeat, r_index);
		return;
	}

	RS::CanvasItemTextureFilter filter = ct->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT ? ct->texture_filter : p_base_filter;
	ERR_FAIL_COND(filter == RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT);

	RS::CanvasItemTextureRepeat repeat = ct->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT ? ct->texture_repeat : p_base_repeat;
	ERR_FAIL_COND(repeat == RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT);

	GLES3::Texture *texture = texture_storage->get_texture(ct->diffuse);

	if (!texture) {
		state.current_tex = texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_WHITE);
		GLES3::Texture *tex = texture_storage->get_texture(state.current_tex);
		state.current_tex_ptr = tex;
		ct->size_cache = Size2i(tex->width, tex->height);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex->tex_id);
	} else {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture->tex_id);

		state.current_tex = p_texture;
		state.current_tex_ptr = texture;
		ct->size_cache = Size2i(texture->width, texture->height);

		texture->gl_set_filter(filter);
		texture->gl_set_repeat(repeat);
	}

	GLES3::Texture *normal_map = texture_storage->get_texture(ct->normal_map);

	if (!normal_map) {
		state.current_normal = RID();
		ct->use_normal_cache = false;
		glActiveTexture(GL_TEXTURE0 + GLES3::Config::get_singleton()->max_texture_image_units - 6);
		GLES3::Texture *tex = texture_storage->get_texture(texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_NORMAL));
		glBindTexture(GL_TEXTURE_2D, tex->tex_id);

	} else {
		glActiveTexture(GL_TEXTURE0 + config->max_texture_image_units - 6);
		glBindTexture(GL_TEXTURE_2D, normal_map->tex_id);
		state.current_normal = ct->normal_map;
		ct->use_normal_cache = true;
		texture->gl_set_filter(filter);
		texture->gl_set_repeat(repeat);
	}

	GLES3::Texture *specular_map = texture_storage->get_texture(ct->specular);

	if (!specular_map) {
		state.current_specular = RID();
		ct->use_specular_cache = false;
		glActiveTexture(GL_TEXTURE0 + config->max_texture_image_units - 7);
		GLES3::Texture *tex = texture_storage->get_texture(texture_storage->texture_gl_get_default(GLES3::DEFAULT_GL_TEXTURE_WHITE));
		glBindTexture(GL_TEXTURE_2D, tex->tex_id);
	} else {
		glActiveTexture(GL_TEXTURE0 + config->max_texture_image_units - 7);
		glBindTexture(GL_TEXTURE_2D, specular_map->tex_id);
		state.current_specular = ct->specular;
		ct->use_specular_cache = true;
		texture->gl_set_filter(filter);
		texture->gl_set_repeat(repeat);
	}

	if (ct->use_specular_cache) {
		state.instance_data_array[r_index].flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
	} else {
		state.instance_data_array[r_index].flags &= ~FLAGS_DEFAULT_SPECULAR_MAP_USED;
	}

	if (ct->use_normal_cache) {
		state.instance_data_array[r_index].flags |= FLAGS_DEFAULT_NORMAL_MAP_USED;
	} else {
		state.instance_data_array[r_index].flags &= ~FLAGS_DEFAULT_NORMAL_MAP_USED;
	}

	state.instance_data_array[r_index].specular_shininess = uint32_t(CLAMP(ct->specular_color.a * 255.0, 0, 255)) << 24;
	state.instance_data_array[r_index].specular_shininess |= uint32_t(CLAMP(ct->specular_color.b * 255.0, 0, 255)) << 16;
	state.instance_data_array[r_index].specular_shininess |= uint32_t(CLAMP(ct->specular_color.g * 255.0, 0, 255)) << 8;
	state.instance_data_array[r_index].specular_shininess |= uint32_t(CLAMP(ct->specular_color.r * 255.0, 0, 255));

	state.current_pixel_size.x = 1.0 / float(ct->size_cache.x);
	state.current_pixel_size.y = 1.0 / float(ct->size_cache.y);

	state.instance_data_array[r_index].color_texture_pixel_size[0] = state.current_pixel_size.x;
	state.instance_data_array[r_index].color_texture_pixel_size[1] = state.current_pixel_size.y;
}

void RasterizerCanvasGLES3::reset_canvas() {
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
	glEnable(GL_BLEND);
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES3::canvas_debug_viewport_shadows(Light *p_lights_with_shadow) {
}

void RasterizerCanvasGLES3::canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, Projection *p_xform_cache) {
}

void RasterizerCanvasGLES3::draw_lens_distortion_rect(const Rect2 &p_rect, float p_k1, float p_k2, const Vector2 &p_eye_center, float p_oversample) {
}

RendererCanvasRender::PolygonID RasterizerCanvasGLES3::request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, const Vector<int> &p_bones, const Vector<float> &p_weights) {
	// We interleave the vertex data into one big VBO to improve cache coherence
	uint32_t vertex_count = p_points.size();
	uint32_t stride = 2;
	if ((uint32_t)p_colors.size() == vertex_count) {
		stride += 4;
	}
	if ((uint32_t)p_uvs.size() == vertex_count) {
		stride += 2;
	}
	if ((uint32_t)p_bones.size() == vertex_count * 4 && (uint32_t)p_weights.size() == vertex_count * 4) {
		stride += 4;
	}

	PolygonBuffers pb;
	glGenBuffers(1, &pb.vertex_buffer);
	glGenVertexArrays(1, &pb.vertex_array);
	glBindVertexArray(pb.vertex_array);
	pb.count = vertex_count;
	pb.index_buffer = 0;

	uint32_t buffer_size = stride * p_points.size();

	Vector<uint8_t> polygon_buffer;
	polygon_buffer.resize(buffer_size * sizeof(float));
	{
		glBindBuffer(GL_ARRAY_BUFFER, pb.vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, stride * vertex_count * sizeof(float), nullptr, GL_STATIC_DRAW); // TODO may not be necessary
		uint8_t *r = polygon_buffer.ptrw();
		float *fptr = reinterpret_cast<float *>(r);
		uint32_t *uptr = (uint32_t *)r;
		uint32_t base_offset = 0;
		{
			// Always uses vertex positions
			glEnableVertexAttribArray(RS::ARRAY_VERTEX);
			glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, stride * sizeof(float), nullptr);
			const Vector2 *points_ptr = p_points.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = points_ptr[i].x;
				fptr[base_offset + i * stride + 1] = points_ptr[i].y;
			}

			base_offset += 2;
		}

		// Next add colors
		if ((uint32_t)p_colors.size() == vertex_count) {
			glEnableVertexAttribArray(RS::ARRAY_COLOR);
			glVertexAttribPointer(RS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(base_offset * sizeof(float)));

			const Color *color_ptr = p_colors.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = color_ptr[i].r;
				fptr[base_offset + i * stride + 1] = color_ptr[i].g;
				fptr[base_offset + i * stride + 2] = color_ptr[i].b;
				fptr[base_offset + i * stride + 3] = color_ptr[i].a;
			}
			base_offset += 4;
		} else {
			glDisableVertexAttribArray(RS::ARRAY_COLOR);
			pb.color_disabled = true;
			pb.color = p_colors.size() == 1 ? p_colors[0] : Color(1.0, 1.0, 1.0, 1.0);
		}

		if ((uint32_t)p_uvs.size() == vertex_count) {
			glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
			glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(base_offset * sizeof(float)));

			const Vector2 *uv_ptr = p_uvs.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = uv_ptr[i].x;
				fptr[base_offset + i * stride + 1] = uv_ptr[i].y;
			}

			base_offset += 2;
		} else {
			glDisableVertexAttribArray(RS::ARRAY_TEX_UV);
		}

		if ((uint32_t)p_indices.size() == vertex_count * 4 && (uint32_t)p_weights.size() == vertex_count * 4) {
			glEnableVertexAttribArray(RS::ARRAY_BONES);
			glVertexAttribPointer(RS::ARRAY_BONES, 4, GL_UNSIGNED_INT, GL_FALSE, stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(base_offset * sizeof(float)));

			const int *bone_ptr = p_bones.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				uint16_t *bone16w = (uint16_t *)&uptr[base_offset + i * stride];

				bone16w[0] = bone_ptr[i * 4 + 0];
				bone16w[1] = bone_ptr[i * 4 + 1];
				bone16w[2] = bone_ptr[i * 4 + 2];
				bone16w[3] = bone_ptr[i * 4 + 3];
			}

			base_offset += 2;
		} else {
			glDisableVertexAttribArray(RS::ARRAY_BONES);
		}

		if ((uint32_t)p_weights.size() == vertex_count * 4) {
			glEnableVertexAttribArray(RS::ARRAY_WEIGHTS);
			glVertexAttribPointer(RS::ARRAY_WEIGHTS, 4, GL_FLOAT, GL_FALSE, stride * sizeof(float), CAST_INT_TO_UCHAR_PTR(base_offset * sizeof(float)));

			const float *weight_ptr = p_weights.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				uint16_t *weight16w = (uint16_t *)&uptr[base_offset + i * stride];

				weight16w[0] = CLAMP(weight_ptr[i * 4 + 0] * 65535, 0, 65535);
				weight16w[1] = CLAMP(weight_ptr[i * 4 + 1] * 65535, 0, 65535);
				weight16w[2] = CLAMP(weight_ptr[i * 4 + 2] * 65535, 0, 65535);
				weight16w[3] = CLAMP(weight_ptr[i * 4 + 3] * 65535, 0, 65535);
			}

			base_offset += 2;
		} else {
			glDisableVertexAttribArray(RS::ARRAY_WEIGHTS);
		}

		ERR_FAIL_COND_V(base_offset != stride, 0);
		glBufferData(GL_ARRAY_BUFFER, vertex_count * stride * sizeof(float), polygon_buffer.ptr(), GL_STATIC_DRAW);
	}

	if (p_indices.size()) {
		//create indices, as indices were requested
		Vector<uint8_t> index_buffer;
		index_buffer.resize(p_indices.size() * sizeof(int32_t));
		{
			uint8_t *w = index_buffer.ptrw();
			memcpy(w, p_indices.ptr(), sizeof(int32_t) * p_indices.size());
		}
		glGenBuffers(1, &pb.index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pb.index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, p_indices.size() * 4, nullptr, GL_STATIC_DRAW); // TODO may not be necessary
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, p_indices.size() * 4, index_buffer.ptr(), GL_STATIC_DRAW);
		pb.count = p_indices.size();
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	PolygonID id = polygon_buffers.last_id++;

	polygon_buffers.polygons[id] = pb;

	return id;
}
void RasterizerCanvasGLES3::free_polygon(PolygonID p_polygon) {
	PolygonBuffers *pb_ptr = polygon_buffers.polygons.getptr(p_polygon);
	ERR_FAIL_COND(!pb_ptr);

	PolygonBuffers &pb = *pb_ptr;

	if (pb.index_buffer != 0) {
		glDeleteBuffers(1, &pb.index_buffer);
	}

	glDeleteVertexArrays(1, &pb.vertex_array);
	glDeleteBuffers(1, &pb.vertex_buffer);

	polygon_buffers.polygons.erase(p_polygon);
}

// Creates a new uniform buffer and uses it right away
// This expands the instance buffer continually
// In theory allocations can reach as high as number_of_draw_calls * 3 frames
// because OpenGL can start rendering subsequent frames before finishing the current one
void RasterizerCanvasGLES3::_allocate_instance_data_buffer() {
	GLuint new_buffer;
	glGenBuffers(1, &new_buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, new_buffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(InstanceData) * state.max_instances_per_batch, nullptr, GL_DYNAMIC_DRAW);
	state.current_buffer = (state.current_buffer + 1);
	state.canvas_instance_data_buffers.insert(state.current_buffer, new_buffer);
	state.fences.insert(state.current_buffer, GLsync());
	state.current_buffer = state.current_buffer % state.canvas_instance_data_buffers.size();
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void RasterizerCanvasGLES3::set_time(double p_time) {
	state.time = p_time;
}

RasterizerCanvasGLES3 *RasterizerCanvasGLES3::singleton = nullptr;

RasterizerCanvasGLES3 *RasterizerCanvasGLES3::get_singleton() {
	return singleton;
}

RasterizerCanvasGLES3::RasterizerCanvasGLES3() {
	singleton = this;
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

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

	// ninepatch buffers
	{
		// array buffer
		glGenBuffers(1, &data.ninepatch_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, data.ninepatch_vertices);

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (16 + 16) * 2, nullptr, GL_DYNAMIC_DRAW);

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

	int uniform_max_size = config->max_uniform_buffer_size;
	if (uniform_max_size < 65536) {
		state.max_lights_per_render = 64;
		state.max_instances_per_batch = 128;
	} else {
		state.max_lights_per_render = 256;
		state.max_instances_per_batch = 512;
	}

	// Reserve 64 Uniform Buffers for instance data
	state.canvas_instance_data_buffers.resize(64);
	state.fences.resize(64);
	glGenBuffers(64, state.canvas_instance_data_buffers.ptr());
	for (int i = 0; i < 64; i++) {
		state.fences[i] = GLsync();
		glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_instance_data_buffers[i]);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(InstanceData) * state.max_instances_per_batch, nullptr, GL_DYNAMIC_DRAW);
	}
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	state.instance_data_array = memnew_arr(InstanceData, state.max_instances_per_batch);

	glGenBuffers(1, &state.canvas_state_buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, state.canvas_state_buffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(StateBuffer), nullptr, GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	String global_defines;
	global_defines += "#define MAX_GLOBAL_SHADER_UNIFORMS 256\n"; // TODO: this is arbitrary for now
	global_defines += "#define MAX_LIGHTS " + itos(state.max_instances_per_batch) + "\n";
	global_defines += "#define MAX_DRAW_DATA_INSTANCES " + itos(state.max_instances_per_batch) + "\n";

	GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.initialize(global_defines);
	state.canvas_shader_default_version = GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_create();
	GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_bind_shader(state.canvas_shader_default_version, CanvasShaderGLES3::MODE_QUAD);

	{
		default_canvas_group_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(default_canvas_group_shader);

		material_storage->shader_set_code(default_canvas_group_shader, R"(
// Default CanvasGroup shader.

shader_type canvas_item;

void fragment() {
	vec4 c = textureLod(SCREEN_TEXTURE, SCREEN_UV, 0.0);

	if (c.a > 0.0001) {
		c.rgb /= c.a;
	}

	COLOR *= c;
}
)");
		default_canvas_group_material = material_storage->material_allocate();
		material_storage->material_initialize(default_canvas_group_material);

		material_storage->material_set_shader(default_canvas_group_material, default_canvas_group_shader);
	}

	state.current_shader_version = state.canvas_shader_default_version;
	state.time = 0.0;
}

RasterizerCanvasGLES3::~RasterizerCanvasGLES3() {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();

	memdelete_arr(state.instance_data_array);

	GLES3::MaterialStorage::get_singleton()->shaders.canvas_shader.version_free(state.canvas_shader_default_version);
	material_storage->material_free(default_canvas_group_material);
	material_storage->shader_free(default_canvas_group_shader);
	singleton = nullptr;

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);
}

#endif // GLES3_ENABLED

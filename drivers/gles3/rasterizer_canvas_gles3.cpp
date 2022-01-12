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
#include "rasterizer_storage_gles3.h"

#include "core/config/project_settings.h"
#include "servers/rendering/rendering_server_default.h"

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
	p_mat4[0] = p_transform.elements[0][0];
	p_mat4[1] = p_transform.elements[0][1];
	p_mat4[2] = 0;
	p_mat4[3] = 0;
	p_mat4[4] = p_transform.elements[1][0];
	p_mat4[5] = p_transform.elements[1][1];
	p_mat4[6] = 0;
	p_mat4[7] = 0;
	p_mat4[8] = 0;
	p_mat4[9] = 0;
	p_mat4[10] = 1;
	p_mat4[11] = 0;
	p_mat4[12] = p_transform.elements[2][0];
	p_mat4[13] = p_transform.elements[2][1];
	p_mat4[14] = 0;
	p_mat4[15] = 1;
}

void RasterizerCanvasGLES3::_update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4) {
	p_mat2x4[0] = p_transform.elements[0][0];
	p_mat2x4[1] = p_transform.elements[1][0];
	p_mat2x4[2] = 0;
	p_mat2x4[3] = p_transform.elements[2][0];

	p_mat2x4[4] = p_transform.elements[0][1];
	p_mat2x4[5] = p_transform.elements[1][1];
	p_mat2x4[6] = 0;
	p_mat2x4[7] = p_transform.elements[2][1];
}

void RasterizerCanvasGLES3::_update_transform_2d_to_mat2x3(const Transform2D &p_transform, float *p_mat2x3) {
	p_mat2x3[0] = p_transform.elements[0][0];
	p_mat2x3[1] = p_transform.elements[0][1];
	p_mat2x3[2] = p_transform.elements[1][0];
	p_mat2x3[3] = p_transform.elements[1][1];
	p_mat2x3[4] = p_transform.elements[2][0];
	p_mat2x3[5] = p_transform.elements[2][1];
}

void RasterizerCanvasGLES3::_update_transform_to_mat4(const Transform3D &p_transform, float *p_mat4) {
	p_mat4[0] = p_transform.basis.elements[0][0];
	p_mat4[1] = p_transform.basis.elements[1][0];
	p_mat4[2] = p_transform.basis.elements[2][0];
	p_mat4[3] = 0;
	p_mat4[4] = p_transform.basis.elements[0][1];
	p_mat4[5] = p_transform.basis.elements[1][1];
	p_mat4[6] = p_transform.basis.elements[2][1];
	p_mat4[7] = 0;
	p_mat4[8] = p_transform.basis.elements[0][2];
	p_mat4[9] = p_transform.basis.elements[1][2];
	p_mat4[10] = p_transform.basis.elements[2][2];
	p_mat4[11] = 0;
	p_mat4[12] = p_transform.origin.x;
	p_mat4[13] = p_transform.origin.y;
	p_mat4[14] = p_transform.origin.z;
	p_mat4[15] = 1;
}

void RasterizerCanvasGLES3::canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) {
	storage->frame.current_rt = nullptr;

	storage->_set_current_render_target(p_to_render_target);

	Transform2D canvas_transform_inverse = p_canvas_transform.affine_inverse();

	// TODO: Setup Directional Lights

	// TODO: Setup lights

	{
		//update canvas state uniform buffer
		StateBuffer state_buffer;

		Size2i ssize = storage->render_target_get_size(p_to_render_target);

		Transform3D screen_transform;
		screen_transform.translate(-(ssize.width / 2.0f), -(ssize.height / 2.0f), 0.0f);
		screen_transform.scale(Vector3(2.0f / ssize.width, 2.0f / ssize.height, 1.0f));
		_update_transform_to_mat4(screen_transform, state_buffer.screen_transform);
		_update_transform_2d_to_mat4(p_canvas_transform, state_buffer.canvas_transform);

		Transform2D normal_transform = p_canvas_transform;
		normal_transform.elements[0].normalize();
		normal_transform.elements[1].normalize();
		normal_transform.elements[2] = Vector2();
		_update_transform_2d_to_mat4(normal_transform, state_buffer.canvas_normal_transform);

		state_buffer.canvas_modulate[0] = p_modulate.r;
		state_buffer.canvas_modulate[1] = p_modulate.g;
		state_buffer.canvas_modulate[2] = p_modulate.b;
		state_buffer.canvas_modulate[3] = p_modulate.a;

		Size2 render_target_size = storage->render_target_get_size(p_to_render_target);
		state_buffer.screen_pixel_size[0] = 1.0 / render_target_size.x;
		state_buffer.screen_pixel_size[1] = 1.0 / render_target_size.y;

		state_buffer.time = storage->frame.time;
		state_buffer.use_pixel_snap = p_snap_2d_vertices_to_pixel;

		state_buffer.directional_light_count = 0; //directional_light_count;

		Vector2 canvas_scale = p_canvas_transform.get_scale();

		state_buffer.sdf_to_screen[0] = render_target_size.width / canvas_scale.x;
		state_buffer.sdf_to_screen[1] = render_target_size.height / canvas_scale.y;

		state_buffer.screen_to_sdf[0] = 1.0 / state_buffer.sdf_to_screen[0];
		state_buffer.screen_to_sdf[1] = 1.0 / state_buffer.sdf_to_screen[1];

		Rect2 sdf_rect = storage->render_target_get_sdf_rect(p_to_render_target);
		Rect2 sdf_tex_rect(sdf_rect.position / canvas_scale, sdf_rect.size / canvas_scale);

		state_buffer.sdf_to_tex[0] = 1.0 / sdf_tex_rect.size.width;
		state_buffer.sdf_to_tex[1] = 1.0 / sdf_tex_rect.size.height;
		state_buffer.sdf_to_tex[2] = -sdf_tex_rect.position.x / sdf_tex_rect.size.width;
		state_buffer.sdf_to_tex[3] = -sdf_tex_rect.position.y / sdf_tex_rect.size.height;

		//print_line("w: " + itos(ssize.width) + " s: " + rtos(canvas_scale));
		state_buffer.tex_to_sdf = 1.0 / ((canvas_scale.x + canvas_scale.y) * 0.5);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, state.canvas_state_buffer);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(StateBuffer), &state_buffer, GL_STREAM_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	{
		state.default_filter = p_default_filter;
		state.default_repeat = p_default_repeat;
	}

	state.current_tex = RID();
	state.current_tex_ptr = NULL;
	state.current_normal = RID();
	state.current_specular = RID();
	state.canvas_texscreen_used = false;

	r_sdf_used = false;
	int item_count = 0;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
	Item *ci = p_item_list;
	while (ci) {
		// just add all items for now
		items[item_count++] = ci;

		if (!ci->next || item_count == MAX_RENDER_ITEMS - 1) {
			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
			//then reset
			item_count = 0;
		}

		ci = ci->next;
	}
}

void RasterizerCanvasGLES3::_render_items(RID p_to_render_target, int p_item_count, const Transform2D &p_canvas_transform_inverse, Light *p_lights, bool p_to_backbuffer) {
	Item *current_clip = nullptr;

	Transform2D canvas_transform_inverse = p_canvas_transform_inverse;

	RID framebuffer;
	Vector<Color> clear_colors;

	canvas_begin();

	RID prev_material;
	uint32_t index = 0;

	for (int i = 0; i < p_item_count; i++) {
		Item *ci = items[i];

		RID material = ci->material_owner == nullptr ? ci->material : ci->material_owner->material;
		RasterizerStorageGLES3::Material *material_ptr = storage->material_owner.get_or_null(material);

		if (material.is_null() && ci->canvas_group != nullptr) {
			material = default_canvas_group_material;
		}

		if (material != prev_material) {
			RasterizerStorageGLES3::Shader *shader_ptr = NULL;

			if (material_ptr) {
				shader_ptr = material_ptr->shader;

				if (shader_ptr && shader_ptr->mode != RS::SHADER_CANVAS_ITEM) {
					shader_ptr = NULL; // not a canvas item shader, don't use.
				}
			}

			if (shader_ptr) {
				if (true) { //check that shader has changed
					if (shader_ptr->canvas_item.uses_time) {
						RenderingServerDefault::redraw_request();
					}
					//state.canvas_shader.version_bind_shader(shader_ptr->version, CanvasShaderGLES3::MODE_QUAD);
					state.current_shader_version = shader_ptr->version;
				}

				int tc = material_ptr->textures.size();
				Pair<StringName, RID> *textures = material_ptr->textures.ptrw();

				ShaderCompiler::GeneratedCode::Texture *texture_uniforms = shader_ptr->texture_uniforms.ptrw();

				for (int ti = 0; ti < tc; i++) {
					glActiveTexture(GL_TEXTURE0 + ti);

					RasterizerStorageGLES3::Texture *t = storage->texture_owner.get_or_null(textures[ti].second);

					if (!t) {
						switch (texture_uniforms[i].hint) {
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

					//Set texture filter and repeat texture_uniforms[i].filter texture_uniforms[i].repeat

					if (t->redraw_if_visible) {
						RenderingServerDefault::redraw_request();
					}

					t = t->get_ptr();

#ifdef TOOLS_ENABLED
					if (t->detect_normal && texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL) {
						t->detect_normal(t->detect_normal_ud);
					}
#endif
					if (t->render_target)
						t->render_target->used_in_frame = true;

					glBindTexture(t->target, t->tex_id);
				}

			} else {
				//state.canvas_shader.version_bind_shader(state.canvas_shader_default_version, CanvasShaderGLES3::MODE_QUAD);
				state.current_shader_version = state.canvas_shader_default_version;
			}
			prev_material = material;
		}

		_render_item(p_to_render_target, ci, canvas_transform_inverse, current_clip, p_lights, index);
	}
	// Render last command
	state.end_batch = true;
	_render_batch(index);

	canvas_end();
}

void RasterizerCanvasGLES3::_render_item(RID p_render_target, const Item *p_item, const Transform2D &p_canvas_transform_inverse, Item *&current_clip, Light *p_lights, uint32_t &r_index) {
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

	RID last_texture;
	Size2 texpixel_size;

	bool skipping = false;

	const Item::Command *c = p_item->commands;
	while (c) {
		if (skipping && c->type != Item::Command::TYPE_ANIMATION_SLICE) {
			c = c->next;
			continue;
		}

		_update_transform_2d_to_mat2x3(base_transform * draw_transform, state.instance_data_array[r_index].world);

		for (int i = 0; i < 4; i++) {
			state.instance_data_array[r_index].modulation[i] = 0.0;
			state.instance_data_array[r_index].ninepatch_margins[i] = 0.0;
			state.instance_data_array[r_index].src_rect[i] = 0.0;
			state.instance_data_array[r_index].dst_rect[i] = 0.0;
			state.instance_data_array[r_index].lights[i] = uint32_t(0);
		}
		state.instance_data_array[r_index].flags = base_flags;
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

				if (rect->texture != last_texture || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_RECT) {
					state.end_batch = true;
					_render_batch(r_index);

					state.current_primitive_points = 0;
					state.current_command = Item::Command::TYPE_RECT;
				}
				_bind_canvas_texture(rect->texture, current_filter, current_repeat, r_index, last_texture, texpixel_size);
				state.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_QUAD);

				Rect2 src_rect;
				Rect2 dst_rect;

				if (rect->texture != RID()) {
					src_rect = (rect->flags & CANVAS_RECT_REGION) ? Rect2(rect->source.position * texpixel_size, rect->source.size * texpixel_size) : Rect2(0, 0, 1, 1);
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
				//_render_batch(r_index);
				r_index++;
				if (r_index >= state.max_instances_per_batch - 1) {
					//r_index--;
					state.end_batch = true;
					_render_batch(r_index);
				}
			} break;

			case Item::Command::TYPE_NINEPATCH: {
				/*
				const Item::CommandNinePatch *np = static_cast<const Item::CommandNinePatch *>(c);

				//bind pipeline
				{
					RID pipeline = pipeline_variants->variants[light_mode][PIPELINE_VARIANT_NINEPATCH].get_render_pipeline(RD::INVALID_ID, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				_bind_canvas_texture(p_draw_list, np->texture, current_filter, current_repeat, index, last_texture, texpixel_size);

				Rect2 src_rect;
				Rect2 dst_rect(np->rect.position.x, np->rect.position.y, np->rect.size.x, np->rect.size.y);

				if (np->texture == RID()) {
					texpixel_size = Size2(1, 1);
					src_rect = Rect2(0, 0, 1, 1);

				} else {
					if (np->source != Rect2()) {
						src_rect = Rect2(np->source.position.x * texpixel_size.width, np->source.position.y * texpixel_size.height, np->source.size.x * texpixel_size.width, np->source.size.y * texpixel_size.height);
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

				RD::get_singleton()->draw_list_set_state.instance_data_array[r_index](p_draw_list, &state.instance_data_array[r_index], sizeof(PushConstant));
				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, shader.quad_index_array);
				RD::get_singleton()->draw_list_draw(p_draw_list, true);

				// Restore if overridden.
				state.instance_data_array[r_index].color_texture_pixel_size[0] = texpixel_size.x;
				state.instance_data_array[r_index].color_texture_pixel_size[1] = texpixel_size.y;
*/
			} break;

			case Item::Command::TYPE_POLYGON: {
				const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(c);

				PolygonBuffers *pb = polygon_buffers.polygons.getptr(polygon->polygon.polygon_id);
				ERR_CONTINUE(!pb);

				if (polygon->texture != last_texture || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_POLYGON) {
					state.end_batch = true;
					_render_batch(r_index);

					state.current_primitive_points = 0;
					state.current_command = Item::Command::TYPE_POLYGON;
				}
				_bind_canvas_texture(polygon->texture, current_filter, current_repeat, r_index, last_texture, texpixel_size);
				state.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_ATTRIBUTES);

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

				// If the previous operation is not done yet, allocated a new buffer
				GLint syncStatus;
				glGetSynciv(state.fences[state.current_buffer], GL_SYNC_STATUS, sizeof(GLint), nullptr, &syncStatus);
				if (syncStatus == GL_UNSIGNALED) {
					_allocate_instance_data_buffer();
				} else {
					glDeleteSync(state.fences[state.current_buffer]);
				}

				glBindBufferBase(GL_UNIFORM_BUFFER, 3, state.canvas_instance_data_buffers[state.current_buffer]);
#ifdef JAVASCRIPT_ENABLED
				//WebGL 2.0 does not support mapping buffers, so use slow glBufferData instead
				glBufferData(GL_UNIFORM_BUFFER, sizeof(InstanceData), &state.instance_data_array[0], GL_DYNAMIC_DRAW);
#else
				void *ubo = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(InstanceData), GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
				memcpy(ubo, &state.instance_data_array[0], sizeof(InstanceData));
				glUnmapBuffer(GL_UNIFORM_BUFFER);
#endif
				glBindVertexArray(pb->vertex_array);

				static const GLenum prim[5] = { GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_TRIANGLES, GL_TRIANGLE_STRIP };

				if (pb->index_buffer != 0) {
					glDrawElements(prim[polygon->primitive], pb->count, GL_UNSIGNED_INT, 0);
				} else {
					glDrawArrays(prim[polygon->primitive], 0, pb->count);
				}
				glBindVertexArray(0);
				state.fences[state.current_buffer] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

				state.current_buffer = (state.current_buffer + 1) % state.canvas_instance_data_buffers.size();
			} break;

			case Item::Command::TYPE_PRIMITIVE: {
				const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(c);

				if (last_texture != default_canvas_texture || state.current_primitive_points != primitive->point_count || state.current_command != Item::Command::TYPE_PRIMITIVE) {
					state.end_batch = true;
					_render_batch(r_index);
					state.current_primitive_points = primitive->point_count;
					state.current_command = Item::Command::TYPE_PRIMITIVE;
				}
				_bind_canvas_texture(RID(), current_filter, current_repeat, r_index, last_texture, texpixel_size);
				state.canvas_shader.version_bind_shader(state.current_shader_version, CanvasShaderGLES3::MODE_PRIMITIVE);

				for (uint32_t j = 0; j < MIN(3, primitive->point_count); j++) {
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
					//r_index--;
					state.end_batch = true;
					_render_batch(r_index);
				}
			} break;

			case Item::Command::TYPE_MESH:
			case Item::Command::TYPE_MULTIMESH:
			case Item::Command::TYPE_PARTICLES: {
				/*
				RID mesh;
				RID mesh_instance;
				RID texture;
				Color modulate(1, 1, 1, 1);
				int instance_count = 1;

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
					mesh = storage->multimesh_get_mesh(multimesh);
					texture = mm->texture;

					if (storage->multimesh_get_transform_format(multimesh) != RS::MULTIMESH_TRANSFORM_2D) {
						break;
					}

					instance_count = storage->multimesh_get_instances_to_draw(multimesh);

					if (instance_count == 0) {
						break;
					}

					state.instance_data_array[r_index].flags |= 1; //multimesh, trails disabled
					if (storage->multimesh_uses_colors(multimesh)) {
						state.instance_data_array[r_index].flags |= FLAGS_INSTANCING_HAS_COLORS;
					}
					if (storage->multimesh_uses_custom_data(multimesh)) {
						state.instance_data_array[r_index].flags |= FLAGS_INSTANCING_HAS_CUSTOM_DATA;
					}
				}

				// TODO: implement particles here

				if (mesh.is_null()) {
					break;
				}

				if (texture != last_texture || state.current_primitive_points != 0 || state.current_command != Item::Command::TYPE_PRIMITIVE) {
					state.end_batch = true;
					_render_batch(r_index);
					state.current_primitive_points = 0;
					state.current_command = c->type;
				}

				_bind_canvas_texture(texture, current_filter, current_repeat, r_index, last_texture, texpixel_size);

				uint32_t surf_count = storage->mesh_get_surface_count(mesh);

				state.instance_data_array[r_index].modulation[0] = base_color.r * modulate.r;
				state.instance_data_array[r_index].modulation[1] = base_color.g * modulate.g;
				state.instance_data_array[r_index].modulation[2] = base_color.b * modulate.b;
				state.instance_data_array[r_index].modulation[3] = base_color.a * modulate.a;

				for (int j = 0; j < 4; j++) {
					state.instance_data_array[r_index].src_rect[j] = 0;
					state.instance_data_array[r_index].dst_rect[j] = 0;
					state.instance_data_array[r_index].ninepatch_margins[j] = 0;
				}

				for (uint32_t j = 0; j < surf_count; j++) {
					RS::SurfaceData *surface = storage->mesh_get_surface(mesh, j);

					RS::PrimitiveType primitive = storage->mesh_surface_get_primitive(surface);
					ERR_CONTINUE(primitive < 0 || primitive >= RS::PRIMITIVE_MAX);

					glBindVertexArray(surface->vertex_array);
					static const GLenum prim[5] = { GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_TRIANGLES, GL_TRIANGLE_STRIP };

					// Draw directly, no need to batch
				}
				*/
			} break;
			case Item::Command::TYPE_TRANSFORM: {
				const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
				draw_transform = transform->xform;
			} break;

			case Item::Command::TYPE_CLIP_IGNORE: {
				/*
				const Item::CommandClipIgnore *ci = static_cast<const Item::CommandClipIgnore *>(c);
				if (current_clip) {
					if (ci->ignore != reclip) {
						if (ci->ignore) {
							RD::get_singleton()->draw_list_disable_scissor(p_draw_list);
							reclip = true;
						} else {
							RD::get_singleton()->draw_list_enable_scissor(p_draw_list, current_clip->final_clip_rect);
							reclip = false;
						}
					}
				}
				*/
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
}

void RasterizerCanvasGLES3::_render_batch(uint32_t &r_index) {
	if (state.end_batch && r_index > 0) {
		// If the previous operation is not done yet, allocate a new buffer
		GLint syncStatus;
		glGetSynciv(state.fences[state.current_buffer], GL_SYNC_STATUS, sizeof(GLint), nullptr, &syncStatus);
		if (syncStatus == GL_UNSIGNALED) {
			_allocate_instance_data_buffer();
		} else {
			glDeleteSync(state.fences[state.current_buffer]);
		}

		glBindBufferBase(GL_UNIFORM_BUFFER, 3, state.canvas_instance_data_buffers[state.current_buffer]);
#ifdef JAVASCRIPT_ENABLED
		//WebGL 2.0 does not support mapping buffers, so use slow glBufferData instead
		glBufferData(GL_UNIFORM_BUFFER, sizeof(InstanceData) * r_index, state.instance_data_array, GL_DYNAMIC_DRAW);
#else
		void *ubo = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(InstanceData) * r_index, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
		memcpy(ubo, state.instance_data_array, sizeof(InstanceData) * r_index);
		glUnmapBuffer(GL_UNIFORM_BUFFER);
#endif
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
		state.end_batch = false;
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

// TODO maybe dont use
void RasterizerCanvasGLES3::_end_batch(uint32_t &r_index) {
	for (int i = 0; i < 4; i++) {
		state.instance_data_array[r_index].modulation[i] = 0.0;
		state.instance_data_array[r_index].ninepatch_margins[i] = 0.0;
		state.instance_data_array[r_index].src_rect[i] = 0.0;
		state.instance_data_array[r_index].dst_rect[i] = 0.0;
	}
	state.instance_data_array[r_index].flags = uint32_t(0);
	state.instance_data_array[r_index].color_texture_pixel_size[0] = 0.0;
	state.instance_data_array[r_index].color_texture_pixel_size[1] = 0.0;

	state.instance_data_array[r_index].pad[0] = 0.0;
	state.instance_data_array[r_index].pad[1] = 0.0;

	state.instance_data_array[r_index].lights[0] = uint32_t(0);
	state.instance_data_array[r_index].lights[1] = uint32_t(0);
	state.instance_data_array[r_index].lights[2] = uint32_t(0);
	state.instance_data_array[r_index].lights[3] = uint32_t(0);
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

void RasterizerCanvasGLES3::canvas_begin() {
	state.using_transparent_rt = false;

	if (storage->frame.current_rt) {
		storage->bind_framebuffer(storage->frame.current_rt->fbo);
		state.using_transparent_rt = storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT];
	}

	if (storage->frame.current_rt && storage->frame.current_rt->clear_requested) {
		const Color &col = storage->frame.current_rt->clear_color;
		glClearColor(col.r, col.g, col.b, col.a);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		storage->frame.current_rt->clear_requested = false;
	}

	reset_canvas();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
}

void RasterizerCanvasGLES3::canvas_end() {
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void RasterizerCanvasGLES3::_bind_canvas_texture(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, uint32_t &r_index, RID &r_last_texture, Size2 &r_texpixel_size) {
	if (p_texture == RID()) {
		p_texture = default_canvas_texture;
	}

	if (r_last_texture == p_texture) {
		return; //nothing to do, its the same
	}

	state.end_batch = true;
	_render_batch(r_index);

	RasterizerStorageGLES3::CanvasTexture *ct = nullptr;

	RasterizerStorageGLES3::Texture *t = storage->texture_owner.get_or_null(p_texture);

	if (t) {
		//regular texture
		if (!t->canvas_texture) {
			t->canvas_texture = memnew(RasterizerStorageGLES3::CanvasTexture);
			t->canvas_texture->diffuse = p_texture;
		}

		ct = t->canvas_texture;
	} else {
		ct = storage->canvas_texture_owner.get_or_null(p_texture);
	}

	if (!ct) {
		// Invalid Texture RID.
		_bind_canvas_texture(default_canvas_texture, p_base_filter, p_base_repeat, r_index, r_last_texture, r_texpixel_size);
		return;
	}

	RS::CanvasItemTextureFilter filter = ct->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT ? ct->texture_filter : p_base_filter;
	ERR_FAIL_COND(filter == RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT);

	RS::CanvasItemTextureRepeat repeat = ct->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT ? ct->texture_repeat : p_base_repeat;
	ERR_FAIL_COND(repeat == RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT);

	RasterizerStorageGLES3::Texture *texture = storage->texture_owner.get_or_null(ct->diffuse);

	if (!texture) {
		state.current_tex = RID();
		state.current_tex_ptr = NULL;
		ct->size_cache = Size2i(1, 1);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	} else {
		texture = texture->get_ptr();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture->tex_id);

		state.current_tex = ct->diffuse;
		state.current_tex_ptr = texture;
		ct->size_cache = Size2i(texture->width, texture->height);

		texture->GLSetFilter(GL_TEXTURE_2D, filter);
		texture->GLSetRepeat(GL_TEXTURE_2D, repeat);
	}

	RasterizerStorageGLES3::Texture *normal_map = storage->texture_owner.get_or_null(ct->normal_map);

	if (!normal_map) {
		state.current_normal = RID();
		ct->use_normal_cache = false;
		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 6);
		glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);

	} else {
		normal_map = normal_map->get_ptr();

		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 6);
		glBindTexture(GL_TEXTURE_2D, normal_map->tex_id);
		state.current_normal = ct->normal_map;
		ct->use_normal_cache = true;
		texture->GLSetFilter(GL_TEXTURE_2D, filter);
		texture->GLSetRepeat(GL_TEXTURE_2D, repeat);
	}

	RasterizerStorageGLES3::Texture *specular_map = storage->texture_owner.get_or_null(ct->specular);

	if (!specular_map) {
		state.current_specular = RID();
		ct->use_specular_cache = false;
		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 7);
		glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);

	} else {
		specular_map = specular_map->get_ptr();

		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 7);
		glBindTexture(GL_TEXTURE_2D, specular_map->tex_id);
		state.current_specular = ct->specular;
		ct->use_specular_cache = true;
		texture->GLSetFilter(GL_TEXTURE_2D, filter);
		texture->GLSetRepeat(GL_TEXTURE_2D, repeat);
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

	r_texpixel_size.x = 1.0 / float(ct->size_cache.x);
	r_texpixel_size.y = 1.0 / float(ct->size_cache.y);

	state.instance_data_array[r_index].color_texture_pixel_size[0] = r_texpixel_size.x;
	state.instance_data_array[r_index].color_texture_pixel_size[1] = r_texpixel_size.y;

	r_last_texture = p_texture;
}

void RasterizerCanvasGLES3::_set_uniforms() {
}

void RasterizerCanvasGLES3::reset_canvas() {
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glEnable(GL_BLEND);

	// Default to Mix.
	glBlendEquation(GL_FUNC_ADD);
	if (storage->frame.current_rt && storage->frame.current_rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RasterizerCanvasGLES3::canvas_debug_viewport_shadows(Light *p_lights_with_shadow) {
}

void RasterizerCanvasGLES3::canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {
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
		const uint8_t *r = polygon_buffer.ptr();
		float *fptr = (float *)r;
		uint32_t *uptr = (uint32_t *)r;
		uint32_t base_offset = 0;
		{
			// Always uses vertex positions
			glEnableVertexAttribArray(RS::ARRAY_VERTEX);
			glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, stride * sizeof(float), NULL);
			const Vector2 *points_ptr = p_points.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = points_ptr[i].x;
				fptr[base_offset + i * stride + 1] = points_ptr[i].y;
			}

			base_offset += 2;
		}

		// Next add colors
		if (p_colors.size() == 1) {
			glDisableVertexAttribArray(RS::ARRAY_COLOR);
			Color m = p_colors[0];
			glVertexAttrib4f(RS::ARRAY_COLOR, m.r, m.g, m.b, m.a);
		} else if ((uint32_t)p_colors.size() == vertex_count) {
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
			glVertexAttrib4f(RS::ARRAY_COLOR, 1.0, 1.0, 1.0, 1.0);
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

void RasterizerCanvasGLES3::initialize() {
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

	//state.canvas_shadow_shader.init();

	int uniform_max_size;
	glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &uniform_max_size);
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
	global_defines += "#define MAX_GLOBAL_VARIABLES 256\n"; // TODO: this is arbitrary for now
	global_defines += "#define MAX_LIGHTS " + itos(state.max_instances_per_batch) + "\n";
	global_defines += "#define MAX_DRAW_DATA_INSTANCES " + itos(state.max_instances_per_batch) + "\n";

	state.canvas_shader.initialize(global_defines);
	state.canvas_shader_default_version = state.canvas_shader.version_create();
	state.canvas_shader.version_bind_shader(state.canvas_shader_default_version, CanvasShaderGLES3::MODE_QUAD);

	//state.canvas_shader.set_conditional(CanvasOldShaderGLES3::USE_RGBA_SHADOWS, storage->config.use_rgba_2d_shadows);

	//state.canvas_shader.bind();

	//state.lens_shader.init();

	//state.canvas_shader.set_conditional(CanvasOldShaderGLES3::USE_PIXEL_SNAP, GLOBAL_DEF("rendering/quality/2d/use_pixel_snap", false));

	{
		default_canvas_group_shader = storage->shader_allocate();
		storage->shader_initialize(default_canvas_group_shader);

		storage->shader_set_code(default_canvas_group_shader, R"(
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
		default_canvas_group_material = storage->material_allocate();
		storage->material_initialize(default_canvas_group_material);

		storage->material_set_shader(default_canvas_group_material, default_canvas_group_shader);
	}

	default_canvas_texture = storage->canvas_texture_allocate();
	storage->canvas_texture_initialize(default_canvas_texture);

	state.using_light = NULL;
	state.using_transparent_rt = false;
	state.using_skeleton = false;
	state.current_shader_version = state.canvas_shader_default_version;
}

RasterizerCanvasGLES3::RasterizerCanvasGLES3() {
}
RasterizerCanvasGLES3::~RasterizerCanvasGLES3() {
	state.canvas_shader.version_free(state.canvas_shader_default_version);
	storage->free(default_canvas_group_material);
	storage->free(default_canvas_group_shader);
	storage->free(default_canvas_texture);
}

void RasterizerCanvasGLES3::finalize() {
	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);

	glDeleteBuffers(1, &data.canvas_quad_vertices);
	glDeleteVertexArrays(1, &data.canvas_quad_array);
}

#endif // GLES3_ENABLED

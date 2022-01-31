/*************************************************************************/
/*  renderer_canvas_render_rd.cpp                                        */
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

#include "renderer_canvas_render_rd.h"

#include "core/config/project_settings.h"
#include "core/math/geometry_2d.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "renderer_compositor_rd.h"
#include "servers/rendering/rendering_server_default.h"

void RendererCanvasRenderRD::_update_transform_2d_to_mat4(const Transform2D &p_transform, float *p_mat4) {
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

void RendererCanvasRenderRD::_update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4) {
	p_mat2x4[0] = p_transform.elements[0][0];
	p_mat2x4[1] = p_transform.elements[1][0];
	p_mat2x4[2] = 0;
	p_mat2x4[3] = p_transform.elements[2][0];

	p_mat2x4[4] = p_transform.elements[0][1];
	p_mat2x4[5] = p_transform.elements[1][1];
	p_mat2x4[6] = 0;
	p_mat2x4[7] = p_transform.elements[2][1];
}

void RendererCanvasRenderRD::_update_transform_2d_to_mat2x3(const Transform2D &p_transform, float *p_mat2x3) {
	p_mat2x3[0] = p_transform.elements[0][0];
	p_mat2x3[1] = p_transform.elements[0][1];
	p_mat2x3[2] = p_transform.elements[1][0];
	p_mat2x3[3] = p_transform.elements[1][1];
	p_mat2x3[4] = p_transform.elements[2][0];
	p_mat2x3[5] = p_transform.elements[2][1];
}

void RendererCanvasRenderRD::_update_transform_to_mat4(const Transform3D &p_transform, float *p_mat4) {
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

RendererCanvasRender::PolygonID RendererCanvasRenderRD::request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, const Vector<int> &p_bones, const Vector<float> &p_weights) {
	// Care must be taken to generate array formats
	// in ways where they could be reused, so we will
	// put single-occuring elements first, and repeated
	// elements later. This way the generated formats are
	// the same no matter the length of the arrays.
	// This dramatically reduces the amount of pipeline objects
	// that need to be created for these formats.

	uint32_t vertex_count = p_points.size();
	uint32_t stride = 2; //vertices always repeat
	if ((uint32_t)p_colors.size() == vertex_count || p_colors.size() == 1) {
		stride += 4;
	}
	if ((uint32_t)p_uvs.size() == vertex_count) {
		stride += 2;
	}
	if ((uint32_t)p_bones.size() == vertex_count * 4 && (uint32_t)p_weights.size() == vertex_count * 4) {
		stride += 4;
	}

	uint32_t buffer_size = stride * p_points.size();

	Vector<uint8_t> polygon_buffer;
	polygon_buffer.resize(buffer_size * sizeof(float));
	Vector<RD::VertexAttribute> descriptions;
	descriptions.resize(5);
	Vector<RID> buffers;
	buffers.resize(5);

	{
		const uint8_t *r = polygon_buffer.ptr();
		float *fptr = (float *)r;
		uint32_t *uptr = (uint32_t *)r;
		uint32_t base_offset = 0;
		{ //vertices
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = RS::ARRAY_VERTEX;
			vd.stride = stride * sizeof(float);

			descriptions.write[0] = vd;

			const Vector2 *points_ptr = p_points.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = points_ptr[i].x;
				fptr[base_offset + i * stride + 1] = points_ptr[i].y;
			}

			base_offset += 2;
		}

		//colors
		if ((uint32_t)p_colors.size() == vertex_count || p_colors.size() == 1) {
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = RS::ARRAY_COLOR;
			vd.stride = stride * sizeof(float);

			descriptions.write[1] = vd;

			if (p_colors.size() == 1) {
				Color color = p_colors[0];
				for (uint32_t i = 0; i < vertex_count; i++) {
					fptr[base_offset + i * stride + 0] = color.r;
					fptr[base_offset + i * stride + 1] = color.g;
					fptr[base_offset + i * stride + 2] = color.b;
					fptr[base_offset + i * stride + 3] = color.a;
				}
			} else {
				const Color *color_ptr = p_colors.ptr();

				for (uint32_t i = 0; i < vertex_count; i++) {
					fptr[base_offset + i * stride + 0] = color_ptr[i].r;
					fptr[base_offset + i * stride + 1] = color_ptr[i].g;
					fptr[base_offset + i * stride + 2] = color_ptr[i].b;
					fptr[base_offset + i * stride + 3] = color_ptr[i].a;
				}
			}
			base_offset += 4;
		} else {
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = 0;
			vd.location = RS::ARRAY_COLOR;
			vd.stride = 0;

			descriptions.write[1] = vd;
			buffers.write[1] = storage->mesh_get_default_rd_buffer(RendererStorageRD::DEFAULT_RD_BUFFER_COLOR);
		}

		//uvs
		if ((uint32_t)p_uvs.size() == vertex_count) {
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = RS::ARRAY_TEX_UV;
			vd.stride = stride * sizeof(float);

			descriptions.write[2] = vd;

			const Vector2 *uv_ptr = p_uvs.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = uv_ptr[i].x;
				fptr[base_offset + i * stride + 1] = uv_ptr[i].y;
			}
			base_offset += 2;
		} else {
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			vd.offset = 0;
			vd.location = RS::ARRAY_TEX_UV;
			vd.stride = 0;

			descriptions.write[2] = vd;
			buffers.write[2] = storage->mesh_get_default_rd_buffer(RendererStorageRD::DEFAULT_RD_BUFFER_TEX_UV);
		}

		//bones
		if ((uint32_t)p_indices.size() == vertex_count * 4 && (uint32_t)p_weights.size() == vertex_count * 4) {
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R16G16B16A16_UINT;
			vd.offset = base_offset * sizeof(float);
			vd.location = RS::ARRAY_BONES;
			vd.stride = stride * sizeof(float);

			descriptions.write[3] = vd;

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
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
			vd.offset = 0;
			vd.location = RS::ARRAY_BONES;
			vd.stride = 0;

			descriptions.write[3] = vd;
			buffers.write[3] = storage->mesh_get_default_rd_buffer(RendererStorageRD::DEFAULT_RD_BUFFER_BONES);
		}

		//weights
		if ((uint32_t)p_weights.size() == vertex_count * 4) {
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
			vd.offset = base_offset * sizeof(float);
			vd.location = RS::ARRAY_WEIGHTS;
			vd.stride = stride * sizeof(float);

			descriptions.write[4] = vd;

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
			RD::VertexAttribute vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = 0;
			vd.location = RS::ARRAY_WEIGHTS;
			vd.stride = 0;

			descriptions.write[4] = vd;
			buffers.write[4] = storage->mesh_get_default_rd_buffer(RendererStorageRD::DEFAULT_RD_BUFFER_WEIGHTS);
		}

		//check that everything is as it should be
		ERR_FAIL_COND_V(base_offset != stride, 0); //bug
	}

	RD::VertexFormatID vertex_id = RD::get_singleton()->vertex_format_create(descriptions);
	ERR_FAIL_COND_V(vertex_id == RD::INVALID_ID, 0);

	PolygonBuffers pb;
	pb.vertex_buffer = RD::get_singleton()->vertex_buffer_create(polygon_buffer.size(), polygon_buffer);
	for (int i = 0; i < descriptions.size(); i++) {
		if (buffers[i] == RID()) { //if put in vertex, use as vertex
			buffers.write[i] = pb.vertex_buffer;
		}
	}

	pb.vertex_array = RD::get_singleton()->vertex_array_create(p_points.size(), vertex_id, buffers);

	if (p_indices.size()) {
		//create indices, as indices were requested
		Vector<uint8_t> index_buffer;
		index_buffer.resize(p_indices.size() * sizeof(int32_t));
		{
			uint8_t *w = index_buffer.ptrw();
			memcpy(w, p_indices.ptr(), sizeof(int32_t) * p_indices.size());
		}
		pb.index_buffer = RD::get_singleton()->index_buffer_create(p_indices.size(), RD::INDEX_BUFFER_FORMAT_UINT32, index_buffer);
		pb.indices = RD::get_singleton()->index_array_create(pb.index_buffer, 0, p_indices.size());
	}

	pb.vertex_format_id = vertex_id;

	PolygonID id = polygon_buffers.last_id++;

	polygon_buffers.polygons[id] = pb;

	return id;
}

void RendererCanvasRenderRD::free_polygon(PolygonID p_polygon) {
	PolygonBuffers *pb_ptr = polygon_buffers.polygons.getptr(p_polygon);
	ERR_FAIL_COND(!pb_ptr);

	PolygonBuffers &pb = *pb_ptr;

	if (pb.indices.is_valid()) {
		RD::get_singleton()->free(pb.indices);
	}
	if (pb.index_buffer.is_valid()) {
		RD::get_singleton()->free(pb.index_buffer);
	}

	RD::get_singleton()->free(pb.vertex_array);
	RD::get_singleton()->free(pb.vertex_buffer);

	polygon_buffers.polygons.erase(p_polygon);
}

////////////////////

void RendererCanvasRenderRD::_bind_canvas_texture(RD::DrawListID p_draw_list, RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID &r_last_texture, PushConstant &push_constant, Size2 &r_texpixel_size) {
	if (p_texture == RID()) {
		p_texture = default_canvas_texture;
	}

	if (r_last_texture == p_texture) {
		return; //nothing to do, its the same
	}

	RID uniform_set;
	Color specular_shininess;
	Size2i size;
	bool use_normal;
	bool use_specular;

	bool success = storage->canvas_texture_get_uniform_set(p_texture, p_base_filter, p_base_repeat, shader.default_version_rd_shader, CANVAS_TEXTURE_UNIFORM_SET, uniform_set, size, specular_shininess, use_normal, use_specular);
	//something odd happened
	if (!success) {
		_bind_canvas_texture(p_draw_list, default_canvas_texture, p_base_filter, p_base_repeat, r_last_texture, push_constant, r_texpixel_size);
		return;
	}

	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, uniform_set, CANVAS_TEXTURE_UNIFORM_SET);

	if (specular_shininess.a < 0.999) {
		push_constant.flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
	} else {
		push_constant.flags &= ~FLAGS_DEFAULT_SPECULAR_MAP_USED;
	}

	if (use_normal) {
		push_constant.flags |= FLAGS_DEFAULT_NORMAL_MAP_USED;
	} else {
		push_constant.flags &= ~FLAGS_DEFAULT_NORMAL_MAP_USED;
	}

	push_constant.specular_shininess = uint32_t(CLAMP(specular_shininess.a * 255.0, 0, 255)) << 24;
	push_constant.specular_shininess |= uint32_t(CLAMP(specular_shininess.b * 255.0, 0, 255)) << 16;
	push_constant.specular_shininess |= uint32_t(CLAMP(specular_shininess.g * 255.0, 0, 255)) << 8;
	push_constant.specular_shininess |= uint32_t(CLAMP(specular_shininess.r * 255.0, 0, 255));

	r_texpixel_size.x = 1.0 / float(size.x);
	r_texpixel_size.y = 1.0 / float(size.y);

	push_constant.color_texture_pixel_size[0] = r_texpixel_size.x;
	push_constant.color_texture_pixel_size[1] = r_texpixel_size.y;

	r_last_texture = p_texture;
}

void RendererCanvasRenderRD::_render_item(RD::DrawListID p_draw_list, RID p_render_target, const Item *p_item, RD::FramebufferFormatID p_framebuffer_format, const Transform2D &p_canvas_transform_inverse, Item *&current_clip, Light *p_lights, PipelineVariants *p_pipeline_variants) {
	//create an empty push constant

	RS::CanvasItemTextureFilter current_filter = default_filter;
	RS::CanvasItemTextureRepeat current_repeat = default_repeat;

	if (p_item->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT) {
		current_filter = p_item->texture_filter;
	}

	if (p_item->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) {
		current_repeat = p_item->texture_repeat;
	}

	PushConstant push_constant;
	Transform2D base_transform = p_canvas_transform_inverse * p_item->final_transform;
	Transform2D draw_transform;
	_update_transform_2d_to_mat2x3(base_transform, push_constant.world);

	Color base_color = p_item->final_modulate;

	for (int i = 0; i < 4; i++) {
		push_constant.modulation[i] = 0;
		push_constant.ninepatch_margins[i] = 0;
		push_constant.src_rect[i] = 0;
		push_constant.dst_rect[i] = 0;
	}
	push_constant.flags = 0;
	push_constant.color_texture_pixel_size[0] = 0;
	push_constant.color_texture_pixel_size[1] = 0;

	push_constant.pad[0] = 0;
	push_constant.pad[1] = 0;

	push_constant.lights[0] = 0;
	push_constant.lights[1] = 0;
	push_constant.lights[2] = 0;
	push_constant.lights[3] = 0;

	uint32_t base_flags = 0;

	uint16_t light_count = 0;
	PipelineLightMode light_mode;

	{
		Light *light = p_lights;

		while (light) {
			if (light->render_index_cache >= 0 && p_item->light_mask & light->item_mask && p_item->z_final >= light->z_min && p_item->z_final <= light->z_max && p_item->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {
				uint32_t light_index = light->render_index_cache;
				push_constant.lights[light_count >> 2] |= light_index << ((light_count & 3) * 8);

				light_count++;

				if (light_count == MAX_LIGHTS_PER_ITEM) {
					break;
				}
			}
			light = light->next_ptr;
		}

		base_flags |= light_count << FLAGS_LIGHT_COUNT_SHIFT;
	}

	light_mode = (light_count > 0 || using_directional_lights) ? PIPELINE_LIGHT_MODE_ENABLED : PIPELINE_LIGHT_MODE_DISABLED;

	PipelineVariants *pipeline_variants = p_pipeline_variants;

	bool reclip = false;

	RID last_texture;
	Size2 texpixel_size;

	bool skipping = false;

	const Item::Command *c = p_item->commands;
	while (c) {
		if (skipping && c->type != Item::Command::TYPE_ANIMATION_SLICE) {
			c = c->next;
			continue;
		}

		push_constant.flags = base_flags | (push_constant.flags & (FLAGS_DEFAULT_NORMAL_MAP_USED | FLAGS_DEFAULT_SPECULAR_MAP_USED)); //reset on each command for sanity, keep canvastexture binding config

		switch (c->type) {
			case Item::Command::TYPE_RECT: {
				const Item::CommandRect *rect = static_cast<const Item::CommandRect *>(c);

				if (rect->flags & CANVAS_RECT_TILE) {
					current_repeat = RenderingServer::CanvasItemTextureRepeat::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED;
				}

				//bind pipeline
				{
					RID pipeline = pipeline_variants->variants[light_mode][PIPELINE_VARIANT_QUAD].get_render_pipeline(RD::INVALID_ID, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				_bind_canvas_texture(p_draw_list, rect->texture, current_filter, current_repeat, last_texture, push_constant, texpixel_size);

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
						push_constant.flags |= FLAGS_CLIP_RECT_UV;
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
					push_constant.flags |= FLAGS_USE_MSDF;
					push_constant.msdf[0] = rect->px_range; // Pixel range.
					push_constant.msdf[1] = rect->outline; // Outline size.
					push_constant.msdf[2] = 0.f; // Reserved.
					push_constant.msdf[3] = 0.f; // Reserved.
				}

				push_constant.modulation[0] = rect->modulate.r * base_color.r;
				push_constant.modulation[1] = rect->modulate.g * base_color.g;
				push_constant.modulation[2] = rect->modulate.b * base_color.b;
				push_constant.modulation[3] = rect->modulate.a * base_color.a;

				push_constant.src_rect[0] = src_rect.position.x;
				push_constant.src_rect[1] = src_rect.position.y;
				push_constant.src_rect[2] = src_rect.size.width;
				push_constant.src_rect[3] = src_rect.size.height;

				push_constant.dst_rect[0] = dst_rect.position.x;
				push_constant.dst_rect[1] = dst_rect.position.y;
				push_constant.dst_rect[2] = dst_rect.size.width;
				push_constant.dst_rect[3] = dst_rect.size.height;

				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, shader.quad_index_array);
				RD::get_singleton()->draw_list_draw(p_draw_list, true);

			} break;

			case Item::Command::TYPE_NINEPATCH: {
				const Item::CommandNinePatch *np = static_cast<const Item::CommandNinePatch *>(c);

				//bind pipeline
				{
					RID pipeline = pipeline_variants->variants[light_mode][PIPELINE_VARIANT_NINEPATCH].get_render_pipeline(RD::INVALID_ID, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				_bind_canvas_texture(p_draw_list, np->texture, current_filter, current_repeat, last_texture, push_constant, texpixel_size);

				Rect2 src_rect;
				Rect2 dst_rect(np->rect.position.x, np->rect.position.y, np->rect.size.x, np->rect.size.y);

				if (np->texture == RID()) {
					texpixel_size = Size2(1, 1);
					src_rect = Rect2(0, 0, 1, 1);

				} else {
					if (np->source != Rect2()) {
						src_rect = Rect2(np->source.position.x * texpixel_size.width, np->source.position.y * texpixel_size.height, np->source.size.x * texpixel_size.width, np->source.size.y * texpixel_size.height);
						push_constant.color_texture_pixel_size[0] = 1.0 / np->source.size.width;
						push_constant.color_texture_pixel_size[1] = 1.0 / np->source.size.height;

					} else {
						src_rect = Rect2(0, 0, 1, 1);
					}
				}

				push_constant.modulation[0] = np->color.r * base_color.r;
				push_constant.modulation[1] = np->color.g * base_color.g;
				push_constant.modulation[2] = np->color.b * base_color.b;
				push_constant.modulation[3] = np->color.a * base_color.a;

				push_constant.src_rect[0] = src_rect.position.x;
				push_constant.src_rect[1] = src_rect.position.y;
				push_constant.src_rect[2] = src_rect.size.width;
				push_constant.src_rect[3] = src_rect.size.height;

				push_constant.dst_rect[0] = dst_rect.position.x;
				push_constant.dst_rect[1] = dst_rect.position.y;
				push_constant.dst_rect[2] = dst_rect.size.width;
				push_constant.dst_rect[3] = dst_rect.size.height;

				push_constant.flags |= int(np->axis_x) << FLAGS_NINEPATCH_H_MODE_SHIFT;
				push_constant.flags |= int(np->axis_y) << FLAGS_NINEPATCH_V_MODE_SHIFT;

				if (np->draw_center) {
					push_constant.flags |= FLAGS_NINEPACH_DRAW_CENTER;
				}

				push_constant.ninepatch_margins[0] = np->margin[SIDE_LEFT];
				push_constant.ninepatch_margins[1] = np->margin[SIDE_TOP];
				push_constant.ninepatch_margins[2] = np->margin[SIDE_RIGHT];
				push_constant.ninepatch_margins[3] = np->margin[SIDE_BOTTOM];

				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, shader.quad_index_array);
				RD::get_singleton()->draw_list_draw(p_draw_list, true);

				// Restore if overridden.
				push_constant.color_texture_pixel_size[0] = texpixel_size.x;
				push_constant.color_texture_pixel_size[1] = texpixel_size.y;

			} break;
			case Item::Command::TYPE_POLYGON: {
				const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(c);

				PolygonBuffers *pb = polygon_buffers.polygons.getptr(polygon->polygon.polygon_id);
				ERR_CONTINUE(!pb);
				//bind pipeline
				{
					static const PipelineVariant variant[RS::PRIMITIVE_MAX] = { PIPELINE_VARIANT_ATTRIBUTE_POINTS, PIPELINE_VARIANT_ATTRIBUTE_LINES, PIPELINE_VARIANT_ATTRIBUTE_LINES_STRIP, PIPELINE_VARIANT_ATTRIBUTE_TRIANGLES, PIPELINE_VARIANT_ATTRIBUTE_TRIANGLE_STRIP };
					ERR_CONTINUE(polygon->primitive < 0 || polygon->primitive >= RS::PRIMITIVE_MAX);
					RID pipeline = pipeline_variants->variants[light_mode][variant[polygon->primitive]].get_render_pipeline(pb->vertex_format_id, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				if (polygon->primitive == RS::PRIMITIVE_LINES) {
					//not supported in most hardware, so pointless
					//RD::get_singleton()->draw_list_set_line_width(p_draw_list, polygon->line_width);
				}

				//bind textures

				_bind_canvas_texture(p_draw_list, polygon->texture, current_filter, current_repeat, last_texture, push_constant, texpixel_size);

				push_constant.modulation[0] = base_color.r;
				push_constant.modulation[1] = base_color.g;
				push_constant.modulation[2] = base_color.b;
				push_constant.modulation[3] = base_color.a;

				for (int j = 0; j < 4; j++) {
					push_constant.src_rect[j] = 0;
					push_constant.dst_rect[j] = 0;
					push_constant.ninepatch_margins[j] = 0;
				}

				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
				RD::get_singleton()->draw_list_bind_vertex_array(p_draw_list, pb->vertex_array);
				if (pb->indices.is_valid()) {
					RD::get_singleton()->draw_list_bind_index_array(p_draw_list, pb->indices);
				}
				RD::get_singleton()->draw_list_draw(p_draw_list, pb->indices.is_valid());

			} break;
			case Item::Command::TYPE_PRIMITIVE: {
				const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(c);

				//bind pipeline
				{
					static const PipelineVariant variant[4] = { PIPELINE_VARIANT_PRIMITIVE_POINTS, PIPELINE_VARIANT_PRIMITIVE_LINES, PIPELINE_VARIANT_PRIMITIVE_TRIANGLES, PIPELINE_VARIANT_PRIMITIVE_TRIANGLES };
					ERR_CONTINUE(primitive->point_count == 0 || primitive->point_count > 4);
					RID pipeline = pipeline_variants->variants[light_mode][variant[primitive->point_count - 1]].get_render_pipeline(RD::INVALID_ID, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				_bind_canvas_texture(p_draw_list, RID(), current_filter, current_repeat, last_texture, push_constant, texpixel_size);

				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, primitive_arrays.index_array[MIN(3, primitive->point_count) - 1]);

				for (uint32_t j = 0; j < MIN(3, primitive->point_count); j++) {
					push_constant.points[j * 2 + 0] = primitive->points[j].x;
					push_constant.points[j * 2 + 1] = primitive->points[j].y;
					push_constant.uvs[j * 2 + 0] = primitive->uvs[j].x;
					push_constant.uvs[j * 2 + 1] = primitive->uvs[j].y;
					Color col = primitive->colors[j] * base_color;
					push_constant.colors[j * 2 + 0] = (uint32_t(Math::make_half_float(col.g)) << 16) | Math::make_half_float(col.r);
					push_constant.colors[j * 2 + 1] = (uint32_t(Math::make_half_float(col.a)) << 16) | Math::make_half_float(col.b);
				}
				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
				RD::get_singleton()->draw_list_draw(p_draw_list, true);

				if (primitive->point_count == 4) {
					for (uint32_t j = 1; j < 3; j++) {
						//second half of triangle
						push_constant.points[j * 2 + 0] = primitive->points[j + 1].x;
						push_constant.points[j * 2 + 1] = primitive->points[j + 1].y;
						push_constant.uvs[j * 2 + 0] = primitive->uvs[j + 1].x;
						push_constant.uvs[j * 2 + 1] = primitive->uvs[j + 1].y;
						Color col = primitive->colors[j + 1] * base_color;
						push_constant.colors[j * 2 + 0] = (uint32_t(Math::make_half_float(col.g)) << 16) | Math::make_half_float(col.r);
						push_constant.colors[j * 2 + 1] = (uint32_t(Math::make_half_float(col.a)) << 16) | Math::make_half_float(col.b);
					}

					RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
					RD::get_singleton()->draw_list_draw(p_draw_list, true);
				}

			} break;
			case Item::Command::TYPE_MESH:
			case Item::Command::TYPE_MULTIMESH:
			case Item::Command::TYPE_PARTICLES: {
				RID mesh;
				RID mesh_instance;
				RID texture;
				Color modulate(1, 1, 1, 1);
				float world_backup[6];
				int instance_count = 1;

				for (int j = 0; j < 6; j++) {
					world_backup[j] = push_constant.world[j];
				}

				if (c->type == Item::Command::TYPE_MESH) {
					const Item::CommandMesh *m = static_cast<const Item::CommandMesh *>(c);
					mesh = m->mesh;
					mesh_instance = m->mesh_instance;
					texture = m->texture;
					modulate = m->modulate;
					_update_transform_2d_to_mat2x3(base_transform * draw_transform * m->transform, push_constant.world);
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

					RID uniform_set = storage->multimesh_get_2d_uniform_set(multimesh, shader.default_version_rd_shader, TRANSFORMS_UNIFORM_SET);
					RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, uniform_set, TRANSFORMS_UNIFORM_SET);
					push_constant.flags |= 1; //multimesh, trails disabled
					if (storage->multimesh_uses_colors(multimesh)) {
						push_constant.flags |= FLAGS_INSTANCING_HAS_COLORS;
					}
					if (storage->multimesh_uses_custom_data(multimesh)) {
						push_constant.flags |= FLAGS_INSTANCING_HAS_CUSTOM_DATA;
					}
				} else if (c->type == Item::Command::TYPE_PARTICLES) {
					const Item::CommandParticles *pt = static_cast<const Item::CommandParticles *>(c);
					ERR_BREAK(storage->particles_get_mode(pt->particles) != RS::PARTICLES_MODE_2D);
					storage->particles_request_process(pt->particles);

					if (storage->particles_is_inactive(pt->particles)) {
						break;
					}

					RenderingServerDefault::redraw_request(); // active particles means redraw request

					bool local_coords = true;
					int dpc = storage->particles_get_draw_passes(pt->particles);
					if (dpc == 0) {
						break; //nothing to draw
					}
					uint32_t divisor = 1;
					instance_count = storage->particles_get_amount(pt->particles, divisor);

					RID uniform_set = storage->particles_get_instance_buffer_uniform_set(pt->particles, shader.default_version_rd_shader, TRANSFORMS_UNIFORM_SET);
					RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, uniform_set, TRANSFORMS_UNIFORM_SET);

					push_constant.flags |= divisor;
					instance_count /= divisor;

					push_constant.flags |= FLAGS_INSTANCING_HAS_COLORS;
					push_constant.flags |= FLAGS_INSTANCING_HAS_CUSTOM_DATA;

					mesh = storage->particles_get_draw_pass_mesh(pt->particles, 0); //higher ones are ignored
					texture = pt->texture;

					if (storage->particles_has_collision(pt->particles) && storage->render_target_is_sdf_enabled(p_render_target)) {
						//pass collision information
						Transform2D xform;
						if (local_coords) {
							xform = p_item->final_transform;
						} else {
							xform = p_canvas_transform_inverse;
						}

						RID sdf_texture = storage->render_target_get_sdf_texture(p_render_target);

						Rect2 to_screen;
						{
							Rect2 sdf_rect = storage->render_target_get_sdf_rect(p_render_target);

							to_screen.size = Vector2(1.0 / sdf_rect.size.width, 1.0 / sdf_rect.size.height);
							to_screen.position = -sdf_rect.position * to_screen.size;
						}

						storage->particles_set_canvas_sdf_collision(pt->particles, true, xform, to_screen, sdf_texture);
					} else {
						storage->particles_set_canvas_sdf_collision(pt->particles, false, Transform2D(), Rect2(), RID());
					}
				}

				if (mesh.is_null()) {
					break;
				}

				_bind_canvas_texture(p_draw_list, texture, current_filter, current_repeat, last_texture, push_constant, texpixel_size);

				uint32_t surf_count = storage->mesh_get_surface_count(mesh);
				static const PipelineVariant variant[RS::PRIMITIVE_MAX] = { PIPELINE_VARIANT_ATTRIBUTE_POINTS, PIPELINE_VARIANT_ATTRIBUTE_LINES, PIPELINE_VARIANT_ATTRIBUTE_LINES_STRIP, PIPELINE_VARIANT_ATTRIBUTE_TRIANGLES, PIPELINE_VARIANT_ATTRIBUTE_TRIANGLE_STRIP };

				push_constant.modulation[0] = base_color.r * modulate.r;
				push_constant.modulation[1] = base_color.g * modulate.g;
				push_constant.modulation[2] = base_color.b * modulate.b;
				push_constant.modulation[3] = base_color.a * modulate.a;

				for (int j = 0; j < 4; j++) {
					push_constant.src_rect[j] = 0;
					push_constant.dst_rect[j] = 0;
					push_constant.ninepatch_margins[j] = 0;
				}

				for (uint32_t j = 0; j < surf_count; j++) {
					void *surface = storage->mesh_get_surface(mesh, j);

					RS::PrimitiveType primitive = storage->mesh_surface_get_primitive(surface);
					ERR_CONTINUE(primitive < 0 || primitive >= RS::PRIMITIVE_MAX);

					uint32_t input_mask = pipeline_variants->variants[light_mode][variant[primitive]].get_vertex_input_mask();

					RID vertex_array;
					RD::VertexFormatID vertex_format = RD::INVALID_FORMAT_ID;

					if (mesh_instance.is_valid()) {
						storage->mesh_instance_surface_get_vertex_arrays_and_format(mesh_instance, j, input_mask, vertex_array, vertex_format);
					} else {
						storage->mesh_surface_get_vertex_arrays_and_format(surface, input_mask, vertex_array, vertex_format);
					}

					RID pipeline = pipeline_variants->variants[light_mode][variant[primitive]].get_render_pipeline(vertex_format, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);

					RID index_array = storage->mesh_surface_get_index_array(surface, 0);

					if (index_array.is_valid()) {
						RD::get_singleton()->draw_list_bind_index_array(p_draw_list, index_array);
					}

					RD::get_singleton()->draw_list_bind_vertex_array(p_draw_list, vertex_array);
					RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));

					RD::get_singleton()->draw_list_draw(p_draw_list, index_array.is_valid(), instance_count);
				}

				for (int j = 0; j < 6; j++) {
					push_constant.world[j] = world_backup[j];
				}
			} break;
			case Item::Command::TYPE_TRANSFORM: {
				const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
				draw_transform = transform->xform;
				_update_transform_2d_to_mat2x3(base_transform * transform->xform, push_constant.world);

			} break;
			case Item::Command::TYPE_CLIP_IGNORE: {
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

			} break;
			case Item::Command::TYPE_ANIMATION_SLICE: {
				const Item::CommandAnimationSlice *as = static_cast<const Item::CommandAnimationSlice *>(c);
				double current_time = RendererCompositorRD::singleton->get_total_time();
				double local_time = Math::fposmod(current_time - as->offset, as->animation_length);
				skipping = !(local_time >= as->slice_begin && local_time < as->slice_end);

				RenderingServerDefault::redraw_request(); // animation visible means redraw request
			} break;
		}

		c = c->next;
	}

	if (current_clip && reclip) {
		//will make it re-enable clipping if needed afterwards
		current_clip = nullptr;
	}
}

RID RendererCanvasRenderRD::_create_base_uniform_set(RID p_to_render_target, bool p_backbuffer) {
	//re create canvas state
	Vector<RD::Uniform> uniforms;

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.binding = 1;
		u.ids.push_back(state.canvas_state_buffer);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.binding = 2;
		u.ids.push_back(state.lights_uniform_buffer);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 3;
		u.ids.push_back(storage->decal_atlas_get_texture());
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 4;
		u.ids.push_back(state.shadow_texture);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 5;
		u.ids.push_back(state.shadow_sampler);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 6;
		RID screen;
		if (p_backbuffer) {
			screen = storage->render_target_get_rd_texture(p_to_render_target);
		} else {
			screen = storage->render_target_get_rd_backbuffer(p_to_render_target);
			if (screen.is_null()) { //unallocated backbuffer
				screen = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE);
			}
		}
		u.ids.push_back(screen);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 7;
		RID sdf = storage->render_target_get_sdf_texture(p_to_render_target);
		u.ids.push_back(sdf);
		uniforms.push_back(u);
	}

	{
		//needs samplers for the material (uses custom textures) create them
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 8;
		u.ids.resize(12);
		RID *ids_ptr = u.ids.ptrw();
		ids_ptr[0] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		ids_ptr[1] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		ids_ptr[2] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		ids_ptr[3] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		ids_ptr[4] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		ids_ptr[5] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
		ids_ptr[6] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		ids_ptr[7] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		ids_ptr[8] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		ids_ptr[9] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		ids_ptr[10] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		ids_ptr[11] = storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.binding = 9;
		u.ids.push_back(storage->global_variables_get_storage_buffer());
		uniforms.push_back(u);
	}

	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader.default_version_rd_shader, BASE_UNIFORM_SET);
	if (p_backbuffer) {
		storage->render_target_set_backbuffer_uniform_set(p_to_render_target, uniform_set);
	} else {
		storage->render_target_set_framebuffer_uniform_set(p_to_render_target, uniform_set);
	}

	return uniform_set;
}

void RendererCanvasRenderRD::_render_items(RID p_to_render_target, int p_item_count, const Transform2D &p_canvas_transform_inverse, Light *p_lights, bool p_to_backbuffer) {
	Item *current_clip = nullptr;

	Transform2D canvas_transform_inverse = p_canvas_transform_inverse;

	RID framebuffer;
	RID fb_uniform_set;
	bool clear = false;
	Vector<Color> clear_colors;

	if (p_to_backbuffer) {
		framebuffer = storage->render_target_get_rd_backbuffer_framebuffer(p_to_render_target);
		fb_uniform_set = storage->render_target_get_backbuffer_uniform_set(p_to_render_target);
	} else {
		framebuffer = storage->render_target_get_rd_framebuffer(p_to_render_target);

		if (storage->render_target_is_clear_requested(p_to_render_target)) {
			clear = true;
			clear_colors.push_back(storage->render_target_get_clear_request_color(p_to_render_target));
			storage->render_target_disable_clear_request(p_to_render_target);
		}
#ifndef _MSC_VER
#warning TODO obtain from framebuffer format eventually when this is implemented
#endif

		fb_uniform_set = storage->render_target_get_framebuffer_uniform_set(p_to_render_target);
	}

	if (fb_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(fb_uniform_set)) {
		fb_uniform_set = _create_base_uniform_set(p_to_render_target, p_to_backbuffer);
	}

	RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(framebuffer);

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer, clear ? RD::INITIAL_ACTION_CLEAR : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, clear_colors);

	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, fb_uniform_set, BASE_UNIFORM_SET);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, state.default_transforms_uniform_set, TRANSFORMS_UNIFORM_SET);

	RID prev_material;

	PipelineVariants *pipeline_variants = &shader.pipeline_variants;

	for (int i = 0; i < p_item_count; i++) {
		Item *ci = items[i];

		if (current_clip != ci->final_clip_owner) {
			current_clip = ci->final_clip_owner;

			//setup clip
			if (current_clip) {
				RD::get_singleton()->draw_list_enable_scissor(draw_list, current_clip->final_clip_rect);

			} else {
				RD::get_singleton()->draw_list_disable_scissor(draw_list);
			}
		}

		RID material = ci->material_owner == nullptr ? ci->material : ci->material_owner->material;

		if (material.is_null() && ci->canvas_group != nullptr) {
			material = default_canvas_group_material;
		}

		if (material != prev_material) {
			MaterialData *material_data = nullptr;
			if (material.is_valid()) {
				material_data = (MaterialData *)storage->material_get_data(material, RendererStorageRD::SHADER_TYPE_2D);
			}

			if (material_data) {
				if (material_data->shader_data->version.is_valid() && material_data->shader_data->valid) {
					pipeline_variants = &material_data->shader_data->pipeline_variants;
					// Update uniform set.
					if (material_data->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(material_data->uniform_set)) { // Material may not have a uniform set.
						RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_data->uniform_set, MATERIAL_UNIFORM_SET);
					}
				} else {
					pipeline_variants = &shader.pipeline_variants;
				}
			} else {
				pipeline_variants = &shader.pipeline_variants;
			}
		}

		_render_item(draw_list, p_to_render_target, ci, fb_format, canvas_transform_inverse, current_clip, p_lights, pipeline_variants);

		prev_material = material;
	}

	RD::get_singleton()->draw_list_end();
}

void RendererCanvasRenderRD::canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_light_list, const Transform2D &p_canvas_transform, RenderingServer::CanvasItemTextureFilter p_default_filter, RenderingServer::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) {
	r_sdf_used = false;
	int item_count = 0;

	//setup canvas state uniforms if needed

	Transform2D canvas_transform_inverse = p_canvas_transform.affine_inverse();

	//setup directional lights if exist

	uint32_t light_count = 0;
	uint32_t directional_light_count = 0;
	{
		Light *l = p_directional_light_list;
		uint32_t index = 0;

		while (l) {
			if (index == state.max_lights_per_render) {
				l->render_index_cache = -1;
				l = l->next_ptr;
				continue;
			}

			CanvasLight *clight = canvas_light_owner.get_or_null(l->light_internal);
			if (!clight) { //unused or invalid texture
				l->render_index_cache = -1;
				l = l->next_ptr;
				ERR_CONTINUE(!clight);
			}

			Vector2 canvas_light_dir = l->xform_cache.elements[1].normalized();

			state.light_uniforms[index].position[0] = -canvas_light_dir.x;
			state.light_uniforms[index].position[1] = -canvas_light_dir.y;

			_update_transform_2d_to_mat2x4(clight->shadow.directional_xform, state.light_uniforms[index].shadow_matrix);

			state.light_uniforms[index].height = l->height; //0..1 here

			for (int i = 0; i < 4; i++) {
				state.light_uniforms[index].shadow_color[i] = uint8_t(CLAMP(int32_t(l->shadow_color[i] * 255.0), 0, 255));
				state.light_uniforms[index].color[i] = l->color[i];
			}

			state.light_uniforms[index].color[3] = l->energy; //use alpha for energy, so base color can go separate

			if (state.shadow_fb.is_valid()) {
				state.light_uniforms[index].shadow_pixel_size = (1.0 / state.shadow_texture_size) * (1.0 + l->shadow_smooth);
				state.light_uniforms[index].shadow_z_far_inv = 1.0 / clight->shadow.z_far;
				state.light_uniforms[index].shadow_y_ofs = clight->shadow.y_offset;
			} else {
				state.light_uniforms[index].shadow_pixel_size = 1.0;
				state.light_uniforms[index].shadow_z_far_inv = 1.0;
				state.light_uniforms[index].shadow_y_ofs = 0;
			}

			state.light_uniforms[index].flags = l->blend_mode << LIGHT_FLAGS_BLEND_SHIFT;
			state.light_uniforms[index].flags |= l->shadow_filter << LIGHT_FLAGS_FILTER_SHIFT;
			if (clight->shadow.enabled) {
				state.light_uniforms[index].flags |= LIGHT_FLAGS_HAS_SHADOW;
			}

			l->render_index_cache = index;

			index++;
			l = l->next_ptr;
		}

		light_count = index;
		directional_light_count = light_count;
		using_directional_lights = directional_light_count > 0;
	}

	//setup lights if exist

	{
		Light *l = p_light_list;
		uint32_t index = light_count;

		while (l) {
			if (index == state.max_lights_per_render) {
				l->render_index_cache = -1;
				l = l->next_ptr;
				continue;
			}

			CanvasLight *clight = canvas_light_owner.get_or_null(l->light_internal);
			if (!clight) { //unused or invalid texture
				l->render_index_cache = -1;
				l = l->next_ptr;
				ERR_CONTINUE(!clight);
			}
			Transform2D to_light_xform = (p_canvas_transform * l->light_shader_xform).affine_inverse();

			Vector2 canvas_light_pos = p_canvas_transform.xform(l->xform.get_origin()); //convert light position to canvas coordinates, as all computation is done in canvas coords to avoid precision loss
			state.light_uniforms[index].position[0] = canvas_light_pos.x;
			state.light_uniforms[index].position[1] = canvas_light_pos.y;

			_update_transform_2d_to_mat2x4(to_light_xform, state.light_uniforms[index].matrix);
			_update_transform_2d_to_mat2x4(l->xform_cache.affine_inverse(), state.light_uniforms[index].shadow_matrix);

			state.light_uniforms[index].height = l->height * (p_canvas_transform.elements[0].length() + p_canvas_transform.elements[1].length()) * 0.5; //approximate height conversion to the canvas size, since all calculations are done in canvas coords to avoid precision loss
			for (int i = 0; i < 4; i++) {
				state.light_uniforms[index].shadow_color[i] = uint8_t(CLAMP(int32_t(l->shadow_color[i] * 255.0), 0, 255));
				state.light_uniforms[index].color[i] = l->color[i];
			}

			state.light_uniforms[index].color[3] = l->energy; //use alpha for energy, so base color can go separate

			if (state.shadow_fb.is_valid()) {
				state.light_uniforms[index].shadow_pixel_size = (1.0 / state.shadow_texture_size) * (1.0 + l->shadow_smooth);
				state.light_uniforms[index].shadow_z_far_inv = 1.0 / clight->shadow.z_far;
				state.light_uniforms[index].shadow_y_ofs = clight->shadow.y_offset;
			} else {
				state.light_uniforms[index].shadow_pixel_size = 1.0;
				state.light_uniforms[index].shadow_z_far_inv = 1.0;
				state.light_uniforms[index].shadow_y_ofs = 0;
			}

			state.light_uniforms[index].flags = l->blend_mode << LIGHT_FLAGS_BLEND_SHIFT;
			state.light_uniforms[index].flags |= l->shadow_filter << LIGHT_FLAGS_FILTER_SHIFT;
			if (clight->shadow.enabled) {
				state.light_uniforms[index].flags |= LIGHT_FLAGS_HAS_SHADOW;
			}

			if (clight->texture.is_valid()) {
				Rect2 atlas_rect = storage->decal_atlas_get_texture_rect(clight->texture);
				state.light_uniforms[index].atlas_rect[0] = atlas_rect.position.x;
				state.light_uniforms[index].atlas_rect[1] = atlas_rect.position.y;
				state.light_uniforms[index].atlas_rect[2] = atlas_rect.size.width;
				state.light_uniforms[index].atlas_rect[3] = atlas_rect.size.height;

			} else {
				state.light_uniforms[index].atlas_rect[0] = 0;
				state.light_uniforms[index].atlas_rect[1] = 0;
				state.light_uniforms[index].atlas_rect[2] = 0;
				state.light_uniforms[index].atlas_rect[3] = 0;
			}

			l->render_index_cache = index;

			index++;
			l = l->next_ptr;
		}

		light_count = index;
	}

	if (light_count > 0) {
		RD::get_singleton()->buffer_update(state.lights_uniform_buffer, 0, sizeof(LightUniform) * light_count, &state.light_uniforms[0]);
	}

	{
		//update canvas state uniform buffer
		State::Buffer state_buffer;

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

		state_buffer.time = state.time;
		state_buffer.use_pixel_snap = p_snap_2d_vertices_to_pixel;

		state_buffer.directional_light_count = directional_light_count;

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

		RD::get_singleton()->buffer_update(state.canvas_state_buffer, 0, sizeof(State::Buffer), &state_buffer);
	}

	{ //default filter/repeat
		default_filter = p_default_filter;
		default_repeat = p_default_repeat;
	}

	//fill the list until rendering is possible.
	bool material_screen_texture_found = false;
	Item *ci = p_item_list;
	Rect2 back_buffer_rect;
	bool backbuffer_copy = false;

	Item *canvas_group_owner = nullptr;

	bool update_skeletons = false;
	bool time_used = false;

	while (ci) {
		if (ci->copy_back_buffer && canvas_group_owner == nullptr) {
			backbuffer_copy = true;

			if (ci->copy_back_buffer->full) {
				back_buffer_rect = Rect2();
			} else {
				back_buffer_rect = ci->copy_back_buffer->rect;
			}
		}

		RID material = ci->material_owner == nullptr ? ci->material : ci->material_owner->material;

		if (material.is_valid()) {
			MaterialData *md = (MaterialData *)storage->material_get_data(material, RendererStorageRD::SHADER_TYPE_2D);
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

		if (ci->skeleton.is_valid()) {
			const Item::Command *c = ci->commands;

			while (c) {
				if (c->type == Item::Command::TYPE_MESH) {
					const Item::CommandMesh *cm = static_cast<const Item::CommandMesh *>(c);
					if (cm->mesh_instance.is_valid()) {
						storage->mesh_instance_check_for_update(cm->mesh_instance);
						update_skeletons = true;
					}
				}
				c = c->next;
			}
		}

		if (ci->canvas_group_owner != nullptr) {
			if (canvas_group_owner == nullptr) {
				//Canvas group begins here, render until before this item
				if (update_skeletons) {
					storage->update_mesh_instances();
					update_skeletons = false;
				}
				_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
				item_count = 0;

				Rect2i group_rect = ci->canvas_group_owner->global_rect_cache;

				if (ci->canvas_group_owner->canvas_group->mode == RS::CANVAS_GROUP_MODE_OPAQUE) {
					storage->render_target_copy_to_back_buffer(p_to_render_target, group_rect, false);
				} else {
					storage->render_target_clear_back_buffer(p_to_render_target, group_rect, Color(0, 0, 0, 0));
				}

				backbuffer_copy = false;
				canvas_group_owner = ci->canvas_group_owner; //continue until owner found
			}

			ci->canvas_group_owner = nullptr; //must be cleared
		}

		if (ci == canvas_group_owner) {
			if (update_skeletons) {
				storage->update_mesh_instances();
				update_skeletons = false;
			}

			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list, true);
			item_count = 0;

			if (ci->canvas_group->blur_mipmaps) {
				storage->render_target_gen_back_buffer_mipmaps(p_to_render_target, ci->global_rect_cache);
			}

			canvas_group_owner = nullptr;
		}

		if (backbuffer_copy) {
			//render anything pending, including clearing if no items
			if (update_skeletons) {
				storage->update_mesh_instances();
				update_skeletons = false;
			}
			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
			item_count = 0;

			storage->render_target_copy_to_back_buffer(p_to_render_target, back_buffer_rect, true);

			backbuffer_copy = false;
			material_screen_texture_found = true; //after a backbuffer copy, screen texture makes no further copies
		}

		items[item_count++] = ci;

		if (!ci->next || item_count == MAX_RENDER_ITEMS - 1) {
			if (update_skeletons) {
				storage->update_mesh_instances();
				update_skeletons = false;
			}

			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list);
			//then reset
			item_count = 0;
		}

		ci = ci->next;
	}

	if (time_used) {
		RenderingServerDefault::redraw_request();
	}
}

RID RendererCanvasRenderRD::light_create() {
	CanvasLight canvas_light;
	return canvas_light_owner.make_rid(canvas_light);
}

void RendererCanvasRenderRD::light_set_texture(RID p_rid, RID p_texture) {
	CanvasLight *cl = canvas_light_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!cl);
	if (cl->texture == p_texture) {
		return;
	}
	if (cl->texture.is_valid()) {
		storage->texture_remove_from_decal_atlas(cl->texture);
	}
	cl->texture = p_texture;

	if (cl->texture.is_valid()) {
		storage->texture_add_to_decal_atlas(cl->texture);
	}
}

void RendererCanvasRenderRD::light_set_use_shadow(RID p_rid, bool p_enable) {
	CanvasLight *cl = canvas_light_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!cl);

	cl->shadow.enabled = p_enable;
}

void RendererCanvasRenderRD::_update_shadow_atlas() {
	if (state.shadow_fb == RID()) {
		//ah, we lack the shadow texture..
		RD::get_singleton()->free(state.shadow_texture); //erase placeholder

		Vector<RID> fb_textures;

		{ //texture
			RD::TextureFormat tf;
			tf.texture_type = RD::TEXTURE_TYPE_2D;
			tf.width = state.shadow_texture_size;
			tf.height = state.max_lights_per_render * 2;
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;

			state.shadow_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			fb_textures.push_back(state.shadow_texture);
		}
		{
			RD::TextureFormat tf;
			tf.texture_type = RD::TEXTURE_TYPE_2D;
			tf.width = state.shadow_texture_size;
			tf.height = state.max_lights_per_render * 2;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			tf.format = RD::DATA_FORMAT_D32_SFLOAT;
			//chunks to write
			state.shadow_depth_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			fb_textures.push_back(state.shadow_depth_texture);
		}

		state.shadow_fb = RD::get_singleton()->framebuffer_create(fb_textures);
	}
}
void RendererCanvasRenderRD::light_update_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) {
	CanvasLight *cl = canvas_light_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!cl->shadow.enabled);

	_update_shadow_atlas();

	cl->shadow.z_far = p_far;
	cl->shadow.y_offset = float(p_shadow_index * 2 + 1) / float(state.max_lights_per_render * 2);
	Vector<Color> cc;
	cc.push_back(Color(p_far, p_far, p_far, 1.0));

	for (int i = 0; i < 4; i++) {
		//make sure it remains orthogonal, makes easy to read angle later

		//light.basis.scale(Vector3(to_light.elements[0].length(),to_light.elements[1].length(),1));

		Rect2i rect((state.shadow_texture_size / 4) * i, p_shadow_index * 2, (state.shadow_texture_size / 4), 2);
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(state.shadow_fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, cc, 1.0, 0, rect);

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

		Vector3 cam_target = Basis(Vector3(0, 0, Math_TAU * ((i + 3) / 4.0))).xform(Vector3(0, 1, 0));
		projection = projection * CameraMatrix(Transform3D().looking_at(cam_target, Vector3(0, 0, -1)).affine_inverse());

		ShadowRenderPushConstant push_constant;
		for (int y = 0; y < 4; y++) {
			for (int x = 0; x < 4; x++) {
				push_constant.projection[y * 4 + x] = projection.matrix[y][x];
			}
		}
		static const Vector2 directions[4] = { Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1) };
		push_constant.direction[0] = directions[i].x;
		push_constant.direction[1] = directions[i].y;
		push_constant.z_far = p_far;
		push_constant.pad = 0;

		LightOccluderInstance *instance = p_occluders;

		while (instance) {
			OccluderPolygon *co = occluder_polygon_owner.get_or_null(instance->occluder);

			if (!co || co->index_array.is_null() || !(p_light_mask & instance->light_mask)) {
				instance = instance->next;
				continue;
			}

			_update_transform_2d_to_mat2x4(p_light_xform * instance->xform_cache, push_constant.modelview);

			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, shadow_render.render_pipelines[co->cull_mode]);
			RD::get_singleton()->draw_list_bind_vertex_array(draw_list, co->vertex_array);
			RD::get_singleton()->draw_list_bind_index_array(draw_list, co->index_array);
			RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(ShadowRenderPushConstant));

			RD::get_singleton()->draw_list_draw(draw_list, true);

			instance = instance->next;
		}

		RD::get_singleton()->draw_list_end();
	}
}

void RendererCanvasRenderRD::light_update_directional_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_cull_distance, const Rect2 &p_clip_rect, LightOccluderInstance *p_occluders) {
	CanvasLight *cl = canvas_light_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!cl->shadow.enabled);

	_update_shadow_atlas();

	Vector2 light_dir = p_light_xform.elements[1].normalized();

	Vector2 center = p_clip_rect.get_center();

	float to_edge_distance = ABS(light_dir.dot(p_clip_rect.get_support(light_dir)) - light_dir.dot(center));

	Vector2 from_pos = center - light_dir * (to_edge_distance + p_cull_distance);
	float distance = to_edge_distance * 2.0 + p_cull_distance;
	float half_size = p_clip_rect.size.length() * 0.5; //shadow length, must keep this no matter the angle

	cl->shadow.z_far = distance;
	cl->shadow.y_offset = float(p_shadow_index * 2 + 1) / float(state.max_lights_per_render * 2);

	Transform2D to_light_xform;

	to_light_xform[2] = from_pos;
	to_light_xform[1] = light_dir;
	to_light_xform[0] = -light_dir.orthogonal();

	to_light_xform.invert();

	Vector<Color> cc;
	cc.push_back(Color(1, 1, 1, 1));

	Rect2i rect(0, p_shadow_index * 2, state.shadow_texture_size, 2);
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(state.shadow_fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, cc, 1.0, 0, rect);

	CameraMatrix projection;
	projection.set_orthogonal(-half_size, half_size, -0.5, 0.5, 0.0, distance);
	projection = projection * CameraMatrix(Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, -1)).affine_inverse());

	ShadowRenderPushConstant push_constant;
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			push_constant.projection[y * 4 + x] = projection.matrix[y][x];
		}
	}

	push_constant.direction[0] = 0.0;
	push_constant.direction[1] = 1.0;
	push_constant.z_far = distance;
	push_constant.pad = 0;

	LightOccluderInstance *instance = p_occluders;

	while (instance) {
		OccluderPolygon *co = occluder_polygon_owner.get_or_null(instance->occluder);

		if (!co || co->index_array.is_null() || !(p_light_mask & instance->light_mask)) {
			instance = instance->next;
			continue;
		}

		_update_transform_2d_to_mat2x4(to_light_xform * instance->xform_cache, push_constant.modelview);

		RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, shadow_render.render_pipelines[co->cull_mode]);
		RD::get_singleton()->draw_list_bind_vertex_array(draw_list, co->vertex_array);
		RD::get_singleton()->draw_list_bind_index_array(draw_list, co->index_array);
		RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(ShadowRenderPushConstant));

		RD::get_singleton()->draw_list_draw(draw_list, true);

		instance = instance->next;
	}

	RD::get_singleton()->draw_list_end();

	Transform2D to_shadow;
	to_shadow.elements[0].x = 1.0 / -(half_size * 2.0);
	to_shadow.elements[2].x = 0.5;

	cl->shadow.directional_xform = to_shadow * to_light_xform;
}

void RendererCanvasRenderRD::render_sdf(RID p_render_target, LightOccluderInstance *p_occluders) {
	RID fb = storage->render_target_get_sdf_framebuffer(p_render_target);
	Rect2i rect = storage->render_target_get_sdf_rect(p_render_target);

	Transform2D to_sdf;
	to_sdf.elements[0] *= rect.size.width;
	to_sdf.elements[1] *= rect.size.height;
	to_sdf.elements[2] = rect.position;

	Transform2D to_clip;
	to_clip.elements[0] *= 2.0;
	to_clip.elements[1] *= 2.0;
	to_clip.elements[2] = -Vector2(1.0, 1.0);

	to_clip = to_clip * to_sdf.affine_inverse();

	Vector<Color> cc;
	cc.push_back(Color(0, 0, 0, 0));

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, cc);

	CameraMatrix projection;

	ShadowRenderPushConstant push_constant;
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			push_constant.projection[y * 4 + x] = projection.matrix[y][x];
		}
	}

	push_constant.direction[0] = 0.0;
	push_constant.direction[1] = 0.0;
	push_constant.z_far = 0;
	push_constant.pad = 0;

	LightOccluderInstance *instance = p_occluders;

	while (instance) {
		OccluderPolygon *co = occluder_polygon_owner.get_or_null(instance->occluder);

		if (!co || co->sdf_index_array.is_null()) {
			instance = instance->next;
			continue;
		}

		_update_transform_2d_to_mat2x4(to_clip * instance->xform_cache, push_constant.modelview);

		RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, shadow_render.sdf_render_pipelines[co->sdf_is_lines ? SHADOW_RENDER_SDF_LINES : SHADOW_RENDER_SDF_TRIANGLES]);
		RD::get_singleton()->draw_list_bind_vertex_array(draw_list, co->sdf_vertex_array);
		RD::get_singleton()->draw_list_bind_index_array(draw_list, co->sdf_index_array);
		RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(ShadowRenderPushConstant));

		RD::get_singleton()->draw_list_draw(draw_list, true);

		instance = instance->next;
	}

	RD::get_singleton()->draw_list_end();

	storage->render_target_sdf_process(p_render_target); //done rendering, process it
}

RID RendererCanvasRenderRD::occluder_polygon_create() {
	OccluderPolygon occluder;
	occluder.line_point_count = 0;
	occluder.sdf_point_count = 0;
	occluder.sdf_index_count = 0;
	occluder.cull_mode = RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;
	return occluder_polygon_owner.make_rid(occluder);
}

void RendererCanvasRenderRD::occluder_polygon_set_shape(RID p_occluder, const Vector<Vector2> &p_points, bool p_closed) {
	OccluderPolygon *oc = occluder_polygon_owner.get_or_null(p_occluder);
	ERR_FAIL_COND(!oc);

	Vector<Vector2> lines;

	if (p_points.size()) {
		int lc = p_points.size() * 2;

		lines.resize(lc - (p_closed ? 0 : 2));
		{
			Vector2 *w = lines.ptrw();
			const Vector2 *r = p_points.ptr();

			int max = lc / 2;
			if (!p_closed) {
				max--;
			}
			for (int i = 0; i < max; i++) {
				Vector2 a = r[i];
				Vector2 b = r[(i + 1) % (lc / 2)];
				w[i * 2 + 0] = a;
				w[i * 2 + 1] = b;
			}
		}
	}

	if (oc->line_point_count != lines.size() && oc->vertex_array.is_valid()) {
		RD::get_singleton()->free(oc->vertex_array);
		RD::get_singleton()->free(oc->vertex_buffer);
		RD::get_singleton()->free(oc->index_array);
		RD::get_singleton()->free(oc->index_buffer);

		oc->vertex_array = RID();
		oc->vertex_buffer = RID();
		oc->index_array = RID();
		oc->index_buffer = RID();

		oc->line_point_count = lines.size();
	}

	if (lines.size()) {
		Vector<uint8_t> geometry;
		Vector<uint8_t> indices;
		int lc = lines.size();

		geometry.resize(lc * 6 * sizeof(float));
		indices.resize(lc * 3 * sizeof(uint16_t));

		{
			uint8_t *vw = geometry.ptrw();
			float *vwptr = (float *)vw;
			uint8_t *iw = indices.ptrw();
			uint16_t *iwptr = (uint16_t *)iw;

			const Vector2 *lr = lines.ptr();

			const int POLY_HEIGHT = 16384;

			for (int i = 0; i < lc / 2; i++) {
				vwptr[i * 12 + 0] = lr[i * 2 + 0].x;
				vwptr[i * 12 + 1] = lr[i * 2 + 0].y;
				vwptr[i * 12 + 2] = POLY_HEIGHT;

				vwptr[i * 12 + 3] = lr[i * 2 + 1].x;
				vwptr[i * 12 + 4] = lr[i * 2 + 1].y;
				vwptr[i * 12 + 5] = POLY_HEIGHT;

				vwptr[i * 12 + 6] = lr[i * 2 + 1].x;
				vwptr[i * 12 + 7] = lr[i * 2 + 1].y;
				vwptr[i * 12 + 8] = -POLY_HEIGHT;

				vwptr[i * 12 + 9] = lr[i * 2 + 0].x;
				vwptr[i * 12 + 10] = lr[i * 2 + 0].y;
				vwptr[i * 12 + 11] = -POLY_HEIGHT;

				iwptr[i * 6 + 0] = i * 4 + 0;
				iwptr[i * 6 + 1] = i * 4 + 1;
				iwptr[i * 6 + 2] = i * 4 + 2;

				iwptr[i * 6 + 3] = i * 4 + 2;
				iwptr[i * 6 + 4] = i * 4 + 3;
				iwptr[i * 6 + 5] = i * 4 + 0;
			}
		}

		//if same buffer len is being set, just use BufferSubData to avoid a pipeline flush

		if (oc->vertex_array.is_null()) {
			//create from scratch
			//vertices
			oc->vertex_buffer = RD::get_singleton()->vertex_buffer_create(lc * 6 * sizeof(real_t), geometry);

			Vector<RID> buffer;
			buffer.push_back(oc->vertex_buffer);
			oc->vertex_array = RD::get_singleton()->vertex_array_create(4 * lc / 2, shadow_render.vertex_format, buffer);
			//indices

			oc->index_buffer = RD::get_singleton()->index_buffer_create(3 * lc, RD::INDEX_BUFFER_FORMAT_UINT16, indices);
			oc->index_array = RD::get_singleton()->index_array_create(oc->index_buffer, 0, 3 * lc);

		} else {
			//update existing
			const uint8_t *vr = geometry.ptr();
			RD::get_singleton()->buffer_update(oc->vertex_buffer, 0, geometry.size(), vr);
			const uint8_t *ir = indices.ptr();
			RD::get_singleton()->buffer_update(oc->index_buffer, 0, indices.size(), ir);
		}
	}

	// sdf

	Vector<int> sdf_indices;

	if (p_points.size()) {
		if (p_closed) {
			sdf_indices = Geometry2D::triangulate_polygon(p_points);
			oc->sdf_is_lines = false;
		} else {
			int max = p_points.size();
			sdf_indices.resize(max * 2);

			int *iw = sdf_indices.ptrw();
			for (int i = 0; i < max; i++) {
				iw[i * 2 + 0] = i;
				iw[i * 2 + 1] = (i + 1) % max;
			}
			oc->sdf_is_lines = true;
		}
	}

	if (oc->sdf_index_count != sdf_indices.size() && oc->sdf_point_count != p_points.size() && oc->sdf_vertex_array.is_valid()) {
		RD::get_singleton()->free(oc->sdf_vertex_array);
		RD::get_singleton()->free(oc->sdf_vertex_buffer);
		RD::get_singleton()->free(oc->sdf_index_array);
		RD::get_singleton()->free(oc->sdf_index_buffer);

		oc->sdf_vertex_array = RID();
		oc->sdf_vertex_buffer = RID();
		oc->sdf_index_array = RID();
		oc->sdf_index_buffer = RID();

		oc->sdf_index_count = sdf_indices.size();
		oc->sdf_point_count = p_points.size();

		oc->sdf_is_lines = false;
	}

	if (sdf_indices.size()) {
		if (oc->sdf_vertex_array.is_null()) {
			//create from scratch
			//vertices
			oc->sdf_vertex_buffer = RD::get_singleton()->vertex_buffer_create(p_points.size() * 2 * sizeof(real_t), p_points.to_byte_array());
			oc->sdf_index_buffer = RD::get_singleton()->index_buffer_create(sdf_indices.size(), RD::INDEX_BUFFER_FORMAT_UINT32, sdf_indices.to_byte_array());
			oc->sdf_index_array = RD::get_singleton()->index_array_create(oc->sdf_index_buffer, 0, sdf_indices.size());

			Vector<RID> buffer;
			buffer.push_back(oc->sdf_vertex_buffer);
			oc->sdf_vertex_array = RD::get_singleton()->vertex_array_create(p_points.size(), shadow_render.sdf_vertex_format, buffer);
			//indices

		} else {
			//update existing
			RD::get_singleton()->buffer_update(oc->vertex_buffer, 0, sizeof(real_t) * 2 * p_points.size(), p_points.ptr());
			RD::get_singleton()->buffer_update(oc->index_buffer, 0, sdf_indices.size() * sizeof(int32_t), sdf_indices.ptr());
		}
	}
}

void RendererCanvasRenderRD::occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) {
	OccluderPolygon *oc = occluder_polygon_owner.get_or_null(p_occluder);
	ERR_FAIL_COND(!oc);
	oc->cull_mode = p_mode;
}

void RendererCanvasRenderRD::ShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_screen_texture = false;
	uses_sdf = false;
	uses_time = false;

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;

	int blend_mode = BLEND_MODE_MIX;
	uses_screen_texture = false;

	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);
	actions.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&blend_mode, BLEND_MODE_PMALPHA);
	actions.render_mode_values["blend_disabled"] = Pair<int *, int>(&blend_mode, BLEND_MODE_DISABLED);

	actions.usage_flag_pointers["SCREEN_TEXTURE"] = &uses_screen_texture;
	actions.usage_flag_pointers["texture_sdf"] = &uses_sdf;
	actions.usage_flag_pointers["TIME"] = &uses_time;

	actions.uniforms = &uniforms;

	RendererCanvasRenderRD *canvas_singleton = (RendererCanvasRenderRD *)RendererCanvasRender::singleton;

	Error err = canvas_singleton->shader.compiler.compile(RS::SHADER_CANVAS_ITEM, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = canvas_singleton->shader.canvas_shader.version_create();
	}

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}
	print_line("\n**uniforms:\n" + gen_code.uniforms);
	print_line("\n**vertex_globals:\n" + gen_code.vertex_global);
	print_line("\n**vertex_code:\n" + gen_code.vertex);
	print_line("\n**fragment_globals:\n" + gen_code.fragment_global);
	print_line("\n**fragment_code:\n" + gen_code.fragment);
	print_line("\n**light_code:\n" + gen_code.light);
#endif
	canvas_singleton->shader.canvas_shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines);
	ERR_FAIL_COND(!canvas_singleton->shader.canvas_shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//update them pipelines

	RD::PipelineColorBlendState::Attachment attachment;

	switch (blend_mode) {
		case BLEND_MODE_DISABLED: {
			// nothing to do here, disabled by default

		} break;
		case BLEND_MODE_MIX: {
			attachment.enable_blend = true;
			attachment.color_blend_op = RD::BLEND_OP_ADD;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

			attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		} break;
		case BLEND_MODE_ADD: {
			attachment.enable_blend = true;
			attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			attachment.color_blend_op = RD::BLEND_OP_ADD;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;

		} break;
		case BLEND_MODE_SUB: {
			attachment.enable_blend = true;
			attachment.alpha_blend_op = RD::BLEND_OP_SUBTRACT;
			attachment.color_blend_op = RD::BLEND_OP_SUBTRACT;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;

		} break;
		case BLEND_MODE_MUL: {
			attachment.enable_blend = true;
			attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			attachment.color_blend_op = RD::BLEND_OP_ADD;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_DST_COLOR;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ZERO;
			attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_DST_ALPHA;
			attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;

		} break;
		case BLEND_MODE_PMALPHA: {
			attachment.enable_blend = true;
			attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			attachment.color_blend_op = RD::BLEND_OP_ADD;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		} break;
	}

	RD::PipelineColorBlendState blend_state;
	blend_state.attachments.push_back(attachment);

	//update pipelines

	for (int i = 0; i < PIPELINE_LIGHT_MODE_MAX; i++) {
		for (int j = 0; j < PIPELINE_VARIANT_MAX; j++) {
			RD::RenderPrimitive primitive[PIPELINE_VARIANT_MAX] = {
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::RENDER_PRIMITIVE_LINES,
				RD::RENDER_PRIMITIVE_POINTS,
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS,
				RD::RENDER_PRIMITIVE_LINES,
				RD::RENDER_PRIMITIVE_LINESTRIPS,
				RD::RENDER_PRIMITIVE_POINTS,
			};

			ShaderVariant shader_variants[PIPELINE_LIGHT_MODE_MAX][PIPELINE_VARIANT_MAX] = {
				{ //non lit
						SHADER_VARIANT_QUAD,
						SHADER_VARIANT_NINEPATCH,
						SHADER_VARIANT_PRIMITIVE,
						SHADER_VARIANT_PRIMITIVE,
						SHADER_VARIANT_PRIMITIVE_POINTS,
						SHADER_VARIANT_ATTRIBUTES,
						SHADER_VARIANT_ATTRIBUTES,
						SHADER_VARIANT_ATTRIBUTES,
						SHADER_VARIANT_ATTRIBUTES,
						SHADER_VARIANT_ATTRIBUTES_POINTS },
				{ //lit
						SHADER_VARIANT_QUAD_LIGHT,
						SHADER_VARIANT_NINEPATCH_LIGHT,
						SHADER_VARIANT_PRIMITIVE_LIGHT,
						SHADER_VARIANT_PRIMITIVE_LIGHT,
						SHADER_VARIANT_PRIMITIVE_POINTS_LIGHT,
						SHADER_VARIANT_ATTRIBUTES_LIGHT,
						SHADER_VARIANT_ATTRIBUTES_LIGHT,
						SHADER_VARIANT_ATTRIBUTES_LIGHT,
						SHADER_VARIANT_ATTRIBUTES_LIGHT,
						SHADER_VARIANT_ATTRIBUTES_POINTS_LIGHT },
			};

			RID shader_variant = canvas_singleton->shader.canvas_shader.version_get_shader(version, shader_variants[i][j]);
			pipeline_variants.variants[i][j].setup(shader_variant, primitive[j], RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), blend_state, 0);
		}
	}

	valid = true;
}

void RendererCanvasRenderRD::ShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
	if (!p_texture.is_valid()) {
		if (default_texture_params.has(p_name) && default_texture_params[p_name].has(p_index)) {
			default_texture_params[p_name].erase(p_index);

			if (default_texture_params[p_name].is_empty()) {
				default_texture_params.erase(p_name);
			}
		}
	} else {
		if (!default_texture_params.has(p_name)) {
			default_texture_params[p_name] = Map<int, RID>();
		}
		default_texture_params[p_name][p_index] = p_texture;
	}
}

void RendererCanvasRenderRD::ShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_LOCAL) {
			continue;
		}
		if (E.value.texture_order >= 0) {
			order[E.value.texture_order + 100000] = E.key;
		} else {
			order[E.value.order] = E.key;
		}
	}

	for (const KeyValue<int, StringName> &E : order) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E.value]);
		pi.name = E.value;
		p_param_list->push_back(pi);
	}
}

void RendererCanvasRenderRD::ShaderData::get_instance_param_list(List<RendererStorage::InstanceShaderParam> *p_param_list) const {
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E.value);
		p.info.name = E.key; //supply name
		p.index = E.value.instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E.value.default_value, E.value.type, E.value.array_size, E.value.hint);
		p_param_list->push_back(p);
	}
}

bool RendererCanvasRenderRD::ShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RendererCanvasRenderRD::ShaderData::is_animated() const {
	return false;
}

bool RendererCanvasRenderRD::ShaderData::casts_shadows() const {
	return false;
}

Variant RendererCanvasRenderRD::ShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode RendererCanvasRenderRD::ShaderData::get_native_source_code() const {
	RendererCanvasRenderRD *canvas_singleton = (RendererCanvasRenderRD *)RendererCanvasRender::singleton;
	return canvas_singleton->shader.canvas_shader.version_get_native_source_code(version);
}

RendererCanvasRenderRD::ShaderData::ShaderData() {
	valid = false;
	uses_screen_texture = false;
	uses_sdf = false;
}

RendererCanvasRenderRD::ShaderData::~ShaderData() {
	RendererCanvasRenderRD *canvas_singleton = (RendererCanvasRenderRD *)RendererCanvasRender::singleton;
	ERR_FAIL_COND(!canvas_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		canvas_singleton->shader.canvas_shader.version_free(version);
	}
}

RendererStorageRD::ShaderData *RendererCanvasRenderRD::_create_shader_func() {
	ShaderData *shader_data = memnew(ShaderData);
	return shader_data;
}

bool RendererCanvasRenderRD::MaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	RendererCanvasRenderRD *canvas_singleton = (RendererCanvasRenderRD *)RendererCanvasRender::singleton;

	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, canvas_singleton->shader.canvas_shader.version_get_shader(shader_data->version, 0), MATERIAL_UNIFORM_SET);
}

RendererCanvasRenderRD::MaterialData::~MaterialData() {
	free_parameters_uniform_set(uniform_set);
}

RendererStorageRD::MaterialData *RendererCanvasRenderRD::_create_material_func(ShaderData *p_shader) {
	MaterialData *material_data = memnew(MaterialData);
	material_data->shader_data = p_shader;
	//update will happen later anyway so do nothing.
	return material_data;
}

void RendererCanvasRenderRD::set_time(double p_time) {
	state.time = p_time;
}

void RendererCanvasRenderRD::update() {
}

RendererCanvasRenderRD::RendererCanvasRenderRD(RendererStorageRD *p_storage) {
	storage = p_storage;

	{ //create default samplers

		default_samplers.default_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
		default_samplers.default_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
	}

	{ //shader variants

		String global_defines;

		uint32_t uniform_max_size = RD::get_singleton()->limit_get(RD::LIMIT_MAX_UNIFORM_BUFFER_SIZE);
		if (uniform_max_size < 65536) {
			//Yes, you guessed right, ARM again
			state.max_lights_per_render = 64;
			global_defines += "#define MAX_LIGHTS 64\n";
		} else {
			state.max_lights_per_render = DEFAULT_MAX_LIGHTS_PER_RENDER;
			global_defines += "#define MAX_LIGHTS " + itos(DEFAULT_MAX_LIGHTS_PER_RENDER) + "\n";
		}

		state.light_uniforms = memnew_arr(LightUniform, state.max_lights_per_render);
		Vector<String> variants;
		//non light variants
		variants.push_back(""); //none by default is first variant
		variants.push_back("#define USE_NINEPATCH\n"); //ninepatch is the second variant
		variants.push_back("#define USE_PRIMITIVE\n"); //primitive is the third
		variants.push_back("#define USE_PRIMITIVE\n#define USE_POINT_SIZE\n"); //points need point size
		variants.push_back("#define USE_ATTRIBUTES\n"); // attributes for vertex arrays
		variants.push_back("#define USE_ATTRIBUTES\n#define USE_POINT_SIZE\n"); //attributes with point size
		//light variants
		variants.push_back("#define USE_LIGHTING\n"); //none by default is first variant
		variants.push_back("#define USE_LIGHTING\n#define USE_NINEPATCH\n"); //ninepatch is the second variant
		variants.push_back("#define USE_LIGHTING\n#define USE_PRIMITIVE\n"); //primitive is the third
		variants.push_back("#define USE_LIGHTING\n#define USE_PRIMITIVE\n#define USE_POINT_SIZE\n"); //points need point size
		variants.push_back("#define USE_LIGHTING\n#define USE_ATTRIBUTES\n"); // attributes for vertex arrays
		variants.push_back("#define USE_LIGHTING\n#define USE_ATTRIBUTES\n#define USE_POINT_SIZE\n"); //attributes with point size

		shader.canvas_shader.initialize(variants, global_defines);

		shader.default_version = shader.canvas_shader.version_create();
		shader.default_version_rd_shader = shader.canvas_shader.version_get_shader(shader.default_version, SHADER_VARIANT_QUAD);

		RD::PipelineColorBlendState blend_state;
		RD::PipelineColorBlendState::Attachment blend_attachment;

		blend_attachment.enable_blend = true;
		blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
		blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
		blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
		blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
		blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		blend_state.attachments.push_back(blend_attachment);

		for (int i = 0; i < PIPELINE_LIGHT_MODE_MAX; i++) {
			for (int j = 0; j < PIPELINE_VARIANT_MAX; j++) {
				RD::RenderPrimitive primitive[PIPELINE_VARIANT_MAX] = {
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_LINES,
					RD::RENDER_PRIMITIVE_POINTS,
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS,
					RD::RENDER_PRIMITIVE_LINES,
					RD::RENDER_PRIMITIVE_LINESTRIPS,
					RD::RENDER_PRIMITIVE_POINTS,
				};

				ShaderVariant shader_variants[PIPELINE_LIGHT_MODE_MAX][PIPELINE_VARIANT_MAX] = {
					{ //non lit
							SHADER_VARIANT_QUAD,
							SHADER_VARIANT_NINEPATCH,
							SHADER_VARIANT_PRIMITIVE,
							SHADER_VARIANT_PRIMITIVE,
							SHADER_VARIANT_PRIMITIVE_POINTS,
							SHADER_VARIANT_ATTRIBUTES,
							SHADER_VARIANT_ATTRIBUTES,
							SHADER_VARIANT_ATTRIBUTES,
							SHADER_VARIANT_ATTRIBUTES,
							SHADER_VARIANT_ATTRIBUTES_POINTS },
					{ //lit
							SHADER_VARIANT_QUAD_LIGHT,
							SHADER_VARIANT_NINEPATCH_LIGHT,
							SHADER_VARIANT_PRIMITIVE_LIGHT,
							SHADER_VARIANT_PRIMITIVE_LIGHT,
							SHADER_VARIANT_PRIMITIVE_POINTS_LIGHT,
							SHADER_VARIANT_ATTRIBUTES_LIGHT,
							SHADER_VARIANT_ATTRIBUTES_LIGHT,
							SHADER_VARIANT_ATTRIBUTES_LIGHT,
							SHADER_VARIANT_ATTRIBUTES_LIGHT,
							SHADER_VARIANT_ATTRIBUTES_POINTS_LIGHT },
				};

				RID shader_variant = shader.canvas_shader.version_get_shader(shader.default_version, shader_variants[i][j]);
				shader.pipeline_variants.variants[i][j].setup(shader_variant, primitive[j], RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), blend_state, 0);
			}
		}
	}

	{
		//shader compiler
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["VERTEX"] = "vertex";
		actions.renames["LIGHT_VERTEX"] = "light_vertex";
		actions.renames["SHADOW_VERTEX"] = "shadow_vertex";
		actions.renames["UV"] = "uv";
		actions.renames["POINT_SIZE"] = "gl_PointSize";

		actions.renames["WORLD_MATRIX"] = "world_matrix";
		actions.renames["CANVAS_MATRIX"] = "canvas_data.canvas_transform";
		actions.renames["SCREEN_MATRIX"] = "canvas_data.screen_transform";
		actions.renames["TIME"] = "canvas_data.time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["AT_LIGHT_PASS"] = "false";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";

		actions.renames["COLOR"] = "color";
		actions.renames["NORMAL"] = "normal";
		actions.renames["NORMAL_MAP"] = "normal_map";
		actions.renames["NORMAL_MAP_DEPTH"] = "normal_map_depth";
		actions.renames["TEXTURE"] = "color_texture";
		actions.renames["TEXTURE_PIXEL_SIZE"] = "draw_data.color_texture_pixel_size";
		actions.renames["NORMAL_TEXTURE"] = "normal_texture";
		actions.renames["SPECULAR_SHININESS_TEXTURE"] = "specular_texture";
		actions.renames["SPECULAR_SHININESS"] = "specular_shininess";
		actions.renames["SCREEN_UV"] = "screen_uv";
		actions.renames["SCREEN_TEXTURE"] = "screen_texture";
		actions.renames["SCREEN_PIXEL_SIZE"] = "canvas_data.screen_pixel_size";
		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["INSTANCE_ID"] = "gl_InstanceIndex";
		actions.renames["VERTEX_ID"] = "gl_VertexIndex";

		actions.renames["LIGHT_POSITION"] = "light_position";
		actions.renames["LIGHT_COLOR"] = "light_color";
		actions.renames["LIGHT_ENERGY"] = "light_energy";
		actions.renames["LIGHT"] = "light";
		actions.renames["SHADOW_MODULATE"] = "shadow_modulate";

		actions.renames["texture_sdf"] = "texture_sdf";
		actions.renames["texture_sdf_normal"] = "texture_sdf_normal";
		actions.renames["sdf_to_screen_uv"] = "sdf_to_screen_uv";
		actions.renames["screen_uv_to_sdf"] = "screen_uv_to_sdf";

		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
		actions.usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";
		actions.usage_defines["SCREEN_PIXEL_SIZE"] = "@SCREEN_UV";
		actions.usage_defines["NORMAL"] = "#define NORMAL_USED\n";
		actions.usage_defines["NORMAL_MAP"] = "#define NORMAL_MAP_USED\n";
		actions.usage_defines["LIGHT"] = "#define LIGHT_SHADER_CODE_USED\n";

		actions.render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
		actions.render_mode_defines["unshaded"] = "#define MODE_UNSHADED\n";
		actions.render_mode_defines["light_only"] = "#define MODE_LIGHT_ONLY\n";

		actions.custom_samplers["TEXTURE"] = "texture_sampler";
		actions.custom_samplers["NORMAL_TEXTURE"] = "texture_sampler";
		actions.custom_samplers["SPECULAR_SHININESS_TEXTURE"] = "texture_sampler";
		actions.custom_samplers["SCREEN_TEXTURE"] = "material_samplers[3]"; //mipmap and filter for screen texture
		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = MATERIAL_UNIFORM_SET;
		actions.base_uniform_string = "material.";
		actions.default_filter = ShaderLanguage::FILTER_LINEAR;
		actions.default_repeat = ShaderLanguage::REPEAT_DISABLE;
		actions.base_varying_index = 4;

		actions.global_buffer_array_variable = "global_variables.data";

		shader.compiler.initialize(actions);
	}

	{ //shadow rendering
		Vector<String> versions;
		versions.push_back("\n#define MODE_SHADOW\n"); //shadow
		versions.push_back("\n#define MODE_SDF\n"); //sdf
		shadow_render.shader.initialize(versions);

		{
			Vector<RD::AttachmentFormat> attachments;

			RD::AttachmentFormat af_color;
			af_color.format = RD::DATA_FORMAT_R32_SFLOAT;
			af_color.usage_flags = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

			attachments.push_back(af_color);

			RD::AttachmentFormat af_depth;
			af_depth.format = RD::DATA_FORMAT_D32_SFLOAT;
			af_depth.usage_flags = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			attachments.push_back(af_depth);

			shadow_render.framebuffer_format = RD::get_singleton()->framebuffer_format_create(attachments);
		}

		{
			Vector<RD::AttachmentFormat> attachments;

			RD::AttachmentFormat af_color;
			af_color.format = RD::DATA_FORMAT_R8_UNORM;
			af_color.usage_flags = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

			attachments.push_back(af_color);

			shadow_render.sdf_framebuffer_format = RD::get_singleton()->framebuffer_format_create(attachments);
		}

		//pipelines
		Vector<RD::VertexAttribute> vf;
		RD::VertexAttribute vd;
		vd.format = sizeof(real_t) == sizeof(float) ? RD::DATA_FORMAT_R32G32B32_SFLOAT : RD::DATA_FORMAT_R64G64B64_SFLOAT;
		vd.location = 0;
		vd.offset = 0;
		vd.stride = sizeof(real_t) * 3;
		vf.push_back(vd);
		shadow_render.vertex_format = RD::get_singleton()->vertex_format_create(vf);

		vd.format = sizeof(real_t) == sizeof(float) ? RD::DATA_FORMAT_R32G32_SFLOAT : RD::DATA_FORMAT_R64G64_SFLOAT;
		vd.stride = sizeof(real_t) * 2;

		vf.write[0] = vd;
		shadow_render.sdf_vertex_format = RD::get_singleton()->vertex_format_create(vf);

		shadow_render.shader_version = shadow_render.shader.version_create();

		for (int i = 0; i < 3; i++) {
			RD::PipelineRasterizationState rs;
			rs.cull_mode = i == 0 ? RD::POLYGON_CULL_DISABLED : (i == 1 ? RD::POLYGON_CULL_FRONT : RD::POLYGON_CULL_BACK);
			RD::PipelineDepthStencilState ds;
			ds.enable_depth_write = true;
			ds.enable_depth_test = true;
			ds.depth_compare_operator = RD::COMPARE_OP_LESS;
			shadow_render.render_pipelines[i] = RD::get_singleton()->render_pipeline_create(shadow_render.shader.version_get_shader(shadow_render.shader_version, SHADOW_RENDER_MODE_SHADOW), shadow_render.framebuffer_format, shadow_render.vertex_format, RD::RENDER_PRIMITIVE_TRIANGLES, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
		}

		for (int i = 0; i < 2; i++) {
			shadow_render.sdf_render_pipelines[i] = RD::get_singleton()->render_pipeline_create(shadow_render.shader.version_get_shader(shadow_render.shader_version, SHADOW_RENDER_MODE_SDF), shadow_render.sdf_framebuffer_format, shadow_render.sdf_vertex_format, i == 0 ? RD::RENDER_PRIMITIVE_TRIANGLES : RD::RENDER_PRIMITIVE_LINES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{ //bindings

		state.canvas_state_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(State::Buffer));
		state.lights_uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(LightUniform) * state.max_lights_per_render);

		RD::SamplerState shadow_sampler_state;
		shadow_sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		shadow_sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
		shadow_sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT; //shadow wrap around
		shadow_sampler_state.compare_op = RD::COMPARE_OP_GREATER;
		shadow_sampler_state.enable_compare = true;
		state.shadow_sampler = RD::get_singleton()->sampler_create(shadow_sampler_state);
	}

	{
		//polygon buffers
		polygon_buffers.last_id = 1;
	}

	{ // default index buffer

		Vector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			uint8_t *w = pv.ptrw();
			int *p32 = (int *)w;
			p32[0] = 0;
			p32[1] = 1;
			p32[2] = 2;
			p32[3] = 0;
			p32[4] = 2;
			p32[5] = 3;
		}
		shader.quad_index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		shader.quad_index_array = RD::get_singleton()->index_array_create(shader.quad_index_buffer, 0, 6);
	}

	{ //primitive
		primitive_arrays.index_array[0] = shader.quad_index_array = RD::get_singleton()->index_array_create(shader.quad_index_buffer, 0, 1);
		primitive_arrays.index_array[1] = shader.quad_index_array = RD::get_singleton()->index_array_create(shader.quad_index_buffer, 0, 2);
		primitive_arrays.index_array[2] = shader.quad_index_array = RD::get_singleton()->index_array_create(shader.quad_index_buffer, 0, 3);
		primitive_arrays.index_array[3] = shader.quad_index_array = RD::get_singleton()->index_array_create(shader.quad_index_buffer, 0, 6);
	}

	{ //default skeleton buffer

		shader.default_skeleton_uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(SkeletonUniform));
		SkeletonUniform su;
		_update_transform_2d_to_mat4(Transform2D(), su.skeleton_inverse);
		_update_transform_2d_to_mat4(Transform2D(), su.skeleton_transform);
		RD::get_singleton()->buffer_update(shader.default_skeleton_uniform_buffer, 0, sizeof(SkeletonUniform), &su);

		shader.default_skeleton_texture_buffer = RD::get_singleton()->texture_buffer_create(32, RD::DATA_FORMAT_R32G32B32A32_SFLOAT);
	}
	{
		//default shadow texture to keep uniform set happy
		RD::TextureFormat tf;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.width = 4;
		tf.height = 4;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;

		state.shadow_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	{
		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.ids.push_back(storage->get_default_rd_storage_buffer());
			uniforms.push_back(u);
		}

		state.default_transforms_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader.default_version_rd_shader, TRANSFORMS_UNIFORM_SET);
	}

	default_canvas_texture = storage->canvas_texture_allocate();
	storage->canvas_texture_initialize(default_canvas_texture);

	state.shadow_texture_size = GLOBAL_GET("rendering/2d/shadow_atlas/size");

	//create functions for shader and material
	storage->shader_set_data_request_function(RendererStorageRD::SHADER_TYPE_2D, _create_shader_funcs);
	storage->material_set_data_request_function(RendererStorageRD::SHADER_TYPE_2D, _create_material_funcs);

	state.time = 0;

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

	static_assert(sizeof(PushConstant) == 128);
}

bool RendererCanvasRenderRD::free(RID p_rid) {
	if (canvas_light_owner.owns(p_rid)) {
		CanvasLight *cl = canvas_light_owner.get_or_null(p_rid);
		ERR_FAIL_COND_V(!cl, false);
		light_set_use_shadow(p_rid, false);
		canvas_light_owner.free(p_rid);
	} else if (occluder_polygon_owner.owns(p_rid)) {
		occluder_polygon_set_shape(p_rid, Vector<Vector2>(), false);
		occluder_polygon_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

void RendererCanvasRenderRD::set_shadow_texture_size(int p_size) {
	p_size = nearest_power_of_2_templated(p_size);
	if (p_size == state.shadow_texture_size) {
		return;
	}
	state.shadow_texture_size = p_size;
	if (state.shadow_fb.is_valid()) {
		RD::get_singleton()->free(state.shadow_texture);
		RD::get_singleton()->free(state.shadow_depth_texture);
		state.shadow_fb = RID();

		{
			//create a default shadow texture to keep uniform set happy (and that it gets erased when a new one is created)
			RD::TextureFormat tf;
			tf.texture_type = RD::TEXTURE_TYPE_2D;
			tf.width = 4;
			tf.height = 4;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;

			state.shadow_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
	}
}

RendererCanvasRenderRD::~RendererCanvasRenderRD() {
	//canvas state

	storage->free(default_canvas_group_material);
	storage->free(default_canvas_group_shader);

	{
		if (state.canvas_state_buffer.is_valid()) {
			RD::get_singleton()->free(state.canvas_state_buffer);
		}

		memdelete_arr(state.light_uniforms);
		RD::get_singleton()->free(state.lights_uniform_buffer);
		RD::get_singleton()->free(shader.default_skeleton_uniform_buffer);
		RD::get_singleton()->free(shader.default_skeleton_texture_buffer);
	}

	//shadow rendering
	{
		shadow_render.shader.version_free(shadow_render.shader_version);
		//this will also automatically clear all pipelines
		RD::get_singleton()->free(state.shadow_sampler);
	}
	//bindings

	//shaders

	shader.canvas_shader.version_free(shader.default_version);

	//buffers
	{
		RD::get_singleton()->free(shader.quad_index_array);
		RD::get_singleton()->free(shader.quad_index_buffer);
		//primitives are erase by dependency
	}

	if (state.shadow_fb.is_valid()) {
		RD::get_singleton()->free(state.shadow_depth_texture);
	}
	RD::get_singleton()->free(state.shadow_texture);

	storage->free(default_canvas_texture);
	//pipelines don't need freeing, they are all gone after shaders are gone
}

/*************************************************************************/
/*  rasterizer_canvas_rd.cpp                                             */
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

#include "rasterizer_canvas_rd.h"
#include "core/math/math_funcs.h"
#include "core/project_settings.h"
#include "rasterizer_rd.h"

void RasterizerCanvasRD::_update_transform_2d_to_mat4(const Transform2D &p_transform, float *p_mat4) {

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

void RasterizerCanvasRD::_update_transform_2d_to_mat2x4(const Transform2D &p_transform, float *p_mat2x4) {

	p_mat2x4[0] = p_transform.elements[0][0];
	p_mat2x4[1] = p_transform.elements[1][0];
	p_mat2x4[2] = 0;
	p_mat2x4[3] = p_transform.elements[2][0];

	p_mat2x4[4] = p_transform.elements[0][1];
	p_mat2x4[5] = p_transform.elements[1][1];
	p_mat2x4[6] = 0;
	p_mat2x4[7] = p_transform.elements[2][1];
}

void RasterizerCanvasRD::_update_transform_2d_to_mat2x3(const Transform2D &p_transform, float *p_mat2x3) {

	p_mat2x3[0] = p_transform.elements[0][0];
	p_mat2x3[1] = p_transform.elements[0][1];
	p_mat2x3[2] = p_transform.elements[1][0];
	p_mat2x3[3] = p_transform.elements[1][1];
	p_mat2x3[4] = p_transform.elements[2][0];
	p_mat2x3[5] = p_transform.elements[2][1];
}

void RasterizerCanvasRD::_update_transform_to_mat4(const Transform &p_transform, float *p_mat4) {

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

void RasterizerCanvasRD::_update_specular_shininess(const Color &p_transform, uint32_t *r_ss) {

	*r_ss = uint32_t(CLAMP(p_transform.a * 255.0, 0, 255)) << 24;
	*r_ss |= uint32_t(CLAMP(p_transform.b * 255.0, 0, 255)) << 16;
	*r_ss |= uint32_t(CLAMP(p_transform.g * 255.0, 0, 255)) << 8;
	*r_ss |= uint32_t(CLAMP(p_transform.r * 255.0, 0, 255));
}

RID RasterizerCanvasRD::_create_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, RenderingServer::CanvasItemTextureFilter p_filter, RenderingServer::CanvasItemTextureRepeat p_repeat, RID p_multimesh) {

	Vector<RD::Uniform> uniform_set;

	{ // COLOR TEXTURE
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 1;
		RID texture = storage->texture_get_rd_texture(p_texture);
		if (!texture.is_valid()) {
			//use default white texture
			texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}
		u.ids.push_back(texture);
		uniform_set.push_back(u);
	}

	{ // NORMAL TEXTURE
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 2;
		RID texture = storage->texture_get_rd_texture(p_normalmap);
		if (!texture.is_valid()) {
			//use default normal texture
			texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_NORMAL);
		}
		u.ids.push_back(texture);
		uniform_set.push_back(u);
	}

	{ // SPECULAR TEXTURE
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 3;
		RID texture = storage->texture_get_rd_texture(p_specular);
		if (!texture.is_valid()) {
			//use default white texture
			texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}
		u.ids.push_back(texture);
		uniform_set.push_back(u);
	}

	{ // SAMPLER
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 4;
		RID sampler = storage->sampler_rd_get_default(p_filter, p_repeat);
		ERR_FAIL_COND_V(sampler.is_null(), RID());
		u.ids.push_back(sampler);
		uniform_set.push_back(u);
	}

	{ // MULTIMESH TEXTURE BUFFER
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE_BUFFER;
		u.binding = 5;
		u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_MULTIMESH_BUFFER));
		uniform_set.push_back(u);
	}

	return RD::get_singleton()->uniform_set_create(uniform_set, shader.default_version_rd_shader, 0);
}

RasterizerCanvas::TextureBindingID RasterizerCanvasRD::request_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, RenderingServer::CanvasItemTextureFilter p_filter, RenderingServer::CanvasItemTextureRepeat p_repeat, RID p_multimesh) {

	if (p_filter == RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT) {
		p_filter = default_samplers.default_filter;
	}

	if (p_repeat == RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) {
		p_repeat = default_samplers.default_repeat;
	}

	TextureBindingKey key;
	key.texture = p_texture;
	key.normalmap = p_normalmap;
	key.specular = p_specular;
	key.multimesh = p_multimesh;
	key.texture_filter = p_filter;
	key.texture_repeat = p_repeat;

	TextureBinding *binding;
	TextureBindingID id;
	{
		TextureBindingID *idptr = bindings.texture_key_bindings.getptr(key);

		if (!idptr) {
			id = bindings.id_generator++;
			bindings.texture_key_bindings[key] = id;
			binding = memnew(TextureBinding);
			binding->key = key;
			binding->id = id;

			bindings.texture_bindings[id] = binding;

		} else {
			id = *idptr;
			binding = bindings.texture_bindings[id];
		}
	}

	binding->reference_count++;

	if (binding->to_dispose.in_list()) {
		//was queued for disposal previously, but ended up reused.
		bindings.to_dispose_list.remove(&binding->to_dispose);
	}

	if (binding->uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(binding->uniform_set)) {
		binding->uniform_set = _create_texture_binding(p_texture, p_normalmap, p_specular, p_filter, p_repeat, p_multimesh);
	}

	return id;
}

void RasterizerCanvasRD::free_texture_binding(TextureBindingID p_binding) {

	TextureBinding **binding_ptr = bindings.texture_bindings.getptr(p_binding);
	ERR_FAIL_COND(!binding_ptr);
	TextureBinding *binding = *binding_ptr;
	ERR_FAIL_COND(binding->reference_count == 0);
	binding->reference_count--;
	if (binding->reference_count == 0) {
		bindings.to_dispose_list.add(&binding->to_dispose);
	}
}

void RasterizerCanvasRD::_dispose_bindings() {

	while (bindings.to_dispose_list.first()) {
		TextureBinding *binding = bindings.to_dispose_list.first()->self();
		if (binding->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(binding->uniform_set)) {
			RD::get_singleton()->free(binding->uniform_set);
		}

		bindings.texture_key_bindings.erase(binding->key);
		bindings.texture_bindings.erase(binding->id);
		bindings.to_dispose_list.remove(&binding->to_dispose);
		memdelete(binding);
	}
}

RasterizerCanvas::PolygonID RasterizerCanvasRD::request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, const Vector<int> &p_bones, const Vector<float> &p_weights) {

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
	Vector<RD::VertexDescription> descriptions;
	descriptions.resize(4);
	Vector<RID> buffers;
	buffers.resize(4);

	{
		const uint8_t *r = polygon_buffer.ptr();
		float *fptr = (float *)r;
		uint32_t *uptr = (uint32_t *)r;
		uint32_t base_offset = 0;
		{ //vertices
			RD::VertexDescription vd;
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
			RD::VertexDescription vd;
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
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = 0;
			vd.location = RS::ARRAY_COLOR;
			vd.stride = 0;

			descriptions.write[1] = vd;
			buffers.write[1] = storage->mesh_get_default_rd_buffer(RasterizerStorageRD::DEFAULT_RD_BUFFER_COLOR);
		}

		//uvs
		if ((uint32_t)p_uvs.size() == vertex_count) {
			RD::VertexDescription vd;
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
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			vd.offset = 0;
			vd.location = RS::ARRAY_TEX_UV;
			vd.stride = 0;

			descriptions.write[2] = vd;
			buffers.write[2] = storage->mesh_get_default_rd_buffer(RasterizerStorageRD::DEFAULT_RD_BUFFER_TEX_UV);
		}

		//bones
		if ((uint32_t)p_indices.size() == vertex_count * 4 && (uint32_t)p_weights.size() == vertex_count * 4) {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
			vd.offset = base_offset * sizeof(float);
			vd.location = RS::ARRAY_BONES;
			vd.stride = stride * sizeof(float);

			descriptions.write[3] = vd;

			const int *bone_ptr = p_bones.ptr();
			const float *weight_ptr = p_weights.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {

				uint16_t *bone16w = (uint16_t *)&uptr[base_offset + i * stride];
				uint16_t *weight16w = (uint16_t *)&uptr[base_offset + i * stride + 2];

				bone16w[0] = bone_ptr[i * 4 + 0];
				bone16w[1] = bone_ptr[i * 4 + 1];
				bone16w[2] = bone_ptr[i * 4 + 2];
				bone16w[3] = bone_ptr[i * 4 + 3];

				weight16w[0] = CLAMP(weight_ptr[i * 4 + 0] * 65535, 0, 65535);
				weight16w[1] = CLAMP(weight_ptr[i * 4 + 1] * 65535, 0, 65535);
				weight16w[2] = CLAMP(weight_ptr[i * 4 + 2] * 65535, 0, 65535);
				weight16w[3] = CLAMP(weight_ptr[i * 4 + 3] * 65535, 0, 65535);
			}

			base_offset += 4;
		} else {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
			vd.offset = 0;
			vd.location = RS::ARRAY_BONES;
			vd.stride = 0;

			descriptions.write[3] = vd;
			buffers.write[3] = storage->mesh_get_default_rd_buffer(RasterizerStorageRD::DEFAULT_RD_BUFFER_BONES);
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
			copymem(w, p_indices.ptr(), sizeof(int32_t) * p_indices.size());
		}
		pb.index_buffer = RD::get_singleton()->index_buffer_create(p_indices.size(), RD::INDEX_BUFFER_FORMAT_UINT32, index_buffer);
		pb.indices = RD::get_singleton()->index_array_create(pb.index_buffer, 0, p_indices.size());
	}

	pb.vertex_format_id = vertex_id;

	PolygonID id = polygon_buffers.last_id++;

	polygon_buffers.polygons[id] = pb;

	return id;
}

void RasterizerCanvasRD::free_polygon(PolygonID p_polygon) {

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

Size2i RasterizerCanvasRD::_bind_texture_binding(TextureBindingID p_binding, RD::DrawListID p_draw_list, uint32_t &flags) {

	TextureBinding **texture_binding_ptr = bindings.texture_bindings.getptr(p_binding);
	ERR_FAIL_COND_V(!texture_binding_ptr, Size2i());
	TextureBinding *texture_binding = *texture_binding_ptr;

	if (texture_binding->key.normalmap.is_valid()) {
		flags |= FLAGS_DEFAULT_NORMAL_MAP_USED;
	}
	if (texture_binding->key.specular.is_valid()) {
		flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
	}

	if (!RD::get_singleton()->uniform_set_is_valid(texture_binding->uniform_set)) {
		//texture may have changed (erased or replaced, see if we can fix)
		texture_binding->uniform_set = _create_texture_binding(texture_binding->key.texture, texture_binding->key.normalmap, texture_binding->key.specular, texture_binding->key.texture_filter, texture_binding->key.texture_repeat, texture_binding->key.multimesh);
		ERR_FAIL_COND_V(!texture_binding->uniform_set.is_valid(), Size2i(1, 1));
	}

	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, texture_binding->uniform_set, 0);
	if (texture_binding->key.texture.is_valid()) {
		return storage->texture_2d_get_size(texture_binding->key.texture);
	} else {
		return Size2i(1, 1);
	}
}

////////////////////
void RasterizerCanvasRD::_render_item(RD::DrawListID p_draw_list, const Item *p_item, RD::FramebufferFormatID p_framebuffer_format, const Transform2D &p_canvas_transform_inverse, Item *&current_clip, Light *p_lights, PipelineVariants *p_pipeline_variants) {

	//create an empty push constant

	PushConstant push_constant;
	Transform2D base_transform = p_canvas_transform_inverse * p_item->final_transform;
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

	bool light_uniform_set_dirty = false;

	if (!p_item->custom_data) {
		p_item->custom_data = memnew(ItemStateData);
		light_uniform_set_dirty = true;
	}

	ItemStateData *state_data = (ItemStateData *)p_item->custom_data;

	Light *light_cache[DEFAULT_MAX_LIGHTS_PER_ITEM];
	uint16_t light_count = 0;
	PipelineLightMode light_mode;

	{

		Light *light = p_lights;

		while (light) {

			if (light->render_index_cache >= 0 && p_item->light_mask & light->item_mask && p_item->z_final >= light->z_min && p_item->z_final <= light->z_max && p_item->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {

				uint32_t light_index = light->render_index_cache;
				push_constant.lights[light_count >> 2] |= light_index << ((light_count & 3) * 8);

				if (!light_uniform_set_dirty && (state_data->light_cache[light_count].light != light || state_data->light_cache[light_count].light_version != light->version)) {
					light_uniform_set_dirty = true;
				}

				light_cache[light_count] = light;

				light_count++;
				if (light->mode == RS::CANVAS_LIGHT_MODE_MASK) {
					base_flags |= FLAGS_USING_LIGHT_MASK;
				}
				if (light_count == state.max_lights_per_item) {
					break;
				}
			}
			light = light->next_ptr;
		}

		if (light_count != state_data->light_cache_count) {
			light_uniform_set_dirty = true;
		}
		base_flags |= light_count << FLAGS_LIGHT_COUNT_SHIFT;
	}

	{

		RID &canvas_item_state = light_count ? state_data->state_uniform_set_with_light : state_data->state_uniform_set;

		bool invalid_uniform = canvas_item_state.is_valid() && !RD::get_singleton()->uniform_set_is_valid(canvas_item_state);

		if (canvas_item_state.is_null() || invalid_uniform || (light_count > 0 && light_uniform_set_dirty)) {
			//re create canvas state
			Vector<RD::Uniform> uniforms;

			if (state_data->state_uniform_set_with_light.is_valid() && !invalid_uniform) {
				RD::get_singleton()->free(canvas_item_state);
			}

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
				u.binding = 0;
				u.ids.push_back(state.canvas_state_buffer);
				uniforms.push_back(u);
			}

			if (false && p_item->skeleton.is_valid()) {
				//bind skeleton stuff
			} else {
				//bind default

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_TEXTURE_BUFFER;
					u.binding = 1;
					u.ids.push_back(shader.default_skeleton_texture_buffer);
					uniforms.push_back(u);
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
					u.binding = 2;
					u.ids.push_back(shader.default_skeleton_uniform_buffer);
					uniforms.push_back(u);
				}
			}

			//validate and update lighs if they are being used

			if (light_count > 0) {
				//recreate uniform set

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
					u.binding = 3;
					u.ids.push_back(state.lights_uniform_buffer);
					uniforms.push_back(u);
				}

				{

					RD::Uniform u_lights;
					u_lights.type = RD::UNIFORM_TYPE_TEXTURE;
					u_lights.binding = 4;

					RD::Uniform u_shadows;
					u_shadows.type = RD::UNIFORM_TYPE_TEXTURE;
					u_shadows.binding = 5;

					//lights
					for (uint32_t i = 0; i < state.max_lights_per_item; i++) {
						if (i < light_count) {

							CanvasLight *cl = canvas_light_owner.getornull(light_cache[i]->light_internal);
							ERR_CONTINUE(!cl);

							RID rd_texture;

							if (cl->texture.is_valid()) {
								rd_texture = storage->texture_get_rd_texture(cl->texture);
							}
							if (rd_texture.is_valid()) {
								u_lights.ids.push_back(rd_texture);
							} else {
								u_lights.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
							}
							if (cl->shadow.texture.is_valid()) {
								u_shadows.ids.push_back(cl->shadow.texture);
							} else {
								u_shadows.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK));
							}
						} else {
							u_lights.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
							u_shadows.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK));
						}
					}

					uniforms.push_back(u_lights);
					uniforms.push_back(u_shadows);
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_SAMPLER;
					u.binding = 6;
					u.ids.push_back(state.shadow_sampler);
					uniforms.push_back(u);
				}

				canvas_item_state = RD::get_singleton()->uniform_set_create(uniforms, shader.default_version_rd_shader_light, 2);
			} else {
				canvas_item_state = RD::get_singleton()->uniform_set_create(uniforms, shader.default_version_rd_shader, 2);
			}
		}

		RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, canvas_item_state, 2);
	}

	light_mode = light_count > 0 ? PIPELINE_LIGHT_MODE_ENABLED : PIPELINE_LIGHT_MODE_DISABLED;

	PipelineVariants *pipeline_variants = p_pipeline_variants;

	bool reclip = false;

	const Item::Command *c = p_item->commands;
	while (c) {
		push_constant.flags = base_flags; //reset on each command for sanity
		push_constant.specular_shininess = 0xFFFFFFFF;

		switch (c->type) {
			case Item::Command::TYPE_RECT: {

				const Item::CommandRect *rect = static_cast<const Item::CommandRect *>(c);

				//bind pipeline
				{
					RID pipeline = pipeline_variants->variants[light_mode][PIPELINE_VARIANT_QUAD].get_render_pipeline(RD::INVALID_ID, p_framebuffer_format);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				Size2 texpixel_size;
				{
					texpixel_size = _bind_texture_binding(rect->texture_binding.binding_id, p_draw_list, push_constant.flags);
					texpixel_size.x = 1.0 / texpixel_size.x;
					texpixel_size.y = 1.0 / texpixel_size.y;
				}

				if (rect->specular_shininess.a < 0.999) {
					push_constant.flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
				}

				_update_specular_shininess(rect->specular_shininess, &push_constant.specular_shininess);

				Rect2 src_rect;
				Rect2 dst_rect;

				if (texpixel_size != Vector2()) {
					push_constant.color_texture_pixel_size[0] = texpixel_size.x;
					push_constant.color_texture_pixel_size[1] = texpixel_size.y;

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
					texpixel_size = Vector2(1, 1);
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

				push_constant.color_texture_pixel_size[0] = texpixel_size.x;
				push_constant.color_texture_pixel_size[1] = texpixel_size.y;

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

				Size2 texpixel_size;
				{
					texpixel_size = _bind_texture_binding(np->texture_binding.binding_id, p_draw_list, push_constant.flags);
					texpixel_size.x = 1.0 / texpixel_size.x;
					texpixel_size.y = 1.0 / texpixel_size.y;
				}

				if (np->specular_shininess.a < 0.999) {
					push_constant.flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
				}

				_update_specular_shininess(np->specular_shininess, &push_constant.specular_shininess);

				Rect2 src_rect;
				Rect2 dst_rect(np->rect.position.x, np->rect.position.y, np->rect.size.x, np->rect.size.y);

				if (texpixel_size == Size2()) {

					texpixel_size = Size2(1, 1);
					src_rect = Rect2(0, 0, 1, 1);

				} else {

					if (np->source != Rect2()) {
						src_rect = Rect2(np->source.position.x * texpixel_size.width, np->source.position.y * texpixel_size.height, np->source.size.x * texpixel_size.width, np->source.size.y * texpixel_size.height);
						texpixel_size = Size2(1.0 / np->source.size.width, 1.0 / np->source.size.height);
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

				push_constant.color_texture_pixel_size[0] = texpixel_size.x;
				push_constant.color_texture_pixel_size[1] = texpixel_size.y;

				push_constant.flags |= int(np->axis_x) << FLAGS_NINEPATCH_H_MODE_SHIFT;
				push_constant.flags |= int(np->axis_y) << FLAGS_NINEPATCH_V_MODE_SHIFT;

				if (np->draw_center) {
					push_constant.flags |= FLAGS_NINEPACH_DRAW_CENTER;
				}

				push_constant.ninepatch_margins[0] = np->margin[MARGIN_LEFT];
				push_constant.ninepatch_margins[1] = np->margin[MARGIN_TOP];
				push_constant.ninepatch_margins[2] = np->margin[MARGIN_RIGHT];
				push_constant.ninepatch_margins[3] = np->margin[MARGIN_BOTTOM];

				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, shader.quad_index_array);
				RD::get_singleton()->draw_list_draw(p_draw_list, true);

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

				Size2 texpixel_size;
				{
					texpixel_size = _bind_texture_binding(polygon->texture_binding.binding_id, p_draw_list, push_constant.flags);
					texpixel_size.x = 1.0 / texpixel_size.x;
					texpixel_size.y = 1.0 / texpixel_size.y;
				}

				if (polygon->specular_shininess.a < 0.999) {
					push_constant.flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
				}

				_update_specular_shininess(polygon->specular_shininess, &push_constant.specular_shininess);

				push_constant.modulation[0] = base_color.r;
				push_constant.modulation[1] = base_color.g;
				push_constant.modulation[2] = base_color.b;
				push_constant.modulation[3] = base_color.a;

				for (int j = 0; j < 4; j++) {
					push_constant.src_rect[j] = 0;
					push_constant.dst_rect[j] = 0;
					push_constant.ninepatch_margins[j] = 0;
				}

				push_constant.color_texture_pixel_size[0] = texpixel_size.x;
				push_constant.color_texture_pixel_size[1] = texpixel_size.y;

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

				{
					_bind_texture_binding(primitive->texture_binding.binding_id, p_draw_list, push_constant.flags);
				}

				if (primitive->specular_shininess.a < 0.999) {
					push_constant.flags |= FLAGS_DEFAULT_SPECULAR_MAP_USED;
				}

				_update_specular_shininess(primitive->specular_shininess, &push_constant.specular_shininess);

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
				ERR_PRINT("FIXME: Mesh, MultiMesh and Particles render commands are unimplemented currently, they need to be ported to the 4.0 rendering architecture.");
#ifndef _MSC_VER
#warning Item::Command types for Mesh, MultiMesh and Particles need to be implemented.
#endif
				// See #if 0'ed code below to port from GLES3.
			} break;

#if 0
			case Item::Command::TYPE_MESH: {

				Item::CommandMesh *mesh = static_cast<Item::CommandMesh *>(c);
				_set_texture_rect_mode(false);

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(mesh->texture, mesh->normal_map);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				}

				state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform * mesh->transform);

				RasterizerStorageGLES3::Mesh *mesh_data = storage->mesh_owner.getornull(mesh->mesh);
				if (mesh_data) {

					for (int j = 0; j < mesh_data->surfaces.size(); j++) {
						RasterizerStorageGLES3::Surface *s = mesh_data->surfaces[j];
						// materials are ignored in 2D meshes, could be added but many things (ie, lighting mode, reading from screen, etc) would break as they are not meant be set up at this point of drawing
						glBindVertexArray(s->array_id);

						glVertexAttrib4f(RS::ARRAY_COLOR, mesh->modulate.r, mesh->modulate.g, mesh->modulate.b, mesh->modulate.a);

						if (s->index_array_len) {
							glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
						} else {
							glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
						}

						glBindVertexArray(0);
					}
				}
				state.canvas_shader.set_uniform(CanvasShaderGLES3::MODELVIEW_MATRIX, state.final_transform);

			} break;
			case Item::Command::TYPE_MULTIMESH: {

				Item::CommandMultiMesh *mmesh = static_cast<Item::CommandMultiMesh *>(c);

				RasterizerStorageGLES3::MultiMesh *multi_mesh = storage->multimesh_owner.getornull(mmesh->multimesh);

				if (!multi_mesh)
					break;

				RasterizerStorageGLES3::Mesh *mesh_data = storage->mesh_owner.getornull(multi_mesh->mesh);

				if (!mesh_data)
					break;

				RasterizerStorageGLES3::Texture *texture = _bind_canvas_texture(mmesh->texture, mmesh->normal_map);

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, multi_mesh->custom_data_format != RS::MULTIMESH_CUSTOM_DATA_NONE);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, true);
				//reset shader and force rebind
				state.using_texture_rect = true;
				_set_texture_rect_mode(false);

				if (texture) {
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
					state.canvas_shader.set_uniform(CanvasShaderGLES3::COLOR_TEXPIXEL_SIZE, texpixel_size);
				}

				int amount = MIN(multi_mesh->size, multi_mesh->visible_instances);

				if (amount == -1) {
					amount = multi_mesh->size;
				}

				for (int j = 0; j < mesh_data->surfaces.size(); j++) {
					RasterizerStorageGLES3::Surface *s = mesh_data->surfaces[j];
					// materials are ignored in 2D meshes, could be added but many things (ie, lighting mode, reading from screen, etc) would break as they are not meant be set up at this point of drawing
					glBindVertexArray(s->instancing_array_id);

					glBindBuffer(GL_ARRAY_BUFFER, multi_mesh->buffer); //modify the buffer

					int stride = (multi_mesh->xform_floats + multi_mesh->color_floats + multi_mesh->custom_data_floats) * 4;
					glEnableVertexAttribArray(8);
					glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(0));
					glVertexAttribDivisor(8, 1);
					glEnableVertexAttribArray(9);
					glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(4 * 4));
					glVertexAttribDivisor(9, 1);

					int color_ofs;

					if (multi_mesh->transform_format == RS::MULTIMESH_TRANSFORM_3D) {
						glEnableVertexAttribArray(10);
						glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(8 * 4));
						glVertexAttribDivisor(10, 1);
						color_ofs = 12 * 4;
					} else {
						glDisableVertexAttribArray(10);
						glVertexAttrib4f(10, 0, 0, 1, 0);
						color_ofs = 8 * 4;
					}

					int custom_data_ofs = color_ofs;

					switch (multi_mesh->color_format) {

						case RS::MULTIMESH_COLOR_NONE: {
							glDisableVertexAttribArray(11);
							glVertexAttrib4f(11, 1, 1, 1, 1);
						} break;
						case RS::MULTIMESH_COLOR_8BIT: {
							glEnableVertexAttribArray(11);
							glVertexAttribPointer(11, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, CAST_INT_TO_UCHAR_PTR(color_ofs));
							glVertexAttribDivisor(11, 1);
							custom_data_ofs += 4;

						} break;
						case RS::MULTIMESH_COLOR_FLOAT: {
							glEnableVertexAttribArray(11);
							glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(color_ofs));
							glVertexAttribDivisor(11, 1);
							custom_data_ofs += 4 * 4;
						} break;
					}

					switch (multi_mesh->custom_data_format) {

						case RS::MULTIMESH_CUSTOM_DATA_NONE: {
							glDisableVertexAttribArray(12);
							glVertexAttrib4f(12, 1, 1, 1, 1);
						} break;
						case RS::MULTIMESH_CUSTOM_DATA_8BIT: {
							glEnableVertexAttribArray(12);
							glVertexAttribPointer(12, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, CAST_INT_TO_UCHAR_PTR(custom_data_ofs));
							glVertexAttribDivisor(12, 1);

						} break;
						case RS::MULTIMESH_CUSTOM_DATA_FLOAT: {
							glEnableVertexAttribArray(12);
							glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(custom_data_ofs));
							glVertexAttribDivisor(12, 1);
						} break;
					}

					if (s->index_array_len) {
						glDrawElementsInstanced(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0, amount);
					} else {
						glDrawArraysInstanced(gl_primitive[s->primitive], 0, s->array_len, amount);
					}

					glBindVertexArray(0);
				}

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, false);
				state.using_texture_rect = true;
				_set_texture_rect_mode(false);

			} break;
			case Item::Command::TYPE_PARTICLES: {

				Item::CommandParticles *particles_cmd = static_cast<Item::CommandParticles *>(c);

				RasterizerStorageGLES3::Particles *particles = storage->particles_owner.getornull(particles_cmd->particles);
				if (!particles)
					break;

				if (particles->inactive && !particles->emitting)
					break;

				glVertexAttrib4f(RS::ARRAY_COLOR, 1, 1, 1, 1); //not used, so keep white

				RenderingServerRaster::redraw_request();

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
					Size2 texpixel_size(1.0 / texture->width, 1.0 / texture->height);
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

				glBindVertexArray(data.particle_quad_array); //use particle quad array
				glBindBuffer(GL_ARRAY_BUFFER, particles->particle_buffers[0]); //bind particle buffer

				int stride = sizeof(float) * 4 * 6;

				int amount = particles->amount;

				if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_LIFETIME) {

					glEnableVertexAttribArray(8); //xform x
					glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 3));
					glVertexAttribDivisor(8, 1);
					glEnableVertexAttribArray(9); //xform y
					glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 4));
					glVertexAttribDivisor(9, 1);
					glEnableVertexAttribArray(10); //xform z
					glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 5));
					glVertexAttribDivisor(10, 1);
					glEnableVertexAttribArray(11); //color
					glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, nullptr);
					glVertexAttribDivisor(11, 1);
					glEnableVertexAttribArray(12); //custom
					glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 2));
					glVertexAttribDivisor(12, 1);

					glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, amount);
				} else {
					//split
					int split = int(Math::ceil(particles->phase * particles->amount));

					if (amount - split > 0) {
						glEnableVertexAttribArray(8); //xform x
						glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(stride * split + sizeof(float) * 4 * 3));
						glVertexAttribDivisor(8, 1);
						glEnableVertexAttribArray(9); //xform y
						glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(stride * split + sizeof(float) * 4 * 4));
						glVertexAttribDivisor(9, 1);
						glEnableVertexAttribArray(10); //xform z
						glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(stride * split + sizeof(float) * 4 * 5));
						glVertexAttribDivisor(10, 1);
						glEnableVertexAttribArray(11); //color
						glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(stride * split + 0));
						glVertexAttribDivisor(11, 1);
						glEnableVertexAttribArray(12); //custom
						glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(stride * split + sizeof(float) * 4 * 2));
						glVertexAttribDivisor(12, 1);

						glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, amount - split);
					}

					if (split > 0) {
						glEnableVertexAttribArray(8); //xform x
						glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 3));
						glVertexAttribDivisor(8, 1);
						glEnableVertexAttribArray(9); //xform y
						glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 4));
						glVertexAttribDivisor(9, 1);
						glEnableVertexAttribArray(10); //xform z
						glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 5));
						glVertexAttribDivisor(10, 1);
						glEnableVertexAttribArray(11); //color
						glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, nullptr);
						glVertexAttribDivisor(11, 1);
						glEnableVertexAttribArray(12); //custom
						glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(sizeof(float) * 4 * 2));
						glVertexAttribDivisor(12, 1);

						glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, split);
					}
				}

				glBindVertexArray(0);

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_PARTICLES, false);
				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCING, false);
				state.using_texture_rect = true;
				_set_texture_rect_mode(false);

			} break;
#endif
			case Item::Command::TYPE_TRANSFORM: {

				const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
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
		}

		c = c->next;
	}

	if (current_clip && reclip) {
		//will make it re-enable clipping if needed afterwards
		current_clip = nullptr;
	}
}

void RasterizerCanvasRD::_render_items(RID p_to_render_target, int p_item_count, const Transform2D &p_canvas_transform_inverse, Light *p_lights, RID p_screen_uniform_set) {

	Item *current_clip = nullptr;

	Transform2D canvas_transform_inverse = p_canvas_transform_inverse;

	RID framebuffer = storage->render_target_get_rd_framebuffer(p_to_render_target);

	Vector<Color> clear_colors;
	bool clear = false;
	if (storage->render_target_is_clear_requested(p_to_render_target)) {
		clear = true;
		clear_colors.push_back(storage->render_target_get_clear_request_color(p_to_render_target));
		storage->render_target_disable_clear_request(p_to_render_target);
	}
#ifndef _MSC_VER
#warning TODO obtain from framebuffer format eventually when this is implemented
#endif

	RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(framebuffer);

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer, clear ? RD::INITIAL_ACTION_CLEAR : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, clear_colors);

	if (p_screen_uniform_set.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_screen_uniform_set, 3);
	}
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

		if (ci->material != prev_material) {

			MaterialData *material_data = nullptr;
			if (ci->material.is_valid()) {
				material_data = (MaterialData *)storage->material_get_data(ci->material, RasterizerStorageRD::SHADER_TYPE_2D);
			}

			if (material_data) {

				if (material_data->shader_data->version.is_valid() && material_data->shader_data->valid) {
					pipeline_variants = &material_data->shader_data->pipeline_variants;
					if (material_data->uniform_set.is_valid()) {
						RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_data->uniform_set, 1);
					}
				} else {
					pipeline_variants = &shader.pipeline_variants;
				}
			} else {
				pipeline_variants = &shader.pipeline_variants;
			}
		}

		_render_item(draw_list, ci, fb_format, canvas_transform_inverse, current_clip, p_lights, pipeline_variants);

		prev_material = ci->material;
	}

	RD::get_singleton()->draw_list_end();
}

void RasterizerCanvasRD::canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, const Transform2D &p_canvas_transform) {

	int item_count = 0;

	//setup canvas state uniforms if needed

	Transform2D canvas_transform_inverse = p_canvas_transform.affine_inverse();

	{
		//update canvas state uniform buffer
		State::Buffer state_buffer;

		Size2i ssize = storage->render_target_get_size(p_to_render_target);

		Transform screen_transform;
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
		RD::get_singleton()->buffer_update(state.canvas_state_buffer, 0, sizeof(State::Buffer), &state_buffer, true);
	}

	//setup lights if exist

	{

		Light *l = p_light_list;
		uint32_t index = 0;

		while (l) {

			if (index == state.max_lights_per_render) {
				l->render_index_cache = -1;
				l = l->next_ptr;
				continue;
			}

			CanvasLight *clight = canvas_light_owner.getornull(l->light_internal);
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
				state.light_uniforms[index].shadow_color[i] = l->shadow_color[i];
				state.light_uniforms[index].color[i] = l->color[i];
			}

			state.light_uniforms[index].color[3] = l->energy; //use alpha for energy, so base color can go separate

			if (clight->shadow.texture.is_valid()) {
				state.light_uniforms[index].shadow_pixel_size = (1.0 / clight->shadow.size) * (1.0 + l->shadow_smooth);
			} else {
				state.light_uniforms[index].shadow_pixel_size = 1.0;
			}

			state.light_uniforms[index].flags |= l->mode << LIGHT_FLAGS_BLEND_SHIFT;
			state.light_uniforms[index].flags |= l->shadow_filter << LIGHT_FLAGS_FILTER_SHIFT;
			if (clight->shadow.texture.is_valid()) {
				state.light_uniforms[index].flags |= LIGHT_FLAGS_HAS_SHADOW;
			}

			l->render_index_cache = index;

			index++;
			l = l->next_ptr;
		}

		if (index > 0) {
			RD::get_singleton()->buffer_update(state.lights_uniform_buffer, 0, sizeof(LightUniform) * index, &state.light_uniforms[0], true);
		}
	}

	//fill the list until rendering is possible.
	bool material_screen_texture_found = false;
	Item *ci = p_item_list;
	Rect2 back_buffer_rect;
	bool backbuffer_copy = false;
	RID screen_uniform_set;

	while (ci) {

		if (ci->copy_back_buffer) {
			backbuffer_copy = true;

			if (ci->copy_back_buffer->full) {
				back_buffer_rect = Rect2();
			} else {
				back_buffer_rect = ci->copy_back_buffer->rect;
			}
		}

		if (ci->material.is_valid()) {
			MaterialData *md = (MaterialData *)storage->material_get_data(ci->material, RasterizerStorageRD::SHADER_TYPE_2D);
			if (md && md->shader_data->valid) {

				if (md->shader_data->uses_screen_texture) {
					if (!material_screen_texture_found) {
						backbuffer_copy = true;
						back_buffer_rect = Rect2();
					}
					if (screen_uniform_set.is_null()) {
						RID backbuffer_shader = shader.canvas_shader.version_get_shader(md->shader_data->version, 0); //any version is fine
						screen_uniform_set = storage->render_target_get_back_buffer_uniform_set(p_to_render_target, backbuffer_shader);
					}
				}

				if (md->last_frame != RasterizerRD::get_frame_number()) {
					md->last_frame = RasterizerRD::get_frame_number();
					if (!RD::get_singleton()->uniform_set_is_valid(md->uniform_set)) {
						// uniform set may be gone because a dependency was erased. In this case, it will happen
						// if a texture is deleted, so just re-create it.
						storage->material_force_update_textures(ci->material, RasterizerStorageRD::SHADER_TYPE_2D);
					}
				}
			}
		}

		if (backbuffer_copy) {
			//render anything pending, including clearing if no items
			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list, screen_uniform_set);
			item_count = 0;

			storage->render_target_copy_to_back_buffer(p_to_render_target, back_buffer_rect);

			backbuffer_copy = false;
			material_screen_texture_found = true; //after a backbuffer copy, screen texture makes no further copies
		}

		items[item_count++] = ci;

		if (!ci->next || item_count == MAX_RENDER_ITEMS - 1) {
			_render_items(p_to_render_target, item_count, canvas_transform_inverse, p_light_list, screen_uniform_set);
			//then reset
			item_count = 0;
		}

		ci = ci->next;
	}
}

RID RasterizerCanvasRD::light_create() {

	CanvasLight canvas_light;
	canvas_light.shadow.size = 0;
	return canvas_light_owner.make_rid(canvas_light);
}

void RasterizerCanvasRD::light_set_texture(RID p_rid, RID p_texture) {
	CanvasLight *cl = canvas_light_owner.getornull(p_rid);
	ERR_FAIL_COND(!cl);
	if (cl->texture == p_texture) {
		return;
	}

	cl->texture = p_texture;
}
void RasterizerCanvasRD::light_set_use_shadow(RID p_rid, bool p_enable, int p_resolution) {
	CanvasLight *cl = canvas_light_owner.getornull(p_rid);
	ERR_FAIL_COND(!cl);
	ERR_FAIL_COND(p_resolution < 64);
	if (cl->shadow.texture.is_valid() == p_enable && p_resolution == cl->shadow.size) {
		return;
	}

	if (cl->shadow.texture.is_valid()) {

		RD::get_singleton()->free(cl->shadow.fb);
		RD::get_singleton()->free(cl->shadow.depth);
		RD::get_singleton()->free(cl->shadow.texture);
		cl->shadow.fb = RID();
		cl->shadow.texture = RID();
		cl->shadow.depth = RID();
	}

	if (p_enable) {

		Vector<RID> fb_textures;

		{ //texture
			RD::TextureFormat tf;
			tf.type = RD::TEXTURE_TYPE_2D;
			tf.width = p_resolution;
			tf.height = 1;
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;

			cl->shadow.texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			fb_textures.push_back(cl->shadow.texture);
		}
		{
			RD::TextureFormat tf;
			tf.type = RD::TEXTURE_TYPE_2D;
			tf.width = p_resolution;
			tf.height = 1;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_X8_D24_UNORM_PACK32, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_X8_D24_UNORM_PACK32 : RD::DATA_FORMAT_D32_SFLOAT;
			//chunks to write
			cl->shadow.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
			fb_textures.push_back(cl->shadow.depth);
		}

		cl->shadow.fb = RD::get_singleton()->framebuffer_create(fb_textures);
	}

	cl->shadow.size = p_resolution;
}

void RasterizerCanvasRD::light_update_shadow(RID p_rid, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) {

	CanvasLight *cl = canvas_light_owner.getornull(p_rid);
	ERR_FAIL_COND(cl->shadow.texture.is_null());

	for (int i = 0; i < 4; i++) {

		//make sure it remains orthogonal, makes easy to read angle later

		//light.basis.scale(Vector3(to_light.elements[0].length(),to_light.elements[1].length(),1));

		Vector<Color> cc;
		cc.push_back(Color(p_far, p_far, p_far, 1.0));
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(cl->shadow.fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, cc, 1.0, 0, Rect2i((cl->shadow.size / 4) * i, 0, (cl->shadow.size / 4), 1));

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

		Vector3 cam_target = Basis(Vector3(0, 0, Math_PI * 2 * ((i + 3) / 4.0))).xform(Vector3(0, 1, 0));
		projection = projection * CameraMatrix(Transform().looking_at(cam_target, Vector3(0, 0, -1)).affine_inverse());

		ShadowRenderPushConstant push_constant;
		for (int y = 0; y < 4; y++) {
			for (int x = 0; x < 4; x++) {
				push_constant.projection[y * 4 + x] = projection.matrix[y][x];
			}
		}
		static const Vector2 directions[4] = { Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1) };
		push_constant.direction[0] = directions[i].x;
		push_constant.direction[1] = directions[i].y;
		push_constant.pad[0] = 0;
		push_constant.pad[1] = 0;

		/*if (i == 0)
			*p_xform_cache = projection;*/

		LightOccluderInstance *instance = p_occluders;

		while (instance) {

			OccluderPolygon *co = occluder_polygon_owner.getornull(instance->occluder);

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

RID RasterizerCanvasRD::occluder_polygon_create() {

	OccluderPolygon occluder;
	occluder.point_count = 0;
	occluder.cull_mode = RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;
	return occluder_polygon_owner.make_rid(occluder);
}

void RasterizerCanvasRD::occluder_polygon_set_shape_as_lines(RID p_occluder, const Vector<Vector2> &p_lines) {

	OccluderPolygon *oc = occluder_polygon_owner.getornull(p_occluder);
	ERR_FAIL_COND(!oc);

	if (oc->point_count != p_lines.size() && oc->vertex_array.is_valid()) {

		RD::get_singleton()->free(oc->vertex_array);
		RD::get_singleton()->free(oc->vertex_buffer);
		RD::get_singleton()->free(oc->index_array);
		RD::get_singleton()->free(oc->index_buffer);

		oc->vertex_array = RID();
		oc->vertex_buffer = RID();
		oc->index_array = RID();
		oc->index_buffer = RID();
	}

	if (p_lines.size()) {

		Vector<uint8_t> geometry;
		Vector<uint8_t> indices;
		int lc = p_lines.size();

		geometry.resize(lc * 6 * sizeof(float));
		indices.resize(lc * 3 * sizeof(uint16_t));

		{
			uint8_t *vw = geometry.ptrw();
			float *vwptr = (float *)vw;
			uint8_t *iw = indices.ptrw();
			uint16_t *iwptr = (uint16_t *)iw;

			const Vector2 *lr = p_lines.ptr();

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
}
void RasterizerCanvasRD::occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) {
	OccluderPolygon *oc = occluder_polygon_owner.getornull(p_occluder);
	ERR_FAIL_COND(!oc);
	oc->cull_mode = p_mode;
}

void RasterizerCanvasRD::ShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_screen_texture = false;
	uses_material_samplers = false;

	if (code == String()) {
		return; //just invalid, but no error
	}

	ShaderCompilerRD::GeneratedCode gen_code;

	int light_mode = LIGHT_MODE_NORMAL;
	int blend_mode = BLEND_MODE_MIX;
	uses_screen_texture = false;

	ShaderCompilerRD::IdentifierActions actions;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);
	actions.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&blend_mode, BLEND_MODE_PMALPHA);
	actions.render_mode_values["blend_disabled"] = Pair<int *, int>(&blend_mode, BLEND_MODE_DISABLED);

	actions.render_mode_values["unshaded"] = Pair<int *, int>(&light_mode, LIGHT_MODE_UNSHADED);
	actions.render_mode_values["light_only"] = Pair<int *, int>(&light_mode, LIGHT_MODE_LIGHT_ONLY);

	actions.usage_flag_pointers["SCREEN_TEXTURE"] = &uses_screen_texture;

	actions.uniforms = &uniforms;

	RasterizerCanvasRD *canvas_singleton = (RasterizerCanvasRD *)RasterizerCanvas::singleton;

	Error err = canvas_singleton->shader.compiler.compile(RS::SHADER_CANVAS_ITEM, code, &actions, path, gen_code);

	ERR_FAIL_COND(err != OK);

	if (version.is_null()) {
		version = canvas_singleton->shader.canvas_shader.version_create();
	}

	if (gen_code.texture_uniforms.size() || uses_screen_texture) { //requires the samplers
		gen_code.defines.push_back("\n#define USE_MATERIAL_SAMPLERS\n");
		uses_material_samplers = true;
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
	canvas_singleton->shader.canvas_shader.version_set_code(version, gen_code.uniforms, gen_code.vertex_global, gen_code.vertex, gen_code.fragment_global, gen_code.light, gen_code.fragment, gen_code.defines);
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
			attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			attachment.color_blend_op = RD::BLEND_OP_ADD;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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

void RasterizerCanvasRD::ShaderData::set_default_texture_param(const StringName &p_name, RID p_texture) {
	if (!p_texture.is_valid()) {
		default_texture_params.erase(p_name);
	} else {
		default_texture_params[p_name] = p_texture;
	}
}
void RasterizerCanvasRD::ShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {

	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {

		if (E->get().texture_order >= 0) {
			order[E->get().texture_order + 100000] = E->key();
		} else {
			order[E->get().order] = E->key();
		}
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {

		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E->get()]);
		pi.name = E->get();
		p_param_list->push_back(pi);
	}
}

bool RasterizerCanvasRD::ShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RasterizerCanvasRD::ShaderData::is_animated() const {
	return false;
}
bool RasterizerCanvasRD::ShaderData::casts_shadows() const {
	return false;
}
Variant RasterizerCanvasRD::ShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.hint);
	}
	return Variant();
}

RasterizerCanvasRD::ShaderData::ShaderData() {
	valid = false;
	uses_screen_texture = false;
	uses_material_samplers = false;
}

RasterizerCanvasRD::ShaderData::~ShaderData() {
	RasterizerCanvasRD *canvas_singleton = (RasterizerCanvasRD *)RasterizerCanvas::singleton;
	ERR_FAIL_COND(!canvas_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		canvas_singleton->shader.canvas_shader.version_free(version);
	}
}

RasterizerStorageRD::ShaderData *RasterizerCanvasRD::_create_shader_func() {
	ShaderData *shader_data = memnew(ShaderData);
	return shader_data;
}
void RasterizerCanvasRD::MaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {

	RasterizerCanvasRD *canvas_singleton = (RasterizerCanvasRD *)RasterizerCanvas::singleton;

	if ((uint32_t)ubo_data.size() != shader_data->ubo_size) {
		p_uniform_dirty = true;
		if (uniform_buffer.is_valid()) {
			RD::get_singleton()->free(uniform_buffer);
			uniform_buffer = RID();
		}

		ubo_data.resize(shader_data->ubo_size);
		if (ubo_data.size()) {
			uniform_buffer = RD::get_singleton()->uniform_buffer_create(ubo_data.size());
			memset(ubo_data.ptrw(), 0, ubo_data.size()); //clear
		}

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	//check whether buffer changed
	if (p_uniform_dirty && ubo_data.size()) {

		update_uniform_buffer(shader_data->uniforms, shader_data->ubo_offsets.ptr(), p_parameters, ubo_data.ptrw(), ubo_data.size(), false);
		RD::get_singleton()->buffer_update(uniform_buffer, 0, ubo_data.size(), ubo_data.ptrw());
	}

	uint32_t tex_uniform_count = shader_data->texture_uniforms.size();

	if ((uint32_t)texture_cache.size() != tex_uniform_count) {
		texture_cache.resize(tex_uniform_count);
		p_textures_dirty = true;

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	if (p_textures_dirty && tex_uniform_count) {

		update_textures(p_parameters, shader_data->default_texture_params, shader_data->texture_uniforms, texture_cache.ptrw(), false);
	}

	if (shader_data->ubo_size == 0 && !shader_data->uses_material_samplers) {
		// This material does not require an uniform set, so don't create it.
		return;
	}

	if (!p_textures_dirty && uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//no reason to update uniform set, only UBO (or nothing) was needed to update
		return;
	}

	Vector<RD::Uniform> uniforms;

	{
		if (shader_data->uses_material_samplers) {
			//needs samplers for the material (uses custom textures) create them
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 0;
			u.ids.resize(12);
			RID *ids_ptr = u.ids.ptrw();
			ids_ptr[0] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[1] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[2] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[3] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[4] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[5] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[6] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[7] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[8] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[9] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[10] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[11] = canvas_singleton->storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			uniforms.push_back(u);
		}

		if (shader_data->ubo_size) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 1;
			u.ids.push_back(uniform_buffer);
			uniforms.push_back(u);
		}

		const RID *textures = texture_cache.ptrw();
		for (uint32_t i = 0; i < tex_uniform_count; i++) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2 + i;
			u.ids.push_back(textures[i]);
			uniforms.push_back(u);
		}
	}

	uniform_set = RD::get_singleton()->uniform_set_create(uniforms, canvas_singleton->shader.canvas_shader.version_get_shader(shader_data->version, 0), 1);
}
RasterizerCanvasRD::MaterialData::~MaterialData() {
	if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		RD::get_singleton()->free(uniform_set);
	}

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
	}
}

RasterizerStorageRD::MaterialData *RasterizerCanvasRD::_create_material_func(ShaderData *p_shader) {
	MaterialData *material_data = memnew(MaterialData);
	material_data->shader_data = p_shader;
	material_data->last_frame = false;
	//update will happen later anyway so do nothing.
	return material_data;
}

void RasterizerCanvasRD::set_time(double p_time) {
	state.time = p_time;
}

void RasterizerCanvasRD::update() {
	_dispose_bindings();
}

RasterizerCanvasRD::RasterizerCanvasRD(RasterizerStorageRD *p_storage) {
	storage = p_storage;

	{ //create default samplers

		default_samplers.default_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
		default_samplers.default_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
	}

	{ //shader variants

		uint32_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

		String global_defines;
		if (textures_per_stage <= 16) {
			//ARM pretty much, and very old Intel GPUs under Linux
			state.max_lights_per_item = 4; //sad
			global_defines += "#define MAX_LIGHT_TEXTURES 4\n";
		} else if (textures_per_stage <= 32) {
			//Apple (Metal)
			state.max_lights_per_item = 8; //sad
			global_defines += "#define MAX_LIGHT_TEXTURES 8\n";
		} else {
			//Anything else (16 lights per item)
			state.max_lights_per_item = DEFAULT_MAX_LIGHTS_PER_ITEM;
			global_defines += "#define MAX_LIGHT_TEXTURES " + itos(DEFAULT_MAX_LIGHTS_PER_ITEM) + "\n";
		}

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
		shader.default_version_rd_shader_light = shader.canvas_shader.version_get_shader(shader.default_version, SHADER_VARIANT_QUAD_LIGHT);

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
				shader.pipeline_variants.variants[i][j].setup(shader_variant, primitive[j], RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_blend(), 0);
			}
		}
	}

	{
		//shader compiler
		ShaderCompilerRD::DefaultIdentifierActions actions;

		actions.renames["VERTEX"] = "vertex";
		actions.renames["LIGHT_VERTEX"] = "light_vertex";
		actions.renames["SHADOW_VERTEX"] = "shadow_vertex";
		actions.renames["UV"] = "uv";
		actions.renames["POINT_SIZE"] = "gl_PointSize";

		actions.renames["WORLD_MATRIX"] = "world_matrix";
		actions.renames["CANVAS_MATRIX"] = "canvas_data.canvas_transform";
		actions.renames["SCREEN_MATRIX"] = "canvas_data.screen_transform";
		actions.renames["TIME"] = "canvas_data.time";
		actions.renames["AT_LIGHT_PASS"] = "false";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";

		actions.renames["COLOR"] = "color";
		actions.renames["NORMAL"] = "normal";
		actions.renames["NORMALMAP"] = "normal_map";
		actions.renames["NORMALMAP_DEPTH"] = "normal_depth";
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

		actions.renames["LIGHT_POSITION"] = "light_pos";
		actions.renames["LIGHT_COLOR"] = "light_color";
		actions.renames["LIGHT_ENERGY"] = "light_energy";
		actions.renames["LIGHT"] = "light";
		actions.renames["SHADOW_MODULATE"] = "shadow_modulate";

		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
		actions.usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";
		actions.usage_defines["SCREEN_PIXEL_SIZE"] = "@SCREEN_UV";
		actions.usage_defines["NORMAL"] = "#define NORMAL_USED\n";
		actions.usage_defines["NORMALMAP"] = "#define NORMALMAP_USED\n";
		actions.usage_defines["LIGHT"] = "#define LIGHT_SHADER_CODE_USED\n";

		actions.render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";

		actions.custom_samplers["TEXTURE"] = "texture_sampler";
		actions.custom_samplers["NORMAL_TEXTURE"] = "texture_sampler";
		actions.custom_samplers["SPECULAR_SHININESS_TEXTURE"] = "texture_sampler";
		actions.custom_samplers["SCREEN_TEXTURE"] = "material_samplers[3]"; //mipmap and filter for screen texture
		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 2;
		actions.texture_layout_set = 1;
		actions.base_uniform_string = "material.";
		actions.default_filter = ShaderLanguage::FILTER_LINEAR;
		actions.default_repeat = ShaderLanguage::REPEAT_DISABLE;
		actions.base_varying_index = 4;

		shader.compiler.initialize(actions);
	}

	{ //shadow rendering
		Vector<String> versions;
		versions.push_back(String()); //no versions
		shadow_render.shader.initialize(versions);

		{
			Vector<RD::AttachmentFormat> attachments;

			RD::AttachmentFormat af_color;
			af_color.format = RD::DATA_FORMAT_R32_SFLOAT;
			af_color.usage_flags = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

			attachments.push_back(af_color);

			RD::AttachmentFormat af_depth;
			af_depth.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			af_depth.usage_flags = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			attachments.push_back(af_depth);

			shadow_render.framebuffer_format = RD::get_singleton()->framebuffer_format_create(attachments);
		}

		//pipelines
		Vector<RD::VertexDescription> vf;
		RD::VertexDescription vd;
		vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
		vd.location = 0;
		vd.offset = 0;
		vd.stride = sizeof(float) * 3;
		vf.push_back(vd);
		shadow_render.vertex_format = RD::get_singleton()->vertex_format_create(vf);

		shadow_render.shader_version = shadow_render.shader.version_create();

		for (int i = 0; i < 3; i++) {
			RD::PipelineRasterizationState rs;
			rs.cull_mode = i == 0 ? RD::POLYGON_CULL_DISABLED : (i == 1 ? RD::POLYGON_CULL_FRONT : RD::POLYGON_CULL_BACK);
			RD::PipelineDepthStencilState ds;
			ds.enable_depth_write = true;
			ds.enable_depth_test = true;
			ds.depth_compare_operator = RD::COMPARE_OP_LESS;
			shadow_render.render_pipelines[i] = RD::get_singleton()->render_pipeline_create(shadow_render.shader.version_get_shader(shadow_render.shader_version, 0), shadow_render.framebuffer_format, shadow_render.vertex_format, RD::RENDER_PRIMITIVE_TRIANGLES, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{ //bindings
		bindings.id_generator = 0;
		//generate for 0
		bindings.default_empty = request_texture_binding(RID(), RID(), RID(), RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, RID());

		{ //state allocate
			state.canvas_state_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(State::Buffer));
			state.lights_uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(LightUniform) * state.max_lights_per_render);

			RD::SamplerState shadow_sampler_state;
			shadow_sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
			shadow_sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
			shadow_sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT; //shadow wrap around
			shadow_sampler_state.compare_op = RD::COMPARE_OP_GREATER;
			state.shadow_sampler = RD::get_singleton()->sampler_create(shadow_sampler_state);
		}
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

	//create functions for shader and material
	storage->shader_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_2D, _create_shader_funcs);
	storage->material_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_2D, _create_material_funcs);

	state.time = 0;

	static_assert(sizeof(PushConstant) == 128);
}

bool RasterizerCanvasRD::free(RID p_rid) {

	if (canvas_light_owner.owns(p_rid)) {
		CanvasLight *cl = canvas_light_owner.getornull(p_rid);
		ERR_FAIL_COND_V(!cl, false);
		light_set_use_shadow(p_rid, false, 64);
		canvas_light_owner.free(p_rid);
	} else if (occluder_polygon_owner.owns(p_rid)) {
		occluder_polygon_set_shape_as_lines(p_rid, Vector<Vector2>());
		occluder_polygon_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

RasterizerCanvasRD::~RasterizerCanvasRD() {

	//canvas state

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
	{

		free_texture_binding(bindings.default_empty);

		//dispose pending
		_dispose_bindings();
		//anything remains?
		if (bindings.texture_bindings.size()) {
			ERR_PRINT("Some texture bindings were not properly freed (leaked canvasitems?");
			const TextureBindingID *key = nullptr;
			while ((key = bindings.texture_bindings.next(key))) {
				TextureBinding *tb = bindings.texture_bindings[*key];
				tb->reference_count = 1;
				free_texture_binding(*key);
			}
			//dispose pending
			_dispose_bindings();
		}
	}

	//shaders

	shader.canvas_shader.version_free(shader.default_version);

	//buffers
	{
		RD::get_singleton()->free(shader.quad_index_array);
		RD::get_singleton()->free(shader.quad_index_buffer);
		//primitives are erase by dependency
	}

	//pipelines don't need freeing, they are all gone after shaders are gone
}

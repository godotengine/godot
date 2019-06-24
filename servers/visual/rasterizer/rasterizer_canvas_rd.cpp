#include "rasterizer_canvas_rd.h"
#include "core/math/math_funcs.h"
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

RID RasterizerCanvasRD::_create_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, VisualServer::CanvasItemTextureFilter p_filter, VisualServer::CanvasItemTextureRepeat p_repeat, RID p_multimesh) {

	Vector<RD::Uniform> uniform_set;

	{ // COLOR TEXTURE
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 1;
		RID texture = storage->texture_get_rd_texture(p_texture);
		if (!texture.is_valid()) {
			//use default white texture
			texture = default_textures.white_texture;
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
			texture = default_textures.normal_texture;
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
			texture = default_textures.white_texture;
		}
		u.ids.push_back(texture);
		uniform_set.push_back(u);
	}

	{ // SAMPLER
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 4;
		RID sampler = default_samplers.samplers[p_filter][p_repeat];
		ERR_FAIL_COND_V(sampler.is_null(), RID());
		u.ids.push_back(sampler);
		uniform_set.push_back(u);
	}

	{ // MULTIMESH TEXTURE BUFFER
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE_BUFFER;
		u.binding = 5;
		u.ids.push_back(default_textures.default_multimesh_tb);
		uniform_set.push_back(u);
	}

	return RD::get_singleton()->uniform_set_create(uniform_set, shader.default_version_rd_shader, 0);
}

RasterizerCanvas::TextureBindingID RasterizerCanvasRD::request_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, VisualServer::CanvasItemTextureFilter p_filter, VisualServer::CanvasItemTextureRepeat p_repeat, RID p_multimesh) {

	if (p_filter == VS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT) {
		p_filter = default_samplers.default_filter;
	}

	if (p_repeat == VS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) {
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
	uint32_t base_offset = 0;
	uint32_t stride = 2; //vertices always repeat
	if ((uint32_t)p_colors.size() == vertex_count) {
		stride += 4;
	} else {
		base_offset += 4;
	}
	if ((uint32_t)p_uvs.size() == vertex_count) {
		stride += 2;
	} else {
		base_offset += 2;
	}
	if ((uint32_t)p_bones.size() == vertex_count * 4) {
		stride += 4;
	} else {
		base_offset += 4;
	}
	if ((uint32_t)p_weights.size() == vertex_count * 4) {
		stride += 4;
	} else {
		base_offset += 4;
	}

	uint32_t buffer_size = base_offset + stride * p_points.size();

	PoolVector<uint8_t> polygon_buffer;
	polygon_buffer.resize(buffer_size * sizeof(float));
	Vector<RD::VertexDescription> descriptions;
	descriptions.resize(5);

	{
		PoolVector<uint8_t>::Read r = polygon_buffer.read();
		float *fptr = (float *)r.ptr();
		uint32_t *uptr = (uint32_t *)r.ptr();
		uint32_t single_offset = 0;

		{ //vertices
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = VS::ARRAY_VERTEX;
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
		if ((uint32_t)p_colors.size() == vertex_count) {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = VS::ARRAY_COLOR;
			vd.stride = stride * sizeof(float);

			descriptions.write[1] = vd;

			const Color *color_ptr = p_colors.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = color_ptr[i].r;
				fptr[base_offset + i * stride + 1] = color_ptr[i].g;
				fptr[base_offset + i * stride + 2] = color_ptr[i].b;
				fptr[base_offset + i * stride + 3] = color_ptr[i].a;
			}
			base_offset += 4;
		} else {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = single_offset * sizeof(float);
			vd.location = VS::ARRAY_COLOR;
			vd.stride = 0;

			descriptions.write[1] = vd;

			Color color = p_colors.size() ? p_colors[0] : Color(1, 1, 1, 1);
			fptr[single_offset + 0] = color.r;
			fptr[single_offset + 1] = color.g;
			fptr[single_offset + 2] = color.b;
			fptr[single_offset + 3] = color.a;

			single_offset += 4;
		}

		//uvs
		if ((uint32_t)p_uvs.size() == vertex_count) {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = VS::ARRAY_TEX_UV;
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
			vd.offset = single_offset * sizeof(float);
			vd.location = VS::ARRAY_TEX_UV;
			vd.stride = 0;

			descriptions.write[2] = vd;

			Vector2 uv;
			fptr[single_offset + 0] = uv.x;
			fptr[single_offset + 1] = uv.y;

			single_offset += 2;
		}

		//bones
		if ((uint32_t)p_indices.size() == vertex_count * 4) {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
			vd.offset = base_offset * sizeof(float);
			vd.location = VS::ARRAY_BONES;
			vd.stride = stride * sizeof(float);

			descriptions.write[3] = vd;

			const int *bone_ptr = p_bones.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				uptr[base_offset + i * stride + 0] = bone_ptr[i * 4 + 0];
				uptr[base_offset + i * stride + 1] = bone_ptr[i * 4 + 1];
				uptr[base_offset + i * stride + 2] = bone_ptr[i * 4 + 2];
				uptr[base_offset + i * stride + 3] = bone_ptr[i * 4 + 3];
			}

			base_offset += 4;
		} else {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
			vd.offset = single_offset * sizeof(float);
			vd.location = VS::ARRAY_BONES;
			vd.stride = 0;

			descriptions.write[3] = vd;

			uptr[single_offset + 0] = 0;
			uptr[single_offset + 1] = 0;
			uptr[single_offset + 2] = 0;
			uptr[single_offset + 3] = 0;
			single_offset += 4;
		}

		//bones
		if ((uint32_t)p_weights.size() == vertex_count * 4) {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = base_offset * sizeof(float);
			vd.location = VS::ARRAY_WEIGHTS;
			vd.stride = stride * sizeof(float);

			descriptions.write[4] = vd;

			const float *weight_ptr = p_weights.ptr();

			for (uint32_t i = 0; i < vertex_count; i++) {
				fptr[base_offset + i * stride + 0] = weight_ptr[i * 4 + 0];
				fptr[base_offset + i * stride + 1] = weight_ptr[i * 4 + 1];
				fptr[base_offset + i * stride + 2] = weight_ptr[i * 4 + 2];
				fptr[base_offset + i * stride + 3] = weight_ptr[i * 4 + 3];
			}

			base_offset += 4;
		} else {
			RD::VertexDescription vd;
			vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			vd.offset = single_offset * sizeof(float);
			vd.location = VS::ARRAY_WEIGHTS;
			vd.stride = 0;

			descriptions.write[4] = vd;

			fptr[single_offset + 0] = 0.0;
			fptr[single_offset + 1] = 0.0;
			fptr[single_offset + 2] = 0.0;
			fptr[single_offset + 3] = 0.0;
			single_offset += 4;
		}

		//check that everything is as it should be
		ERR_FAIL_COND_V(single_offset != (base_offset - stride), 0);
		ERR_FAIL_COND_V(((base_offset - stride) + stride * vertex_count) != buffer_size, 0);
	}

	RD::VertexFormatID vertex_id = RD::get_singleton()->vertex_format_create(descriptions);
	ERR_FAIL_COND_V(vertex_id == RD::INVALID_ID, 0);

	PolygonBuffers pb;
	pb.vertex_buffer = RD::get_singleton()->vertex_buffer_create(polygon_buffer.size(), polygon_buffer);
	Vector<RID> buffers;
	buffers.resize(descriptions.size());
	for (int i = 0; i < descriptions.size(); i++) {
		buffers.write[i] = pb.vertex_buffer;
	}
	pb.vertex_array = RD::get_singleton()->vertex_array_create(p_points.size(), vertex_id, buffers);

	if (p_indices.size()) {
		//create indices, as indices were requested
		PoolVector<uint8_t> index_buffer;
		index_buffer.resize(p_indices.size() * sizeof(int32_t));
		{
			PoolVector<uint8_t>::Write w = index_buffer.write();
			copymem(w.ptr(), p_indices.ptr(), sizeof(int32_t) * p_indices.size());
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

Size2i RasterizerCanvasRD::_bind_texture_binding(TextureBindingID p_binding, RD::DrawListID p_draw_list) {

	TextureBinding **texture_binding_ptr = bindings.texture_bindings.getptr(p_binding);
	ERR_FAIL_COND_V(!texture_binding_ptr, Size2i());
	TextureBinding *texture_binding = *texture_binding_ptr;

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
void RasterizerCanvasRD::_render_item(RD::DrawListID p_draw_list, const Item *p_item, RenderTargetFormat p_render_target_format, RD::TextureSamples p_samples, const Color &p_modulate, const Transform2D &p_canvas_transform_inverse, Item *&current_clip) {

	int cc = p_item->commands.size();
	const Item::Command *const *commands = p_item->commands.ptr();

	//create an empty push constant
	PushConstant push_constant;
	Transform2D base_transform = p_canvas_transform_inverse * p_item->final_transform;
	_update_transform_2d_to_mat2x3(base_transform, push_constant.world);
	for (int i = 0; i < 4; i++) {
		push_constant.modulation[i] = 0;
		push_constant.ninepatch_margins[i] = 0;
		push_constant.src_rect[i] = 0;
		push_constant.dst_rect[i] = 0;
	}
	push_constant.flags = 0;
	push_constant.specular_shininess = 0xFFFFFFFF;
	push_constant.color_texture_pixel_size[0] = 0;
	push_constant.color_texture_pixel_size[1] = 0;
	push_constant.pad[0] = 0;
	push_constant.pad[1] = 0;
	push_constant.pad[2] = 0;

	PipelineVariants *pipeline_variants = &shader.pipeline_variants;

	bool reclip = false;

	for (int i = 0; i < cc; i++) {

		const Item::Command *c = commands[i];
		push_constant.flags = 0; //reset on each command for sanity

		switch (c->type) {
			case Item::Command::TYPE_RECT: {

				const Item::CommandRect *rect = static_cast<const Item::CommandRect *>(c);

				//bind pipeline
				{
					RID pipeline = pipeline_variants->variants[p_render_target_format][PIPELINE_VARIANT_QUAD].get_render_pipeline(RD::INVALID_ID, p_samples);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				Size2 texpixel_size;
				{
					texpixel_size = _bind_texture_binding(rect->texture_binding.binding_id, p_draw_list);
					texpixel_size.x = 1.0 / texpixel_size.x;
					texpixel_size.y = 1.0 / texpixel_size.y;
				}

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

				push_constant.modulation[0] = rect->modulate.r * p_modulate.r;
				push_constant.modulation[1] = rect->modulate.g * p_modulate.g;
				push_constant.modulation[2] = rect->modulate.b * p_modulate.b;
				push_constant.modulation[3] = rect->modulate.a;

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
					RID pipeline = pipeline_variants->variants[p_render_target_format][PIPELINE_VARIANT_NINEPATCH].get_render_pipeline(RD::INVALID_ID, p_samples);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				Size2 texpixel_size;
				{
					texpixel_size = _bind_texture_binding(np->texture_binding.binding_id, p_draw_list);
					texpixel_size.x = 1.0 / texpixel_size.x;
					texpixel_size.y = 1.0 / texpixel_size.y;
				}

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

				push_constant.modulation[0] = np->color.r * p_modulate.r;
				push_constant.modulation[1] = np->color.g * p_modulate.g;
				push_constant.modulation[2] = np->color.b * p_modulate.b;
				push_constant.modulation[3] = np->color.a * p_modulate.a;

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
					static const PipelineVariant variant[VS::PRIMITIVE_MAX] = { PIPELINE_VARIANT_ATTRIBUTE_POINTS, PIPELINE_VARIANT_ATTRIBUTE_LINES, PIPELINE_VARIANT_ATTRIBUTE_TRIANGLES };
					ERR_CONTINUE(polygon->primitive < 0 || polygon->primitive >= VS::PRIMITIVE_MAX);
					RID pipeline = pipeline_variants->variants[p_render_target_format][variant[polygon->primitive]].get_render_pipeline(pb->vertex_format_id, p_samples);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				if (polygon->primitive == VS::PRIMITIVE_LINES) {
					//not supported in most hardware, so pointless
					//RD::get_singleton()->draw_list_set_line_width(p_draw_list, polygon->line_width);
				}

				//bind textures

				Size2 texpixel_size;
				{
					texpixel_size = _bind_texture_binding(polygon->texture_binding.binding_id, p_draw_list);
					texpixel_size.x = 1.0 / texpixel_size.x;
					texpixel_size.y = 1.0 / texpixel_size.y;
				}

				push_constant.modulation[0] = p_modulate.r;
				push_constant.modulation[1] = p_modulate.g;
				push_constant.modulation[2] = p_modulate.b;
				push_constant.modulation[3] = p_modulate.a;

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
					RID pipeline = pipeline_variants->variants[p_render_target_format][variant[primitive->point_count - 1]].get_render_pipeline(RD::INVALID_ID, p_samples);
					RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
				}

				//bind textures

				{
					_bind_texture_binding(primitive->texture_binding.binding_id, p_draw_list);
				}

				for (uint32_t j = 0; j < primitive->point_count; j++) {
					push_constant.points[j * 2 + 0] = primitive->points[j].x;
					push_constant.points[j * 2 + 1] = primitive->points[j].y;
					push_constant.uvs[j * 2 + 0] = primitive->uvs[j].x;
					push_constant.uvs[j * 2 + 1] = primitive->uvs[j].y;
					Color col = primitive->colors[j] * p_modulate;
					push_constant.colors[j * 2 + 0] = (uint32_t(Math::make_half_float(col.g)) << 16) | Math::make_half_float(col.r);
					push_constant.colors[j * 2 + 1] = (uint32_t(Math::make_half_float(col.a)) << 16) | Math::make_half_float(col.b);
				}
				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));

				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, primitive_arrays.index_array[primitive->point_count - 1]);

				RD::get_singleton()->draw_list_draw(p_draw_list, true);

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

						glVertexAttrib4f(VS::ARRAY_COLOR, mesh->modulate.r, mesh->modulate.g, mesh->modulate.b, mesh->modulate.a);

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

				state.canvas_shader.set_conditional(CanvasShaderGLES3::USE_INSTANCE_CUSTOM, multi_mesh->custom_data_format != VS::MULTIMESH_CUSTOM_DATA_NONE);
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

					if (multi_mesh->transform_format == VS::MULTIMESH_TRANSFORM_3D) {
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

						case VS::MULTIMESH_COLOR_NONE: {
							glDisableVertexAttribArray(11);
							glVertexAttrib4f(11, 1, 1, 1, 1);
						} break;
						case VS::MULTIMESH_COLOR_8BIT: {
							glEnableVertexAttribArray(11);
							glVertexAttribPointer(11, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, CAST_INT_TO_UCHAR_PTR(color_ofs));
							glVertexAttribDivisor(11, 1);
							custom_data_ofs += 4;

						} break;
						case VS::MULTIMESH_COLOR_FLOAT: {
							glEnableVertexAttribArray(11);
							glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, CAST_INT_TO_UCHAR_PTR(color_ofs));
							glVertexAttribDivisor(11, 1);
							custom_data_ofs += 4 * 4;
						} break;
					}

					switch (multi_mesh->custom_data_format) {

						case VS::MULTIMESH_CUSTOM_DATA_NONE: {
							glDisableVertexAttribArray(12);
							glVertexAttrib4f(12, 1, 1, 1, 1);
						} break;
						case VS::MULTIMESH_CUSTOM_DATA_8BIT: {
							glEnableVertexAttribArray(12);
							glVertexAttribPointer(12, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, CAST_INT_TO_UCHAR_PTR(custom_data_ofs));
							glVertexAttribDivisor(12, 1);

						} break;
						case VS::MULTIMESH_CUSTOM_DATA_FLOAT: {
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

				if (particles->draw_order != VS::PARTICLES_DRAW_ORDER_LIFETIME) {

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
					glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, NULL);
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
						glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, stride, NULL);
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
				base_transform = base_transform * transform->xform;
				_update_transform_2d_to_mat2x3(base_transform, push_constant.world);

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
	}

	if (current_clip && reclip) {
		//will make it re-enable clipping if needed afterwards
		current_clip = NULL;
	}
}

void RasterizerCanvasRD::_render_items(RID p_to_render_target, int p_item_count, const Color &p_modulate, const Transform2D &p_transform) {

	Item *current_clip = NULL;

	RenderTargetFormat render_target_format = RENDER_TARGET_FORMAT_8_BIT_INT;
	Transform2D canvas_transform_inverse = p_transform.affine_inverse();

	RID framebuffer = storage->render_target_get_rd_framebuffer(p_to_render_target);

	Vector<Color> clear_colors;
	bool clear = false;
	if (storage->render_target_is_clear_requested(p_to_render_target)) {
		clear = true;
		clear_colors.push_back(storage->render_target_get_clear_request_color(p_to_render_target));
		storage->render_target_disable_clear_request(p_to_render_target);
	}
#warning TODO obtain from framebuffer format eventually when this is implemented
	RD::TextureSamples texture_samples = RD::TEXTURE_SAMPLES_1;
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer, clear ? RD::INITIAL_ACTION_CLEAR : RD::INITIAL_ACTION_KEEP_COLOR, RD::FINAL_ACTION_READ_COLOR_DISCARD_DEPTH, clear_colors);

	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, state.canvas_state_uniform_set, 3);

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

		if (false) { //not skeleton

			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, shader.default_skeleton_uniform_set, 1);
		}

		_render_item(draw_list, ci, render_target_format, texture_samples, p_modulate, canvas_transform_inverse, current_clip);
	}

	RD::get_singleton()->draw_list_end();
}

void RasterizerCanvasRD::_update_canvas_state_uniform_set() {

	if (state.canvas_state_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(state.canvas_state_uniform_set)) {
		return; //nothing to update
	}

	Vector<RD::Uniform> uniforms;

	RD::Uniform u;
	u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
	u.binding = 0;
	u.ids.push_back(state.canvas_state_buffer);
	uniforms.push_back(u);

	state.canvas_state_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader.default_version_rd_shader, 3); // uses index 3
}

void RasterizerCanvasRD::canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, const Transform2D &p_canvas_transform) {

	int item_count = 0;

	//setup canvas state uniforms if needed
	_update_canvas_state_uniform_set();

	{
		//update canvas state uniform buffer
		State::Buffer state_buffer;

		Size2i ssize = storage->render_target_get_size(p_to_render_target);

		Transform screen_transform;
		screen_transform.translate(-(ssize.width / 2.0f), -(ssize.height / 2.0f), 0.0f);
		screen_transform.scale(Vector3(2.0f / ssize.width, 2.0f / ssize.height, 1.0f));
		_update_transform_to_mat4(screen_transform, state_buffer.screen_transform);
		_update_transform_2d_to_mat4(p_canvas_transform, state_buffer.canvas_transform);
		RD::get_singleton()->buffer_update(state.canvas_state_buffer, 0, sizeof(State::Buffer), &state_buffer, true);
	}

	//fill the list until rendering is possible.
	Item *ci = p_item_list;
	while (ci) {

		items[item_count++] = ci;

		bool backbuffer_copy = ci->copy_back_buffer; // || shader uses SCREEN_TEXTURE
		if (!ci->next || backbuffer_copy || item_count == MAX_RENDER_ITEMS - 1) {
			_render_items(p_to_render_target, item_count, p_modulate, p_canvas_transform);
			//then reset
			item_count = 0;
		}

		if (ci->copy_back_buffer) {

			if (ci->copy_back_buffer->full) {

				//_copy_texscreen(Rect2());
			} else {
				//_copy_texscreen(ci->copy_back_buffer->rect);
			}
		}

		ci = ci->next;
	}
}

void RasterizerCanvasRD::update() {
	_dispose_bindings();
}

RasterizerCanvasRD::RasterizerCanvasRD(RasterizerStorageRD *p_storage) {
	storage = p_storage;

	{ //create default textures

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.type = RD::TEXTURE_TYPE_2D;

		PoolVector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<PoolVector<uint8_t> > vpv;
			vpv.push_back(pv);
			default_textures.white_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<PoolVector<uint8_t> > vpv;
			vpv.push_back(pv);
			default_textures.black_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 128);
			pv.set(i * 4 + 1, 128);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<PoolVector<uint8_t> > vpv;
			vpv.push_back(pv);
			default_textures.normal_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 128);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<PoolVector<uint8_t> > vpv;
			vpv.push_back(pv);
			default_textures.aniso_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		default_textures.default_multimesh_tb = RD::get_singleton()->texture_buffer_create(16, RD::DATA_FORMAT_R8G8B8A8_UNORM, pv);
	}

	{ //create default samplers

		for (int i = 1; i < VS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
			for (int j = 1; j < VS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
				RD::SamplerState sampler_state;
				switch (i) {
					case VS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
						sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
						sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
						sampler_state.max_lod = 0;
					} break;
					case VS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR: {

						sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
						sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
						sampler_state.max_lod = 0;
					} break;
					case VS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
						sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
						sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;

					} break;
					case VS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIMPAMPS: {
						sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
						sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					} break;
					default: {
					}
				}
				switch (j) {
					case VS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {

						sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
						sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

					} break;
					case VS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
						sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
						sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
					} break;
					case VS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
						sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
						sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					} break;
					default: {
					}
				}

				default_samplers.samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
			}
		}

		default_samplers.default_filter = VS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
		default_samplers.default_repeat = VS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
	}

	{ //shader variants
		Vector<String> variants;
		variants.push_back(""); //none by default is first variant
		variants.push_back("#define USE_NINEPATCH\n"); //ninepatch is the second variant
		variants.push_back("#define USE_PRIMITIVE\n"); //primitve is the third
		variants.push_back("#define USE_PRIMITIVE\n#define USE_POINT_SIZE\n"); //points need point size
		variants.push_back("#define USE_ATTRIBUTES\n"); // attributes for vertex arrays
		variants.push_back("#define USE_ATTRIBUTES\n#define USE_POINT_SIZE\n"); //attributes with point size
		shader.canvas_shader.initialize(variants);

		shader.default_version = shader.canvas_shader.version_create();

		{
			//framebuffer formats
			RD::AttachmentFormat af;
			af.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			af.samples = RD::TEXTURE_SAMPLES_1;
			af.usage_flags = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			Vector<RD::AttachmentFormat> formats;
			formats.push_back(af);
			shader.framebuffer_formats[RENDER_TARGET_FORMAT_8_BIT_INT] = RD::get_singleton()->framebuffer_format_create(formats);

			formats.clear();
			af.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			formats.push_back(af);
			shader.framebuffer_formats[RENDER_TARGET_FORMAT_16_BIT_FLOAT] = RD::get_singleton()->framebuffer_format_create(formats);
		}

		for (int i = 0; i < RENDER_TARGET_FORMAT_MAX; i++) {
			RD::FramebufferFormatID fb_format = shader.framebuffer_formats[i];
			for (int j = 0; j < PIPELINE_VARIANT_MAX; j++) {
				RD::RenderPrimitive primitive[PIPELINE_VARIANT_MAX] = {
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_LINES,
					RD::RENDER_PRIMITIVE_POINTS,
					RD::RENDER_PRIMITIVE_TRIANGLES,
					RD::RENDER_PRIMITIVE_LINES,
					RD::RENDER_PRIMITIVE_POINTS,
				};
				ShaderVariant shader_variants[PIPELINE_VARIANT_MAX] = {
					SHADER_VARIANT_QUAD,
					SHADER_VARIANT_NINEPATCH,
					SHADER_VARIANT_PRIMITIVE,
					SHADER_VARIANT_PRIMITIVE,
					SHADER_VARIANT_PRIMITIVE_POINTS,
					SHADER_VARIANT_ATTRIBUTES,
					SHADER_VARIANT_ATTRIBUTES,
					SHADER_VARIANT_ATTRIBUTES_POINTS
				};

				RID shader_variant = shader.canvas_shader.version_get_shader(shader.default_version, shader_variants[j]);
				shader.pipeline_variants.variants[i][j].setup(shader_variant, fb_format, primitive[j], RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_blend(), 0);
			}
		}

		shader.default_version_rd_shader = shader.canvas_shader.version_get_shader(shader.default_version, 0);
	}

	{ //bindings
		bindings.id_generator = 0;
		//generate for 0
		bindings.default_empty = request_texture_binding(RID(), RID(), RID(), VS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, VS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, RID());

		{ //state allocate
			state.canvas_state_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(State::Buffer));
		}
	}

	{
		//polygon buffers
		polygon_buffers.last_id = 1;
	}

	{ // default index buffer

		PoolVector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			PoolVector<uint8_t>::Write w = pv.write();
			int *p32 = (int *)w.ptr();
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

		shader.default_skeleton_uniform = RD::get_singleton()->uniform_buffer_create(sizeof(SkeletonUniform));
		SkeletonUniform su;
		_update_transform_2d_to_mat4(Transform2D(), su.skeleton_inverse);
		_update_transform_2d_to_mat4(Transform2D(), su.skeleton_transform);
		RD::get_singleton()->buffer_update(shader.default_skeleton_uniform, 0, sizeof(SkeletonUniform), &su);
	}

	{ //default material uniform set
		Vector<RD::Uniform> default_material_uniforms;
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		u.binding = 2;
		u.ids.push_back(shader.default_skeleton_uniform);
		default_material_uniforms.push_back(u);

		u.ids.clear();
		u.type = RD::UNIFORM_TYPE_TEXTURE_BUFFER;
		u.binding = 1;
		u.ids.push_back(default_textures.default_multimesh_tb);
		default_material_uniforms.push_back(u);

		shader.default_skeleton_uniform_set = RD::get_singleton()->uniform_set_create(default_material_uniforms, shader.canvas_shader.version_get_shader(shader.default_version, SHADER_VARIANT_ATTRIBUTES), 2);
	}

	ERR_FAIL_COND(sizeof(PushConstant) != 128);
}

RasterizerCanvasRD::~RasterizerCanvasRD() {

	//canvas state

	if (state.canvas_state_buffer.is_valid()) {
		RD::get_singleton()->free(state.canvas_state_buffer);
	}

	if (state.canvas_state_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(state.canvas_state_uniform_set)) {
		RD::get_singleton()->free(state.canvas_state_uniform_set);
	}

	//bindings
	{

		free_texture_binding(bindings.default_empty);

		//dispose pending
		_dispose_bindings();
		//anything remains?
		if (bindings.texture_bindings.size()) {
			ERR_PRINT("Some texture bindings were not properly freed (leaked canvasitems?");
			const TextureBindingID *key = NULL;
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

	RD::get_singleton()->free(shader.default_skeleton_uniform_set);
	RD::get_singleton()->free(shader.default_skeleton_uniform);
	shader.canvas_shader.version_free(shader.default_version);

	//buffers
	RD::get_singleton()->free(shader.quad_index_array);
	RD::get_singleton()->free(shader.quad_index_buffer);

	//pipelines don't need freeing, they are all gone after shaders are gone

	//samplers
	for (int i = 1; i < VS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < VS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::get_singleton()->free(default_samplers.samplers[i][j]);
		}
	}

	//textures
	RD::get_singleton()->free(default_textures.white_texture);
	RD::get_singleton()->free(default_textures.black_texture);
	RD::get_singleton()->free(default_textures.normal_texture);
	RD::get_singleton()->free(default_textures.aniso_texture);
	RD::get_singleton()->free(default_textures.default_multimesh_tb);
}

/**************************************************************************/
/*  mesh_rasterizer_rd.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "mesh_rasterizer_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/utilities.h"

namespace RendererRD {

MeshRasterizerRD *MeshRasterizerRD::singleton = nullptr;

MeshRasterizerRD *MeshRasterizerRD::get_singleton() {
	return singleton;
}

void MeshRasterizerRD::RasterizeMeshShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;
	ShaderCompiler::IdentifierActions actions;

	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.render_mode_values["cull_disabled"] = Pair<int *, int>(&cull_modei, RS::CULL_MODE_DISABLED);
	actions.render_mode_values["cull_front"] = Pair<int *, int>(&cull_modei, RS::CULL_MODE_FRONT);
	actions.render_mode_values["cull_back"] = Pair<int *, int>(&cull_modei, RS::CULL_MODE_BACK);

	actions.uniforms = &uniforms;

	Error err = singleton->compiler.compile(RS::SHADER_RASTERIZE_MESH, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = singleton->shader_file_rd.version_create();
	}

	singleton->shader_file_rd.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines);

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	{
		// Global shader uniforms.
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.binding = 0;
		u.append_id(RendererRD::MaterialStorage::get_singleton()->global_shader_uniforms_get_storage_buffer());

		Vector<RD::Uniform> us = { u };
		MaterialStorage::get_singleton()->samplers_rd_get_default().append_uniforms(us, SAMPLERS_BINDING_FIRST_INDEX);
		base_uniforms = RD::get_singleton()->uniform_set_create(us, singleton->shader_file_rd.version_get_shader(version, 0), BASE_UNIFORM_SET);
	}

	shader_rd = singleton->shader_file_rd.version_get_shader(version, 0);

	RD::PipelineRasterizationState pipeline_rasterization_state;
	pipeline_rasterization_state.cull_mode = (RD::PolygonCullMode)cull_modei;
	pipeline_cache.setup(shader_rd, RD::RENDER_PRIMITIVE_TRIANGLES, pipeline_rasterization_state, {}, {}, singleton->pipeline_color_blend_state);

	valid = true;
}

MeshRasterizerRD::RasterizeMeshShaderData::~RasterizeMeshShaderData() {
	MeshRasterizerRD *rasterizer = MeshRasterizerRD::get_singleton();
	rasterizer->shader_file_rd.version_free(version);
}

bool MeshRasterizerRD::RasterizeMeshMaterialData::update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	MeshRasterizerRD *rasterizer = MeshRasterizerRD::get_singleton();
	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, material_uniforms, rasterizer->shader_file_rd.version_get_shader(shader_data->version, 0), MATERIAL_UNIFORM_SET, false, false);
}

RendererRD::MaterialStorage::ShaderData *MeshRasterizerRD::_create_rasterize_mesh_shader_funcs() {
	RasterizeMeshShaderData *shader_data = memnew(RasterizeMeshShaderData);
	return shader_data;
}

RendererRD::MaterialStorage::MaterialData *MeshRasterizerRD::_create_rasterize_mesh_material_funcs(RendererRD::MaterialStorage::ShaderData *p_shader) {
	RasterizeMeshMaterialData *material_data = memnew(RasterizeMeshMaterialData);
	material_data->shader_data = static_cast<RasterizeMeshShaderData *>(p_shader);
	return material_data;
}

RID MeshRasterizerRD::mesh_rasterizer_allocate() {
	return mesh_rasterizer_owner.allocate_rid();
}

void MeshRasterizerRD::mesh_rasterizer_initialize(RID p_mesh_rasterizer, int p_width, int p_height, RS::RasterizedTextureFormat p_texture_format, bool p_generate_mipmaps) {
	mesh_rasterizer_owner.initialize_rid(p_mesh_rasterizer);
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);

	uint32_t w = p_width;
	uint32_t h = p_height;
	uint32_t mipmaps = 1;
	while (true) {
		if (w == 1 && h == 1) {
			break;
		}
		w = MAX(1u, w >> 1);
		h = MAX(1u, h >> 1);
		mipmaps++;
	}
	RD::TextureFormat tex_format;
	tex_format.width = p_width;
	tex_format.height = p_height;
	tex_format.mipmaps = p_generate_mipmaps ? mipmaps : 1;
	tex_format.texture_type = RD::TEXTURE_TYPE_2D;
	tex_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
	switch (p_texture_format) {
		case RenderingServer::RASTERIZED_TEXTURE_FORMAT_RGBA8:
			tex_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			break;
		case RenderingServer::RASTERIZED_TEXTURE_FORMAT_RGBA8_SRGB:
			tex_format.format = RD::DATA_FORMAT_R8G8B8A8_SRGB;
			break;
		case RenderingServer::RASTERIZED_TEXTURE_FORMAT_RGBAH:
			tex_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			break;
		case RenderingServer::RASTERIZED_TEXTURE_FORMAT_RGBAF:
			tex_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			break;
	}
	mesh_rasterizer->framebuffer_texture_id = RD::get_singleton()->texture_create(tex_format, {});
	mesh_rasterizer->framebuffer_id = RD::get_singleton()->framebuffer_create({ mesh_rasterizer->framebuffer_texture_id });
	mesh_rasterizer->update_vertex();
}

void MeshRasterizerRD::mesh_rasterizer_set_bg_color(RID p_mesh_rasterizer, const Color &p_bg_color) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	mesh_rasterizer->bg_color = p_bg_color;
}

void MeshRasterizerRD::mesh_rasterizer_set_mesh(RID p_mesh_rasterizer, RID p_mesh, int p_surface_index) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	mesh_rasterizer->mesh = p_mesh;
	mesh_rasterizer->surface_index = p_surface_index;
	mesh_rasterizer->update_vertex();
	if (p_mesh.is_valid()) {
		Utilities::get_singleton()->base_update_dependency(p_mesh, &mesh_rasterizer->dependency_tracker);
	}
}

void MeshRasterizerRD::mesh_rasterizer_set_material(RID p_mesh_rasterizer, RID p_material) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	mesh_rasterizer->material = p_material;
	mesh_rasterizer->update_material();
	if (p_material.is_valid()) {
		MaterialStorage::get_singleton()->material_update_dependency(p_material, &mesh_rasterizer->dependency_tracker);
	}
}

void MeshRasterizerRD::mesh_rasterizer_draw(RID p_mesh_rasterizer) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	mesh_rasterizer->draw();
}

void MeshRasterizerRD::MeshRasterizerData::update_vertex() {
	Variant vertex_array = Vector<Vector3>{ Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0) };
	Vector<int> index_array;
	Vector<Vector2> uv_array;
	Vector<Color> color_array;

	if (mesh.is_valid() && surface_index < RS::get_singleton()->mesh_get_surface_count(mesh)) {
		RS::SurfaceData surface = RS::get_singleton()->mesh_get_surface(mesh, surface_index);

		if (surface.primitive == RS::PRIMITIVE_TRIANGLES) {
			Array surface_array = RS::get_singleton()->mesh_surface_get_arrays(mesh, surface_index);

			vertex_array = surface_array[RS::ARRAY_VERTEX];
			index_array = surface_array[RS::ARRAY_INDEX];
			uv_array = surface_array[RS::ARRAY_TEX_UV];
			color_array = surface_array[RS::ARRAY_COLOR];
		}
	}

	Vector<Vector3> vertex_array_vec3;
	Vector<uint8_t> vertex_data;

	if (vertex_array.get_type() == Variant::PACKED_VECTOR2_ARRAY) {
		// Convert 2D Mesh to 3D;
		Vector<Vector2> array_vec2 = vertex_array;
		vertex_array_vec3.resize(array_vec2.size());
		for (int i = 0; i < array_vec2.size(); i++) {
			Vector2 vec2 = array_vec2[i];
			vertex_array_vec3.write[i] = Vector3(vec2.x, vec2.y, 0);
		}
	} else {
		vertex_array_vec3 = vertex_array;
	}

	Vector3 max = Vector3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
	Vector3 min = Vector3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	for (int i = 0; i < vertex_array_vec3.size(); i++) {
		Vector3 v = vertex_array_vec3[i];
		max.x = MAX(max.x, v.x);
		max.y = MAX(max.y, v.y);
		max.z = MAX(max.z, v.z);
		min.x = MIN(min.x, v.x);
		min.y = MIN(min.y, v.y);
		min.z = MIN(min.z, v.z);
	}

	Vector3 center = (max + min) / 2;
	Vector3 s = max - min;

	// Normalize x,y to [-1,1].
	float scale = 0;
	if (s.x != 0) {
		scale = MAX(s.x, scale);
	}
	if (s.y != 0) {
		scale = MAX(s.y, scale);
	}
	float scale_factor = scale == 0 ? 1 : (2 / scale);

	for (int i = 0; i < vertex_array_vec3.size(); i++) {
		Vector3 v = vertex_array_vec3[i];
		v -= center;
		v *= scale_factor;
		// Invert y.
		v.y = -v.y;
		vertex_array_vec3.write[i] = v;
	}

	uint32_t vertex_count = vertex_array_vec3.size();

	vertex_data.resize(sizeof(float) * 3 * vertex_count);
	memcpy(vertex_data.ptrw(), vertex_array_vec3.ptr(), vertex_data.size());

	bool is_index_16 = vertex_count <= 0xffff;
	Vector<int16_t> index_array_16;
	if (is_index_16) {
		index_array_16.resize(index_array.size());
		for (int i = 0; i < index_array.size(); i++) {
			index_array_16.write[i] = index_array[i];
		}
	}
	if (!index_array.is_empty()) {
		Vector<uint8_t> index_data;
		index_data.resize((is_index_16 ? sizeof(int16_t) : sizeof(int32_t)) * index_array.size());
		if (is_index_16) {
			memcpy(index_data.ptrw(), index_array_16.ptr(), index_data.size());
		} else {
			memcpy(index_data.ptrw(), index_array.ptr(), index_data.size());
		}

		if (index_buffer_id.is_valid()) {
			RD::get_singleton()->free(index_buffer_id);
		}
		index_buffer_id = RD::get_singleton()->index_buffer_create(index_array.size(), is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, index_data);

		index_array_id = RD::get_singleton()->index_array_create(index_buffer_id, 0, index_array.size());
	}
	Vector<uint8_t> uv_data;
	uv_data.resize_zeroed(sizeof(float) * 2 * vertex_count);
	if (!uv_array.is_empty()) {
		memcpy(uv_data.ptrw(), uv_array.ptr(), uv_data.size());
	}

	Vector<uint8_t> color_data;
	color_data.resize(vertex_count * 4);
	color_data.fill(255);
	const Color *src = color_array.ptr();
	for (uint32_t i = 0; i < color_array.size(); i++) {
		uint8_t color8[4] = { (uint8_t)src[i].get_r8(), (uint8_t)src[i].get_g8(), (uint8_t)src[i].get_b8(), (uint8_t)src[i].get_a8() };
		memcpy(color_data.ptrw() + i * 4, color8, 4);
	}

	if (vertex_buffer_pos_id.is_valid()) {
		RD::get_singleton()->free(vertex_buffer_pos_id);
	}
	if (vertex_buffer_uv_id.is_valid()) {
		RD::get_singleton()->free(vertex_buffer_uv_id);
	}
	if (vertex_buffer_color_id.is_valid()) {
		RD::get_singleton()->free(vertex_buffer_color_id);
	}
	vertex_buffer_pos_id = RD::get_singleton()->vertex_buffer_create(vertex_data.size(), vertex_data);
	vertex_buffer_uv_id = RD::get_singleton()->vertex_buffer_create(uv_data.size(), uv_data);
	vertex_buffer_color_id = RD::get_singleton()->vertex_buffer_create(color_data.size(), color_data);

	Vector<RID> vertex_buffers = { vertex_buffer_pos_id, vertex_buffer_uv_id, vertex_buffer_color_id };

	vertex_array_id = RD::get_singleton()->vertex_array_create(vertex_count, singleton->vertex_format, vertex_buffers);
}

void MeshRasterizerRD::MeshRasterizerData::update_material() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	if (material.is_valid()) {
		MaterialStorage::MaterialData *md = material_storage->material_get_data(material, MaterialStorage::SHADER_TYPE_RASTERIZE_MESH);
		if (md != nullptr) {
			material_data = static_cast<RasterizeMeshMaterialData *>(md);
			shader_data = static_cast<RasterizeMeshShaderData *>(material_storage->material_get_shader_data(material));
			return;
		}
	}
	material_data = singleton->default_material_data;
	shader_data = singleton->default_shader_data;
}

void MeshRasterizerRD::MeshRasterizerData::draw() {
	Utilities::get_singleton()->update_dirty_resources();

	RID pipeline = shader_data->pipeline_cache.get_render_pipeline(singleton->vertex_format, RD::get_singleton()->framebuffer_get_format(framebuffer_id));
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer_id, RD::DrawFlags::DRAW_CLEAR_COLOR_ALL, { bg_color });
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipeline);

	// Vertex
	RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array_id);
	if (index_array_id.is_valid()) {
		RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array_id);
	}

	// Uniforms
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, shader_data->base_uniforms, BASE_UNIFORM_SET);
	if (material_data->material_uniforms.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_data->material_uniforms, MATERIAL_UNIFORM_SET);
	}

	RD::get_singleton()->draw_list_draw(draw_list, index_array_id.is_valid(), 1);
	RD::get_singleton()->draw_list_end();

	RD::TextureFormat tex_fmt = RD::get_singleton()->texture_get_format(framebuffer_texture_id);
	if (tex_fmt.mipmaps <= 1) {
		return;
	}

	// Generate mipmaps.
	Image::Format img_fmt;
	switch (tex_fmt.format) {
		case RD::DATA_FORMAT_R16G16B16_SFLOAT:
			img_fmt = Image::FORMAT_RGBAH;
			break;
		case RD::DATA_FORMAT_R32G32B32A32_SFLOAT:
			img_fmt = Image::FORMAT_RGBAF;
			break;
		default:
			img_fmt = Image::FORMAT_RGBA8;
	}
	Ref<Image> img = Image::create_from_data(tex_fmt.width, tex_fmt.height, true, img_fmt, RD::get_singleton()->texture_get_data(framebuffer_texture_id, 0));
	img->generate_mipmaps();
	Vector<uint8_t> data = img->get_data();
	int mipmap_count = tex_fmt.mipmaps;
	tex_fmt.mipmaps = 1;
	for (int i = 1; i < mipmap_count; i++) {
		Size2i mipmap_size = Size2i(MAX(1u, tex_fmt.width / (1 << i)), MAX(1u, tex_fmt.height / (1 << i)));
		tex_fmt.width = mipmap_size.x;
		tex_fmt.height = mipmap_size.y;
		int start = img->get_mipmap_offset(i);
		Vector<uint8_t> d;
		d.resize(mipmap_size.x * mipmap_size.y * 4);
		memcpy(d.ptrw(), data.ptr() + start, d.size());

		RID tex = RD::get_singleton()->texture_create(tex_fmt, {}, { d });
		Error err = RD::get_singleton()->texture_copy(tex, framebuffer_texture_id, Vector3(), Vector3(), Vector3(mipmap_size.x, mipmap_size.y, 0), 0, i, 0, 0);

		ERR_FAIL_COND_MSG(err != OK, vformat("Failed to generate mipmaps: %s", error_names[err]));
	}
}

RID MeshRasterizerRD::mesh_rasterizer_get_rd_texture(RID p_mesh_rasterizer) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	return mesh_rasterizer->framebuffer_texture_id;
}

bool MeshRasterizerRD::free(RID p_mesh_rasterizer) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	if (mesh_rasterizer == nullptr) {
		return false;
	}
	if (mesh_rasterizer->framebuffer_texture_id.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->framebuffer_texture_id);
	}
	if (mesh_rasterizer->index_buffer_id.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->index_buffer_id);
	}
	if (mesh_rasterizer->vertex_buffer_pos_id.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->vertex_buffer_pos_id);
	}
	if (mesh_rasterizer->vertex_buffer_uv_id.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->vertex_buffer_uv_id);
	}
	if (mesh_rasterizer->vertex_buffer_color_id.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->vertex_buffer_color_id);
	}
	mesh_rasterizer_owner.free(p_mesh_rasterizer);
	return true;
}

void MeshRasterizerRD::_dependency_changed(Dependency::DependencyChangedNotification p_notification, DependencyTracker *p_tracker) {
	MeshRasterizerData *mesh_rasterizer = (MeshRasterizerData *)p_tracker->userdata;
	switch (p_notification) {
		case Dependency::DEPENDENCY_CHANGED_MATERIAL: {
			mesh_rasterizer->update_material();
		} break;
		case Dependency::DEPENDENCY_CHANGED_MATERIAL_PARAM: {
			if (Engine::get_singleton()->is_editor_hint()) {
				mesh_rasterizer->draw();
			}
		} break;
		case Dependency::DEPENDENCY_CHANGED_MESH: {
			mesh_rasterizer->update_vertex();
			if (Engine::get_singleton()->is_editor_hint()) {
				mesh_rasterizer->draw();
			}
		} break;
		default: {
		}
	}
}

void MeshRasterizerRD::_dependency_deleted(const RID &p_dependency, DependencyTracker *p_tracker) {
	MeshRasterizerData *mesh_rasterizer = (MeshRasterizerData *)p_tracker->userdata;
	if (p_dependency == mesh_rasterizer->mesh) {
		mesh_rasterizer->mesh = RID();
		mesh_rasterizer->update_vertex();
	} else if (p_dependency == mesh_rasterizer->material) {
		mesh_rasterizer->material = RID();
		mesh_rasterizer->update_material();
	}
}

MeshRasterizerRD::MeshRasterizerData::MeshRasterizerData() {
	dependency_tracker.userdata = this;
	dependency_tracker.changed_callback = &MeshRasterizerRD::_dependency_changed;
	dependency_tracker.deleted_callback = &MeshRasterizerRD::_dependency_deleted;

	update_vertex();
	update_material();
}

MeshRasterizerRD::MeshRasterizerRD() {
	singleton = this;

	MaterialStorage *material_storage = MaterialStorage::get_singleton();

	// register our shader funds
	material_storage->shader_set_data_request_function(MaterialStorage::SHADER_TYPE_RASTERIZE_MESH, _create_rasterize_mesh_shader_funcs);
	material_storage->material_set_data_request_function(MaterialStorage::SHADER_TYPE_RASTERIZE_MESH, _create_rasterize_mesh_material_funcs);

	{
		//shader compiler
		ShaderCompiler::DefaultIdentifierActions actions;
		actions.renames["VERTEX"] = "vertex";
		actions.renames["UV"] = "uv";
		actions.renames["COLOR"] = "color";

		actions.renames["PI"] = String::num(Math::PI);
		actions.renames["TAU"] = String::num(Math::TAU);
		actions.renames["E"] = String::num(Math::E);

		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = MATERIAL_UNIFORM_SET;
		actions.base_uniform_string = "material.";
		actions.default_filter = ShaderLanguage::FILTER_LINEAR;
		actions.default_repeat = ShaderLanguage::REPEAT_DISABLE;
		actions.base_varying_index = 2;

		actions.global_buffer_array_variable = "global_shader_uniforms.data";

		compiler.initialize(actions);
	}

	String defines = "\n#define SAMPLERS_BINDING_FIRST_INDEX " + itos(SAMPLERS_BINDING_FIRST_INDEX) + "\n";
	shader_file_rd.initialize({ "" }, defines);

	default_shader = material_storage->shader_allocate();
	default_material = material_storage->material_allocate();
	material_storage->shader_initialize(default_shader);
	material_storage->material_initialize(default_material);
	material_storage->shader_set_code(default_shader, "shader_type rasterize_mesh;");
	material_storage->material_set_shader(default_material, default_shader);

	default_shader_data = static_cast<RasterizeMeshShaderData *>(material_storage->material_get_shader_data(default_material));
	default_material_data = static_cast<RasterizeMeshMaterialData *>(material_storage->material_get_data(default_material, MaterialStorage::SHADER_TYPE_RASTERIZE_MESH));

	RD::VertexAttribute pos;
	pos.location = 0,
	pos.format = RD::DATA_FORMAT_R32G32B32_SFLOAT,
	pos.stride = sizeof(float) * 3;

	RD::VertexAttribute uv;
	uv.location = 1,
	uv.format = RD::DATA_FORMAT_R32G32_SFLOAT,
	uv.stride = sizeof(float) * 2;

	RD::VertexAttribute color;
	color.location = 2,
	color.format = RD::DATA_FORMAT_R8G8B8A8_UNORM,
	color.stride = sizeof(uint8_t) * 4;

	Vector<RD::VertexAttribute> vertex_attrs = { pos, uv, color };

	vertex_format = RD::get_singleton()->vertex_format_create(vertex_attrs);

	pipeline_color_blend_state.attachments.append({});

	default_shader_data->pipeline_cache.setup(default_shader_data->shader_rd, RD::RENDER_PRIMITIVE_TRIANGLES, {}, {}, {}, singleton->pipeline_color_blend_state);
}

void MeshRasterizerRD::free_shader() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	material_storage->shader_free(default_shader);
	material_storage->material_free(default_material);
}

} //namespace RendererRD

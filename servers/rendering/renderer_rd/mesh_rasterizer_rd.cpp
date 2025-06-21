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
#include "framebuffer_cache_rd.h"
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

	Error err = singleton->compiler.compile(RS::SHADER_MESH_RASTERIZER, code, &actions, path, gen_code);
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

	valid = true;
}

bool MeshRasterizerRD::RasterizeMeshShaderData::is_animated() const {
	return false;
}

bool MeshRasterizerRD::RasterizeMeshShaderData::casts_shadows() const {
	return false;
}

RS::ShaderNativeSourceCode MeshRasterizerRD::RasterizeMeshShaderData::get_native_source_code() const {
	return singleton->shader_file_rd.version_get_native_source_code(version);
}

Pair<ShaderRD *, RID> MeshRasterizerRD::RasterizeMeshShaderData::get_native_shader_and_version() const {
	return { &singleton->shader_file_rd, version };
}

MeshRasterizerRD::RasterizeMeshShaderData::~RasterizeMeshShaderData() {
	MeshRasterizerRD *rasterizer = MeshRasterizerRD::get_singleton();
	rasterizer->shader_file_rd.version_free(version);
}

bool MeshRasterizerRD::RasterizeMeshMaterialData::update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	MeshRasterizerRD *rasterizer = MeshRasterizerRD::get_singleton();
	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, material_uniforms, rasterizer->shader_file_rd.version_get_shader(shader_data->version, 0), MATERIAL_UNIFORM_SET, false, false);
}

RendererRD::MaterialStorage::ShaderData *MeshRasterizerRD::_create_mesh_rasterizer_shader_funcs() {
	RasterizeMeshShaderData *shader_data = memnew(RasterizeMeshShaderData);
	return shader_data;
}

RendererRD::MaterialStorage::MaterialData *MeshRasterizerRD::_create_mesh_rasterizer_material_funcs(RendererRD::MaterialStorage::ShaderData *p_shader) {
	RasterizeMeshMaterialData *material_data = memnew(RasterizeMeshMaterialData);
	material_data->shader_data = static_cast<RasterizeMeshShaderData *>(p_shader);
	return material_data;
}

RID MeshRasterizerRD::mesh_rasterizer_allocate() {
	return mesh_rasterizer_owner.allocate_rid();
}

void MeshRasterizerRD::mesh_rasterizer_initialize(RID p_mesh_rasterizer, RID p_mesh, int p_surface_index) {
	mesh_rasterizer_owner.initialize_rid(p_mesh_rasterizer);
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	mesh_rasterizer->mesh = p_mesh;
	mesh_rasterizer->surface_index = p_surface_index;
	mesh_rasterizer->update_vertex();
	if (p_mesh.is_valid()) {
		Utilities::get_singleton()->base_update_dependency(p_mesh, &mesh_rasterizer->dependency_tracker);
	}
}

void MeshRasterizerRD::mesh_rasterizer_draw(RID p_mesh_rasterizer, RID p_material, RID p_texture_drawable, Ref<RasterizerBlendState> p_blend_state, const Color &p_bg_color, RD::TextureSamples p_multisample) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	ERR_FAIL_COND(p_mesh_rasterizer.is_null());
	ERR_FAIL_COND(p_material.is_null());

	MaterialStorage::get_singleton()->_update_global_shader_uniforms(); //must do before materials, so it can queue them for update
	MaterialStorage::get_singleton()->_update_queued_materials();

	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	MaterialStorage::MaterialData *md = material_storage->material_get_data(p_material, MaterialStorage::SHADER_TYPE_MESH_RASTERIZER);
	RasterizeMeshMaterialData *material_data = nullptr;
	RasterizeMeshShaderData *shader_data = nullptr;
	if (md != nullptr) {
		material_data = static_cast<RasterizeMeshMaterialData *>(md);
		shader_data = static_cast<RasterizeMeshShaderData *>(material_storage->material_get_shader_data(p_material));
	}
	ERR_FAIL_COND(material_data == nullptr);
	ERR_FAIL_COND(shader_data == nullptr);

	TextureStorage *texture_storage = TextureStorage::get_singleton();
	RID rd_texture = texture_storage->texture_get_rd_texture(p_texture_drawable, false);
	RD::TextureFormat tex_fmt = RD::get_singleton()->texture_get_format(rd_texture);

	// MSAA.
	RID rd_texture_samples;
	bool is_msaa = p_multisample > RD::TEXTURE_SAMPLES_1;
	if (is_msaa) {
		if (mesh_rasterizer->rd_texture_samples_cache.second.is_valid() && rd_texture == mesh_rasterizer->rd_texture_samples_cache.first) {
			rd_texture_samples = mesh_rasterizer->rd_texture_samples_cache.second;
		} else {
			tex_fmt.samples = p_multisample;
			tex_fmt.mipmaps = 1;
			tex_fmt.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			if (mesh_rasterizer->rd_texture_samples_cache.second.is_valid()) {
				RD::get_singleton()->free(mesh_rasterizer->rd_texture_samples_cache.second);
			}
			rd_texture_samples = RD::get_singleton()->texture_create(tex_fmt, {});

			mesh_rasterizer->rd_texture_samples_cache = { rd_texture, rd_texture_samples };
		}
	}

	RID framebuffer_rid;
	if (is_msaa) {
		framebuffer_rid = FramebufferCacheRD::get_singleton()->get_cache(rd_texture_samples);
	} else {
		framebuffer_rid = FramebufferCacheRD::get_singleton()->get_cache(rd_texture);
	}

	RD::PipelineRasterizationState rasterization_state;
	rasterization_state.cull_mode = (RD::PolygonCullMode)shader_data->cull_modei;
	RD::PipelineMultisampleState pipline_multisample_state;
	pipline_multisample_state.sample_count = p_multisample;
	RD::FramebufferFormatID fb_fmt = RD::get_singleton()->framebuffer_get_format(framebuffer_rid);

	RD::PipelineColorBlendState blend_state;
	blend_state.attachments.push_back({});
	if (p_blend_state.is_valid()) {
		p_blend_state->get_rd_blend_state(blend_state);
	}

	PipelineCacheKey k = {
		shader_data->shader_rd.get_id(),
		fb_fmt,
		mesh_rasterizer->primitive,
		p_multisample,
		p_blend_state
	};

	RID pipeline;
	if (RD::get_singleton()->render_pipeline_is_valid(mesh_rasterizer->pipeline_cache.second) && mesh_rasterizer->pipeline_cache.first == k) {
		pipeline = mesh_rasterizer->pipeline_cache.second;
	} else {
		if (RD::get_singleton()->render_pipeline_is_valid(mesh_rasterizer->pipeline_cache.second)) {
			RD::get_singleton()->free(mesh_rasterizer->pipeline_cache.second);
		}

		pipeline = RD::get_singleton()->render_pipeline_create(shader_data->shader_rd, fb_fmt, singleton->vertex_format, mesh_rasterizer->primitive, rasterization_state, pipline_multisample_state, {}, blend_state);

		mesh_rasterizer->pipeline_cache = { k, pipeline };
	}

	LocalVector<Color> clear_colors = { p_bg_color };
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer_rid, p_blend_state.is_null() ? RD::DRAW_CLEAR_ALL : RD::DRAW_DEFAULT_ALL, clear_colors);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipeline);

	// Vertex
	RD::get_singleton()->draw_list_bind_vertex_array(draw_list, mesh_rasterizer->vertex_array_rid);
	if (mesh_rasterizer->index_array_rid.is_valid()) {
		RD::get_singleton()->draw_list_bind_index_array(draw_list, mesh_rasterizer->index_array_rid);
	}

	// Uniforms
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, shader_data->base_uniforms, BASE_UNIFORM_SET);
	if (material_data->material_uniforms.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_data->material_uniforms, MATERIAL_UNIFORM_SET);
	}

	RD::get_singleton()->draw_list_draw(draw_list, mesh_rasterizer->index_array_rid.is_valid(), 1);
	RD::get_singleton()->draw_list_end();

	if (is_msaa) {
		Error err = RD::get_singleton()->texture_resolve_multisample(rd_texture_samples, rd_texture);
		ERR_FAIL_COND_MSG(err != OK, vformat("Resolve multisample texture fails: %s", error_names[err]));
	}
}

static RD::RenderPrimitive _primitive_type_to_render_primitive(RS::PrimitiveType p_primitive) {
	switch (p_primitive) {
		case RS::PRIMITIVE_POINTS:
			return RD::RENDER_PRIMITIVE_POINTS;
		case RS::PRIMITIVE_LINES:
			return RD::RENDER_PRIMITIVE_LINES;
		case RS::PRIMITIVE_LINE_STRIP:
			return RD::RENDER_PRIMITIVE_LINESTRIPS;
		case RS::PRIMITIVE_TRIANGLES:
			return RD::RENDER_PRIMITIVE_TRIANGLES;
		case RS::PRIMITIVE_TRIANGLE_STRIP:
			return RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS;
		default:
			return RD::RENDER_PRIMITIVE_MAX;
	}
}

void MeshRasterizerRD::MeshRasterizerData::update_vertex() {
	Variant vertex_array = Vector<Vector3>{ Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0) };
	Vector<int> index_array;
	Vector<Vector2> uv_array;
	Vector<Color> color_array;

	if (mesh.is_valid() && surface_index < RS::get_singleton()->mesh_get_surface_count(mesh)) {
		RS::SurfaceData surface = RS::get_singleton()->mesh_get_surface(mesh, surface_index);

		primitive = _primitive_type_to_render_primitive(surface.primitive);
		if (primitive != RD::RENDER_PRIMITIVE_MAX) {
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
		// Flip 2D Mesh y-axis and convert to 3D.
		Vector<Vector2> array_vec2 = vertex_array;
		vertex_array_vec3.resize(array_vec2.size());
		for (int i = 0; i < array_vec2.size(); i++) {
			Vector2 vec2 = array_vec2[i];
			vertex_array_vec3.write[i] = Vector3(vec2.x, -vec2.y, 0);
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

	// Normalize x,y,z to [-1,1]. We will normalize z to [0,1] and flip y in glsl.
	float scale = 0;
	if (s.x != 0) {
		scale = MAX(s.x, scale);
	}
	if (s.y != 0) {
		scale = MAX(s.y, scale);
	}
	if (s.z != 0) {
		scale = MAX(s.z, scale);
	}
	float scale_factor = scale == 0 ? 1 : (2 / scale);

	for (int i = 0; i < vertex_array_vec3.size(); i++) {
		Vector3 v = vertex_array_vec3[i];
		v -= center;
		v *= scale_factor;
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

		if (index_buffer_rid.is_valid()) {
			RD::get_singleton()->free(index_buffer_rid);
		}
		index_buffer_rid = RD::get_singleton()->index_buffer_create(index_array.size(), is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, index_data);

		index_array_rid = RD::get_singleton()->index_array_create(index_buffer_rid, 0, index_array.size());
	}
	Vector<uint8_t> uv_data;
	uv_data.resize_initialized(sizeof(float) * 2 * vertex_count);
	if (!uv_array.is_empty()) {
		memcpy(uv_data.ptrw(), uv_array.ptr(), uv_data.size());
	}

	Vector<uint8_t> color_data;
	color_data.resize(vertex_count * 4);
	uint8_t *color_data_ptr = color_data.ptrw();
	memset(color_data_ptr, 255, color_data.size());
	const Color *src = color_array.ptr();
	for (uint32_t i = 0; i < color_array.size(); i++) {
		uint8_t color8[4] = { (uint8_t)src[i].get_r8(), (uint8_t)src[i].get_g8(), (uint8_t)src[i].get_b8(), (uint8_t)src[i].get_a8() };
		memcpy(color_data_ptr + i * 4, color8, 4);
	}

	if (vertex_buffer_pos_rid.is_valid()) {
		RD::get_singleton()->free(vertex_buffer_pos_rid);
	}
	if (vertex_buffer_uv_rid.is_valid()) {
		RD::get_singleton()->free(vertex_buffer_uv_rid);
	}
	if (vertex_buffer_color_rid.is_valid()) {
		RD::get_singleton()->free(vertex_buffer_color_rid);
	}
	vertex_buffer_pos_rid = RD::get_singleton()->vertex_buffer_create(vertex_data.size(), vertex_data);
	vertex_buffer_uv_rid = RD::get_singleton()->vertex_buffer_create(uv_data.size(), uv_data);
	vertex_buffer_color_rid = RD::get_singleton()->vertex_buffer_create(color_data.size(), color_data);

	Vector<RID> vertex_buffers = { vertex_buffer_pos_rid, vertex_buffer_uv_rid, vertex_buffer_color_rid };

	vertex_array_rid = RD::get_singleton()->vertex_array_create(vertex_count, singleton->vertex_format, vertex_buffers);
}

bool MeshRasterizerRD::free(RID p_mesh_rasterizer) {
	MeshRasterizerData *mesh_rasterizer = mesh_rasterizer_owner.get_or_null(p_mesh_rasterizer);
	if (mesh_rasterizer == nullptr) {
		return false;
	}
	if (mesh_rasterizer->rd_texture_samples_cache.second.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->rd_texture_samples_cache.second);
	}
	if (mesh_rasterizer->index_buffer_rid.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->index_buffer_rid);
	}
	if (mesh_rasterizer->vertex_buffer_pos_rid.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->vertex_buffer_pos_rid);
	}
	if (mesh_rasterizer->vertex_buffer_uv_rid.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->vertex_buffer_uv_rid);
	}
	if (mesh_rasterizer->vertex_buffer_color_rid.is_valid()) {
		RD::get_singleton()->free(mesh_rasterizer->vertex_buffer_color_rid);
	}
	mesh_rasterizer_owner.free(p_mesh_rasterizer);
	return true;
}

void MeshRasterizerRD::_dependency_changed(Dependency::DependencyChangedNotification p_notification, DependencyTracker *p_tracker) {
	MeshRasterizerData *mesh_rasterizer = (MeshRasterizerData *)p_tracker->userdata;
	switch (p_notification) {
		case Dependency::DEPENDENCY_CHANGED_MESH: {
			mesh_rasterizer->update_vertex();
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
	}
}

MeshRasterizerRD::MeshRasterizerData::MeshRasterizerData() {
	dependency_tracker.userdata = this;
	dependency_tracker.changed_callback = &MeshRasterizerRD::_dependency_changed;
	dependency_tracker.deleted_callback = &MeshRasterizerRD::_dependency_deleted;

	update_vertex();
}

MeshRasterizerRD::MeshRasterizerRD() {
	singleton = this;

	MaterialStorage *material_storage = MaterialStorage::get_singleton();

	// register our shader funds
	material_storage->shader_set_data_request_function(MaterialStorage::SHADER_TYPE_MESH_RASTERIZER, _create_mesh_rasterizer_shader_funcs);
	material_storage->material_set_data_request_function(MaterialStorage::SHADER_TYPE_MESH_RASTERIZER, _create_mesh_rasterizer_material_funcs);

	{
		//shader compiler
		ShaderCompiler::DefaultIdentifierActions actions;
		actions.renames["VERTEX"] = "vertex";
		actions.renames["UV"] = "uv";
		actions.renames["COLOR"] = "color";
		actions.renames["POINT_SIZE"] = "gl_PointSize";
		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["FRONT_FACING"] = "gl_FrontFacing";
		actions.renames["VERTEX_ID"] = "gl_VertexIndex";

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

	RD::FramebufferPass pass;
	pass.resolve_attachments.append(0);
	pass.color_attachments.append(1);
	render_passes = { pass };
}

} //namespace RendererRD

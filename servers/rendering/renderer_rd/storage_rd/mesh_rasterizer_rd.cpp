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
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

namespace RendererRD {

MeshRasterizerRD *MeshRasterizerRD::singleton = nullptr;

MeshRasterizerRD *MeshRasterizerRD::get_singleton() {
	return singleton;
}

void MeshRasterizerRD::MeshRasterizerShaderData::set_code(const String &p_code) {
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
		LocalVector<RD::Uniform> us = { u };
		// Samplers.
		MaterialStorage::get_singleton()->samplers_rd_get_default().append_uniforms(us, SAMPLERS_BINDING_FIRST_INDEX);

		base_uniforms = UniformSetCacheRD::get_singleton()->get_cache_vec(singleton->shader_file_rd.version_get_shader(version, 0), BASE_UNIFORM_SET, us);
	}

	shader_rd = singleton->shader_file_rd.version_get_shader(version, 0);

	valid = true;
}

bool MeshRasterizerRD::MeshRasterizerShaderData::is_animated() const {
	return false;
}

bool MeshRasterizerRD::MeshRasterizerShaderData::casts_shadows() const {
	return false;
}

RS::ShaderNativeSourceCode MeshRasterizerRD::MeshRasterizerShaderData::get_native_source_code() const {
	return singleton->shader_file_rd.version_get_native_source_code(version);
}

Pair<ShaderRD *, RID> MeshRasterizerRD::MeshRasterizerShaderData::get_native_shader_and_version() const {
	return { &singleton->shader_file_rd, version };
}

uint64_t MeshRasterizerRD::MeshRasterizerShaderData::get_vertex_input_mask() {
	return RD::get_singleton()->shader_get_vertex_input_attribute_mask(shader_rd);
}

MeshRasterizerRD::MeshRasterizerShaderData::~MeshRasterizerShaderData() {
	MeshRasterizerRD *rasterizer = MeshRasterizerRD::get_singleton();
	rasterizer->shader_file_rd.version_free(version);
}

bool MeshRasterizerRD::MeshRasterizerMaterialData::update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	MeshRasterizerRD *rasterizer = MeshRasterizerRD::get_singleton();
	bool uniform_set_changed = update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, material_uniforms, rasterizer->shader_file_rd.version_get_shader(shader_data->version, 0), MATERIAL_UNIFORM_SET, false, false);
	bool uniform_set_changed_srgb = update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, material_uniforms_srgb, rasterizer->shader_file_rd.version_get_shader(shader_data->version, 0), MATERIAL_UNIFORM_SET, true, false);
	return uniform_set_changed || uniform_set_changed_srgb;
}

MeshRasterizerRD::MeshRasterizerMaterialData::~MeshRasterizerMaterialData() {
	free_parameters_uniform_set(material_uniforms);
	free_parameters_uniform_set(material_uniforms_srgb);
}

RendererRD::MaterialStorage::ShaderData *MeshRasterizerRD::_create_mesh_rasterizer_shader_funcs() {
	MeshRasterizerShaderData *shader_data = memnew(MeshRasterizerShaderData);
	return shader_data;
}

RendererRD::MaterialStorage::MaterialData *MeshRasterizerRD::_create_mesh_rasterizer_material_funcs(RendererRD::MaterialStorage::ShaderData *p_shader) {
	MeshRasterizerMaterialData *material_data = memnew(MeshRasterizerMaterialData);
	material_data->shader_data = static_cast<MeshRasterizerShaderData *>(p_shader);
	return material_data;
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

void MeshRasterizerRD::texture_drawable_blit_mesh_advanced(RID p_texture_drawable, RID p_material, RID p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer) {
	MaterialStorage::get_singleton()->_update_global_shader_uniforms(); //must do before materials, so it can queue them for update
	MaterialStorage::get_singleton()->_update_queued_materials();

	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	MaterialStorage::MaterialData *md = material_storage->material_get_data(p_material, MaterialStorage::SHADER_TYPE_MESH_RASTERIZER);
	ERR_FAIL_NULL(md);
	MeshRasterizerMaterialData *material_data = static_cast<MeshRasterizerMaterialData *>(md);
	MeshRasterizerShaderData *shader_data = static_cast<MeshRasterizerShaderData *>(material_storage->material_get_shader_data(p_material));
	ERR_FAIL_COND(shader_data->shader_rd.is_null());

	RID index_array_rid;
	RID vertex_array_rid;
	RD::VertexFormatID vertex_format;
	MeshStorage *mesh_storage = MeshStorage::get_singleton();
	ERR_FAIL_COND(p_surface_index >= (uint32_t)mesh_storage->mesh_get_surface_count(p_mesh));

	void *surface = mesh_storage->mesh_get_surface(p_mesh, p_surface_index);
	RD::RenderPrimitive primitive = _primitive_type_to_render_primitive(mesh_storage->mesh_surface_get_primitive(surface));
	ERR_FAIL_COND(primitive == RD::RENDER_PRIMITIVE_MAX);

	index_array_rid = mesh_storage->mesh_surface_get_index_array(surface, 0);
	uint64_t input_mask = shader_data->get_vertex_input_mask();
	mesh_storage->mesh_surface_get_vertex_arrays_and_format(surface, input_mask, false, vertex_array_rid, vertex_format);

	TextureStorage *texture_storage = TextureStorage::get_singleton();
	bool use_srgb = texture_storage->texture_drawable_is_srgb(p_texture_drawable);
	RID rd_texture_slice = texture_storage->texture_drawable_get_slice(p_texture_drawable, p_layer, 0);

	RID framebuffer_rid = FramebufferCacheRD::get_singleton()->get_cache(rd_texture_slice);
	RD::FramebufferFormatID fb_fmt = RD::get_singleton()->framebuffer_get_format(framebuffer_rid);

	RD::PipelineRasterizationState rasterization_state;
	rasterization_state.cull_mode = (RD::PolygonCullMode)shader_data->cull_modei;

	RD::PipelineColorBlendState blend_state;
	RD::PipelineColorBlendState::Attachment blend_attachment;
	switch (p_blend_mode) {
		case RS::TEXTURE_DRAWABLE_BLEND_MIX:
			blend_attachment = MaterialStorage::ShaderData::blend_mode_to_blend_attachment(MaterialStorage::ShaderData::BLEND_MODE_MIX);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_ADD:
			blend_attachment = MaterialStorage::ShaderData::blend_mode_to_blend_attachment(MaterialStorage::ShaderData::BLEND_MODE_ADD);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_SUB:
			blend_attachment = MaterialStorage::ShaderData::blend_mode_to_blend_attachment(MaterialStorage::ShaderData::BLEND_MODE_SUB);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_MUL:
			blend_attachment = MaterialStorage::ShaderData::blend_mode_to_blend_attachment(MaterialStorage::ShaderData::BLEND_MODE_MUL);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_PREMULTIPLIED_ALPHA:
			blend_attachment = MaterialStorage::ShaderData::blend_mode_to_blend_attachment(MaterialStorage::ShaderData::BLEND_MODE_PREMULTIPLIED_ALPHA);
			break;
		case RS::TEXTURE_DRAWABLE_BLEND_CLEAR:
		case RS::TEXTURE_DRAWABLE_BLEND_DISABLED:
			blend_attachment = MaterialStorage::ShaderData::blend_mode_to_blend_attachment(MaterialStorage::ShaderData::BLEND_MODE_DISABLED);
			break;
	}
	blend_state.attachments = { blend_attachment };

	Vector<RD::PipelineSpecializationConstant> specialization_constants;
	RD::PipelineSpecializationConstant sc;
	sc.constant_id = 0;
	sc.bool_value = use_srgb;
	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
	specialization_constants.push_back(sc);

	RID pipeline_rid = RD::get_singleton()->render_pipeline_create(shader_data->shader_rd, fb_fmt, vertex_format, primitive, rasterization_state, RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), blend_state, 0, 0, specialization_constants);

	LocalVector<Color> clear_colors = { use_srgb ? p_clear_color.srgb_to_linear() : p_clear_color };
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer_rid, p_blend_mode == RS::TEXTURE_DRAWABLE_BLEND_CLEAR ? RD::DRAW_CLEAR_ALL : RD::DRAW_DEFAULT_ALL, clear_colors);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipeline_rid);

	// Vertex
	RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array_rid);

	if (index_array_rid.is_valid()) {
		RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array_rid);
	}

	// Uniforms
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, shader_data->base_uniforms, BASE_UNIFORM_SET);
	if (use_srgb && material_data->material_uniforms_srgb.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_data->material_uniforms_srgb, MATERIAL_UNIFORM_SET);
	} else {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material_data->material_uniforms, MATERIAL_UNIFORM_SET);
	}

	RD::get_singleton()->draw_list_draw(draw_list, index_array_rid.is_valid(), 1);
	RD::get_singleton()->draw_list_end();

	RD::get_singleton()->free(pipeline_rid);
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

		actions.renames["VERTEX"] = "vertex_interp";
		actions.renames["NORMAL"] = "normal_interp";
		actions.renames["TANGENT"] = "tangent";
		actions.renames["BINORMAL"] = "binormal";
		actions.renames["POSITION"] = "position";
		actions.renames["UV"] = "uv_interp";
		actions.renames["UV2"] = "uv2_interp";
		actions.renames["COLOR"] = "color_interp";
		actions.renames["OUTPUT_COLOR"] = "output_color";

		actions.renames["POINT_SIZE"] = "gl_PointSize";
		actions.renames["VERTEX_ID"] = "gl_VertexIndex";
		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["FRONT_FACING"] = "gl_FrontFacing";

		actions.renames["PI"] = String::num(Math::PI);
		actions.renames["TAU"] = String::num(Math::TAU);
		actions.renames["E"] = String::num(Math::E);

		actions.usage_defines["NORMAL"] = "#define NORMAL_USED\n";
		actions.usage_defines["TANGENT"] = "#define TANGENT_USED\n";
		actions.usage_defines["BINORMAL"] = "@TANGENT";
		actions.usage_defines["UV"] = "#define UV_USED\n";
		actions.usage_defines["UV2"] = "#define UV2_USED\n";
		actions.usage_defines["BONE_INDICES"] = "#define BONES_USED\n";
		actions.usage_defines["BONE_WEIGHTS"] = "#define WEIGHTS_USED\n";
		actions.usage_defines["CUSTOM0"] = "#define CUSTOM0_USED\n";
		actions.usage_defines["CUSTOM1"] = "#define CUSTOM1_USED\n";
		actions.usage_defines["CUSTOM2"] = "#define CUSTOM2_USED\n";
		actions.usage_defines["CUSTOM3"] = "#define CUSTOM3_USED\n";
		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";

		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = MATERIAL_UNIFORM_SET;
		actions.base_uniform_string = "material.";
		actions.default_filter = ShaderLanguage::FILTER_LINEAR;
		actions.default_repeat = ShaderLanguage::REPEAT_DISABLE;
		actions.base_varying_index = 7;

		actions.global_buffer_array_variable = "global_shader_uniforms.data";

		compiler.initialize(actions);
	}

	String defines = "\n#define SAMPLERS_BINDING_FIRST_INDEX " + itos(SAMPLERS_BINDING_FIRST_INDEX) + "\n";
	shader_file_rd.initialize({ "" }, defines);
}

} //namespace RendererRD

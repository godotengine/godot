/*************************************************************************/
/*  rasterizer_scene_high_end_rd.cpp                                     */
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

#include "rasterizer_scene_high_end_rd.h"
#include "core/project_settings.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server_raster.h"

static _FORCE_INLINE_ void store_transform(const Transform &p_mtx, float *p_array) {
	p_array[0] = p_mtx.basis.elements[0][0];
	p_array[1] = p_mtx.basis.elements[1][0];
	p_array[2] = p_mtx.basis.elements[2][0];
	p_array[3] = 0;
	p_array[4] = p_mtx.basis.elements[0][1];
	p_array[5] = p_mtx.basis.elements[1][1];
	p_array[6] = p_mtx.basis.elements[2][1];
	p_array[7] = 0;
	p_array[8] = p_mtx.basis.elements[0][2];
	p_array[9] = p_mtx.basis.elements[1][2];
	p_array[10] = p_mtx.basis.elements[2][2];
	p_array[11] = 0;
	p_array[12] = p_mtx.origin.x;
	p_array[13] = p_mtx.origin.y;
	p_array[14] = p_mtx.origin.z;
	p_array[15] = 1;
}

static _FORCE_INLINE_ void store_basis_3x4(const Basis &p_mtx, float *p_array) {
	p_array[0] = p_mtx.elements[0][0];
	p_array[1] = p_mtx.elements[1][0];
	p_array[2] = p_mtx.elements[2][0];
	p_array[3] = 0;
	p_array[4] = p_mtx.elements[0][1];
	p_array[5] = p_mtx.elements[1][1];
	p_array[6] = p_mtx.elements[2][1];
	p_array[7] = 0;
	p_array[8] = p_mtx.elements[0][2];
	p_array[9] = p_mtx.elements[1][2];
	p_array[10] = p_mtx.elements[2][2];
	p_array[11] = 0;
}

static _FORCE_INLINE_ void store_transform_3x3(const Transform &p_mtx, float *p_array) {
	p_array[0] = p_mtx.basis.elements[0][0];
	p_array[1] = p_mtx.basis.elements[1][0];
	p_array[2] = p_mtx.basis.elements[2][0];
	p_array[3] = 0;
	p_array[4] = p_mtx.basis.elements[0][1];
	p_array[5] = p_mtx.basis.elements[1][1];
	p_array[6] = p_mtx.basis.elements[2][1];
	p_array[7] = 0;
	p_array[8] = p_mtx.basis.elements[0][2];
	p_array[9] = p_mtx.basis.elements[1][2];
	p_array[10] = p_mtx.basis.elements[2][2];
	p_array[11] = 0;
}

static _FORCE_INLINE_ void store_camera(const CameraMatrix &p_mtx, float *p_array) {

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {

			p_array[i * 4 + j] = p_mtx.matrix[i][j];
		}
	}
}

static _FORCE_INLINE_ void store_soft_shadow_kernel(const float *p_kernel, float *p_array) {

	for (int i = 0; i < 128; i++) {
		p_array[i] = p_kernel[i];
	}
}

/* SCENE SHADER */
void RasterizerSceneHighEndRD::ShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_screen_texture = false;

	if (code == String()) {
		return; //just invalid, but no error
	}

	ShaderCompilerRD::GeneratedCode gen_code;

	int blend_mode = BLEND_MODE_MIX;
	int depth_testi = DEPTH_TEST_ENABLED;
	int cull = CULL_BACK;

	uses_point_size = false;
	uses_alpha = false;
	uses_blend_alpha = false;
	uses_depth_pre_pass = false;
	uses_discard = false;
	uses_roughness = false;
	uses_normal = false;
	bool wireframe = false;

	unshaded = false;
	uses_vertex = false;
	uses_sss = false;
	uses_transmittance = false;
	uses_screen_texture = false;
	uses_depth_texture = false;
	uses_normal_texture = false;
	uses_time = false;
	writes_modelview_or_projection = false;
	uses_world_coordinates = false;

	int depth_drawi = DEPTH_DRAW_OPAQUE;

	ShaderCompilerRD::IdentifierActions actions;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);

	actions.render_mode_values["depth_draw_never"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_DISABLED);
	actions.render_mode_values["depth_draw_opaque"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_OPAQUE);
	actions.render_mode_values["depth_draw_always"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_ALWAYS);

	actions.render_mode_values["depth_test_disabled"] = Pair<int *, int>(&depth_testi, DEPTH_TEST_DISABLED);

	actions.render_mode_values["cull_disabled"] = Pair<int *, int>(&cull, CULL_DISABLED);
	actions.render_mode_values["cull_front"] = Pair<int *, int>(&cull, CULL_FRONT);
	actions.render_mode_values["cull_back"] = Pair<int *, int>(&cull, CULL_BACK);

	actions.render_mode_flags["unshaded"] = &unshaded;
	actions.render_mode_flags["wireframe"] = &wireframe;

	actions.usage_flag_pointers["ALPHA"] = &uses_alpha;
	actions.render_mode_flags["depth_prepass_alpha"] = &uses_depth_pre_pass;

	actions.usage_flag_pointers["SSS_STRENGTH"] = &uses_sss;
	actions.usage_flag_pointers["SSS_TRANSMITTANCE_DEPTH"] = &uses_transmittance;

	actions.usage_flag_pointers["SCREEN_TEXTURE"] = &uses_screen_texture;
	actions.usage_flag_pointers["DEPTH_TEXTURE"] = &uses_depth_texture;
	actions.usage_flag_pointers["NORMAL_TEXTURE"] = &uses_normal_texture;
	actions.usage_flag_pointers["DISCARD"] = &uses_discard;
	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["ROUGHNESS"] = &uses_roughness;
	actions.usage_flag_pointers["NORMAL"] = &uses_normal;
	actions.usage_flag_pointers["NORMALMAP"] = &uses_normal;

	actions.usage_flag_pointers["POINT_SIZE"] = &uses_point_size;
	actions.usage_flag_pointers["POINT_COORD"] = &uses_point_size;

	actions.write_flag_pointers["MODELVIEW_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["PROJECTION_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["VERTEX"] = &uses_vertex;

	actions.uniforms = &uniforms;

	RasterizerSceneHighEndRD *scene_singleton = (RasterizerSceneHighEndRD *)RasterizerSceneHighEndRD::singleton;

	Error err = scene_singleton->shader.compiler.compile(RS::SHADER_SPATIAL, code, &actions, path, gen_code);

	ERR_FAIL_COND(err != OK);

	if (version.is_null()) {
		version = scene_singleton->shader.scene_shader.version_create();
	}

	depth_draw = DepthDraw(depth_drawi);
	depth_test = DepthTest(depth_testi);

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
	scene_singleton->shader.scene_shader.version_set_code(version, gen_code.uniforms, gen_code.vertex_global, gen_code.vertex, gen_code.fragment_global, gen_code.light, gen_code.fragment, gen_code.defines);
	ERR_FAIL_COND(!scene_singleton->shader.scene_shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//blend modes

	RD::PipelineColorBlendState::Attachment blend_attachment;

	switch (blend_mode) {
		case BLEND_MODE_MIX: {

			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		} break;
		case BLEND_MODE_ADD: {

			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			uses_blend_alpha = true; //force alpha used because of blend

		} break;
		case BLEND_MODE_SUB: {

			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_SUBTRACT;
			blend_attachment.color_blend_op = RD::BLEND_OP_SUBTRACT;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			uses_blend_alpha = true; //force alpha used because of blend

		} break;
		case BLEND_MODE_MUL: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_DST_COLOR;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ZERO;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_DST_ALPHA;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
			uses_blend_alpha = true; //force alpha used because of blend
		} break;
	}

	RD::PipelineColorBlendState blend_state_blend;
	blend_state_blend.attachments.push_back(blend_attachment);
	RD::PipelineColorBlendState blend_state_opaque = RD::PipelineColorBlendState::create_disabled(1);
	RD::PipelineColorBlendState blend_state_opaque_specular = RD::PipelineColorBlendState::create_disabled(2);
	RD::PipelineColorBlendState blend_state_depth_normal = RD::PipelineColorBlendState::create_disabled(1);
	RD::PipelineColorBlendState blend_state_depth_normal_roughness = RD::PipelineColorBlendState::create_disabled(2);

	//update pipelines

	RD::PipelineDepthStencilState depth_stencil_state;

	if (depth_test != DEPTH_TEST_DISABLED) {
		depth_stencil_state.enable_depth_test = true;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;
		depth_stencil_state.enable_depth_write = depth_draw != DEPTH_DRAW_DISABLED ? true : false;
	}

	for (int i = 0; i < CULL_VARIANT_MAX; i++) {

		RD::PolygonCullMode cull_mode_rd_table[CULL_VARIANT_MAX][3] = {
			{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_FRONT, RD::POLYGON_CULL_BACK },
			{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_BACK, RD::POLYGON_CULL_FRONT },
			{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_DISABLED }
		};

		RD::PolygonCullMode cull_mode_rd = cull_mode_rd_table[i][cull];

		for (int j = 0; j < RS::PRIMITIVE_MAX; j++) {

			RD::RenderPrimitive primitive_rd_table[RS::PRIMITIVE_MAX] = {
				RD::RENDER_PRIMITIVE_POINTS,
				RD::RENDER_PRIMITIVE_LINES,
				RD::RENDER_PRIMITIVE_LINESTRIPS,
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS,
			};

			RD::RenderPrimitive primitive_rd = uses_point_size ? RD::RENDER_PRIMITIVE_POINTS : primitive_rd_table[j];

			for (int k = 0; k < SHADER_VERSION_MAX; k++) {

				RD::PipelineRasterizationState raster_state;
				raster_state.cull_mode = cull_mode_rd;
				raster_state.wireframe = wireframe;

				RD::PipelineColorBlendState blend_state;
				RD::PipelineDepthStencilState depth_stencil = depth_stencil_state;

				if (uses_alpha || uses_blend_alpha) {
					if (k == SHADER_VERSION_COLOR_PASS || k == SHADER_VERSION_VCT_COLOR_PASS || k == SHADER_VERSION_LIGHTMAP_COLOR_PASS) {
						blend_state = blend_state_blend;
						if (depth_draw == DEPTH_DRAW_OPAQUE) {
							depth_stencil.enable_depth_write = false; //alpha does not draw depth
						}
					} else if (uses_depth_pre_pass && (k == SHADER_VERSION_DEPTH_PASS || k == SHADER_VERSION_DEPTH_PASS_DP || k == SHADER_VERSION_DEPTH_PASS_WITH_NORMAL || k == SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS || k == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL)) {
						if (k == SHADER_VERSION_DEPTH_PASS || k == SHADER_VERSION_DEPTH_PASS_DP) {
							//none, blend state contains nothing
						} else if (k == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL) {
							blend_state = RD::PipelineColorBlendState::create_disabled(5); //writes to normal and roughness in opaque way
						} else {
							blend_state = blend_state_opaque; //writes to normal and roughness in opaque way
						}
					} else {
						pipelines[i][j][k].clear();
						continue; // do not use this version (will error if using it is attempted)
					}
				} else {

					if (k == SHADER_VERSION_COLOR_PASS || k == SHADER_VERSION_VCT_COLOR_PASS || k == SHADER_VERSION_LIGHTMAP_COLOR_PASS) {
						blend_state = blend_state_opaque;
					} else if (k == SHADER_VERSION_DEPTH_PASS || k == SHADER_VERSION_DEPTH_PASS_DP) {
						//none, leave empty
					} else if (k == SHADER_VERSION_DEPTH_PASS_WITH_NORMAL) {
						blend_state = blend_state_depth_normal;
					} else if (k == SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS) {
						blend_state = blend_state_depth_normal_roughness;
					} else if (k == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL) {
						blend_state = RD::PipelineColorBlendState::create_disabled(5); //writes to normal and roughness in opaque way

					} else {
						//specular write
						blend_state = blend_state_opaque_specular;
					}
				}

				RID shader_variant = scene_singleton->shader.scene_shader.version_get_shader(version, k);
				pipelines[i][j][k].setup(shader_variant, primitive_rd, raster_state, RD::PipelineMultisampleState(), depth_stencil, blend_state, 0);
			}
		}
	}

	valid = true;
}

void RasterizerSceneHighEndRD::ShaderData::set_default_texture_param(const StringName &p_name, RID p_texture) {
	if (!p_texture.is_valid()) {
		default_texture_params.erase(p_name);
	} else {
		default_texture_params[p_name] = p_texture;
	}
}

void RasterizerSceneHighEndRD::ShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {

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

bool RasterizerSceneHighEndRD::ShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RasterizerSceneHighEndRD::ShaderData::is_animated() const {
	return false;
}

bool RasterizerSceneHighEndRD::ShaderData::casts_shadows() const {
	return false;
}
Variant RasterizerSceneHighEndRD::ShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.hint);
	}
	return Variant();
}

RasterizerSceneHighEndRD::ShaderData::ShaderData() {
	valid = false;
	uses_screen_texture = false;
}

RasterizerSceneHighEndRD::ShaderData::~ShaderData() {
	RasterizerSceneHighEndRD *scene_singleton = (RasterizerSceneHighEndRD *)RasterizerSceneHighEndRD::singleton;
	ERR_FAIL_COND(!scene_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		scene_singleton->shader.scene_shader.version_free(version);
	}
}

RasterizerStorageRD::ShaderData *RasterizerSceneHighEndRD::_create_shader_func() {
	ShaderData *shader_data = memnew(ShaderData);
	return shader_data;
}

void RasterizerSceneHighEndRD::MaterialData::set_render_priority(int p_priority) {
	priority = p_priority - RS::MATERIAL_RENDER_PRIORITY_MIN; //8 bits
}

void RasterizerSceneHighEndRD::MaterialData::set_next_pass(RID p_pass) {
	next_pass = p_pass;
}

void RasterizerSceneHighEndRD::MaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {

	RasterizerSceneHighEndRD *scene_singleton = (RasterizerSceneHighEndRD *)RasterizerSceneHighEndRD::singleton;

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

		update_textures(p_parameters, shader_data->default_texture_params, shader_data->texture_uniforms, texture_cache.ptrw(), true);
	}

	if (shader_data->ubo_size == 0 && shader_data->texture_uniforms.size() == 0) {
		// This material does not require an uniform set, so don't create it.
		return;
	}

	if (!p_textures_dirty && uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//no reason to update uniform set, only UBO (or nothing) was needed to update
		return;
	}

	Vector<RD::Uniform> uniforms;

	{

		if (shader_data->ubo_size) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 0;
			u.ids.push_back(uniform_buffer);
			uniforms.push_back(u);
		}

		const RID *textures = texture_cache.ptrw();
		for (uint32_t i = 0; i < tex_uniform_count; i++) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1 + i;
			u.ids.push_back(textures[i]);
			uniforms.push_back(u);
		}
	}

	uniform_set = RD::get_singleton()->uniform_set_create(uniforms, scene_singleton->shader.scene_shader.version_get_shader(shader_data->version, 0), MATERIAL_UNIFORM_SET);
}

RasterizerSceneHighEndRD::MaterialData::~MaterialData() {
	if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		RD::get_singleton()->free(uniform_set);
	}

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
	}
}

RasterizerStorageRD::MaterialData *RasterizerSceneHighEndRD::_create_material_func(ShaderData *p_shader) {
	MaterialData *material_data = memnew(MaterialData);
	material_data->shader_data = p_shader;
	material_data->last_frame = false;
	//update will happen later anyway so do nothing.
	return material_data;
}

RasterizerSceneHighEndRD::RenderBufferDataHighEnd::~RenderBufferDataHighEnd() {
	clear();
}

void RasterizerSceneHighEndRD::RenderBufferDataHighEnd::ensure_specular() {

	if (!specular.is_valid()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.width = width;
		tf.height = height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		if (msaa != RS::VIEWPORT_MSAA_DISABLED) {
			tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
		} else {
			tf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		}

		specular = RD::get_singleton()->texture_create(tf, RD::TextureView());

		if (msaa == RS::VIEWPORT_MSAA_DISABLED) {

			{
				Vector<RID> fb;
				fb.push_back(color);
				fb.push_back(specular);
				fb.push_back(depth);

				color_specular_fb = RD::get_singleton()->framebuffer_create(fb);
			}
			{
				Vector<RID> fb;
				fb.push_back(specular);

				specular_only_fb = RD::get_singleton()->framebuffer_create(fb);
			}

		} else {

			tf.samples = texture_samples;
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			specular_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

			{
				Vector<RID> fb;
				fb.push_back(color_msaa);
				fb.push_back(specular_msaa);
				fb.push_back(depth_msaa);

				color_specular_fb = RD::get_singleton()->framebuffer_create(fb);
			}
			{
				Vector<RID> fb;
				fb.push_back(specular_msaa);

				specular_only_fb = RD::get_singleton()->framebuffer_create(fb);
			}
		}
	}
}

void RasterizerSceneHighEndRD::RenderBufferDataHighEnd::clear() {

	if (color_msaa.is_valid()) {
		RD::get_singleton()->free(color_msaa);
		color_msaa = RID();
	}

	if (depth_msaa.is_valid()) {
		RD::get_singleton()->free(depth_msaa);
		depth_msaa = RID();
	}

	if (specular.is_valid()) {
		if (specular_msaa.is_valid()) {
			RD::get_singleton()->free(specular_msaa);
			specular_msaa = RID();
		}
		RD::get_singleton()->free(specular);
		specular = RID();
	}

	color = RID();
	depth = RID();
	color_specular_fb = RID();
	specular_only_fb = RID();
	color_fb = RID();
	depth_fb = RID();

	if (normal_buffer.is_valid()) {
		RD::get_singleton()->free(normal_buffer);
		if (normal_buffer_msaa.is_valid()) {
			RD::get_singleton()->free(normal_buffer_msaa);
			normal_buffer_msaa = RID();
		}
		normal_buffer = RID();
		depth_normal_fb = RID();
	}

	if (roughness_buffer.is_valid()) {
		RD::get_singleton()->free(roughness_buffer);
		if (roughness_buffer_msaa.is_valid()) {
			RD::get_singleton()->free(roughness_buffer_msaa);
			roughness_buffer_msaa = RID();
		}
		roughness_buffer = RID();
		depth_normal_roughness_fb = RID();
	}
}

void RasterizerSceneHighEndRD::RenderBufferDataHighEnd::configure(RID p_color_buffer, RID p_depth_buffer, int p_width, int p_height, RS::ViewportMSAA p_msaa) {
	clear();

	msaa = p_msaa;

	width = p_width;
	height = p_height;

	color = p_color_buffer;
	depth = p_depth_buffer;

	if (p_msaa == RS::VIEWPORT_MSAA_DISABLED) {

		{
			Vector<RID> fb;
			fb.push_back(p_color_buffer);
			fb.push_back(depth);

			color_fb = RD::get_singleton()->framebuffer_create(fb);
		}
		{
			Vector<RID> fb;
			fb.push_back(depth);

			depth_fb = RD::get_singleton()->framebuffer_create(fb);
		}
	} else {

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.width = p_width;
		tf.height = p_height;
		tf.type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		RD::TextureSamples ts[RS::VIEWPORT_MSAA_MAX] = {
			RD::TEXTURE_SAMPLES_1,
			RD::TEXTURE_SAMPLES_2,
			RD::TEXTURE_SAMPLES_4,
			RD::TEXTURE_SAMPLES_8,
			RD::TEXTURE_SAMPLES_16
		};

		texture_samples = ts[p_msaa];
		tf.samples = texture_samples;

		color_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

		tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
		tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		depth_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

		{
			Vector<RID> fb;
			fb.push_back(color_msaa);
			fb.push_back(depth_msaa);

			color_fb = RD::get_singleton()->framebuffer_create(fb);
		}
		{
			Vector<RID> fb;
			fb.push_back(depth_msaa);

			depth_fb = RD::get_singleton()->framebuffer_create(fb);
		}
	}
}

void RasterizerSceneHighEndRD::_allocate_normal_texture(RenderBufferDataHighEnd *rb) {
	if (rb->normal_buffer.is_valid()) {
		return;
	}

	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_A2B10G10R10_UNORM_PACK32;
	tf.width = rb->width;
	tf.height = rb->height;
	tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;

	if (rb->msaa != RS::VIEWPORT_MSAA_DISABLED) {
		tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	} else {
		tf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	rb->normal_buffer = RD::get_singleton()->texture_create(tf, RD::TextureView());

	if (rb->msaa == RS::VIEWPORT_MSAA_DISABLED) {
		Vector<RID> fb;
		fb.push_back(rb->depth);
		fb.push_back(rb->normal_buffer);
		rb->depth_normal_fb = RD::get_singleton()->framebuffer_create(fb);
	} else {
		tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		tf.samples = rb->texture_samples;
		rb->normal_buffer_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

		Vector<RID> fb;
		fb.push_back(rb->depth_msaa);
		fb.push_back(rb->normal_buffer_msaa);
		rb->depth_normal_fb = RD::get_singleton()->framebuffer_create(fb);
	}

	_render_buffers_clear_uniform_set(rb);
}

void RasterizerSceneHighEndRD::_allocate_roughness_texture(RenderBufferDataHighEnd *rb) {

	if (rb->roughness_buffer.is_valid()) {
		return;
	}

	ERR_FAIL_COND(rb->normal_buffer.is_null());

	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_R8_UNORM;
	tf.width = rb->width;
	tf.height = rb->height;
	tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

	if (rb->msaa != RS::VIEWPORT_MSAA_DISABLED) {
		tf.usage_bits |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	} else {
		tf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	rb->roughness_buffer = RD::get_singleton()->texture_create(tf, RD::TextureView());

	if (rb->msaa == RS::VIEWPORT_MSAA_DISABLED) {

		Vector<RID> fb;
		fb.push_back(rb->depth);
		fb.push_back(rb->normal_buffer);
		fb.push_back(rb->roughness_buffer);
		rb->depth_normal_roughness_fb = RD::get_singleton()->framebuffer_create(fb);
	} else {
		tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		tf.samples = rb->texture_samples;
		rb->roughness_buffer_msaa = RD::get_singleton()->texture_create(tf, RD::TextureView());

		Vector<RID> fb;
		fb.push_back(rb->depth_msaa);
		fb.push_back(rb->normal_buffer_msaa);
		fb.push_back(rb->roughness_buffer_msaa);
		rb->depth_normal_roughness_fb = RD::get_singleton()->framebuffer_create(fb);
	}

	_render_buffers_clear_uniform_set(rb);
}

RasterizerSceneRD::RenderBufferData *RasterizerSceneHighEndRD::_create_render_buffer_data() {
	return memnew(RenderBufferDataHighEnd);
}

bool RasterizerSceneHighEndRD::free(RID p_rid) {
	if (RasterizerSceneRD::free(p_rid)) {
		return true;
	}
	return false;
}

void RasterizerSceneHighEndRD::_fill_instances(RenderList::Element **p_elements, int p_element_count, bool p_for_depth) {

	for (int i = 0; i < p_element_count; i++) {

		const RenderList::Element *e = p_elements[i];
		InstanceData &id = scene_state.instances[i];
		store_transform(e->instance->transform, id.transform);
		store_transform(Transform(e->instance->transform.basis.inverse().transposed()), id.normal_transform);
		id.flags = 0;
		id.mask = e->instance->layer_mask;

		if (e->instance->base_type == RS::INSTANCE_MULTIMESH) {
			id.flags |= INSTANCE_DATA_FLAG_MULTIMESH;
			uint32_t stride;
			if (storage->multimesh_get_transform_format(e->instance->base) == RS::MULTIMESH_TRANSFORM_2D) {
				id.flags |= INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D;
				stride = 2;
			} else {
				stride = 3;
			}
			if (storage->multimesh_uses_colors(e->instance->base)) {
				id.flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR;
				stride += 1;
			}
			if (storage->multimesh_uses_custom_data(e->instance->base)) {
				id.flags |= INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA;
				stride += 1;
			}

			id.flags |= (stride << INSTANCE_DATA_FLAGS_MULTIMESH_STRIDE_SHIFT);
		} else if (e->instance->base_type == RS::INSTANCE_MESH) {
			if (e->instance->skeleton.is_valid()) {
				id.flags |= INSTANCE_DATA_FLAG_SKELETON;
			}
		}

		if (p_for_depth) {
			id.gi_offset = 0xFFFFFFFF;
			continue;
		}

		if (!e->instance->gi_probe_instances.empty()) {
			uint32_t written = 0;
			for (int j = 0; j < e->instance->gi_probe_instances.size(); j++) {
				RID probe = e->instance->gi_probe_instances[j];
				int slot = gi_probe_instance_get_slot(probe);
				if (slot < 0) {
					continue; //unallocated, dont render
				}

				if (render_pass != gi_probe_instance_get_render_pass(probe)) {
					continue; //not rendered in this frame
				}

				uint32_t index = gi_probe_instance_get_render_index(probe);

				if (written == 0) {
					id.gi_offset = index;
					written = 1;
				} else {
					id.gi_offset = index << 16;
					written = 2;
					break;
				}
			}
			if (written == 0) {
				id.gi_offset = 0xFFFFFFFF;
			} else if (written == 1) {
				id.gi_offset |= 0xFFFF0000;
			}
		} else {
			id.gi_offset = 0xFFFFFFFF;
		}
	}

	RD::get_singleton()->buffer_update(scene_state.instance_buffer, 0, sizeof(InstanceData) * p_element_count, scene_state.instances, true);
}

/// RENDERING ///

void RasterizerSceneHighEndRD::_render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderList::Element **p_elements, int p_element_count, bool p_reverse_cull, PassMode p_pass_mode, bool p_no_gi, RID p_radiance_uniform_set, RID p_render_buffers_uniform_set) {

	RD::DrawListID draw_list = p_draw_list;
	RD::FramebufferFormatID framebuffer_format = p_framebuffer_Format;

	//global scope bindings
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, render_base_uniform_set, SCENE_UNIFORM_SET);
	if (p_radiance_uniform_set.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_radiance_uniform_set, RADIANCE_UNIFORM_SET);
	} else {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, default_radiance_uniform_set, RADIANCE_UNIFORM_SET);
	}
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, view_dependant_uniform_set, VIEW_DEPENDANT_UNIFORM_SET);
	if (p_render_buffers_uniform_set.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_render_buffers_uniform_set, RENDER_BUFFERS_UNIFORM_SET);
	} else {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, default_render_buffers_uniform_set, RENDER_BUFFERS_UNIFORM_SET);
	}
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, default_vec4_xform_uniform_set, TRANSFORMS_UNIFORM_SET);

	MaterialData *prev_material = nullptr;

	RID prev_vertex_array_rd;
	RID prev_index_array_rd;
	RID prev_pipeline_rd;
	RID prev_xforms_uniform_set;

	PushConstant push_constant;
	zeromem(&push_constant, sizeof(PushConstant));

	for (int i = 0; i < p_element_count; i++) {

		const RenderList::Element *e = p_elements[i];

		MaterialData *material = e->material;
		ShaderData *shader = material->shader_data;
		RID xforms_uniform_set;

		//find cull variant
		ShaderData::CullVariant cull_variant;

		if ((p_pass_mode == PASS_MODE_SHADOW || p_pass_mode == PASS_MODE_SHADOW_DP) && e->instance->cast_shadows == RS::SHADOW_CASTING_SETTING_DOUBLE_SIDED) {
			cull_variant = ShaderData::CULL_VARIANT_DOUBLE_SIDED;
		} else {
			bool mirror = e->instance->mirror;
			if (p_reverse_cull) {
				mirror = !mirror;
			}
			cull_variant = mirror ? ShaderData::CULL_VARIANT_REVERSED : ShaderData::CULL_VARIANT_NORMAL;
		}

		//find primitive and vertex format
		RS::PrimitiveType primitive;

		switch (e->instance->base_type) {
			case RS::INSTANCE_MESH: {
				primitive = storage->mesh_surface_get_primitive(e->instance->base, e->surface_index);
				if (e->instance->skeleton.is_valid()) {
					xforms_uniform_set = storage->skeleton_get_3d_uniform_set(e->instance->skeleton, default_shader_rd, TRANSFORMS_UNIFORM_SET);
				}
			} break;
			case RS::INSTANCE_MULTIMESH: {
				RID mesh = storage->multimesh_get_mesh(e->instance->base);
				ERR_CONTINUE(!mesh.is_valid()); //should be a bug
				primitive = storage->mesh_surface_get_primitive(mesh, e->surface_index);

				xforms_uniform_set = storage->multimesh_get_3d_uniform_set(e->instance->base, default_shader_rd, TRANSFORMS_UNIFORM_SET);

			} break;
			case RS::INSTANCE_IMMEDIATE: {
				ERR_CONTINUE(true); //should be a bug
			} break;
			case RS::INSTANCE_PARTICLES: {
				ERR_CONTINUE(true); //should be a bug
			} break;
			default: {
				ERR_CONTINUE(true); //should be a bug
			}
		}

		ShaderVersion shader_version = SHADER_VERSION_MAX; // Assigned to silence wrong -Wmaybe-initialized.

		switch (p_pass_mode) {
			case PASS_MODE_COLOR:
			case PASS_MODE_COLOR_TRANSPARENT: {
				if (e->uses_lightmap) {
					shader_version = SHADER_VERSION_LIGHTMAP_COLOR_PASS;
				} else if (e->uses_vct) {
					shader_version = SHADER_VERSION_VCT_COLOR_PASS;
				} else {
					shader_version = SHADER_VERSION_COLOR_PASS;
				}
			} break;
			case PASS_MODE_COLOR_SPECULAR: {
				if (e->uses_lightmap) {
					shader_version = SHADER_VERSION_LIGHTMAP_COLOR_PASS_WITH_SEPARATE_SPECULAR;
				} else if (e->uses_vct) {
					shader_version = SHADER_VERSION_VCT_COLOR_PASS_WITH_SEPARATE_SPECULAR;
				} else {
					shader_version = SHADER_VERSION_COLOR_PASS_WITH_SEPARATE_SPECULAR;
				}
			} break;
			case PASS_MODE_SHADOW:
			case PASS_MODE_DEPTH: {
				shader_version = SHADER_VERSION_DEPTH_PASS;
			} break;
			case PASS_MODE_SHADOW_DP: {
				shader_version = SHADER_VERSION_DEPTH_PASS_DP;
			} break;
			case PASS_MODE_DEPTH_NORMAL: {
				shader_version = SHADER_VERSION_DEPTH_PASS_WITH_NORMAL;
			} break;
			case PASS_MODE_DEPTH_NORMAL_ROUGHNESS: {
				shader_version = SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS;
			} break;
			case PASS_MODE_DEPTH_MATERIAL: {
				shader_version = SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL;
			} break;
		}

		RenderPipelineVertexFormatCacheRD *pipeline = nullptr;

		pipeline = &shader->pipelines[cull_variant][primitive][shader_version];

		RD::VertexFormatID vertex_format = -1;
		RID vertex_array_rd;
		RID index_array_rd;

		switch (e->instance->base_type) {
			case RS::INSTANCE_MESH: {
				storage->mesh_surface_get_arrays_and_format(e->instance->base, e->surface_index, pipeline->get_vertex_input_mask(), vertex_array_rd, index_array_rd, vertex_format);
			} break;
			case RS::INSTANCE_MULTIMESH: {
				RID mesh = storage->multimesh_get_mesh(e->instance->base);
				ERR_CONTINUE(!mesh.is_valid()); //should be a bug
				storage->mesh_surface_get_arrays_and_format(mesh, e->surface_index, pipeline->get_vertex_input_mask(), vertex_array_rd, index_array_rd, vertex_format);
			} break;
			case RS::INSTANCE_IMMEDIATE: {
				ERR_CONTINUE(true); //should be a bug
			} break;
			case RS::INSTANCE_PARTICLES: {
				ERR_CONTINUE(true); //should be a bug
			} break;
			default: {
				ERR_CONTINUE(true); //should be a bug
			}
		}

		if (prev_vertex_array_rd != vertex_array_rd) {
			RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array_rd);
			prev_vertex_array_rd = vertex_array_rd;
		}

		if (prev_index_array_rd != index_array_rd) {
			if (index_array_rd.is_valid()) {
				RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array_rd);
			}
			prev_index_array_rd = index_array_rd;
		}

		RID pipeline_rd = pipeline->get_render_pipeline(vertex_format, framebuffer_format);

		if (pipeline_rd != prev_pipeline_rd) {
			// checking with prev shader does not make so much sense, as
			// the pipeline may still be different.
			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, pipeline_rd);
			prev_pipeline_rd = pipeline_rd;
		}

		if (xforms_uniform_set.is_valid() && prev_xforms_uniform_set != xforms_uniform_set) {
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, xforms_uniform_set, TRANSFORMS_UNIFORM_SET);
			prev_xforms_uniform_set = xforms_uniform_set;
		}

		if (material != prev_material) {
			//update uniform set
			if (material->uniform_set.is_valid()) {
				RD::get_singleton()->draw_list_bind_uniform_set(draw_list, material->uniform_set, MATERIAL_UNIFORM_SET);
			}

			prev_material = material;
		}

		push_constant.index = i;
		RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(PushConstant));

		switch (e->instance->base_type) {
			case RS::INSTANCE_MESH: {
				RD::get_singleton()->draw_list_draw(draw_list, index_array_rd.is_valid());
			} break;
			case RS::INSTANCE_MULTIMESH: {
				uint32_t instances = storage->multimesh_get_instances_to_draw(e->instance->base);
				RD::get_singleton()->draw_list_draw(draw_list, index_array_rd.is_valid(), instances);
			} break;
			case RS::INSTANCE_IMMEDIATE: {

			} break;
			case RS::INSTANCE_PARTICLES: {

			} break;
			default: {
				ERR_CONTINUE(true); //should be a bug
			}
		}
	}
}

void RasterizerSceneHighEndRD::_setup_environment(RID p_environment, const CameraMatrix &p_cam_projection, const Transform &p_cam_transform, RID p_reflection_probe, bool p_no_fog, const Size2 &p_screen_pixel_size, RID p_shadow_atlas, bool p_flip_y, const Color &p_default_bg_color, float p_znear, float p_zfar, bool p_opaque_render_buffers, bool p_pancake_shadows) {

	//CameraMatrix projection = p_cam_projection;
	//projection.flip_y(); // Vulkan and modern APIs use Y-Down
	CameraMatrix correction;
	correction.set_depth_correction(p_flip_y);
	CameraMatrix projection = correction * p_cam_projection;

	//store camera into ubo
	store_camera(projection, scene_state.ubo.projection_matrix);
	store_camera(projection.inverse(), scene_state.ubo.inv_projection_matrix);
	store_transform(p_cam_transform, scene_state.ubo.camera_matrix);
	store_transform(p_cam_transform.affine_inverse(), scene_state.ubo.inv_camera_matrix);

	scene_state.ubo.z_far = p_zfar;
	scene_state.ubo.z_near = p_znear;

	scene_state.ubo.pancake_shadows = p_pancake_shadows;

	store_soft_shadow_kernel(directional_penumbra_shadow_kernel_get(), scene_state.ubo.directional_penumbra_shadow_kernel);
	store_soft_shadow_kernel(directional_soft_shadow_kernel_get(), scene_state.ubo.directional_soft_shadow_kernel);
	store_soft_shadow_kernel(penumbra_shadow_kernel_get(), scene_state.ubo.penumbra_shadow_kernel);
	store_soft_shadow_kernel(soft_shadow_kernel_get(), scene_state.ubo.soft_shadow_kernel);

	scene_state.ubo.directional_penumbra_shadow_samples = directional_penumbra_shadow_samples_get();
	scene_state.ubo.directional_soft_shadow_samples = directional_soft_shadow_samples_get();
	scene_state.ubo.penumbra_shadow_samples = penumbra_shadow_samples_get();
	scene_state.ubo.soft_shadow_samples = soft_shadow_samples_get();

	scene_state.ubo.screen_pixel_size[0] = p_screen_pixel_size.x;
	scene_state.ubo.screen_pixel_size[1] = p_screen_pixel_size.y;

	if (p_shadow_atlas.is_valid()) {
		Vector2 sas = shadow_atlas_get_size(p_shadow_atlas);
		scene_state.ubo.shadow_atlas_pixel_size[0] = 1.0 / sas.x;
		scene_state.ubo.shadow_atlas_pixel_size[1] = 1.0 / sas.y;
	}
	{
		Vector2 dss = directional_shadow_get_size();
		scene_state.ubo.directional_shadow_pixel_size[0] = 1.0 / dss.x;
		scene_state.ubo.directional_shadow_pixel_size[1] = 1.0 / dss.y;
	}
	//time global variables
	scene_state.ubo.time = time;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {

		scene_state.ubo.use_ambient_light = true;
		scene_state.ubo.ambient_light_color_energy[0] = 1;
		scene_state.ubo.ambient_light_color_energy[1] = 1;
		scene_state.ubo.ambient_light_color_energy[2] = 1;
		scene_state.ubo.ambient_light_color_energy[3] = 1.0;
		scene_state.ubo.use_ambient_cubemap = false;
		scene_state.ubo.use_reflection_cubemap = false;
		scene_state.ubo.ssao_enabled = false;

	} else if (is_environment(p_environment)) {

		RS::EnvironmentBG env_bg = environment_get_background(p_environment);
		RS::EnvironmentAmbientSource ambient_src = environment_get_ambient_light_ambient_source(p_environment);

		float bg_energy = environment_get_bg_energy(p_environment);
		scene_state.ubo.ambient_light_color_energy[3] = bg_energy;

		scene_state.ubo.ambient_color_sky_mix = environment_get_ambient_sky_contribution(p_environment);

		//ambient
		if (ambient_src == RS::ENV_AMBIENT_SOURCE_BG && (env_bg == RS::ENV_BG_CLEAR_COLOR || env_bg == RS::ENV_BG_COLOR)) {

			Color color = env_bg == RS::ENV_BG_CLEAR_COLOR ? p_default_bg_color : environment_get_bg_color(p_environment);
			color = color.to_linear();

			scene_state.ubo.ambient_light_color_energy[0] = color.r * bg_energy;
			scene_state.ubo.ambient_light_color_energy[1] = color.g * bg_energy;
			scene_state.ubo.ambient_light_color_energy[2] = color.b * bg_energy;
			scene_state.ubo.use_ambient_light = true;
			scene_state.ubo.use_ambient_cubemap = false;
		} else {

			float energy = environment_get_ambient_light_ambient_energy(p_environment);
			Color color = environment_get_ambient_light_color(p_environment);
			color = color.to_linear();
			scene_state.ubo.ambient_light_color_energy[0] = color.r * energy;
			scene_state.ubo.ambient_light_color_energy[1] = color.g * energy;
			scene_state.ubo.ambient_light_color_energy[2] = color.b * energy;

			Basis sky_transform = environment_get_sky_orientation(p_environment);
			sky_transform = sky_transform.inverse() * p_cam_transform.basis;
			store_transform_3x3(sky_transform, scene_state.ubo.radiance_inverse_xform);

			scene_state.ubo.use_ambient_cubemap = (ambient_src == RS::ENV_AMBIENT_SOURCE_BG && env_bg == RS::ENV_BG_SKY) || ambient_src == RS::ENV_AMBIENT_SOURCE_SKY;
			scene_state.ubo.use_ambient_light = scene_state.ubo.use_ambient_cubemap || ambient_src == RS::ENV_AMBIENT_SOURCE_COLOR;
		}

		//specular
		RS::EnvironmentReflectionSource ref_src = environment_get_reflection_source(p_environment);
		if ((ref_src == RS::ENV_REFLECTION_SOURCE_BG && env_bg == RS::ENV_BG_SKY) || ref_src == RS::ENV_REFLECTION_SOURCE_SKY) {
			scene_state.ubo.use_reflection_cubemap = true;
		} else {
			scene_state.ubo.use_reflection_cubemap = false;
		}

		scene_state.ubo.ssao_enabled = p_opaque_render_buffers && environment_is_ssao_enabled(p_environment);
		scene_state.ubo.ssao_ao_affect = environment_get_ssao_ao_affect(p_environment);
		scene_state.ubo.ssao_light_affect = environment_get_ssao_light_affect(p_environment);

		Color ao_color = environment_get_ao_color(p_environment);
		scene_state.ubo.ao_color[0] = ao_color.r;
		scene_state.ubo.ao_color[1] = ao_color.g;
		scene_state.ubo.ao_color[2] = ao_color.b;
		scene_state.ubo.ao_color[3] = ao_color.a;

	} else {

		if (p_reflection_probe.is_valid() && storage->reflection_probe_is_interior(reflection_probe_instance_get_probe(p_reflection_probe))) {
			scene_state.ubo.use_ambient_light = false;
		} else {
			scene_state.ubo.use_ambient_light = true;
			Color clear_color = p_default_bg_color;
			clear_color = clear_color.to_linear();
			scene_state.ubo.ambient_light_color_energy[0] = clear_color.r;
			scene_state.ubo.ambient_light_color_energy[1] = clear_color.g;
			scene_state.ubo.ambient_light_color_energy[2] = clear_color.b;
			scene_state.ubo.ambient_light_color_energy[3] = 1.0;
		}

		scene_state.ubo.use_ambient_cubemap = false;
		scene_state.ubo.use_reflection_cubemap = false;
	}

	scene_state.ubo.roughness_limiter_enabled = p_opaque_render_buffers && screen_space_roughness_limiter_is_active();

	RD::get_singleton()->buffer_update(scene_state.uniform_buffer, 0, sizeof(SceneState::UBO), &scene_state.ubo, true);
}

void RasterizerSceneHighEndRD::_add_geometry(InstanceBase *p_instance, uint32_t p_surface, RID p_material, PassMode p_pass_mode, uint32_t p_geometry_index) {

	RID m_src;

	m_src = p_instance->material_override.is_valid() ? p_instance->material_override : p_material;

	if (unlikely(get_debug_draw_mode() != RS::VIEWPORT_DEBUG_DRAW_DISABLED)) {
		if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW) {
			m_src = overdraw_material;
		} else if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_WIREFRAME) {
			m_src = wireframe_material;
		} else if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_LIGHTING) {
			m_src = default_material;
		}
	}

	MaterialData *material = nullptr;

	if (m_src.is_valid()) {
		material = (MaterialData *)storage->material_get_data(m_src, RasterizerStorageRD::SHADER_TYPE_3D);
		if (!material || !material->shader_data->valid) {
			material = nullptr;
		}
	}

	if (!material) {
		material = (MaterialData *)storage->material_get_data(default_material, RasterizerStorageRD::SHADER_TYPE_3D);
		m_src = default_material;
	}

	ERR_FAIL_COND(!material);

	_add_geometry_with_material(p_instance, p_surface, material, m_src, p_pass_mode, p_geometry_index);

	while (material->next_pass.is_valid()) {

		material = (MaterialData *)storage->material_get_data(material->next_pass, RasterizerStorageRD::SHADER_TYPE_3D);
		if (!material || !material->shader_data->valid)
			break;
		_add_geometry_with_material(p_instance, p_surface, material, material->next_pass, p_pass_mode, p_geometry_index);
	}
}

void RasterizerSceneHighEndRD::_add_geometry_with_material(InstanceBase *p_instance, uint32_t p_surface, MaterialData *p_material, RID p_material_rid, PassMode p_pass_mode, uint32_t p_geometry_index) {

	bool has_read_screen_alpha = p_material->shader_data->uses_screen_texture || p_material->shader_data->uses_depth_texture || p_material->shader_data->uses_normal_texture;
	bool has_base_alpha = (p_material->shader_data->uses_alpha || has_read_screen_alpha);
	bool has_blend_alpha = p_material->shader_data->uses_blend_alpha;
	bool has_alpha = has_base_alpha || has_blend_alpha;

	if (p_material->shader_data->uses_sss) {
		scene_state.used_sss = true;
	}

	if (p_material->shader_data->uses_screen_texture) {
		scene_state.used_screen_texture = true;
	}

	if (p_material->shader_data->uses_depth_texture) {
		scene_state.used_depth_texture = true;
	}

	if (p_material->shader_data->uses_normal_texture) {
		scene_state.used_normal_texture = true;
	}

	if (p_pass_mode != PASS_MODE_COLOR && p_pass_mode != PASS_MODE_COLOR_SPECULAR) {

		if (has_blend_alpha || has_read_screen_alpha || (has_base_alpha && !p_material->shader_data->uses_depth_pre_pass) || p_material->shader_data->depth_draw == ShaderData::DEPTH_DRAW_DISABLED || p_material->shader_data->depth_test == ShaderData::DEPTH_TEST_DISABLED || p_instance->cast_shadows == RS::SHADOW_CASTING_SETTING_OFF) {
			//conditions in which no depth pass should be processed
			return;
		}

		if (p_pass_mode != PASS_MODE_DEPTH_MATERIAL && !p_material->shader_data->writes_modelview_or_projection && !p_material->shader_data->uses_vertex && !p_material->shader_data->uses_discard && !p_material->shader_data->uses_depth_pre_pass) {
			//shader does not use discard and does not write a vertex position, use generic material
			if (p_pass_mode == PASS_MODE_SHADOW || p_pass_mode == PASS_MODE_DEPTH) {
				p_material = (MaterialData *)storage->material_get_data(default_material, RasterizerStorageRD::SHADER_TYPE_3D);
			} else if (p_pass_mode == PASS_MODE_DEPTH_NORMAL && !p_material->shader_data->uses_normal) {
				p_material = (MaterialData *)storage->material_get_data(default_material, RasterizerStorageRD::SHADER_TYPE_3D);
			} else if (p_pass_mode == PASS_MODE_DEPTH_NORMAL_ROUGHNESS && !p_material->shader_data->uses_normal && !p_material->shader_data->uses_roughness) {
				p_material = (MaterialData *)storage->material_get_data(default_material, RasterizerStorageRD::SHADER_TYPE_3D);
			}
		}

		has_alpha = false;
	}

	RenderList::Element *e = (has_alpha || p_material->shader_data->depth_test == ShaderData::DEPTH_TEST_DISABLED) ? render_list.add_alpha_element() : render_list.add_element();

	if (!e)
		return;

	e->instance = p_instance;
	e->material = p_material;
	e->surface_index = p_surface;
	e->sort_key = 0;

	if (e->material->last_pass != render_pass) {
		if (!RD::get_singleton()->uniform_set_is_valid(e->material->uniform_set)) {
			//uniform set no longer valid, probably a texture changed
			storage->material_force_update_textures(p_material_rid, RasterizerStorageRD::SHADER_TYPE_3D);
		}
		e->material->last_pass = render_pass;
		e->material->index = scene_state.current_material_index++;
		if (e->material->shader_data->last_pass != render_pass) {
			e->material->shader_data->last_pass = scene_state.current_material_index++;
			e->material->shader_data->index = scene_state.current_shader_index++;
		}
	}
	e->geometry_index = p_geometry_index;
	e->material_index = e->material->index;
	e->uses_instancing = e->instance->base_type == RS::INSTANCE_MULTIMESH;
	e->uses_lightmap = e->instance->lightmap.is_valid();
	e->uses_vct = e->instance->gi_probe_instances.size();
	e->shader_index = e->shader_index;
	e->depth_layer = e->instance->depth_layer;
	e->priority = p_material->priority;

	if (p_material->shader_data->uses_time) {
		RenderingServerRaster::redraw_request();
	}
}

void RasterizerSceneHighEndRD::_fill_render_list(InstanceBase **p_cull_result, int p_cull_count, PassMode p_pass_mode, bool p_no_gi) {

	scene_state.current_shader_index = 0;
	scene_state.current_material_index = 0;
	scene_state.used_sss = false;
	scene_state.used_screen_texture = false;
	scene_state.used_normal_texture = false;
	scene_state.used_depth_texture = false;

	uint32_t geometry_index = 0;

	//fill list

	for (int i = 0; i < p_cull_count; i++) {

		InstanceBase *inst = p_cull_result[i];

		//add geometry for drawing
		switch (inst->base_type) {

			case RS::INSTANCE_MESH: {

				const RID *materials = nullptr;
				uint32_t surface_count;

				materials = storage->mesh_get_surface_count_and_materials(inst->base, surface_count);
				if (!materials) {
					continue; //nothing to do
				}

				const RID *inst_materials = inst->materials.ptr();

				for (uint32_t j = 0; j < surface_count; j++) {

					RID material = inst_materials[j].is_valid() ? inst_materials[j] : materials[j];

					uint32_t surface_index = storage->mesh_surface_get_render_pass_index(inst->base, j, render_pass, &geometry_index);
					_add_geometry(inst, j, material, p_pass_mode, surface_index);
				}

				//mesh->last_pass=frame;

			} break;

			case RS::INSTANCE_MULTIMESH: {

				if (storage->multimesh_get_instances_to_draw(inst->base) == 0) {
					//not visible, 0 instances
					continue;
				}

				RID mesh = storage->multimesh_get_mesh(inst->base);
				if (!mesh.is_valid()) {
					continue;
				}

				const RID *materials = nullptr;
				uint32_t surface_count;

				materials = storage->mesh_get_surface_count_and_materials(mesh, surface_count);
				if (!materials) {
					continue; //nothing to do
				}

				for (uint32_t j = 0; j < surface_count; j++) {

					uint32_t surface_index = storage->mesh_surface_get_multimesh_render_pass_index(mesh, j, render_pass, &geometry_index);
					_add_geometry(inst, j, materials[j], p_pass_mode, surface_index);
				}

			} break;
#if 0
			case RS::INSTANCE_IMMEDIATE: {

				RasterizerStorageGLES3::Immediate *immediate = storage->immediate_owner.getornull(inst->base);
				ERR_CONTINUE(!immediate);

				_add_geometry(immediate, inst, nullptr, -1, p_depth_pass, p_shadow_pass);

			} break;
			case RS::INSTANCE_PARTICLES: {

				RasterizerStorageGLES3::Particles *particles = storage->particles_owner.getornull(inst->base);
				ERR_CONTINUE(!particles);

				for (int j = 0; j < particles->draw_passes.size(); j++) {

					RID pmesh = particles->draw_passes[j];
					if (!pmesh.is_valid())
						continue;
					RasterizerStorageGLES3::Mesh *mesh = storage->mesh_owner.getornull(pmesh);
					if (!mesh)
						continue; //mesh not assigned

					int ssize = mesh->surfaces.size();

					for (int k = 0; k < ssize; k++) {

						RasterizerStorageGLES3::Surface *s = mesh->surfaces[k];
						_add_geometry(s, inst, particles, -1, p_depth_pass, p_shadow_pass);
					}
				}

			} break;
#endif
			default: {
			}
		}
	}
}

void RasterizerSceneHighEndRD::_setup_reflections(RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, const Transform &p_camera_inverse_transform, RID p_environment) {

	for (int i = 0; i < p_reflection_probe_cull_count; i++) {

		RID rpi = p_reflection_probe_cull_result[i];

		if (i >= (int)scene_state.max_reflections) {
			reflection_probe_instance_set_render_index(rpi, 0); //invalid, but something needs to be set
			continue;
		}

		reflection_probe_instance_set_render_index(rpi, i);

		RID base_probe = reflection_probe_instance_get_probe(rpi);

		ReflectionData &reflection_ubo = scene_state.reflections[i];

		Vector3 extents = storage->reflection_probe_get_extents(base_probe);

		reflection_ubo.box_extents[0] = extents.x;
		reflection_ubo.box_extents[1] = extents.y;
		reflection_ubo.box_extents[2] = extents.z;
		reflection_ubo.index = reflection_probe_instance_get_atlas_index(rpi);

		Vector3 origin_offset = storage->reflection_probe_get_origin_offset(base_probe);

		reflection_ubo.box_offset[0] = origin_offset.x;
		reflection_ubo.box_offset[1] = origin_offset.y;
		reflection_ubo.box_offset[2] = origin_offset.z;
		reflection_ubo.mask = storage->reflection_probe_get_cull_mask(base_probe);

		float intensity = storage->reflection_probe_get_intensity(base_probe);
		bool interior = storage->reflection_probe_is_interior(base_probe);
		bool box_projection = storage->reflection_probe_is_box_projection(base_probe);

		reflection_ubo.params[0] = intensity;
		reflection_ubo.params[1] = 0;
		reflection_ubo.params[2] = interior ? 1.0 : 0.0;
		reflection_ubo.params[3] = box_projection ? 1.0 : 0.0;

		if (interior) {
			Color ambient_linear = storage->reflection_probe_get_interior_ambient(base_probe).to_linear();
			float interior_ambient_energy = storage->reflection_probe_get_interior_ambient_energy(base_probe);
			float interior_ambient_probe_contrib = storage->reflection_probe_get_interior_ambient_probe_contribution(base_probe);
			reflection_ubo.ambient[0] = ambient_linear.r * interior_ambient_energy;
			reflection_ubo.ambient[1] = ambient_linear.g * interior_ambient_energy;
			reflection_ubo.ambient[2] = ambient_linear.b * interior_ambient_energy;
			reflection_ubo.ambient[3] = interior_ambient_probe_contrib;
		} else {
			Color ambient_linear = storage->reflection_probe_get_interior_ambient(base_probe).to_linear();
			if (is_environment(p_environment)) {
				Color env_ambient_color = environment_get_ambient_light_color(p_environment).to_linear();
				float env_ambient_energy = environment_get_ambient_light_ambient_energy(p_environment);
				ambient_linear = env_ambient_color;
				ambient_linear.r *= env_ambient_energy;
				ambient_linear.g *= env_ambient_energy;
				ambient_linear.b *= env_ambient_energy;
			}

			reflection_ubo.ambient[0] = ambient_linear.r;
			reflection_ubo.ambient[1] = ambient_linear.g;
			reflection_ubo.ambient[2] = ambient_linear.b;
			reflection_ubo.ambient[3] = 0; //not used in exterior mode, since it just blends with regular ambient light
		}

		Transform transform = reflection_probe_instance_get_transform(rpi);
		Transform proj = (p_camera_inverse_transform * transform).inverse();
		store_transform(proj, reflection_ubo.local_matrix);

		cluster_builder.add_reflection_probe(transform, extents);

		reflection_probe_instance_set_render_pass(rpi, render_pass);
	}

	if (p_reflection_probe_cull_count) {
		RD::get_singleton()->buffer_update(scene_state.reflection_buffer, 0, MIN(scene_state.max_reflections, (unsigned int)p_reflection_probe_cull_count) * sizeof(ReflectionData), scene_state.reflections, true);
	}
}

void RasterizerSceneHighEndRD::_setup_gi_probes(RID *p_gi_probe_probe_cull_result, int p_gi_probe_probe_cull_count, const Transform &p_camera_transform) {

	int index = 0;

	for (int i = 0; i < p_gi_probe_probe_cull_count; i++) {

		RID rpi = p_gi_probe_probe_cull_result[i];

		if (index >= (int)scene_state.max_gi_probes) {
			continue;
		}

		int slot = gi_probe_instance_get_slot(rpi);
		if (slot < 0) {
			continue; //not usable
		}

		RID base_probe = gi_probe_instance_get_base_probe(rpi);

		GIProbeData &gi_probe_ubo = scene_state.gi_probes[index];

		Transform to_cell = gi_probe_instance_get_transform_to_cell(rpi) * p_camera_transform;

		store_transform(to_cell, gi_probe_ubo.xform);

		Vector3 bounds = storage->gi_probe_get_octree_size(base_probe);

		gi_probe_ubo.bounds[0] = bounds.x;
		gi_probe_ubo.bounds[1] = bounds.y;
		gi_probe_ubo.bounds[2] = bounds.z;

		gi_probe_ubo.dynamic_range = storage->gi_probe_get_dynamic_range(base_probe) * storage->gi_probe_get_energy(base_probe);
		gi_probe_ubo.bias = storage->gi_probe_get_bias(base_probe);
		gi_probe_ubo.normal_bias = storage->gi_probe_get_normal_bias(base_probe);
		gi_probe_ubo.blend_ambient = !storage->gi_probe_is_interior(base_probe);
		gi_probe_ubo.texture_slot = gi_probe_instance_get_slot(rpi);
		gi_probe_ubo.anisotropy_strength = storage->gi_probe_get_anisotropy_strength(base_probe);
		gi_probe_ubo.ao = storage->gi_probe_get_ao(base_probe);
		gi_probe_ubo.ao_size = Math::pow(storage->gi_probe_get_ao_size(base_probe), 4.0f);

		if (gi_probe_is_anisotropic()) {
			gi_probe_ubo.texture_slot *= 3;
		}

		gi_probe_instance_set_render_index(rpi, index);
		gi_probe_instance_set_render_pass(rpi, render_pass);

		index++;
	}

	if (index) {
		RD::get_singleton()->buffer_update(scene_state.gi_probe_buffer, 0, index * sizeof(GIProbeData), scene_state.gi_probes, true);
	}
}

void RasterizerSceneHighEndRD::_setup_lights(RID *p_light_cull_result, int p_light_cull_count, const Transform &p_camera_inverse_transform, RID p_shadow_atlas, bool p_using_shadows) {

	uint32_t light_count = 0;
	scene_state.ubo.directional_light_count = 0;
	sky_scene_state.directional_light_count = 0;

	for (int i = 0; i < p_light_cull_count; i++) {

		RID li = p_light_cull_result[i];
		RID base = light_instance_get_base_light(li);

		ERR_CONTINUE(base.is_null());

		RS::LightType type = storage->light_get_type(base);
		switch (type) {

			case RS::LIGHT_DIRECTIONAL: {

				if (scene_state.ubo.directional_light_count >= scene_state.max_directional_lights) {
					continue;
				}

				DirectionalLightData &light_data = scene_state.directional_lights[scene_state.ubo.directional_light_count];

				Transform light_transform = light_instance_get_base_transform(li);

				Vector3 direction = p_camera_inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, 1))).normalized();

				light_data.direction[0] = direction.x;
				light_data.direction[1] = direction.y;
				light_data.direction[2] = direction.z;

				float sign = storage->light_is_negative(base) ? -1 : 1;

				light_data.energy = sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI;

				Color linear_col = storage->light_get_color(base).to_linear();
				light_data.color[0] = linear_col.r;
				light_data.color[1] = linear_col.g;
				light_data.color[2] = linear_col.b;

				light_data.specular = storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR);
				light_data.mask = storage->light_get_cull_mask(base);

				float size = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

				light_data.size = 1.0 - Math::cos(Math::deg2rad(size)); //angle to cosine offset

				Color shadow_col = storage->light_get_shadow_color(base).to_linear();

				if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_PSSM_SPLITS) {
					light_data.shadow_color1[0] = 1.0;
					light_data.shadow_color1[1] = 0.0;
					light_data.shadow_color1[2] = 0.0;
					light_data.shadow_color1[3] = 1.0;
					light_data.shadow_color2[0] = 0.0;
					light_data.shadow_color2[1] = 1.0;
					light_data.shadow_color2[2] = 0.0;
					light_data.shadow_color2[3] = 1.0;
					light_data.shadow_color3[0] = 0.0;
					light_data.shadow_color3[1] = 0.0;
					light_data.shadow_color3[2] = 1.0;
					light_data.shadow_color3[3] = 1.0;
					light_data.shadow_color4[0] = 1.0;
					light_data.shadow_color4[1] = 1.0;
					light_data.shadow_color4[2] = 0.0;
					light_data.shadow_color4[3] = 1.0;

				} else {

					light_data.shadow_color1[0] = shadow_col.r;
					light_data.shadow_color1[1] = shadow_col.g;
					light_data.shadow_color1[2] = shadow_col.b;
					light_data.shadow_color1[3] = 1.0;
					light_data.shadow_color2[0] = shadow_col.r;
					light_data.shadow_color2[1] = shadow_col.g;
					light_data.shadow_color2[2] = shadow_col.b;
					light_data.shadow_color2[3] = 1.0;
					light_data.shadow_color3[0] = shadow_col.r;
					light_data.shadow_color3[1] = shadow_col.g;
					light_data.shadow_color3[2] = shadow_col.b;
					light_data.shadow_color3[3] = 1.0;
					light_data.shadow_color4[0] = shadow_col.r;
					light_data.shadow_color4[1] = shadow_col.g;
					light_data.shadow_color4[2] = shadow_col.b;
					light_data.shadow_color4[3] = 1.0;
				}

				light_data.shadow_enabled = p_using_shadows && storage->light_has_shadow(base);

				if (light_data.shadow_enabled) {

					RS::LightDirectionalShadowMode smode = storage->light_directional_get_shadow_mode(base);

					int limit = smode == RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL ? 0 : (smode == RS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS ? 1 : 3);
					light_data.blend_splits = storage->light_directional_get_blend_splits(base);
					for (int j = 0; j < 4; j++) {
						Rect2 atlas_rect = light_instance_get_directional_shadow_atlas_rect(li, j);
						CameraMatrix matrix = light_instance_get_shadow_camera(li, j);
						float split = light_instance_get_directional_shadow_split(li, MIN(limit, j));

						CameraMatrix bias;
						bias.set_light_bias();
						CameraMatrix rectm;
						rectm.set_light_atlas_rect(atlas_rect);

						Transform modelview = (p_camera_inverse_transform * light_instance_get_shadow_transform(li, j)).inverse();

						CameraMatrix shadow_mtx = rectm * bias * matrix * modelview;
						light_data.shadow_split_offsets[j] = split;
						float bias_scale = light_instance_get_shadow_bias_scale(li, j);
						light_data.shadow_bias[j] = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) * bias_scale;
						light_data.shadow_normal_bias[j] = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * light_instance_get_directional_shadow_texel_size(li, j);
						light_data.shadow_transmittance_bias[j] = storage->light_get_transmittance_bias(base) * bias_scale;
						light_data.shadow_transmittance_z_scale[j] = light_instance_get_shadow_range(li, j);
						light_data.shadow_range_begin[j] = light_instance_get_shadow_range_begin(li, j);
						store_camera(shadow_mtx, light_data.shadow_matrices[j]);

						Vector2 uv_scale = light_instance_get_shadow_uv_scale(li, j);
						uv_scale *= atlas_rect.size; //adapt to atlas size
						switch (j) {
							case 0: {
								light_data.uv_scale1[0] = uv_scale.x;
								light_data.uv_scale1[1] = uv_scale.y;
							} break;
							case 1: {
								light_data.uv_scale2[0] = uv_scale.x;
								light_data.uv_scale2[1] = uv_scale.y;
							} break;
							case 2: {
								light_data.uv_scale3[0] = uv_scale.x;
								light_data.uv_scale3[1] = uv_scale.y;
							} break;
							case 3: {
								light_data.uv_scale4[0] = uv_scale.x;
								light_data.uv_scale4[1] = uv_scale.y;
							} break;
						}
					}

					float fade_start = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_FADE_START);
					light_data.fade_from = -light_data.shadow_split_offsets[3] * MIN(fade_start, 0.999); //using 1.0 would break smoothstep
					light_data.fade_to = -light_data.shadow_split_offsets[3];

					light_data.soft_shadow_scale = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BLUR);

					float softshadow_angle = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);
					if (softshadow_angle > 0.0) {
						// I know tan(0) is 0, but let's not risk it with numerical precision.
						// technically this will keep expanding until reaching the sun, but all we care
						// is expand until we reach the radius of the near plane (there can't be more occluders than that)
						light_data.softshadow_angle = Math::tan(Math::deg2rad(softshadow_angle));
					} else {
						light_data.softshadow_angle = 0;
						light_data.soft_shadow_scale *= directional_shadow_quality_radius_get(); // Only use quality radius for PCF
					}
				}

				//	Copy to SkyDirectionalLightData
				if (sky_scene_state.directional_light_count < sky_scene_state.max_directional_lights) {

					SkyDirectionalLightData &sky_light_data = sky_scene_state.directional_lights[sky_scene_state.directional_light_count];

					Vector3 world_direction = light_transform.basis.xform(Vector3(0, 0, 1)).normalized();

					sky_light_data.direction[0] = world_direction.x;
					sky_light_data.direction[1] = world_direction.y;
					sky_light_data.direction[2] = -world_direction.z;

					sky_light_data.energy = light_data.energy / Math_PI;

					sky_light_data.color[0] = light_data.color[0];
					sky_light_data.color[1] = light_data.color[1];
					sky_light_data.color[2] = light_data.color[2];

					sky_light_data.enabled = true;
					sky_scene_state.directional_light_count++;
				}

				scene_state.ubo.directional_light_count++;
			} break;
			case RS::LIGHT_SPOT:
			case RS::LIGHT_OMNI: {

				if (light_count >= scene_state.max_lights) {
					continue;
				}

				Transform light_transform = light_instance_get_base_transform(li);

				LightData &light_data = scene_state.lights[light_count];

				float sign = storage->light_is_negative(base) ? -1 : 1;
				Color linear_col = storage->light_get_color(base).to_linear();

				light_data.attenuation_energy[0] = Math::make_half_float(storage->light_get_param(base, RS::LIGHT_PARAM_ATTENUATION));
				light_data.attenuation_energy[1] = Math::make_half_float(sign * storage->light_get_param(base, RS::LIGHT_PARAM_ENERGY) * Math_PI);

				light_data.color_specular[0] = MIN(uint32_t(linear_col.r * 255), 255);
				light_data.color_specular[1] = MIN(uint32_t(linear_col.g * 255), 255);
				light_data.color_specular[2] = MIN(uint32_t(linear_col.b * 255), 255);
				light_data.color_specular[3] = MIN(uint32_t(storage->light_get_param(base, RS::LIGHT_PARAM_SPECULAR) * 255), 255);

				float radius = MAX(0.001, storage->light_get_param(base, RS::LIGHT_PARAM_RANGE));
				light_data.inv_radius = 1.0 / radius;

				Vector3 pos = p_camera_inverse_transform.xform(light_transform.origin);

				light_data.position[0] = pos.x;
				light_data.position[1] = pos.y;
				light_data.position[2] = pos.z;

				Vector3 direction = p_camera_inverse_transform.basis.xform(light_transform.basis.xform(Vector3(0, 0, -1))).normalized();

				light_data.direction[0] = direction.x;
				light_data.direction[1] = direction.y;
				light_data.direction[2] = direction.z;

				float size = storage->light_get_param(base, RS::LIGHT_PARAM_SIZE);

				light_data.size = size;

				light_data.cone_attenuation_angle[0] = Math::make_half_float(storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ATTENUATION));
				float spot_angle = storage->light_get_param(base, RS::LIGHT_PARAM_SPOT_ANGLE);
				light_data.cone_attenuation_angle[1] = Math::make_half_float(Math::cos(Math::deg2rad(spot_angle)));

				light_data.mask = storage->light_get_cull_mask(base);

				light_data.atlas_rect[0] = 0;
				light_data.atlas_rect[1] = 0;
				light_data.atlas_rect[2] = 0;
				light_data.atlas_rect[3] = 0;

				RID projector = storage->light_get_projector(base);

				if (projector.is_valid()) {
					Rect2 rect = storage->decal_atlas_get_texture_rect(projector);

					if (type == RS::LIGHT_SPOT) {

						light_data.projector_rect[0] = rect.position.x;
						light_data.projector_rect[1] = rect.position.y + rect.size.height; //flip because shadow is flipped
						light_data.projector_rect[2] = rect.size.width;
						light_data.projector_rect[3] = -rect.size.height;
					} else {
						light_data.projector_rect[0] = rect.position.x;
						light_data.projector_rect[1] = rect.position.y;
						light_data.projector_rect[2] = rect.size.width;
						light_data.projector_rect[3] = rect.size.height * 0.5; //used by dp, so needs to be half
					}
				} else {
					light_data.projector_rect[0] = 0;
					light_data.projector_rect[1] = 0;
					light_data.projector_rect[2] = 0;
					light_data.projector_rect[3] = 0;
				}

				if (p_using_shadows && p_shadow_atlas.is_valid() && shadow_atlas_owns_light_instance(p_shadow_atlas, li)) {
					// fill in the shadow information

					Color shadow_color = storage->light_get_shadow_color(base);

					light_data.shadow_color_enabled[0] = MIN(uint32_t(shadow_color.r * 255), 255);
					light_data.shadow_color_enabled[1] = MIN(uint32_t(shadow_color.g * 255), 255);
					light_data.shadow_color_enabled[2] = MIN(uint32_t(shadow_color.b * 255), 255);
					light_data.shadow_color_enabled[3] = 255;

					if (type == RS::LIGHT_SPOT) {
						light_data.shadow_bias = (storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) * radius / 10.0);
						float shadow_texel_size = Math::tan(Math::deg2rad(spot_angle)) * radius * 2.0;
						shadow_texel_size *= light_instance_get_shadow_texel_size(li, p_shadow_atlas);

						light_data.shadow_normal_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * shadow_texel_size;

					} else { //omni
						light_data.shadow_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BIAS) * radius / 10.0;
						float shadow_texel_size = light_instance_get_shadow_texel_size(li, p_shadow_atlas);
						light_data.shadow_normal_bias = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * shadow_texel_size * 2.0; // applied in -1 .. 1 space
					}

					light_data.transmittance_bias = storage->light_get_transmittance_bias(base);

					Rect2 rect = light_instance_get_shadow_atlas_rect(li, p_shadow_atlas);

					light_data.atlas_rect[0] = rect.position.x;
					light_data.atlas_rect[1] = rect.position.y;
					light_data.atlas_rect[2] = rect.size.width;
					light_data.atlas_rect[3] = rect.size.height;

					light_data.soft_shadow_scale = storage->light_get_param(base, RS::LIGHT_PARAM_SHADOW_BLUR);

					if (type == RS::LIGHT_OMNI) {

						light_data.atlas_rect[3] *= 0.5; //one paraboloid on top of another
						Transform proj = (p_camera_inverse_transform * light_transform).inverse();

						store_transform(proj, light_data.shadow_matrix);

						if (size > 0.0) {

							light_data.soft_shadow_size = size;
						} else {
							light_data.soft_shadow_size = 0.0;
							light_data.soft_shadow_scale *= shadows_quality_radius_get(); // Only use quality radius for PCF
						}

					} else if (type == RS::LIGHT_SPOT) {

						Transform modelview = (p_camera_inverse_transform * light_transform).inverse();
						CameraMatrix bias;
						bias.set_light_bias();

						CameraMatrix shadow_mtx = bias * light_instance_get_shadow_camera(li, 0) * modelview;
						store_camera(shadow_mtx, light_data.shadow_matrix);

						if (size > 0.0) {
							CameraMatrix cm = light_instance_get_shadow_camera(li, 0);
							float half_np = cm.get_z_near() * Math::tan(Math::deg2rad(spot_angle));
							light_data.soft_shadow_size = (size * 0.5 / radius) / (half_np / cm.get_z_near()) * rect.size.width;
						} else {
							light_data.soft_shadow_size = 0.0;
							light_data.soft_shadow_scale *= shadows_quality_radius_get(); // Only use quality radius for PCF
						}
					}
				} else {
					light_data.shadow_color_enabled[3] = 0;
				}

				light_instance_set_index(li, light_count);

				cluster_builder.add_light(type == RS::LIGHT_SPOT ? LightClusterBuilder::LIGHT_TYPE_SPOT : LightClusterBuilder::LIGHT_TYPE_OMNI, light_transform, radius, spot_angle);

				light_count++;
			} break;
		}

		light_instance_set_render_pass(li, render_pass);

		//update UBO for forward rendering, blit to texture for clustered
	}

	if (light_count) {
		RD::get_singleton()->buffer_update(scene_state.light_buffer, 0, sizeof(LightData) * light_count, scene_state.lights, true);
	}

	if (scene_state.ubo.directional_light_count) {
		RD::get_singleton()->buffer_update(scene_state.directional_light_buffer, 0, sizeof(DirectionalLightData) * scene_state.ubo.directional_light_count, scene_state.directional_lights, true);
	}
}

void RasterizerSceneHighEndRD::_setup_decals(const RID *p_decal_instances, int p_decal_count, const Transform &p_camera_inverse_xform) {

	Transform uv_xform;
	uv_xform.basis.scale(Vector3(2.0, 1.0, 2.0));
	uv_xform.origin = Vector3(-1.0, 0.0, -1.0);

	p_decal_count = MIN((uint32_t)p_decal_count, scene_state.max_decals);
	int idx = 0;
	for (int i = 0; i < p_decal_count; i++) {

		RID di = p_decal_instances[i];
		RID decal = decal_instance_get_base(di);

		Transform xform = decal_instance_get_transform(di);

		float fade = 1.0;

		if (storage->decal_is_distance_fade_enabled(decal)) {
			real_t distance = -p_camera_inverse_xform.xform(xform.origin).z;
			float fade_begin = storage->decal_get_distance_fade_begin(decal);
			float fade_length = storage->decal_get_distance_fade_length(decal);

			if (distance > fade_begin) {
				if (distance > fade_begin + fade_length) {
					continue; // do not use this decal, its invisible
				}

				fade = 1.0 - (distance - fade_begin) / fade_length;
			}
		}

		DecalData &dd = scene_state.decals[idx];

		Vector3 decal_extents = storage->decal_get_extents(decal);

		Transform scale_xform;
		scale_xform.basis.scale(Vector3(decal_extents.x, decal_extents.y, decal_extents.z));
		Transform to_decal_xform = (p_camera_inverse_xform * decal_instance_get_transform(di) * scale_xform * uv_xform).affine_inverse();
		store_transform(to_decal_xform, dd.xform);

		Vector3 normal = xform.basis.get_axis(Vector3::AXIS_Y).normalized();
		normal = p_camera_inverse_xform.basis.xform(normal); //camera is normalized, so fine

		dd.normal[0] = normal.x;
		dd.normal[1] = normal.y;
		dd.normal[2] = normal.z;
		dd.normal_fade = storage->decal_get_normal_fade(decal);

		RID albedo_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_ALBEDO);
		if (albedo_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(albedo_tex);
			dd.albedo_rect[0] = rect.position.x;
			dd.albedo_rect[1] = rect.position.y;
			dd.albedo_rect[2] = rect.size.x;
			dd.albedo_rect[3] = rect.size.y;
		} else {

			dd.albedo_rect[0] = 0;
			dd.albedo_rect[1] = 0;
			dd.albedo_rect[2] = 0;
			dd.albedo_rect[3] = 0;
		}

		RID normal_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_NORMAL);
		RID emission_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_EMISSION);

		if (normal_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(normal_tex);
			dd.normal_rect[0] = rect.position.x;
			dd.normal_rect[1] = rect.position.y;
			dd.normal_rect[2] = rect.size.x;
			dd.normal_rect[3] = rect.size.y;

			Basis normal_xform = p_camera_inverse_xform.basis * xform.basis.orthonormalized();
			store_basis_3x4(normal_xform, dd.normal_xform);

			//store normal xform
		} else {

			if (!emission_tex.is_valid()) {
				continue; //no albedo, no emission, no decal.
			}
			dd.normal_rect[0] = 0;
			dd.normal_rect[1] = 0;
			dd.normal_rect[2] = 0;
			dd.normal_rect[3] = 0;
		}

		RID orm_tex = storage->decal_get_texture(decal, RS::DECAL_TEXTURE_ORM);
		if (orm_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(orm_tex);
			dd.orm_rect[0] = rect.position.x;
			dd.orm_rect[1] = rect.position.y;
			dd.orm_rect[2] = rect.size.x;
			dd.orm_rect[3] = rect.size.y;
		} else {
			dd.orm_rect[0] = 0;
			dd.orm_rect[1] = 0;
			dd.orm_rect[2] = 0;
			dd.orm_rect[3] = 0;
		}

		if (emission_tex.is_valid()) {
			Rect2 rect = storage->decal_atlas_get_texture_rect(emission_tex);
			dd.emission_rect[0] = rect.position.x;
			dd.emission_rect[1] = rect.position.y;
			dd.emission_rect[2] = rect.size.x;
			dd.emission_rect[3] = rect.size.y;
		} else {
			dd.emission_rect[0] = 0;
			dd.emission_rect[1] = 0;
			dd.emission_rect[2] = 0;
			dd.emission_rect[3] = 0;
		}

		Color modulate = storage->decal_get_modulate(decal);
		dd.modulate[0] = modulate.r;
		dd.modulate[1] = modulate.g;
		dd.modulate[2] = modulate.b;
		dd.modulate[3] = modulate.a * fade;
		dd.emission_energy = storage->decal_get_emission_energy(decal) * fade;
		dd.albedo_mix = storage->decal_get_albedo_mix(decal);
		dd.mask = storage->decal_get_cull_mask(decal);
		dd.upper_fade = storage->decal_get_upper_fade(decal);
		dd.lower_fade = storage->decal_get_lower_fade(decal);

		cluster_builder.add_decal(xform, decal_extents);

		idx++;
	}

	if (idx > 0) {
		RD::get_singleton()->buffer_update(scene_state.decal_buffer, 0, sizeof(DecalData) * idx, scene_state.decals, true);
	}
}

void RasterizerSceneHighEndRD::_render_scene(RID p_render_buffer, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID *p_decal_cull_result, int p_decal_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, const Color &p_default_bg_color) {

	RenderBufferDataHighEnd *render_buffer = nullptr;
	if (p_render_buffer.is_valid()) {
		render_buffer = (RenderBufferDataHighEnd *)render_buffers_get_data(p_render_buffer);
	}

	//first of all, make a new render pass
	render_pass++;

	//fill up ubo

	RENDER_TIMESTAMP("Setup 3D Scene");

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {
		p_light_cull_count = 0;
		p_reflection_probe_cull_count = 0;
		p_gi_probe_cull_count = 0;
	}

	bool using_shadows = true;

	if (p_reflection_probe.is_valid()) {
		scene_state.ubo.reflection_multiplier = 0.0;
		if (!storage->reflection_probe_renders_shadows(reflection_probe_instance_get_probe(p_reflection_probe))) {
			using_shadows = false;
		}
	} else {
		scene_state.ubo.reflection_multiplier = 1.0;
	}

	//scene_state.ubo.subsurface_scatter_width = subsurface_scatter_size;

	Vector2 vp_he = p_cam_projection.get_viewport_half_extents();
	scene_state.ubo.viewport_size[0] = vp_he.x;
	scene_state.ubo.viewport_size[1] = vp_he.y;

	Size2 screen_pixel_size;
	Size2i screen_size;
	RID opaque_framebuffer;
	RID opaque_specular_framebuffer;
	RID depth_framebuffer;
	RID alpha_framebuffer;

	PassMode depth_pass_mode = PASS_MODE_DEPTH;
	Vector<Color> depth_pass_clear;
	bool using_separate_specular = false;
	bool using_ssr = false;

	if (render_buffer) {

		screen_pixel_size.width = 1.0 / render_buffer->width;
		screen_pixel_size.height = 1.0 / render_buffer->height;
		screen_size.x = render_buffer->width;
		screen_size.y = render_buffer->height;

		opaque_framebuffer = render_buffer->color_fb;

		if (p_environment.is_valid() && environment_is_ssr_enabled(p_environment)) {
			depth_pass_mode = PASS_MODE_DEPTH_NORMAL_ROUGHNESS;
			render_buffer->ensure_specular();
			using_separate_specular = true;
			using_ssr = true;
			opaque_specular_framebuffer = render_buffer->color_specular_fb;
		} else if (screen_space_roughness_limiter_is_active()) {
			depth_pass_mode = PASS_MODE_DEPTH_NORMAL;
			//we need to allocate both these, if not allocated
			_allocate_normal_texture(render_buffer);
			_allocate_roughness_texture(render_buffer);
		} else if (p_environment.is_valid() && (environment_is_ssao_enabled(p_environment) || get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_NORMAL_BUFFER)) {
			depth_pass_mode = PASS_MODE_DEPTH_NORMAL;
		}

		switch (depth_pass_mode) {
			case PASS_MODE_DEPTH: {
				depth_framebuffer = render_buffer->depth_fb;
			} break;
			case PASS_MODE_DEPTH_NORMAL: {
				_allocate_normal_texture(render_buffer);
				depth_framebuffer = render_buffer->depth_normal_fb;
				depth_pass_clear.push_back(Color(0.5, 0.5, 0.5, 0));
			} break;
			case PASS_MODE_DEPTH_NORMAL_ROUGHNESS: {
				_allocate_normal_texture(render_buffer);
				_allocate_roughness_texture(render_buffer);
				depth_framebuffer = render_buffer->depth_normal_roughness_fb;
				depth_pass_clear.push_back(Color(0.5, 0.5, 0.5, 0));
				depth_pass_clear.push_back(Color());
			} break;
			default: {
			};
		}

		alpha_framebuffer = opaque_framebuffer;

	} else if (p_reflection_probe.is_valid()) {
		uint32_t resolution = reflection_probe_instance_get_resolution(p_reflection_probe);
		screen_pixel_size.width = 1.0 / resolution;
		screen_pixel_size.height = 1.0 / resolution;
		screen_size.x = resolution;
		screen_size.y = resolution;

		opaque_framebuffer = reflection_probe_instance_get_framebuffer(p_reflection_probe, p_reflection_probe_pass);
		depth_framebuffer = reflection_probe_instance_get_depth_framebuffer(p_reflection_probe, p_reflection_probe_pass);
		alpha_framebuffer = opaque_framebuffer;

		if (storage->reflection_probe_is_interior(reflection_probe_instance_get_probe(p_reflection_probe))) {
			p_environment = RID(); //no environment on interiors
		}

	} else {
		ERR_FAIL(); //bug?
	}

	cluster_builder.begin(p_cam_transform.affine_inverse(), p_cam_projection); //prepare cluster

	_setup_lights(p_light_cull_result, p_light_cull_count, p_cam_transform.affine_inverse(), p_shadow_atlas, using_shadows);
	_setup_decals(p_decal_cull_result, p_decal_cull_count, p_cam_transform.affine_inverse());
	_setup_reflections(p_reflection_probe_cull_result, p_reflection_probe_cull_count, p_cam_transform.affine_inverse(), p_environment);
	_setup_gi_probes(p_gi_probe_cull_result, p_gi_probe_cull_count, p_cam_transform);
	_setup_environment(p_environment, p_cam_projection, p_cam_transform, p_reflection_probe, p_reflection_probe.is_valid(), screen_pixel_size, p_shadow_atlas, !p_reflection_probe.is_valid(), p_default_bg_color, p_cam_projection.get_z_near(), p_cam_projection.get_z_far(), false);

	cluster_builder.bake_cluster(); //bake to cluster

	_update_render_base_uniform_set(); //may have changed due to the above (light buffer enlarged, as an example)

	render_list.clear();
	_fill_render_list(p_cull_result, p_cull_count, PASS_MODE_COLOR, render_buffer == nullptr);

	bool using_sss = render_buffer && scene_state.used_sss && sub_surface_scattering_get_quality() != RS::SUB_SURFACE_SCATTERING_QUALITY_DISABLED;

	if (using_sss) {
		using_separate_specular = true;
		render_buffer->ensure_specular();
		using_separate_specular = true;
		opaque_specular_framebuffer = render_buffer->color_specular_fb;
	}
	RID radiance_uniform_set;
	bool draw_sky = false;

	Color clear_color;
	bool keep_color = false;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW) {
		clear_color = Color(0, 0, 0, 1); //in overdraw mode, BG should always be black
	} else if (is_environment(p_environment)) {
		RS::EnvironmentBG bg_mode = environment_get_background(p_environment);
		float bg_energy = environment_get_bg_energy(p_environment);
		switch (bg_mode) {
			case RS::ENV_BG_CLEAR_COLOR: {
				clear_color = p_default_bg_color;
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
			} break;
			case RS::ENV_BG_COLOR: {
				clear_color = environment_get_bg_color(p_environment);
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
			} break;
			case RS::ENV_BG_SKY: {
				RID sky = environment_get_sky(p_environment);
				if (sky.is_valid()) {

					RENDER_TIMESTAMP("Setup Sky");
					CameraMatrix projection = p_cam_projection;
					if (p_reflection_probe.is_valid()) {
						CameraMatrix correction;
						correction.set_depth_correction(true);
						projection = correction * p_cam_projection;
					}

					_setup_sky(p_environment, p_cam_transform.origin, screen_size);
					_update_sky(p_environment, projection, p_cam_transform);
					radiance_uniform_set = sky_get_radiance_uniform_set_rd(sky, default_shader_rd, RADIANCE_UNIFORM_SET);

					draw_sky = true;
				}
			} break;
			case RS::ENV_BG_CANVAS: {
				keep_color = true;
			} break;
			case RS::ENV_BG_KEEP: {
				keep_color = true;
			} break;
			case RS::ENV_BG_CAMERA_FEED: {

			} break;
			default: {
			}
		}
	} else {

		clear_color = p_default_bg_color;
	}

	_setup_view_dependant_uniform_set(p_shadow_atlas, p_reflection_atlas);

	render_list.sort_by_key(false);

	_fill_instances(render_list.elements, render_list.element_count, false);

	bool debug_giprobes = get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_GI_PROBE_ALBEDO || get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_GI_PROBE_LIGHTING || get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_GI_PROBE_EMISSION;

	bool depth_pre_pass = depth_framebuffer.is_valid();
	RID render_buffers_uniform_set;

	bool using_ssao = depth_pre_pass && p_render_buffer.is_valid() && p_environment.is_valid() && environment_is_ssao_enabled(p_environment);

	if (depth_pre_pass) { //depth pre pass
		RENDER_TIMESTAMP("Render Depth Pre-Pass");

		bool finish_depth = using_ssao;
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(depth_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, finish_depth ? RD::FINAL_ACTION_READ : RD::FINAL_ACTION_CONTINUE, depth_pass_clear);
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(depth_framebuffer), render_list.elements, render_list.element_count, false, depth_pass_mode, render_buffer == nullptr, radiance_uniform_set, RID());
		RD::get_singleton()->draw_list_end();

		if (render_buffer && render_buffer->msaa != RS::VIEWPORT_MSAA_DISABLED) {
			if (finish_depth) {
				RD::get_singleton()->texture_resolve_multisample(render_buffer->depth_msaa, render_buffer->depth, true);
			}

			if (depth_pass_mode == PASS_MODE_DEPTH_NORMAL || depth_pass_mode == PASS_MODE_DEPTH_NORMAL_ROUGHNESS) {
				RD::get_singleton()->texture_resolve_multisample(render_buffer->normal_buffer_msaa, render_buffer->normal_buffer, true);
				if (depth_pass_mode == PASS_MODE_DEPTH_NORMAL_ROUGHNESS) {
					RD::get_singleton()->texture_resolve_multisample(render_buffer->roughness_buffer_msaa, render_buffer->roughness_buffer, true);
				}
			}
		}
	}

	if (using_ssao) {
		_process_ssao(p_render_buffer, p_environment, render_buffer->normal_buffer, p_cam_projection);
	}

	if (p_render_buffer.is_valid() && screen_space_roughness_limiter_is_active()) {
		storage->get_effects()->roughness_limit(render_buffer->normal_buffer, render_buffer->roughness_buffer, Size2(render_buffer->width, render_buffer->height), screen_space_roughness_limiter_get_curve());
	}

	if (p_render_buffer.is_valid()) {
		//update the render buffers uniform set in case it changed
		_update_render_buffers_uniform_set(p_render_buffer);
		render_buffers_uniform_set = render_buffer->uniform_set;
	}

	_setup_environment(p_environment, p_cam_projection, p_cam_transform, p_reflection_probe, p_reflection_probe.is_valid(), screen_pixel_size, p_shadow_atlas, !p_reflection_probe.is_valid(), p_default_bg_color, p_cam_projection.get_z_near(), p_cam_projection.get_z_far(), p_render_buffer.is_valid());

	RENDER_TIMESTAMP("Render Opaque Pass");

	bool can_continue_color = !scene_state.used_screen_texture && !using_ssr && !using_sss;
	bool can_continue_depth = !scene_state.used_depth_texture && !using_ssr && !using_sss;

	{

		bool will_continue_color = (can_continue_color || draw_sky || debug_giprobes);
		bool will_continue_depth = (can_continue_depth || draw_sky || debug_giprobes);

		//regular forward for now
		Vector<Color> c;
		if (using_separate_specular) {
			Color cc = clear_color.to_linear();
			cc.a = 0; //subsurf scatter must be 0
			c.push_back(cc);
			c.push_back(Color(0, 0, 0, 0));
		} else {
			c.push_back(clear_color.to_linear());
		}

		RID framebuffer = using_separate_specular ? opaque_specular_framebuffer : opaque_framebuffer;
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer, keep_color ? RD::INITIAL_ACTION_KEEP : RD::INITIAL_ACTION_CLEAR, will_continue_color ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, depth_pre_pass ? (using_ssao ? RD::INITIAL_ACTION_KEEP : RD::INITIAL_ACTION_CONTINUE) : RD::INITIAL_ACTION_CLEAR, will_continue_depth ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, c, 1.0, 0);
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(framebuffer), render_list.elements, render_list.element_count, false, using_separate_specular ? PASS_MODE_COLOR_SPECULAR : PASS_MODE_COLOR, render_buffer == nullptr, radiance_uniform_set, render_buffers_uniform_set);
		RD::get_singleton()->draw_list_end();

		if (will_continue_color && using_separate_specular) {
			// close the specular framebuffer, as it's no longer used
			draw_list = RD::get_singleton()->draw_list_begin(render_buffer->specular_only_fb, RD::INITIAL_ACTION_CONTINUE, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CONTINUE, RD::FINAL_ACTION_CONTINUE);
			RD::get_singleton()->draw_list_end();
		}
	}

	if (debug_giprobes) {
		//debug giprobes
		bool will_continue_color = (can_continue_color || draw_sky);
		bool will_continue_depth = (can_continue_depth || draw_sky);

		CameraMatrix dc;
		dc.set_depth_correction(true);
		CameraMatrix cm = (dc * p_cam_projection) * CameraMatrix(p_cam_transform.affine_inverse());
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(opaque_framebuffer, RD::INITIAL_ACTION_CONTINUE, will_continue_color ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CONTINUE, will_continue_depth ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ);
		for (int i = 0; i < p_gi_probe_cull_count; i++) {
			_debug_giprobe(p_gi_probe_cull_result[i], draw_list, opaque_framebuffer, cm, get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_GI_PROBE_LIGHTING, get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_GI_PROBE_EMISSION, 1.0);
		}
		RD::get_singleton()->draw_list_end();
	}

	if (draw_sky) {
		RENDER_TIMESTAMP("Render Sky");

		CameraMatrix projection = p_cam_projection;
		if (p_reflection_probe.is_valid()) {
			CameraMatrix correction;
			correction.set_depth_correction(true);
			projection = correction * p_cam_projection;
		}

		_draw_sky(can_continue_color, can_continue_depth, opaque_framebuffer, p_environment, projection, p_cam_transform);
	}

	if (render_buffer && !can_continue_color && render_buffer->msaa != RS::VIEWPORT_MSAA_DISABLED) {

		RD::get_singleton()->texture_resolve_multisample(render_buffer->color_msaa, render_buffer->color, true);
		if (using_separate_specular) {
			RD::get_singleton()->texture_resolve_multisample(render_buffer->specular_msaa, render_buffer->specular, true);
		}
	}

	if (render_buffer && !can_continue_depth && render_buffer->msaa != RS::VIEWPORT_MSAA_DISABLED) {

		RD::get_singleton()->texture_resolve_multisample(render_buffer->depth_msaa, render_buffer->depth, true);
	}

	if (using_separate_specular) {

		if (using_sss) {
			RENDER_TIMESTAMP("Sub Surface Scattering");
			_process_sss(p_render_buffer, p_cam_projection);
		}

		if (using_ssr) {
			RENDER_TIMESTAMP("Screen Space Reflection");
			_process_ssr(p_render_buffer, render_buffer->color_fb, render_buffer->normal_buffer, render_buffer->roughness_buffer, render_buffer->specular, render_buffer->specular, Color(0, 0, 0, 1), p_environment, p_cam_projection, render_buffer->msaa == RS::VIEWPORT_MSAA_DISABLED);
		} else {
			//just mix specular back
			RENDER_TIMESTAMP("Merge Specular");
			storage->get_effects()->merge_specular(render_buffer->color_fb, render_buffer->specular, render_buffer->msaa == RS::VIEWPORT_MSAA_DISABLED ? RID() : render_buffer->color, RID());
		}
	}

	RENDER_TIMESTAMP("Render Transparent Pass");

	_setup_environment(p_environment, p_cam_projection, p_cam_transform, p_reflection_probe, p_reflection_probe.is_valid(), screen_pixel_size, p_shadow_atlas, !p_reflection_probe.is_valid(), p_default_bg_color, p_cam_projection.get_z_near(), p_cam_projection.get_z_far(), false);

	render_list.sort_by_reverse_depth_and_priority(true);

	_fill_instances(&render_list.elements[render_list.max_elements - render_list.alpha_element_count], render_list.alpha_element_count, false);

	{
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(alpha_framebuffer, can_continue_color ? RD::INITIAL_ACTION_CONTINUE : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, can_continue_depth ? RD::INITIAL_ACTION_CONTINUE : RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ);
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(alpha_framebuffer), &render_list.elements[render_list.max_elements - render_list.alpha_element_count], render_list.alpha_element_count, false, PASS_MODE_COLOR, render_buffer == nullptr, radiance_uniform_set, render_buffers_uniform_set);
		RD::get_singleton()->draw_list_end();
	}

	if (render_buffer && render_buffer->msaa != RS::VIEWPORT_MSAA_DISABLED) {

		RD::get_singleton()->texture_resolve_multisample(render_buffer->color_msaa, render_buffer->color, true);
	}
}

void RasterizerSceneHighEndRD::_render_shadow(RID p_framebuffer, InstanceBase **p_cull_result, int p_cull_count, const CameraMatrix &p_projection, const Transform &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake) {

	RENDER_TIMESTAMP("Setup Rendering Shadow");

	_update_render_base_uniform_set();

	render_pass++;

	scene_state.ubo.dual_paraboloid_side = p_use_dp_flip ? -1 : 1;

	_setup_environment(RID(), p_projection, p_transform, RID(), true, Vector2(1, 1), RID(), true, Color(), 0, p_zfar, false, p_use_pancake);

	render_list.clear();

	PassMode pass_mode = p_use_dp ? PASS_MODE_SHADOW_DP : PASS_MODE_SHADOW;

	_fill_render_list(p_cull_result, p_cull_count, pass_mode, true);

	_setup_view_dependant_uniform_set(RID(), RID());

	RENDER_TIMESTAMP("Render Shadow");

	render_list.sort_by_key(false);

	_fill_instances(render_list.elements, render_list.element_count, true);

	{
		//regular forward for now
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ);
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(p_framebuffer), render_list.elements, render_list.element_count, p_use_dp_flip, pass_mode, true, RID(), RID());
		RD::get_singleton()->draw_list_end();
	}
}

void RasterizerSceneHighEndRD::_render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) {
	RENDER_TIMESTAMP("Setup Rendering Shadow");

	_update_render_base_uniform_set();

	render_pass++;

	scene_state.ubo.dual_paraboloid_side = 0;

	_setup_environment(RID(), p_cam_projection, p_cam_transform, RID(), true, Vector2(1, 1), RID(), false, Color(), 0, 0);

	render_list.clear();

	PassMode pass_mode = PASS_MODE_DEPTH_MATERIAL;
	_fill_render_list(p_cull_result, p_cull_count, pass_mode, true);

	_setup_view_dependant_uniform_set(RID(), RID());

	RENDER_TIMESTAMP("Render Material");

	render_list.sort_by_key(false);

	_fill_instances(render_list.elements, render_list.element_count, true);

	{
		//regular forward for now
		Vector<Color> clear;
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		clear.push_back(Color(0, 0, 0, 0));
		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, clear, 1.0, 0, p_region);
		_render_list(draw_list, RD::get_singleton()->framebuffer_get_format(p_framebuffer), render_list.elements, render_list.element_count, true, pass_mode, true, RID(), RID());
		RD::get_singleton()->draw_list_end();
	}
}

void RasterizerSceneHighEndRD::_base_uniforms_changed() {

	if (!render_base_uniform_set.is_null() && RD::get_singleton()->uniform_set_is_valid(render_base_uniform_set)) {
		RD::get_singleton()->free(render_base_uniform_set);
	}
	render_base_uniform_set = RID();
}

void RasterizerSceneHighEndRD::_update_render_base_uniform_set() {

	if (render_base_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(render_base_uniform_set)) {

		if (render_base_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(render_base_uniform_set)) {
			RD::get_singleton()->free(render_base_uniform_set);
		}

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 1;
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
			u.binding = 2;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
			u.ids.push_back(shadow_sampler);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 3;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(scene_state.uniform_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 4;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(scene_state.instance_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 5;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(scene_state.light_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 6;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(scene_state.reflection_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 7;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(scene_state.directional_light_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 8;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(scene_state.gi_probe_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 9;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			int slot_count = gi_probe_get_slots().size();
			if (gi_probe_is_anisotropic()) {
				u.ids.resize(slot_count * 3);
			} else {
				u.ids.resize(slot_count);
			}

			for (int i = 0; i < slot_count; i++) {

				RID probe = gi_probe_get_slots()[i];

				if (gi_probe_is_anisotropic()) {
					if (probe.is_null()) {
						RID empty_tex = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);
						u.ids.write[i * 3 + 0] = empty_tex;
						u.ids.write[i * 3 + 1] = empty_tex;
						u.ids.write[i * 3 + 2] = empty_tex;
					} else {
						u.ids.write[i * 3 + 0] = gi_probe_instance_get_texture(probe);
						u.ids.write[i * 3 + 1] = gi_probe_instance_get_aniso_texture(probe, 0);
						u.ids.write[i * 3 + 2] = gi_probe_instance_get_aniso_texture(probe, 1);
					}
				} else {
					if (probe.is_null()) {
						u.ids.write[i] = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);
					} else {
						u.ids.write[i] = gi_probe_instance_get_texture(probe);
					}
				}
			}

			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 10;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID decal_atlas = storage->decal_atlas_get_texture();
			u.ids.push_back(decal_atlas);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 11;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID decal_atlas = storage->decal_atlas_get_texture_srgb();
			u.ids.push_back(decal_atlas);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 12;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(scene_state.decal_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 13;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.ids.push_back(cluster_builder.get_cluster_texture());
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 14;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(cluster_builder.get_cluster_indices_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 15;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			if (directional_shadow_get_texture().is_valid()) {
				u.ids.push_back(directional_shadow_get_texture());
			} else {
				u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			}
			uniforms.push_back(u);
		}

		render_base_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, SCENE_UNIFORM_SET);
	}
}

void RasterizerSceneHighEndRD::_setup_view_dependant_uniform_set(RID p_shadow_atlas, RID p_reflection_atlas) {

	if (view_dependant_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(view_dependant_uniform_set)) {
		RD::get_singleton()->free(view_dependant_uniform_set);
	}

	//default render buffer and scene state uniform set

	Vector<RD::Uniform> uniforms;

	{

		RID ref_texture = p_reflection_atlas.is_valid() ? reflection_atlas_get_texture(p_reflection_atlas) : RID();
		RD::Uniform u;
		u.binding = 0;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		if (ref_texture.is_valid()) {
			u.ids.push_back(ref_texture);
		} else {
			u.ids.push_back(storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK));
		}
		uniforms.push_back(u);
	}

	{
		RD::Uniform u;
		u.binding = 1;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		RID texture;
		if (p_shadow_atlas.is_valid()) {
			texture = shadow_atlas_get_texture(p_shadow_atlas);
		}
		if (!texture.is_valid()) {
			texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}
		u.ids.push_back(texture);
		uniforms.push_back(u);
	}

	view_dependant_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, VIEW_DEPENDANT_UNIFORM_SET);
}

void RasterizerSceneHighEndRD::_render_buffers_clear_uniform_set(RenderBufferDataHighEnd *rb) {

	if (!rb->uniform_set.is_null() && RD::get_singleton()->uniform_set_is_valid(rb->uniform_set)) {
		RD::get_singleton()->free(rb->uniform_set);
	}
	rb->uniform_set = RID();
}

void RasterizerSceneHighEndRD::_render_buffers_uniform_set_changed(RID p_render_buffers) {

	RenderBufferDataHighEnd *rb = (RenderBufferDataHighEnd *)render_buffers_get_data(p_render_buffers);

	_render_buffers_clear_uniform_set(rb);
}

RID RasterizerSceneHighEndRD::_render_buffers_get_roughness_texture(RID p_render_buffers) {
	RenderBufferDataHighEnd *rb = (RenderBufferDataHighEnd *)render_buffers_get_data(p_render_buffers);

	return rb->roughness_buffer;
}

RID RasterizerSceneHighEndRD::_render_buffers_get_normal_texture(RID p_render_buffers) {
	RenderBufferDataHighEnd *rb = (RenderBufferDataHighEnd *)render_buffers_get_data(p_render_buffers);

	return rb->normal_buffer;
}

void RasterizerSceneHighEndRD::_update_render_buffers_uniform_set(RID p_render_buffers) {

	RenderBufferDataHighEnd *rb = (RenderBufferDataHighEnd *)render_buffers_get_data(p_render_buffers);

	if (rb->uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(rb->uniform_set)) {

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID texture = false && rb->depth.is_valid() ? rb->depth : storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE);
			u.ids.push_back(texture);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 1;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID bbt = render_buffers_get_back_buffer_texture(p_render_buffers);
			RID texture = bbt.is_valid() ? bbt : storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK);
			u.ids.push_back(texture);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID texture = rb->normal_buffer.is_valid() ? rb->normal_buffer : storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_NORMAL);
			u.ids.push_back(texture);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 3;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID texture = rb->roughness_buffer.is_valid() ? rb->roughness_buffer : storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK);
			u.ids.push_back(texture);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 4;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID aot = render_buffers_get_ao_texture(p_render_buffers);
			RID texture = aot.is_valid() ? aot : storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK);
			u.ids.push_back(texture);
			uniforms.push_back(u);
		}

		rb->uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, RENDER_BUFFERS_UNIFORM_SET);
	}
}

RasterizerSceneHighEndRD *RasterizerSceneHighEndRD::singleton = nullptr;

void RasterizerSceneHighEndRD::set_time(double p_time, double p_step) {
	time = p_time;
	RasterizerSceneRD::set_time(p_time, p_step);
}

RasterizerSceneHighEndRD::RasterizerSceneHighEndRD(RasterizerStorageRD *p_storage) :
		RasterizerSceneRD(p_storage) {
	singleton = this;
	storage = p_storage;

	/* SCENE SHADER */

	{
		String defines;
		defines += "\n#define MAX_ROUGHNESS_LOD " + itos(get_roughness_layers() - 1) + ".0\n";
		if (is_using_radiance_cubemap_array()) {
			defines += "\n#define USE_RADIANCE_CUBEMAP_ARRAY \n";
		}

		uint32_t uniform_max_size = RD::get_singleton()->limit_get(RD::LIMIT_MAX_UNIFORM_BUFFER_SIZE);

		{ //reflections
			uint32_t reflection_buffer_size;
			if (uniform_max_size < 65536) {
				//Yes, you guessed right, ARM again
				reflection_buffer_size = uniform_max_size;
			} else {
				reflection_buffer_size = 65536;
			}

			scene_state.max_reflections = reflection_buffer_size / sizeof(ReflectionData);
			scene_state.reflections = memnew_arr(ReflectionData, scene_state.max_reflections);
			scene_state.reflection_buffer = RD::get_singleton()->uniform_buffer_create(reflection_buffer_size);
			defines += "\n#define MAX_REFLECTION_DATA_STRUCTS " + itos(scene_state.max_reflections) + "\n";
		}

		{ //lights
			scene_state.max_lights = MIN(1024 * 1024, uniform_max_size) / sizeof(LightData); //1mb of lights
			uint32_t light_buffer_size = scene_state.max_lights * sizeof(LightData);
			scene_state.lights = memnew_arr(LightData, scene_state.max_lights);
			scene_state.light_buffer = RD::get_singleton()->storage_buffer_create(light_buffer_size);
			//defines += "\n#define MAX_LIGHT_DATA_STRUCTS " + itos(scene_state.max_lights) + "\n";

			scene_state.max_directional_lights = 8;
			uint32_t directional_light_buffer_size = scene_state.max_directional_lights * sizeof(DirectionalLightData);
			scene_state.directional_lights = memnew_arr(DirectionalLightData, scene_state.max_directional_lights);
			scene_state.directional_light_buffer = RD::get_singleton()->uniform_buffer_create(directional_light_buffer_size);
			defines += "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(scene_state.max_directional_lights) + "\n";
		}

		{ //giprobes
			int slot_count = gi_probe_get_slots().size();
			if (gi_probe_is_anisotropic()) {
				slot_count *= 3;
				defines += "\n#define GI_PROBE_USE_ANISOTROPY\n";
			}

			if (gi_probe_get_quality() == GIPROBE_QUALITY_ULTRA_LOW) {
				defines += "\n#define GI_PROBE_LOW_QUALITY\n";
			} else if (gi_probe_get_quality() == GIPROBE_QUALITY_HIGH) {
				defines += "\n#define GI_PROBE_HIGH_QUALITY\n";
			}

			defines += "\n#define MAX_GI_PROBE_TEXTURES " + itos(slot_count) + "\n";

			uint32_t giprobe_buffer_size;
			if (uniform_max_size < 65536) {
				//Yes, you guessed right, ARM again
				giprobe_buffer_size = uniform_max_size;
			} else {
				giprobe_buffer_size = 65536;
			}

			giprobe_buffer_size = MIN(sizeof(GIProbeData) * gi_probe_get_slots().size(), giprobe_buffer_size);
			scene_state.max_gi_probes = giprobe_buffer_size / sizeof(GIProbeData);
			scene_state.gi_probes = memnew_arr(GIProbeData, scene_state.max_gi_probes);
			scene_state.gi_probe_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(GIProbeData) * scene_state.max_gi_probes);
			defines += "\n#define MAX_GI_PROBES " + itos(scene_state.max_gi_probes) + "\n";
		}

		{ //decals
			scene_state.max_decals = MIN(1024 * 1024, uniform_max_size) / sizeof(DecalData); //1mb of decals
			uint32_t decal_buffer_size = scene_state.max_decals * sizeof(DecalData);
			scene_state.decals = memnew_arr(DecalData, scene_state.max_decals);
			scene_state.decal_buffer = RD::get_singleton()->storage_buffer_create(decal_buffer_size);
		}

		Vector<String> shader_versions;
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n");
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n#define MODE_DUAL_PARABOLOID\n");
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n#define MODE_RENDER_NORMAL\n");
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n#define MODE_RENDER_NORMAL\n#define MODE_RENDER_ROUGHNESS\n");
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n#define MODE_RENDER_MATERIAL\n");
		shader_versions.push_back("");
		shader_versions.push_back("\n#define MODE_MULTIPLE_RENDER_TARGETS\n");
		shader_versions.push_back("\n#define USE_VOXEL_CONE_TRACING\n");
		shader_versions.push_back("\n#define MODE_MULTIPLE_RENDER_TARGETS\n#define USE_VOXEL_CONE_TRACING\n");
		shader_versions.push_back("\n#define USE_LIGHTMAP\n");
		shader_versions.push_back("\n#define MODE_MULTIPLE_RENDER_TARGETS\n#define USE_LIGHTMAP\n");
		shader.scene_shader.initialize(shader_versions, defines);
	}

	storage->shader_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_3D, _create_shader_funcs);
	storage->material_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_3D, _create_material_funcs);

	{
		//shader compiler
		ShaderCompilerRD::DefaultIdentifierActions actions;

		actions.renames["WORLD_MATRIX"] = "world_matrix";
		actions.renames["WORLD_NORMAL_MATRIX"] = "world_normal_matrix";
		actions.renames["INV_CAMERA_MATRIX"] = "scene_data.inv_camera_matrix";
		actions.renames["CAMERA_MATRIX"] = "scene_data.camera_matrix";
		actions.renames["PROJECTION_MATRIX"] = "projection_matrix";
		actions.renames["INV_PROJECTION_MATRIX"] = "scene_data.inv_projection_matrix";
		actions.renames["MODELVIEW_MATRIX"] = "modelview";
		actions.renames["MODELVIEW_NORMAL_MATRIX"] = "modelview_normal";

		actions.renames["VERTEX"] = "vertex";
		actions.renames["NORMAL"] = "normal";
		actions.renames["TANGENT"] = "tangent";
		actions.renames["BINORMAL"] = "binormal";
		actions.renames["POSITION"] = "position";
		actions.renames["UV"] = "uv_interp";
		actions.renames["UV2"] = "uv2_interp";
		actions.renames["COLOR"] = "color_interp";
		actions.renames["POINT_SIZE"] = "gl_PointSize";
		actions.renames["INSTANCE_ID"] = "gl_InstanceIndex";

		//builtins

		actions.renames["TIME"] = "scene_data.time";
		actions.renames["VIEWPORT_SIZE"] = "scene_data.viewport_size";

		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["FRONT_FACING"] = "gl_FrontFacing";
		actions.renames["NORMALMAP"] = "normalmap";
		actions.renames["NORMALMAP_DEPTH"] = "normaldepth";
		actions.renames["ALBEDO"] = "albedo";
		actions.renames["ALPHA"] = "alpha";
		actions.renames["METALLIC"] = "metallic";
		actions.renames["SPECULAR"] = "specular";
		actions.renames["ROUGHNESS"] = "roughness";
		actions.renames["RIM"] = "rim";
		actions.renames["RIM_TINT"] = "rim_tint";
		actions.renames["CLEARCOAT"] = "clearcoat";
		actions.renames["CLEARCOAT_GLOSS"] = "clearcoat_gloss";
		actions.renames["ANISOTROPY"] = "anisotropy";
		actions.renames["ANISOTROPY_FLOW"] = "anisotropy_flow";
		actions.renames["SSS_STRENGTH"] = "sss_strength";
		actions.renames["SSS_TRANSMITTANCE_COLOR"] = "transmittance_color";
		actions.renames["SSS_TRANSMITTANCE_DEPTH"] = "transmittance_depth";
		actions.renames["SSS_TRANSMITTANCE_CURVE"] = "transmittance_curve";
		actions.renames["SSS_TRANSMITTANCE_BOOST"] = "transmittance_boost";
		actions.renames["BACKLIGHT"] = "backlight";
		actions.renames["AO"] = "ao";
		actions.renames["AO_LIGHT_AFFECT"] = "ao_light_affect";
		actions.renames["EMISSION"] = "emission";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";
		actions.renames["SCREEN_UV"] = "screen_uv";
		actions.renames["SCREEN_TEXTURE"] = "color_buffer";
		actions.renames["DEPTH_TEXTURE"] = "depth_buffer";
		actions.renames["NORMAL_TEXTURE"] = "normal_buffer";
		actions.renames["DEPTH"] = "gl_FragDepth";
		actions.renames["OUTPUT_IS_SRGB"] = "true";

		//for light
		actions.renames["VIEW"] = "view";
		actions.renames["LIGHT_COLOR"] = "light_color";
		actions.renames["LIGHT"] = "light";
		actions.renames["ATTENUATION"] = "attenuation";
		actions.renames["DIFFUSE_LIGHT"] = "diffuse_light";
		actions.renames["SPECULAR_LIGHT"] = "specular_light";

		actions.usage_defines["TANGENT"] = "#define TANGENT_USED\n";
		actions.usage_defines["BINORMAL"] = "@TANGENT";
		actions.usage_defines["RIM"] = "#define LIGHT_RIM_USED\n";
		actions.usage_defines["RIM_TINT"] = "@RIM";
		actions.usage_defines["CLEARCOAT"] = "#define LIGHT_CLEARCOAT_USED\n";
		actions.usage_defines["CLEARCOAT_GLOSS"] = "@CLEARCOAT";
		actions.usage_defines["ANISOTROPY"] = "#define LIGHT_ANISOTROPY_USED\n";
		actions.usage_defines["ANISOTROPY_FLOW"] = "@ANISOTROPY";
		actions.usage_defines["AO"] = "#define AO_USED\n";
		actions.usage_defines["AO_LIGHT_AFFECT"] = "#define AO_USED\n";
		actions.usage_defines["UV"] = "#define UV_USED\n";
		actions.usage_defines["UV2"] = "#define UV2_USED\n";
		actions.usage_defines["NORMALMAP"] = "#define NORMALMAP_USED\n";
		actions.usage_defines["NORMALMAP_DEPTH"] = "@NORMALMAP";
		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["INSTANCE_CUSTOM"] = "#define ENABLE_INSTANCE_CUSTOM\n";
		actions.usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";

		actions.usage_defines["SSS_STRENGTH"] = "#define ENABLE_SSS\n";
		actions.usage_defines["SSS_TRANSMITTANCE_DEPTH"] = "#define ENABLE_TRANSMITTANCE\n";
		actions.usage_defines["BACKLIGHT"] = "#define LIGHT_BACKLIGHT_USED\n";
		actions.usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
		actions.usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";

		actions.usage_defines["DIFFUSE_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";
		actions.usage_defines["SPECULAR_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";

		actions.render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
		actions.render_mode_defines["world_vertex_coords"] = "#define VERTEX_WORLD_COORDS_USED\n";
		actions.render_mode_defines["ensure_correct_normals"] = "#define ENSURE_CORRECT_NORMALS\n";
		actions.render_mode_defines["cull_front"] = "#define DO_SIDE_CHECK\n";
		actions.render_mode_defines["cull_disabled"] = "#define DO_SIDE_CHECK\n";

		bool force_lambert = GLOBAL_GET("rendering/quality/shading/force_lambert_over_burley");

		if (!force_lambert) {
			actions.render_mode_defines["diffuse_burley"] = "#define DIFFUSE_BURLEY\n";
		}

		actions.render_mode_defines["diffuse_oren_nayar"] = "#define DIFFUSE_OREN_NAYAR\n";
		actions.render_mode_defines["diffuse_lambert_wrap"] = "#define DIFFUSE_LAMBERT_WRAP\n";
		actions.render_mode_defines["diffuse_toon"] = "#define DIFFUSE_TOON\n";

		actions.render_mode_defines["sss_mode_skin"] = "#define SSS_MODE_SKIN\n";

		bool force_blinn = GLOBAL_GET("rendering/quality/shading/force_blinn_over_ggx");

		if (!force_blinn) {
			actions.render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_SCHLICK_GGX\n";
		} else {
			actions.render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_BLINN\n";
		}

		actions.render_mode_defines["specular_blinn"] = "#define SPECULAR_BLINN\n";
		actions.render_mode_defines["specular_phong"] = "#define SPECULAR_PHONG\n";
		actions.render_mode_defines["specular_toon"] = "#define SPECULAR_TOON\n";
		actions.render_mode_defines["specular_disabled"] = "#define SPECULAR_DISABLED\n";
		actions.render_mode_defines["shadows_disabled"] = "#define SHADOWS_DISABLED\n";
		actions.render_mode_defines["ambient_light_disabled"] = "#define AMBIENT_LIGHT_DISABLED\n";
		actions.render_mode_defines["shadow_to_opacity"] = "#define USE_SHADOW_TO_OPACITY\n";
		actions.render_mode_defines["unshaded"] = "#define MODE_UNSHADED\n";

		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = MATERIAL_UNIFORM_SET;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;

		shader.compiler.initialize(actions);
	}

	//render list
	render_list.max_elements = GLOBAL_DEF_RST("rendering/limits/rendering/max_renderable_elements", (int)128000);
	render_list.init();
	render_pass = 0;

	{

		scene_state.max_instances = render_list.max_elements;
		scene_state.instances = memnew_arr(InstanceData, scene_state.max_instances);
		scene_state.instance_buffer = RD::get_singleton()->storage_buffer_create(sizeof(InstanceData) * scene_state.max_instances);
	}

	scene_state.uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(SceneState::UBO));

	{
		//default material and shader
		default_shader = storage->shader_create();
		storage->shader_set_code(default_shader, "shader_type spatial; void vertex() { ROUGHNESS = 0.8; } void fragment() { ALBEDO=vec3(0.6); ROUGHNESS=0.8; METALLIC=0.2; } \n");
		default_material = storage->material_create();
		storage->material_set_shader(default_material, default_shader);

		MaterialData *md = (MaterialData *)storage->material_get_data(default_material, RasterizerStorageRD::SHADER_TYPE_3D);
		default_shader_rd = shader.scene_shader.version_get_shader(md->shader_data->version, SHADER_VERSION_COLOR_PASS);
	}

	{

		overdraw_material_shader = storage->shader_create();
		storage->shader_set_code(overdraw_material_shader, "shader_type spatial;\nrender_mode blend_add,unshaded;\n void fragment() { ALBEDO=vec3(0.4,0.8,0.8); ALPHA=0.2; }");
		overdraw_material = storage->material_create();
		storage->material_set_shader(overdraw_material, overdraw_material_shader);

		wireframe_material_shader = storage->shader_create();
		storage->shader_set_code(wireframe_material_shader, "shader_type spatial;\nrender_mode wireframe,unshaded;\n void fragment() { ALBEDO=vec3(0.0,0.0,0.0); }");
		wireframe_material = storage->material_create();
		storage->material_set_shader(wireframe_material, wireframe_material_shader);
	}

	{
		default_vec4_xform_buffer = RD::get_singleton()->storage_buffer_create(256);
		Vector<RD::Uniform> uniforms;
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.ids.push_back(default_vec4_xform_buffer);
		u.binding = 0;
		uniforms.push_back(u);

		default_vec4_xform_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, TRANSFORMS_UNIFORM_SET);
	}
	{

		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.enable_compare = true;
		sampler.compare_op = RD::COMPARE_OP_LESS;
		shadow_sampler = RD::get_singleton()->sampler_create(sampler);
	}

	{
		Vector<RD::Uniform> uniforms;

		RD::Uniform u;
		u.binding = 0;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		RID texture = storage->texture_rd_get_default(is_using_radiance_cubemap_array() ? RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK : RasterizerStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK);
		u.ids.push_back(texture);
		uniforms.push_back(u);

		default_radiance_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, RADIANCE_UNIFORM_SET);
	}

	{ //render buffers
		Vector<RD::Uniform> uniforms;
		for (int i = 0; i < 5; i++) {
			RD::Uniform u;
			u.binding = i;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			RID texture = storage->texture_rd_get_default(i == 0 ? RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE : (i == 2 ? RasterizerStorageRD::DEFAULT_RD_TEXTURE_NORMAL : RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK));
			u.ids.push_back(texture);
			uniforms.push_back(u);
		}

		default_render_buffers_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, RENDER_BUFFERS_UNIFORM_SET);
	}

	cluster_builder.setup(16, 8, 24);
}

RasterizerSceneHighEndRD::~RasterizerSceneHighEndRD() {
	directional_shadow_atlas_set_size(0);

	//clear base uniform set if still valid
	if (view_dependant_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(view_dependant_uniform_set)) {
		RD::get_singleton()->free(view_dependant_uniform_set);
	}

	RD::get_singleton()->free(default_render_buffers_uniform_set);
	RD::get_singleton()->free(default_radiance_uniform_set);
	RD::get_singleton()->free(default_vec4_xform_buffer);
	RD::get_singleton()->free(shadow_sampler);

	storage->free(wireframe_material_shader);
	storage->free(overdraw_material_shader);
	storage->free(default_shader);

	storage->free(wireframe_material);
	storage->free(overdraw_material);
	storage->free(default_material);

	{
		RD::get_singleton()->free(scene_state.uniform_buffer);
		RD::get_singleton()->free(scene_state.instance_buffer);
		RD::get_singleton()->free(scene_state.gi_probe_buffer);
		RD::get_singleton()->free(scene_state.directional_light_buffer);
		RD::get_singleton()->free(scene_state.light_buffer);
		RD::get_singleton()->free(scene_state.reflection_buffer);
		RD::get_singleton()->free(scene_state.decal_buffer);
		memdelete_arr(scene_state.instances);
		memdelete_arr(scene_state.gi_probes);
		memdelete_arr(scene_state.directional_lights);
		memdelete_arr(scene_state.lights);
		memdelete_arr(scene_state.reflections);
		memdelete_arr(scene_state.decals);
	}
}

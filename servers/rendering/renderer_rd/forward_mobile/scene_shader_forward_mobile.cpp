/**************************************************************************/
/*  scene_shader_forward_mobile.cpp                                       */
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

#include "scene_shader_forward_mobile.h"
#include "core/config/project_settings.h"
#include "core/math/math_defs.h"
#include "render_forward_mobile.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"

using namespace RendererSceneRenderImplementation;

/* ShaderData */

void SceneShaderForwardMobile::ShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;

	int blend_mode = BLEND_MODE_MIX;
	int depth_testi = DEPTH_TEST_ENABLED;
	int alpha_antialiasing_mode = ALPHA_ANTIALIASING_OFF;
	int cull = CULL_BACK;

	uses_point_size = false;
	uses_alpha = false;
	uses_alpha_clip = false;
	uses_alpha_antialiasing = false;
	uses_blend_alpha = false;
	uses_depth_prepass_alpha = false;
	uses_discard = false;
	uses_roughness = false;
	uses_normal = false;
	uses_tangent = false;
	bool uses_normal_map = false;
	bool wireframe = false;

	unshaded = false;
	uses_vertex = false;
	uses_sss = false;
	uses_transmittance = false;
	uses_time = false;
	writes_modelview_or_projection = false;
	uses_world_coordinates = false;
	uses_particle_trails = false;

	int depth_drawi = DEPTH_DRAW_OPAQUE;

	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);
	actions.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&blend_mode, BLEND_MODE_PREMULT_ALPHA);

	actions.render_mode_values["alpha_to_coverage"] = Pair<int *, int>(&alpha_antialiasing_mode, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE);
	actions.render_mode_values["alpha_to_coverage_and_one"] = Pair<int *, int>(&alpha_antialiasing_mode, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE);

	actions.render_mode_values["depth_draw_never"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_DISABLED);
	actions.render_mode_values["depth_draw_opaque"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_OPAQUE);
	actions.render_mode_values["depth_draw_always"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_ALWAYS);

	actions.render_mode_values["depth_test_disabled"] = Pair<int *, int>(&depth_testi, DEPTH_TEST_DISABLED);

	actions.render_mode_values["cull_disabled"] = Pair<int *, int>(&cull, CULL_DISABLED);
	actions.render_mode_values["cull_front"] = Pair<int *, int>(&cull, CULL_FRONT);
	actions.render_mode_values["cull_back"] = Pair<int *, int>(&cull, CULL_BACK);

	actions.render_mode_flags["unshaded"] = &unshaded;
	actions.render_mode_flags["wireframe"] = &wireframe;
	actions.render_mode_flags["particle_trails"] = &uses_particle_trails;
	actions.render_mode_flags["world_vertex_coords"] = &uses_world_coordinates;

	actions.usage_flag_pointers["ALPHA"] = &uses_alpha;
	actions.usage_flag_pointers["ALPHA_SCISSOR_THRESHOLD"] = &uses_alpha_clip;
	actions.usage_flag_pointers["ALPHA_HASH_SCALE"] = &uses_alpha_clip;
	actions.usage_flag_pointers["ALPHA_ANTIALIASING_EDGE"] = &uses_alpha_antialiasing;
	actions.usage_flag_pointers["ALPHA_TEXTURE_COORDINATE"] = &uses_alpha_antialiasing;
	actions.render_mode_flags["depth_prepass_alpha"] = &uses_depth_prepass_alpha;

	actions.usage_flag_pointers["SSS_STRENGTH"] = &uses_sss;
	actions.usage_flag_pointers["SSS_TRANSMITTANCE_DEPTH"] = &uses_transmittance;

	actions.usage_flag_pointers["DISCARD"] = &uses_discard;
	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["ROUGHNESS"] = &uses_roughness;
	actions.usage_flag_pointers["NORMAL"] = &uses_normal;
	actions.usage_flag_pointers["NORMAL_MAP"] = &uses_normal_map;

	actions.usage_flag_pointers["TANGENT"] = &uses_tangent;
	actions.usage_flag_pointers["BINORMAL"] = &uses_tangent;
	actions.usage_flag_pointers["ANISOTROPY"] = &uses_tangent;
	actions.usage_flag_pointers["ANISOTROPY_FLOW"] = &uses_tangent;

	actions.usage_flag_pointers["POINT_SIZE"] = &uses_point_size;
	actions.usage_flag_pointers["POINT_COORD"] = &uses_point_size;

	actions.write_flag_pointers["MODELVIEW_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["PROJECTION_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["VERTEX"] = &uses_vertex;

	actions.uniforms = &uniforms;

	SceneShaderForwardMobile *shader_singleton = (SceneShaderForwardMobile *)SceneShaderForwardMobile::singleton;

	Error err = shader_singleton->compiler.compile(RS::SHADER_SPATIAL, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = shader_singleton->shader.version_create();
	}

	depth_draw = DepthDraw(depth_drawi);
	depth_test = DepthTest(depth_testi);
	uses_vertex_time = gen_code.uses_vertex_time;
	uses_fragment_time = gen_code.uses_fragment_time;
	uses_screen_texture_mipmaps = gen_code.uses_screen_texture_mipmaps;
	uses_screen_texture = gen_code.uses_screen_texture;
	uses_depth_texture = gen_code.uses_depth_texture;
	uses_normal_texture = gen_code.uses_normal_roughness_texture;
	uses_normal |= uses_normal_map;
	uses_tangent |= uses_normal_map;

#ifdef DEBUG_ENABLED
	if (uses_sss) {
		WARN_PRINT_ONCE_ED("Sub-surface scattering is only available when using the Forward+ rendering backend.");
	}

	if (uses_transmittance) {
		WARN_PRINT_ONCE_ED("Transmittance is only available when using the Forward+ rendering backend.");
	}
#endif

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}

	HashMap<String, String>::Iterator el = gen_code.code.begin();
	while (el) {
		print_line("\n**code " + el->key + ":\n" + el->value);
		++el;
	}

	print_line("\n**uniforms:\n" + gen_code.uniforms);
	print_line("\n**vertex_globals:\n" + gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX]);
	print_line("\n**fragment_globals:\n" + gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT]);
#endif

	shader_singleton->shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines);
	ERR_FAIL_COND(!shader_singleton->shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//blend modes

	// if any form of Alpha Antialiasing is enabled, set the blend mode to alpha to coverage
	if (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF) {
		blend_mode = BLEND_MODE_ALPHA_TO_COVERAGE;
	}

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
			blend_attachment.alpha_blend_op = RD::BLEND_OP_REVERSE_SUBTRACT;
			blend_attachment.color_blend_op = RD::BLEND_OP_REVERSE_SUBTRACT;
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
		case BLEND_MODE_ALPHA_TO_COVERAGE: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
		} break;
		case BLEND_MODE_PREMULT_ALPHA: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			uses_blend_alpha = true; // Force alpha used because of blend.
		} break;
	}

	RD::PipelineColorBlendState blend_state_blend;
	blend_state_blend.attachments.push_back(blend_attachment);
	RD::PipelineColorBlendState blend_state_opaque = RD::PipelineColorBlendState::create_disabled(1);
	RD::PipelineColorBlendState blend_state_opaque_specular = RD::PipelineColorBlendState::create_disabled(2);
	RD::PipelineColorBlendState blend_state_depth_normal_roughness = RD::PipelineColorBlendState::create_disabled(1);
	RD::PipelineColorBlendState blend_state_depth_normal_roughness_giprobe = RD::PipelineColorBlendState::create_disabled(2);

	//update pipelines

	RD::PipelineDepthStencilState depth_stencil_state;

	if (depth_test != DEPTH_TEST_DISABLED) {
		depth_stencil_state.enable_depth_test = true;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_GREATER_OR_EQUAL;
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
				if (!static_cast<SceneShaderForwardMobile *>(singleton)->shader.is_variant_enabled(k)) {
					continue;
				}
				RD::PipelineRasterizationState raster_state;
				raster_state.cull_mode = cull_mode_rd;
				raster_state.wireframe = wireframe;

				RD::PipelineColorBlendState blend_state;
				RD::PipelineDepthStencilState depth_stencil = depth_stencil_state;
				RD::PipelineMultisampleState multisample_state;

				if (uses_alpha || uses_blend_alpha) {
					// only allow these flags to go through if we have some form of msaa
					if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE) {
						multisample_state.enable_alpha_to_coverage = true;
					} else if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE) {
						multisample_state.enable_alpha_to_coverage = true;
						multisample_state.enable_alpha_to_one = true;
					}

					if (k == SHADER_VERSION_COLOR_PASS || k == SHADER_VERSION_COLOR_PASS_MULTIVIEW || k == SHADER_VERSION_LIGHTMAP_COLOR_PASS || k == SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW) {
						blend_state = blend_state_blend;
						if (depth_draw == DEPTH_DRAW_OPAQUE && !uses_alpha_clip) {
							depth_stencil.enable_depth_write = false; //alpha does not draw depth
						}
					} else if (k == SHADER_VERSION_SHADOW_PASS || k == SHADER_VERSION_SHADOW_PASS_MULTIVIEW || k == SHADER_VERSION_SHADOW_PASS_DP) {
						//none, blend state contains nothing
					} else if (k == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL) {
						blend_state = RD::PipelineColorBlendState::create_disabled(5); //writes to normal and roughness in opaque way
					} else {
						pipelines[i][j][k].clear();
						continue; // do not use this version (will error if using it is attempted)
					}
				} else {
					if (k == SHADER_VERSION_COLOR_PASS || k == SHADER_VERSION_COLOR_PASS_MULTIVIEW || k == SHADER_VERSION_LIGHTMAP_COLOR_PASS || k == SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW) {
						blend_state = blend_state_opaque;
					} else if (k == SHADER_VERSION_SHADOW_PASS || k == SHADER_VERSION_SHADOW_PASS_MULTIVIEW || k == SHADER_VERSION_SHADOW_PASS_DP) {
						//none, leave empty
					} else if (k == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL) {
						blend_state = RD::PipelineColorBlendState::create_disabled(5); //writes to normal and roughness in opaque way
					} else {
						// ???
					}
				}

				RID shader_variant = shader_singleton->shader.version_get_shader(version, k);
				pipelines[i][j][k].setup(shader_variant, primitive_rd, raster_state, multisample_state, depth_stencil, blend_state, 0, singleton->default_specialization_constants);
			}
		}
	}

	valid = true;
}

bool SceneShaderForwardMobile::ShaderData::is_animated() const {
	return (uses_fragment_time && uses_discard) || (uses_vertex_time && uses_vertex);
}

bool SceneShaderForwardMobile::ShaderData::casts_shadows() const {
	bool has_read_screen_alpha = uses_screen_texture || uses_depth_texture || uses_normal_texture;
	bool has_base_alpha = (uses_alpha && (!uses_alpha_clip || uses_alpha_antialiasing)) || has_read_screen_alpha;
	bool has_alpha = has_base_alpha || uses_blend_alpha;

	return !has_alpha || (uses_depth_prepass_alpha && !(depth_draw == DEPTH_DRAW_DISABLED || depth_test == DEPTH_TEST_DISABLED));
}

RS::ShaderNativeSourceCode SceneShaderForwardMobile::ShaderData::get_native_source_code() const {
	SceneShaderForwardMobile *shader_singleton = (SceneShaderForwardMobile *)SceneShaderForwardMobile::singleton;

	return shader_singleton->shader.version_get_native_source_code(version);
}

SceneShaderForwardMobile::ShaderData::ShaderData() :
		shader_list_element(this) {
}

SceneShaderForwardMobile::ShaderData::~ShaderData() {
	SceneShaderForwardMobile *shader_singleton = (SceneShaderForwardMobile *)SceneShaderForwardMobile::singleton;
	ERR_FAIL_NULL(shader_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		shader_singleton->shader.version_free(version);
	}
}

RendererRD::MaterialStorage::ShaderData *SceneShaderForwardMobile::_create_shader_func() {
	ShaderData *shader_data = memnew(ShaderData);
	singleton->shader_list.add(&shader_data->shader_list_element);
	return shader_data;
}

void SceneShaderForwardMobile::MaterialData::set_render_priority(int p_priority) {
	priority = p_priority - RS::MATERIAL_RENDER_PRIORITY_MIN; //8 bits
}

void SceneShaderForwardMobile::MaterialData::set_next_pass(RID p_pass) {
	next_pass = p_pass;
}

bool SceneShaderForwardMobile::MaterialData::update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	SceneShaderForwardMobile *shader_singleton = (SceneShaderForwardMobile *)SceneShaderForwardMobile::singleton;

	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, shader_singleton->shader.version_get_shader(shader_data->version, 0), RenderForwardMobile::MATERIAL_UNIFORM_SET, true, true);
}

SceneShaderForwardMobile::MaterialData::~MaterialData() {
	free_parameters_uniform_set(uniform_set);
}

RendererRD::MaterialStorage::MaterialData *SceneShaderForwardMobile::_create_material_func(ShaderData *p_shader) {
	MaterialData *material_data = memnew(MaterialData);
	material_data->shader_data = p_shader;
	//update will happen later anyway so do nothing.
	return material_data;
}

/* Scene Shader */

SceneShaderForwardMobile *SceneShaderForwardMobile::singleton = nullptr;

SceneShaderForwardMobile::SceneShaderForwardMobile() {
	// there should be only one of these, contained within our RenderForwardMobile singleton.
	singleton = this;
}

void SceneShaderForwardMobile::init(const String p_defines) {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	/* SCENE SHADER */

	{
		Vector<String> shader_versions;
		shader_versions.push_back(""); // SHADER_VERSION_COLOR_PASS
		shader_versions.push_back("\n#define USE_LIGHTMAP\n"); // SHADER_VERSION_LIGHTMAP_COLOR_PASS
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n"); // SHADER_VERSION_SHADOW_PASS, should probably change this to MODE_RENDER_SHADOW because we don't have a depth pass here...
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n#define MODE_DUAL_PARABOLOID\n"); // SHADER_VERSION_SHADOW_PASS_DP
		shader_versions.push_back("\n#define MODE_RENDER_DEPTH\n#define MODE_RENDER_MATERIAL\n"); // SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL

		// multiview versions of our shaders
		shader_versions.push_back("\n#define USE_MULTIVIEW\n"); // SHADER_VERSION_COLOR_PASS_MULTIVIEW
		shader_versions.push_back("\n#define USE_MULTIVIEW\n#define USE_LIGHTMAP\n"); // SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW
		shader_versions.push_back("\n#define USE_MULTIVIEW\n#define MODE_RENDER_DEPTH\n"); // SHADER_VERSION_SHADOW_PASS_MULTIVIEW

		shader.initialize(shader_versions, p_defines);

		if (!RendererCompositorRD::get_singleton()->is_xr_enabled()) {
			shader.set_variant_enabled(SHADER_VERSION_COLOR_PASS_MULTIVIEW, false);
			shader.set_variant_enabled(SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW, false);
			shader.set_variant_enabled(SHADER_VERSION_SHADOW_PASS_MULTIVIEW, false);
		}
	}

	material_storage->shader_set_data_request_function(RendererRD::MaterialStorage::SHADER_TYPE_3D, _create_shader_funcs);
	material_storage->material_set_data_request_function(RendererRD::MaterialStorage::SHADER_TYPE_3D, _create_material_funcs);

	{
		//shader compiler
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["MODEL_MATRIX"] = "read_model_matrix";
		actions.renames["MODEL_NORMAL_MATRIX"] = "model_normal_matrix";
		actions.renames["VIEW_MATRIX"] = "read_view_matrix";
		actions.renames["INV_VIEW_MATRIX"] = "inv_view_matrix";
		actions.renames["PROJECTION_MATRIX"] = "projection_matrix";
		actions.renames["INV_PROJECTION_MATRIX"] = "inv_projection_matrix";
		actions.renames["MODELVIEW_MATRIX"] = "modelview";
		actions.renames["MODELVIEW_NORMAL_MATRIX"] = "modelview_normal";
		actions.renames["MAIN_CAM_INV_VIEW_MATRIX"] = "scene_data.main_cam_inv_view_matrix";

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
		actions.renames["VERTEX_ID"] = "gl_VertexIndex";

		actions.renames["ALPHA_SCISSOR_THRESHOLD"] = "alpha_scissor_threshold";
		actions.renames["ALPHA_HASH_SCALE"] = "alpha_hash_scale";
		actions.renames["ALPHA_ANTIALIASING_EDGE"] = "alpha_antialiasing_edge";
		actions.renames["ALPHA_TEXTURE_COORDINATE"] = "alpha_texture_coordinate";

		//builtins

		actions.renames["TIME"] = "scene_data_block.data.time";
		actions.renames["EXPOSURE"] = "(1.0 / scene_data_block.data.emissive_exposure_normalization)";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["VIEWPORT_SIZE"] = "read_viewport_size";

		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["FRONT_FACING"] = "gl_FrontFacing";
		actions.renames["NORMAL_MAP"] = "normal_map";
		actions.renames["NORMAL_MAP_DEPTH"] = "normal_map_depth";
		actions.renames["ALBEDO"] = "albedo";
		actions.renames["ALPHA"] = "alpha";
		actions.renames["PREMUL_ALPHA_FACTOR"] = "premul_alpha";
		actions.renames["METALLIC"] = "metallic";
		actions.renames["SPECULAR"] = "specular";
		actions.renames["ROUGHNESS"] = "roughness";
		actions.renames["RIM"] = "rim";
		actions.renames["RIM_TINT"] = "rim_tint";
		actions.renames["CLEARCOAT"] = "clearcoat";
		actions.renames["CLEARCOAT_ROUGHNESS"] = "clearcoat_roughness";
		actions.renames["ANISOTROPY"] = "anisotropy";
		actions.renames["ANISOTROPY_FLOW"] = "anisotropy_flow";
		actions.renames["SSS_STRENGTH"] = "sss_strength";
		actions.renames["SSS_TRANSMITTANCE_COLOR"] = "transmittance_color";
		actions.renames["SSS_TRANSMITTANCE_DEPTH"] = "transmittance_depth";
		actions.renames["SSS_TRANSMITTANCE_BOOST"] = "transmittance_boost";
		actions.renames["BACKLIGHT"] = "backlight";
		actions.renames["AO"] = "ao";
		actions.renames["AO_LIGHT_AFFECT"] = "ao_light_affect";
		actions.renames["EMISSION"] = "emission";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";
		actions.renames["SCREEN_UV"] = "screen_uv";
		actions.renames["DEPTH"] = "gl_FragDepth";
		actions.renames["FOG"] = "fog";
		actions.renames["RADIANCE"] = "custom_radiance";
		actions.renames["IRRADIANCE"] = "custom_irradiance";
		actions.renames["BONE_INDICES"] = "bone_attrib";
		actions.renames["BONE_WEIGHTS"] = "weight_attrib";
		actions.renames["CUSTOM0"] = "custom0_attrib";
		actions.renames["CUSTOM1"] = "custom1_attrib";
		actions.renames["CUSTOM2"] = "custom2_attrib";
		actions.renames["CUSTOM3"] = "custom3_attrib";
		actions.renames["OUTPUT_IS_SRGB"] = "SHADER_IS_SRGB";
		actions.renames["CLIP_SPACE_FAR"] = "SHADER_SPACE_FAR";
		actions.renames["LIGHT_VERTEX"] = "light_vertex";

		actions.renames["NODE_POSITION_WORLD"] = "read_model_matrix[3].xyz";
		actions.renames["CAMERA_POSITION_WORLD"] = "scene_data.inv_view_matrix[3].xyz";
		actions.renames["CAMERA_DIRECTION_WORLD"] = "scene_data.inv_view_matrix[2].xyz";
		actions.renames["CAMERA_VISIBLE_LAYERS"] = "scene_data.camera_visible_layers";
		actions.renames["NODE_POSITION_VIEW"] = "(scene_data.view_matrix * read_model_matrix)[3].xyz";

		actions.renames["VIEW_INDEX"] = "ViewIndex";
		actions.renames["VIEW_MONO_LEFT"] = "0";
		actions.renames["VIEW_RIGHT"] = "1";
		actions.renames["EYE_OFFSET"] = "eye_offset";

		//for light
		actions.renames["VIEW"] = "view";
		actions.renames["SPECULAR_AMOUNT"] = "specular_amount";
		actions.renames["LIGHT_COLOR"] = "light_color";
		actions.renames["LIGHT_IS_DIRECTIONAL"] = "is_directional";
		actions.renames["LIGHT"] = "light";
		actions.renames["ATTENUATION"] = "attenuation";
		actions.renames["DIFFUSE_LIGHT"] = "diffuse_light";
		actions.renames["SPECULAR_LIGHT"] = "specular_light";

		actions.usage_defines["NORMAL"] = "#define NORMAL_USED\n";
		actions.usage_defines["TANGENT"] = "#define TANGENT_USED\n";
		actions.usage_defines["BINORMAL"] = "@TANGENT";
		actions.usage_defines["RIM"] = "#define LIGHT_RIM_USED\n";
		actions.usage_defines["RIM_TINT"] = "@RIM";
		actions.usage_defines["CLEARCOAT"] = "#define LIGHT_CLEARCOAT_USED\n";
		actions.usage_defines["CLEARCOAT_ROUGHNESS"] = "@CLEARCOAT";
		actions.usage_defines["ANISOTROPY"] = "#define LIGHT_ANISOTROPY_USED\n";
		actions.usage_defines["ANISOTROPY_FLOW"] = "@ANISOTROPY";
		actions.usage_defines["AO"] = "#define AO_USED\n";
		actions.usage_defines["AO_LIGHT_AFFECT"] = "#define AO_USED\n";
		actions.usage_defines["UV"] = "#define UV_USED\n";
		actions.usage_defines["UV2"] = "#define UV2_USED\n";
		actions.usage_defines["BONE_INDICES"] = "#define BONES_USED\n";
		actions.usage_defines["BONE_WEIGHTS"] = "#define WEIGHTS_USED\n";
		actions.usage_defines["CUSTOM0"] = "#define CUSTOM0_USED\n";
		actions.usage_defines["CUSTOM1"] = "#define CUSTOM1_USED\n";
		actions.usage_defines["CUSTOM2"] = "#define CUSTOM2_USED\n";
		actions.usage_defines["CUSTOM3"] = "#define CUSTOM3_USED\n";
		actions.usage_defines["NORMAL_MAP"] = "#define NORMAL_MAP_USED\n";
		actions.usage_defines["NORMAL_MAP_DEPTH"] = "@NORMAL_MAP";
		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["INSTANCE_CUSTOM"] = "#define ENABLE_INSTANCE_CUSTOM\n";
		actions.usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";
		actions.usage_defines["LIGHT_VERTEX"] = "#define LIGHT_VERTEX_USED\n";

		actions.usage_defines["ALPHA_SCISSOR_THRESHOLD"] = "#define ALPHA_SCISSOR_USED\n";
		actions.usage_defines["ALPHA_HASH_SCALE"] = "#define ALPHA_HASH_USED\n";
		actions.usage_defines["ALPHA_ANTIALIASING_EDGE"] = "#define ALPHA_ANTIALIASING_EDGE_USED\n";
		actions.usage_defines["ALPHA_TEXTURE_COORDINATE"] = "@ALPHA_ANTIALIASING_EDGE";
		actions.usage_defines["PREMUL_ALPHA_FACTOR"] = "#define PREMUL_ALPHA_USED";

		actions.usage_defines["SSS_STRENGTH"] = "#define ENABLE_SSS\n";
		actions.usage_defines["SSS_TRANSMITTANCE_DEPTH"] = "#define ENABLE_TRANSMITTANCE\n";
		actions.usage_defines["BACKLIGHT"] = "#define LIGHT_BACKLIGHT_USED\n";
		actions.usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";

		actions.usage_defines["FOG"] = "#define CUSTOM_FOG_USED\n";
		actions.usage_defines["RADIANCE"] = "#define CUSTOM_RADIANCE_USED\n";
		actions.usage_defines["IRRADIANCE"] = "#define CUSTOM_IRRADIANCE_USED\n";

		actions.usage_defines["MODEL_MATRIX"] = "#define MODEL_MATRIX_USED\n";

		actions.render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
		actions.render_mode_defines["world_vertex_coords"] = "#define VERTEX_WORLD_COORDS_USED\n";
		actions.render_mode_defines["ensure_correct_normals"] = "#define ENSURE_CORRECT_NORMALS\n";
		actions.render_mode_defines["cull_front"] = "#define DO_SIDE_CHECK\n";
		actions.render_mode_defines["cull_disabled"] = "#define DO_SIDE_CHECK\n";
		actions.render_mode_defines["particle_trails"] = "#define USE_PARTICLE_TRAILS\n";
		actions.render_mode_defines["depth_prepass_alpha"] = "#define USE_OPAQUE_PREPASS\n";

		bool force_lambert = GLOBAL_GET("rendering/shading/overrides/force_lambert_over_burley");
		if (!force_lambert) {
			actions.render_mode_defines["diffuse_burley"] = "#define DIFFUSE_BURLEY\n";
		}

		actions.render_mode_defines["diffuse_lambert_wrap"] = "#define DIFFUSE_LAMBERT_WRAP\n";
		actions.render_mode_defines["diffuse_toon"] = "#define DIFFUSE_TOON\n";

		actions.render_mode_defines["sss_mode_skin"] = "#define SSS_MODE_SKIN\n";

		actions.render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_SCHLICK_GGX\n";

		actions.render_mode_defines["specular_toon"] = "#define SPECULAR_TOON\n";
		actions.render_mode_defines["specular_disabled"] = "#define SPECULAR_DISABLED\n";
		actions.render_mode_defines["shadows_disabled"] = "#define SHADOWS_DISABLED\n";
		actions.render_mode_defines["ambient_light_disabled"] = "#define AMBIENT_LIGHT_DISABLED\n";
		actions.render_mode_defines["shadow_to_opacity"] = "#define USE_SHADOW_TO_OPACITY\n";
		actions.render_mode_defines["unshaded"] = "#define MODE_UNSHADED\n";
		actions.render_mode_defines["debug_shadow_splits"] = "#define DEBUG_DRAW_PSSM_SPLITS\n";
		actions.render_mode_defines["fog_disabled"] = "#define FOG_DISABLED\n";

		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = RenderForwardMobile::MATERIAL_UNIFORM_SET;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_shader_uniforms.data";
		actions.instance_uniform_index_variable = "instances.data[draw_call.instance_index].instance_uniforms_ofs";

		actions.apply_luminance_multiplier = true; // apply luminance multiplier to screen texture
		actions.check_multiview_samplers = RendererCompositorRD::get_singleton()->is_xr_enabled(); // Make sure we check sampling multiview textures.

		compiler.initialize(actions);
	}

	{
		//default material and shader
		default_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(default_shader);
		material_storage->shader_set_code(default_shader, R"(
// Default 3D material shader (Mobile).

shader_type spatial;

void vertex() {
	ROUGHNESS = 0.8;
}

void fragment() {
	ALBEDO = vec3(0.6);
	ROUGHNESS = 0.8;
	METALLIC = 0.2;
}
)");
		default_material = material_storage->material_allocate();
		material_storage->material_initialize(default_material);
		material_storage->material_set_shader(default_material, default_shader);

		MaterialData *md = static_cast<MaterialData *>(material_storage->material_get_data(default_material, RendererRD::MaterialStorage::SHADER_TYPE_3D));
		default_shader_rd = shader.version_get_shader(md->shader_data->version, SHADER_VERSION_COLOR_PASS);

		default_material_shader_ptr = md->shader_data;
		default_material_uniform_set = md->uniform_set;
	}

	{
		overdraw_material_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(overdraw_material_shader);
		// Use relatively low opacity so that more "layers" of overlapping objects can be distinguished.
		material_storage->shader_set_code(overdraw_material_shader, R"(
// 3D editor Overdraw debug draw mode shader (Mobile).

shader_type spatial;

render_mode blend_add, unshaded, fog_disabled;

void fragment() {
	ALBEDO = vec3(0.4, 0.8, 0.8);
	ALPHA = 0.1;
}
)");
		overdraw_material = material_storage->material_allocate();
		material_storage->material_initialize(overdraw_material);
		material_storage->material_set_shader(overdraw_material, overdraw_material_shader);

		MaterialData *md = static_cast<MaterialData *>(material_storage->material_get_data(overdraw_material, RendererRD::MaterialStorage::SHADER_TYPE_3D));
		overdraw_material_shader_ptr = md->shader_data;
		overdraw_material_uniform_set = md->uniform_set;
	}

	{
		debug_shadow_splits_material_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(debug_shadow_splits_material_shader);
		// Use relatively low opacity so that more "layers" of overlapping objects can be distinguished.
		material_storage->shader_set_code(debug_shadow_splits_material_shader, R"(
// 3D debug shadow splits mode shader (Mobile).

shader_type spatial;

render_mode debug_shadow_splits, fog_disabled;

void fragment() {
	ALBEDO = vec3(1.0, 1.0, 1.0);
}
)");
		debug_shadow_splits_material = material_storage->material_allocate();
		material_storage->material_initialize(debug_shadow_splits_material);
		material_storage->material_set_shader(debug_shadow_splits_material, debug_shadow_splits_material_shader);

		MaterialData *md = static_cast<MaterialData *>(material_storage->material_get_data(debug_shadow_splits_material, RendererRD::MaterialStorage::SHADER_TYPE_3D));
		debug_shadow_splits_material_shader_ptr = md->shader_data;
		debug_shadow_splits_material_uniform_set = md->uniform_set;
	}

	{
		default_vec4_xform_buffer = RD::get_singleton()->storage_buffer_create(256);
		Vector<RD::Uniform> uniforms;
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.append_id(default_vec4_xform_buffer);
		u.binding = 0;
		uniforms.push_back(u);

		default_vec4_xform_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, default_shader_rd, RenderForwardMobile::TRANSFORMS_UNIFORM_SET);
	}
	{
		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.enable_compare = true;
		sampler.compare_op = RD::COMPARE_OP_GREATER;
		shadow_sampler = RD::get_singleton()->sampler_create(sampler);
	}
}

void SceneShaderForwardMobile::set_default_specialization_constants(const Vector<RD::PipelineSpecializationConstant> &p_constants) {
	default_specialization_constants = p_constants;
	for (SelfList<ShaderData> *E = shader_list.first(); E; E = E->next()) {
		for (int i = 0; i < ShaderData::CULL_VARIANT_MAX; i++) {
			for (int j = 0; j < RS::PRIMITIVE_MAX; j++) {
				for (int k = 0; k < SHADER_VERSION_MAX; k++) {
					E->self()->pipelines[i][j][k].update_specialization_constants(default_specialization_constants);
				}
			}
		}
	}
}

SceneShaderForwardMobile::~SceneShaderForwardMobile() {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	RD::get_singleton()->free(default_vec4_xform_buffer);
	RD::get_singleton()->free(shadow_sampler);

	material_storage->shader_free(overdraw_material_shader);
	material_storage->shader_free(default_shader);
	material_storage->shader_free(debug_shadow_splits_material_shader);

	material_storage->material_free(overdraw_material);
	material_storage->material_free(default_material);
	material_storage->material_free(debug_shadow_splits_material);
}

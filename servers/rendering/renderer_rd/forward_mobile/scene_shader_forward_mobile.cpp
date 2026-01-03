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
	ubo_size = 0;
	uniforms.clear();
	_clear_vertex_input_mask_cache();

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;

	blend_mode = BLEND_MODE_MIX;
	depth_test_disabledi = 0;
	depth_test_invertedi = 0;
	alpha_antialiasing_mode = ALPHA_ANTIALIASING_OFF;
	cull_mode = RS::CULL_MODE_BACK;

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
	uses_normal_map = false;
	uses_bent_normal_map = false;
	wireframe = false;

	unshaded = false;
	uses_vertex = false;
	uses_sss = false;
	uses_transmittance = false;
	uses_time = false;
	writes_modelview_or_projection = false;
	uses_world_coordinates = false;
	uses_particle_trails = false;

	int depth_drawi = DEPTH_DRAW_OPAQUE;

	int stencil_readi = 0;
	int stencil_writei = 0;
	int stencil_write_depth_faili = 0;
	int stencil_comparei = STENCIL_COMPARE_ALWAYS;
	int stencil_referencei = -1;

	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);
	actions.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&blend_mode, BLEND_MODE_PREMULTIPLIED_ALPHA);
	actions.render_mode_values["blend_min"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MINIMUM);
	actions.render_mode_values["blend_max"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MAXIMUM);

	actions.render_mode_values["alpha_to_coverage"] = Pair<int *, int>(&alpha_antialiasing_mode, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE);
	actions.render_mode_values["alpha_to_coverage_and_one"] = Pair<int *, int>(&alpha_antialiasing_mode, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE);

	actions.render_mode_values["depth_draw_never"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_DISABLED);
	actions.render_mode_values["depth_draw_opaque"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_OPAQUE);
	actions.render_mode_values["depth_draw_always"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_ALWAYS);

	actions.render_mode_values["depth_test_disabled"] = Pair<int *, int>(&depth_test_disabledi, 1);
	actions.render_mode_values["depth_test_inverted"] = Pair<int *, int>(&depth_test_invertedi, 1);

	actions.render_mode_values["cull_disabled"] = Pair<int *, int>(&cull_mode, RS::CULL_MODE_DISABLED);
	actions.render_mode_values["cull_front"] = Pair<int *, int>(&cull_mode, RS::CULL_MODE_FRONT);
	actions.render_mode_values["cull_back"] = Pair<int *, int>(&cull_mode, RS::CULL_MODE_BACK);

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
	actions.usage_flag_pointers["BENT_NORMAL_MAP"] = &uses_bent_normal_map;

	actions.usage_flag_pointers["TANGENT"] = &uses_tangent;
	actions.usage_flag_pointers["BINORMAL"] = &uses_tangent;
	actions.usage_flag_pointers["ANISOTROPY"] = &uses_tangent;
	actions.usage_flag_pointers["ANISOTROPY_FLOW"] = &uses_tangent;

	actions.usage_flag_pointers["POINT_SIZE"] = &uses_point_size;
	actions.usage_flag_pointers["POINT_COORD"] = &uses_point_size;

	actions.write_flag_pointers["MODELVIEW_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["PROJECTION_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["VERTEX"] = &uses_vertex;

	actions.stencil_mode_values["read"] = Pair<int *, int>(&stencil_readi, STENCIL_FLAG_READ);
	actions.stencil_mode_values["write"] = Pair<int *, int>(&stencil_writei, STENCIL_FLAG_WRITE);
	actions.stencil_mode_values["write_depth_fail"] = Pair<int *, int>(&stencil_write_depth_faili, STENCIL_FLAG_WRITE_DEPTH_FAIL);

	actions.stencil_mode_values["compare_less"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_LESS);
	actions.stencil_mode_values["compare_equal"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_EQUAL);
	actions.stencil_mode_values["compare_less_or_equal"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_LESS_OR_EQUAL);
	actions.stencil_mode_values["compare_greater"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_GREATER);
	actions.stencil_mode_values["compare_not_equal"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_NOT_EQUAL);
	actions.stencil_mode_values["compare_greater_or_equal"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_GREATER_OR_EQUAL);
	actions.stencil_mode_values["compare_always"] = Pair<int *, int>(&stencil_comparei, STENCIL_COMPARE_ALWAYS);

	actions.stencil_reference = &stencil_referencei;

	actions.uniforms = &uniforms;

	MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
	Error err = SceneShaderForwardMobile::singleton->compiler.compile(RS::SHADER_SPATIAL, code, &actions, path, gen_code);

	if (err != OK) {
		if (version.is_valid()) {
			SceneShaderForwardMobile::singleton->shader.version_free(version);
			version = RID();
		}
		ERR_FAIL_MSG("Shader compilation failed.");
	}

	if (version.is_null()) {
		version = SceneShaderForwardMobile::singleton->shader.version_create(false);
	}

	depth_draw = DepthDraw(depth_drawi);
	if (depth_test_disabledi) {
		depth_test = DEPTH_TEST_DISABLED;
	} else if (depth_test_invertedi) {
		depth_test = DEPTH_TEST_ENABLED_INVERTED;
	} else {
		depth_test = DEPTH_TEST_ENABLED;
	}
	uses_vertex_time = gen_code.uses_vertex_time;
	uses_fragment_time = gen_code.uses_fragment_time;
	uses_screen_texture_mipmaps = gen_code.uses_screen_texture_mipmaps;
	uses_screen_texture = gen_code.uses_screen_texture;
	uses_depth_texture = gen_code.uses_depth_texture;
	uses_normal_texture = gen_code.uses_normal_roughness_texture;
	uses_normal |= uses_normal_map;
	uses_normal |= uses_bent_normal_map;
	uses_tangent |= uses_normal_map;
	uses_tangent |= uses_bent_normal_map;

	stencil_enabled = stencil_referencei != -1;
	stencil_flags = stencil_readi | stencil_writei | stencil_write_depth_faili;
	stencil_compare = StencilCompare(stencil_comparei);
	stencil_reference = stencil_referencei;

#ifdef DEBUG_ENABLED
	if (uses_sss) {
		WARN_PRINT_ONCE_ED("Subsurface scattering is only available when using the Forward+ renderer.");
	}

	if (uses_transmittance) {
		WARN_PRINT_ONCE_ED("Transmittance is only available when using the Forward+ renderer.");
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

	SceneShaderForwardMobile::singleton->shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines);

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	pipeline_hash_map.clear_pipelines();

	// If any form of Alpha Antialiasing is enabled, set the blend mode to alpha to coverage.
	if (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF) {
		blend_mode = BLEND_MODE_ALPHA_TO_COVERAGE;
	}

	uses_blend_alpha = blend_mode_uses_blend_alpha(BlendMode(blend_mode));
}

bool SceneShaderForwardMobile::ShaderData::is_animated() const {
	return (uses_fragment_time && uses_discard) || (uses_vertex_time && uses_vertex);
}

bool SceneShaderForwardMobile::ShaderData::casts_shadows() const {
	bool has_read_screen_alpha = uses_screen_texture || uses_depth_texture || uses_normal_texture;
	bool has_base_alpha = (uses_alpha && (!uses_alpha_clip || uses_alpha_antialiasing)) || has_read_screen_alpha;
	bool has_alpha = has_base_alpha || uses_blend_alpha;

	return !has_alpha || (uses_depth_prepass_alpha && !(depth_draw == DEPTH_DRAW_DISABLED || depth_test != DEPTH_TEST_ENABLED));
}

RS::ShaderNativeSourceCode SceneShaderForwardMobile::ShaderData::get_native_source_code() const {
	if (version.is_valid()) {
		MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
		return SceneShaderForwardMobile::singleton->shader.version_get_native_source_code(version);
	} else {
		return RS::ShaderNativeSourceCode();
	}
}

Pair<ShaderRD *, RID> SceneShaderForwardMobile::ShaderData::get_native_shader_and_version() const {
	if (version.is_valid()) {
		MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
		return { &SceneShaderForwardMobile::singleton->shader, version };
	} else {
		return {};
	}
}

void SceneShaderForwardMobile::ShaderData::_create_pipeline(PipelineKey p_pipeline_key) {
#if PRINT_PIPELINE_COMPILATION_KEYS
	print_line(
			"HASH:", p_pipeline_key.hash(),
			"VERSION:", version,
			"VERTEX:", p_pipeline_key.vertex_format_id,
			"FRAMEBUFFER:", p_pipeline_key.framebuffer_format_id,
			"CULL:", p_pipeline_key.cull_mode,
			"PRIMITIVE:", p_pipeline_key.primitive_type,
			"VERSION:", p_pipeline_key.version,
			"SPEC PACKED #0:", p_pipeline_key.shader_specialization.packed_0,
			"SPEC PACKED #1:", p_pipeline_key.shader_specialization.packed_1,
			"SPEC PACKED #2:", p_pipeline_key.shader_specialization.packed_2,
			"RENDER PASS:", p_pipeline_key.render_pass,
			"WIREFRAME:", p_pipeline_key.wireframe);
#endif

	RD::PipelineColorBlendState::Attachment blend_attachment = blend_mode_to_blend_attachment(BlendMode(blend_mode));
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
		depth_stencil_state.enable_depth_write = depth_draw != DEPTH_DRAW_DISABLED ? true : false;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_GREATER_OR_EQUAL;

		if (depth_test == DEPTH_TEST_ENABLED_INVERTED) {
			depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_LESS;
		}
	}

	RD::RenderPrimitive primitive_rd_table[RS::PRIMITIVE_MAX] = {
		RD::RENDER_PRIMITIVE_POINTS,
		RD::RENDER_PRIMITIVE_LINES,
		RD::RENDER_PRIMITIVE_LINESTRIPS,
		RD::RENDER_PRIMITIVE_TRIANGLES,
		RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS,
	};

	depth_stencil_state.enable_stencil = stencil_enabled;
	if (stencil_enabled) {
		static const RD::CompareOperator stencil_compare_rd_table[STENCIL_COMPARE_MAX] = {
			RD::COMPARE_OP_LESS,
			RD::COMPARE_OP_EQUAL,
			RD::COMPARE_OP_LESS_OR_EQUAL,
			RD::COMPARE_OP_GREATER,
			RD::COMPARE_OP_NOT_EQUAL,
			RD::COMPARE_OP_GREATER_OR_EQUAL,
			RD::COMPARE_OP_ALWAYS,
		};

		uint32_t stencil_mask = 255;

		RD::PipelineDepthStencilState::StencilOperationState op;
		op.fail = RD::STENCIL_OP_KEEP;
		op.pass = RD::STENCIL_OP_KEEP;
		op.depth_fail = RD::STENCIL_OP_KEEP;
		op.compare = stencil_compare_rd_table[stencil_compare];
		op.compare_mask = 0;
		op.write_mask = 0;
		op.reference = stencil_reference;

		if (stencil_flags & STENCIL_FLAG_READ) {
			op.compare_mask = stencil_mask;
		}

		if (stencil_flags & STENCIL_FLAG_WRITE) {
			op.pass = RD::STENCIL_OP_REPLACE;
			op.write_mask = stencil_mask;
		}

		if (stencil_flags & STENCIL_FLAG_WRITE_DEPTH_FAIL) {
			op.depth_fail = RD::STENCIL_OP_REPLACE;
			op.write_mask = stencil_mask;
		}

		depth_stencil_state.front_op = op;
		depth_stencil_state.back_op = op;
	}

	bool emulate_point_size_flag = uses_point_size && SceneShaderForwardMobile::singleton->emulate_point_size;

	RD::RenderPrimitive primitive_rd;
	if (uses_point_size) {
		primitive_rd = emulate_point_size_flag ? RD::RENDER_PRIMITIVE_TRIANGLES : RD::RENDER_PRIMITIVE_POINTS;
	} else {
		primitive_rd = primitive_rd_table[p_pipeline_key.primitive_type];
	}

	RD::PipelineRasterizationState raster_state;
	raster_state.cull_mode = p_pipeline_key.cull_mode;
	raster_state.wireframe = wireframe || p_pipeline_key.wireframe;

	RD::PipelineMultisampleState multisample_state;
	multisample_state.sample_count = RD::get_singleton()->framebuffer_format_get_texture_samples(p_pipeline_key.framebuffer_format_id, 0);

	RD::PipelineColorBlendState blend_state;
	if (uses_alpha || uses_blend_alpha) {
		// These flags should only go through if we have some form of MSAA.
		if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE) {
			multisample_state.enable_alpha_to_coverage = true;
		} else if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE) {
			multisample_state.enable_alpha_to_coverage = true;
			multisample_state.enable_alpha_to_one = true;
		}

		if (p_pipeline_key.version == SHADER_VERSION_COLOR_PASS || p_pipeline_key.version == SHADER_VERSION_COLOR_PASS_MULTIVIEW || p_pipeline_key.version == SHADER_VERSION_LIGHTMAP_COLOR_PASS || p_pipeline_key.version == SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW || p_pipeline_key.version == SHADER_VERSION_MOTION_VECTORS_MULTIVIEW) {
			blend_state = blend_state_blend;
			if (depth_draw == DEPTH_DRAW_OPAQUE && !uses_alpha_clip) {
				// Alpha does not write to depth.
				depth_stencil_state.enable_depth_write = false;
			}
		} else if (p_pipeline_key.version == SHADER_VERSION_SHADOW_PASS || p_pipeline_key.version == SHADER_VERSION_SHADOW_PASS_MULTIVIEW || p_pipeline_key.version == SHADER_VERSION_SHADOW_PASS_DP) {
			// Contains nothing.
		} else if (p_pipeline_key.version == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL) {
			// Writes to normal and roughness in opaque way.
			blend_state = RD::PipelineColorBlendState::create_disabled(5);
		} else {
			// Do not use this version (error case).
		}
	} else {
		if (p_pipeline_key.version == SHADER_VERSION_COLOR_PASS || p_pipeline_key.version == SHADER_VERSION_COLOR_PASS_MULTIVIEW || p_pipeline_key.version == SHADER_VERSION_LIGHTMAP_COLOR_PASS || p_pipeline_key.version == SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW || p_pipeline_key.version == SHADER_VERSION_MOTION_VECTORS_MULTIVIEW) {
			blend_state = blend_state_opaque;
		} else if (p_pipeline_key.version == SHADER_VERSION_SHADOW_PASS || p_pipeline_key.version == SHADER_VERSION_SHADOW_PASS_MULTIVIEW || p_pipeline_key.version == SHADER_VERSION_SHADOW_PASS_DP) {
			// Contains nothing.
		} else if (p_pipeline_key.version == SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL) {
			// Writes to normal and roughness in opaque way.
			blend_state = RD::PipelineColorBlendState::create_disabled(5);
		} else {
			// Unknown pipeline version.
		}
	}

	// Convert the specialization from the key to pipeline specialization constants.
	Vector<RD::PipelineSpecializationConstant> specialization_constants;
	RD::PipelineSpecializationConstant sc;
	sc.constant_id = 0;
	sc.int_value = p_pipeline_key.shader_specialization.packed_0;
	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
	specialization_constants.push_back(sc);

	sc.constant_id = 1;
	sc.int_value = p_pipeline_key.shader_specialization.packed_1;
	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
	specialization_constants.push_back(sc);

	sc.constant_id = 2;
	sc.float_value = p_pipeline_key.shader_specialization.packed_2;
	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT;
	specialization_constants.push_back(sc);

	sc = {}; // Sanitize value bits. "bool_value" only assigns 8 bits and keeps the remaining bits intact.
	sc.constant_id = 3;
	sc.bool_value = emulate_point_size_flag;
	sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
	specialization_constants.push_back(sc);

	RID shader_rid = get_shader_variant(p_pipeline_key.version, p_pipeline_key.ubershader);
	ERR_FAIL_COND(shader_rid.is_null());

	RID pipeline = RD::get_singleton()->render_pipeline_create(shader_rid, p_pipeline_key.framebuffer_format_id, p_pipeline_key.vertex_format_id, primitive_rd, raster_state, multisample_state, depth_stencil_state, blend_state, 0, p_pipeline_key.render_pass, specialization_constants);
	ERR_FAIL_COND(pipeline.is_null());

	pipeline_hash_map.add_compiled_pipeline(p_pipeline_key.hash(), pipeline);
}

RD::PolygonCullMode SceneShaderForwardMobile::ShaderData::get_cull_mode_from_cull_variant(CullVariant p_cull_variant) {
	const RD::PolygonCullMode cull_mode_rd_table[CULL_VARIANT_MAX][3] = {
		{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_FRONT, RD::POLYGON_CULL_BACK },
		{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_BACK, RD::POLYGON_CULL_FRONT },
		{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_DISABLED }
	};

	return cull_mode_rd_table[p_cull_variant][cull_mode];
}

void SceneShaderForwardMobile::ShaderData::_clear_vertex_input_mask_cache() {
	for (uint32_t i = 0; i < VERTEX_INPUT_MASKS_SIZE; i++) {
		vertex_input_masks[i].store(0);
	}
}

RID SceneShaderForwardMobile::ShaderData::get_shader_variant(ShaderVersion p_shader_version, bool p_ubershader) const {
	if (version.is_valid()) {
		MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
		ERR_FAIL_NULL_V(SceneShaderForwardMobile::singleton, RID());
		return SceneShaderForwardMobile::singleton->shader.version_get_shader(version, p_shader_version + (SceneShaderForwardMobile::singleton->use_fp16 ? SHADER_VERSION_MAX * 2 : 0) + (p_ubershader ? SHADER_VERSION_MAX : 0));
	} else {
		return RID();
	}
}

uint64_t SceneShaderForwardMobile::ShaderData::get_vertex_input_mask(ShaderVersion p_shader_version, bool p_ubershader) {
	// Vertex input masks require knowledge of the shader. Since querying the shader can be expensive due to high contention and the necessary mutex, we cache the result instead.
	// It is intentional for the range of the input masks to be different than the versions available in the shaders as it'll only ever use the regular variants or the FP16 ones.
	uint32_t input_mask_index = p_shader_version + (p_ubershader ? SHADER_VERSION_MAX : 0);
	uint64_t input_mask = vertex_input_masks[input_mask_index].load(std::memory_order_relaxed);
	if (input_mask == 0) {
		RID shader_rid = get_shader_variant(p_shader_version, p_ubershader);
		ERR_FAIL_COND_V(shader_rid.is_null(), 0);

		input_mask = RD::get_singleton()->shader_get_vertex_input_attribute_mask(shader_rid);
		vertex_input_masks[input_mask_index].store(input_mask, std::memory_order_relaxed);
	}

	return input_mask;
}

bool SceneShaderForwardMobile::ShaderData::is_valid() const {
	if (version.is_valid()) {
		MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
		ERR_FAIL_NULL_V(SceneShaderForwardMobile::singleton, false);
		return SceneShaderForwardMobile::singleton->shader.version_is_valid(version);
	} else {
		return false;
	}
}

SceneShaderForwardMobile::ShaderData::ShaderData() :
		shader_list_element(this) {
	pipeline_hash_map.set_creation_object_and_function(this, &ShaderData::_create_pipeline);
	pipeline_hash_map.set_compilations(SceneShaderForwardMobile::singleton->pipeline_compilations, &SceneShaderForwardMobile::singleton_mutex);
}

SceneShaderForwardMobile::ShaderData::~ShaderData() {
	pipeline_hash_map.clear_pipelines();

	if (version.is_valid()) {
		MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
		ERR_FAIL_NULL(SceneShaderForwardMobile::singleton);
		SceneShaderForwardMobile::singleton->shader.version_free(version);
	}
}

RendererRD::MaterialStorage::ShaderData *SceneShaderForwardMobile::_create_shader_func() {
	MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
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
	if (shader_data->version.is_valid()) {
		MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
		RID base_shader = SceneShaderForwardMobile::singleton->shader.version_get_shader(shader_data->version, (SceneShaderForwardMobile::singleton->use_fp16 ? SHADER_VERSION_MAX * 2 : 0));
		return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, base_shader, RenderForwardMobile::MATERIAL_UNIFORM_SET, true, true);
	} else {
		return false;
	}
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
Mutex SceneShaderForwardMobile::singleton_mutex;

SceneShaderForwardMobile::SceneShaderForwardMobile() {
	// there should be only one of these, contained within our RenderForwardMobile singleton.
	singleton = this;
}

void SceneShaderForwardMobile::init(const String p_defines) {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	// Store whether the shader will prefer using the FP16 variant.
	use_fp16 = RD::get_singleton()->has_feature(RD::SUPPORTS_HALF_FLOAT);

	emulate_point_size = !RD::get_singleton()->has_feature(RD::SUPPORTS_POINT_SIZE);

	// Immutable samplers : create the shadow sampler to be passed when creating the pipeline.
	{
		RD::SamplerState sampler;
		sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
		sampler.enable_compare = true;
		sampler.compare_op = RD::COMPARE_OP_GREATER;
		shadow_sampler = RD::get_singleton()->sampler_create(sampler);
	}

	/* SCENE SHADER */

	{
		Vector<ShaderRD::VariantDefine> shader_versions;
		for (uint32_t fp16 = 0; fp16 < 2; fp16++) {
			for (uint32_t ubershader = 0; ubershader < 2; ubershader++) {
				String base_define = fp16 ? "\n#define EXPLICIT_FP16\n" : "";
				int shader_group = fp16 ? SHADER_GROUP_FP16 : SHADER_GROUP_FP32;
				int shader_group_multiview = fp16 ? SHADER_GROUP_FP16_MULTIVIEW : SHADER_GROUP_FP32_MULTIVIEW;
				base_define += ubershader ? "\n#define UBERSHADER\n" : "";

				bool default_enabled = (uint32_t(use_fp16) == fp16);
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group, base_define + "", default_enabled)); // SHADER_VERSION_COLOR_PASS
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group, base_define + "\n#define USE_LIGHTMAP\n", default_enabled)); // SHADER_VERSION_LIGHTMAP_COLOR_PASS
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group, base_define + "\n#define MODE_RENDER_DEPTH\n#define SHADOW_PASS\n", default_enabled)); // SHADER_VERSION_SHADOW_PASS, should probably change this to MODE_RENDER_SHADOW because we don't have a depth pass here...
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group, base_define + "\n#define MODE_RENDER_DEPTH\n#define MODE_DUAL_PARABOLOID\n#define SHADOW_PASS\n", default_enabled)); // SHADER_VERSION_SHADOW_PASS_DP
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group, base_define + "\n#define MODE_RENDER_DEPTH\n#define MODE_RENDER_MATERIAL\n", default_enabled)); // SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL

				// Multiview versions of our shaders.
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group_multiview, base_define + "\n#define USE_MULTIVIEW\n", false)); // SHADER_VERSION_COLOR_PASS_MULTIVIEW
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group_multiview, base_define + "\n#define USE_MULTIVIEW\n#define USE_LIGHTMAP\n", false)); // SHADER_VERSION_LIGHTMAP_COLOR_PASS_MULTIVIEW
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group_multiview, base_define + "\n#define USE_MULTIVIEW\n#define MODE_RENDER_DEPTH\n#define SHADOW_PASS\n", false)); // SHADER_VERSION_SHADOW_PASS_MULTIVIEW
				shader_versions.push_back(ShaderRD::VariantDefine(shader_group_multiview, base_define + "\n#define USE_MULTIVIEW\n#define MODE_RENDER_MOTION_VECTORS\n", false)); // SHADER_VERSION_MOTION_VECTORS_MULTIVIEW
			}
		}

		Vector<RD::PipelineImmutableSampler> immutable_samplers;
		RD::PipelineImmutableSampler immutable_shadow_sampler;
		immutable_shadow_sampler.binding = 2;
		immutable_shadow_sampler.append_id(shadow_sampler);
		immutable_shadow_sampler.uniform_type = RenderingDeviceCommons::UNIFORM_TYPE_SAMPLER;
		immutable_samplers.push_back(immutable_shadow_sampler);
		Vector<uint64_t> dynamic_buffers;
		dynamic_buffers.push_back(ShaderRD::DynamicBuffer::encode(RenderForwardMobile::RENDER_PASS_UNIFORM_SET, 0));
		dynamic_buffers.push_back(ShaderRD::DynamicBuffer::encode(RenderForwardMobile::RENDER_PASS_UNIFORM_SET, 1));
		shader.initialize(shader_versions, p_defines, immutable_samplers, dynamic_buffers);

		if (RendererCompositorRD::get_singleton()->is_xr_enabled()) {
			enable_multiview_shader_group();
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
		actions.renames["NORMAL"] = "normal_highp";
		actions.renames["TANGENT"] = "tangent_highp";
		actions.renames["BINORMAL"] = "binormal_highp";
		actions.renames["POSITION"] = "position";
		actions.renames["UV"] = "uv_interp";
		actions.renames["UV2"] = "uv2_interp";
		actions.renames["COLOR"] = "color_highp";
		actions.renames["POINT_SIZE"] = "point_size";
		actions.renames["INSTANCE_ID"] = "INSTANCE_INDEX";
		actions.renames["VERTEX_ID"] = "VERTEX_INDEX";
		actions.renames["Z_CLIP_SCALE"] = "z_clip_scale";

		actions.renames["ALPHA_SCISSOR_THRESHOLD"] = "alpha_scissor_threshold_highp";
		actions.renames["ALPHA_HASH_SCALE"] = "alpha_hash_scale_highp";
		actions.renames["ALPHA_ANTIALIASING_EDGE"] = "alpha_antialiasing_edge_highp";
		actions.renames["ALPHA_TEXTURE_COORDINATE"] = "alpha_texture_coordinate";

		//builtins

		actions.renames["TIME"] = "scene_data_block.data.time";
		actions.renames["EXPOSURE"] = "(1.0 / scene_data_block.data.emissive_exposure_normalization)";
		actions.renames["PI"] = String::num(Math::PI);
		actions.renames["TAU"] = String::num(Math::TAU);
		actions.renames["E"] = String::num(Math::E);
		actions.renames["OUTPUT_IS_SRGB"] = "SHADER_IS_SRGB";
		actions.renames["CLIP_SPACE_FAR"] = "SHADER_SPACE_FAR";
		actions.renames["IN_SHADOW_PASS"] = "IN_SHADOW_PASS";
		actions.renames["VIEWPORT_SIZE"] = "read_viewport_size";

		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["FRONT_FACING"] = "gl_FrontFacing";
		actions.renames["NORMAL_MAP"] = "normal_map_highp";
		actions.renames["NORMAL_MAP_DEPTH"] = "normal_map_depth_highp";
		actions.renames["BENT_NORMAL_MAP"] = "bent_normal_map_highp";
		actions.renames["ALBEDO"] = "albedo_highp";
		actions.renames["ALPHA"] = "alpha_highp";
		actions.renames["PREMUL_ALPHA_FACTOR"] = "premul_alpha_highp";
		actions.renames["METALLIC"] = "metallic_highp";
		actions.renames["SPECULAR"] = "specular_highp";
		actions.renames["ROUGHNESS"] = "roughness_highp";
		actions.renames["RIM"] = "rim_highp";
		actions.renames["RIM_TINT"] = "rim_tint_highp";
		actions.renames["CLEARCOAT"] = "clearcoat_highp";
		actions.renames["CLEARCOAT_ROUGHNESS"] = "clearcoat_roughness_highp";
		actions.renames["ANISOTROPY"] = "anisotropy_highp";
		actions.renames["ANISOTROPY_FLOW"] = "anisotropy_flow_highp";
		actions.renames["SSS_STRENGTH"] = "sss_strength_highp";
		actions.renames["SSS_TRANSMITTANCE_COLOR"] = "transmittance_color_highp";
		actions.renames["SSS_TRANSMITTANCE_DEPTH"] = "transmittance_depth_highp";
		actions.renames["SSS_TRANSMITTANCE_BOOST"] = "transmittance_boost_highp";
		actions.renames["BACKLIGHT"] = "backlight_highp";
		actions.renames["AO"] = "ao_highp";
		actions.renames["AO_LIGHT_AFFECT"] = "ao_light_affect_highp";
		actions.renames["EMISSION"] = "emission_highp";
		actions.renames["POINT_COORD"] = "point_coord";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";
		actions.renames["SCREEN_UV"] = "screen_uv";
		actions.renames["DEPTH"] = "gl_FragDepth";
		actions.renames["FOG"] = "fog_highp";
		actions.renames["RADIANCE"] = "custom_radiance_highp";
		actions.renames["IRRADIANCE"] = "custom_irradiance_highp";
		actions.renames["BONE_INDICES"] = "bone_attrib";
		actions.renames["BONE_WEIGHTS"] = "weight_attrib";
		actions.renames["CUSTOM0"] = "custom0_attrib";
		actions.renames["CUSTOM1"] = "custom1_attrib";
		actions.renames["CUSTOM2"] = "custom2_attrib";
		actions.renames["CUSTOM3"] = "custom3_attrib";
		actions.renames["LIGHT_VERTEX"] = "light_vertex";

		actions.renames["NODE_POSITION_WORLD"] = "read_model_matrix[3].xyz";
		actions.renames["CAMERA_POSITION_WORLD"] = "inv_view_matrix[3].xyz";
		actions.renames["CAMERA_DIRECTION_WORLD"] = "inv_view_matrix[2].xyz";
		actions.renames["CAMERA_VISIBLE_LAYERS"] = "scene_data.camera_visible_layers";
		actions.renames["NODE_POSITION_VIEW"] = "(read_view_matrix * read_model_matrix)[3].xyz";

		actions.renames["VIEW_INDEX"] = "ViewIndex";
		actions.renames["VIEW_MONO_LEFT"] = "0";
		actions.renames["VIEW_RIGHT"] = "1";
		actions.renames["EYE_OFFSET"] = "eye_offset";

		//for light
		actions.renames["VIEW"] = "view_highp";
		actions.renames["SPECULAR_AMOUNT"] = "specular_amount_highp";
		actions.renames["LIGHT_COLOR"] = "light_color_highp";
		actions.renames["LIGHT_IS_DIRECTIONAL"] = "is_directional";
		actions.renames["LIGHT"] = "light_highp";
		actions.renames["ATTENUATION"] = "attenuation_highp";
		actions.renames["DIFFUSE_LIGHT"] = "diffuse_light_highp";
		actions.renames["SPECULAR_LIGHT"] = "specular_light_highp";

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
		actions.usage_defines["BENT_NORMAL_MAP"] = "#define BENT_NORMAL_MAP_USED\n";
		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["INSTANCE_CUSTOM"] = "#define ENABLE_INSTANCE_CUSTOM\n";
		actions.usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";
		actions.usage_defines["LIGHT_VERTEX"] = "#define LIGHT_VERTEX_USED\n";
		actions.usage_defines["Z_CLIP_SCALE"] = "#define Z_CLIP_SCALE_USED\n";

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

		actions.usage_defines["POINT_SIZE"] = "#define POINT_SIZE_USED\n";
		actions.usage_defines["POINT_COORD"] = "#define POINT_COORD_USED\n";

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

		bool force_vertex_shading = GLOBAL_GET("rendering/shading/overrides/force_vertex_shading");
		if (!force_vertex_shading) {
			// If forcing vertex shading, this will be defined already.
			actions.render_mode_defines["vertex_lighting"] = "#define USE_VERTEX_LIGHTING\n";
		}

		actions.render_mode_defines["debug_shadow_splits"] = "#define DEBUG_DRAW_PSSM_SPLITS\n";
		actions.render_mode_defines["fog_disabled"] = "#define FOG_DISABLED\n";

		actions.render_mode_defines["specular_occlusion_disabled"] = "#define SPECULAR_OCCLUSION_DISABLED\n";

		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = RenderForwardMobile::MATERIAL_UNIFORM_SET;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 15;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_shader_uniforms.data";
		actions.instance_uniform_index_variable = "instances.data[draw_call.instance_index].instance_uniforms_ofs";

		actions.apply_luminance_multiplier = true; // apply luminance multiplier to screen texture
		actions.check_multiview_samplers = true;

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
		default_shader_rd = shader.version_get_shader(md->shader_data->version, (use_fp16 ? SHADER_VERSION_MAX * 2 : 0) + SHADER_VERSION_COLOR_PASS);

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
}

void SceneShaderForwardMobile::set_default_specialization(const ShaderSpecialization &p_specialization) {
	default_specialization = p_specialization;

	for (SelfList<ShaderData> *E = shader_list.first(); E; E = E->next()) {
		E->self()->pipeline_hash_map.clear_pipelines();
	}
}

uint32_t SceneShaderForwardMobile::get_pipeline_compilations(RS::PipelineSource p_source) {
	MutexLock lock(SceneShaderForwardMobile::singleton_mutex);
	return pipeline_compilations[p_source];
}

void SceneShaderForwardMobile::enable_fp32_shader_group() {
	shader.enable_group(SHADER_GROUP_FP32);

	if (is_multiview_shader_group_enabled()) {
		enable_multiview_shader_group();
	}
}

void SceneShaderForwardMobile::enable_fp16_shader_group() {
	shader.enable_group(SHADER_GROUP_FP16);

	if (is_multiview_shader_group_enabled()) {
		enable_multiview_shader_group();
	}
}

void SceneShaderForwardMobile::enable_multiview_shader_group() {
	if (shader.is_group_enabled(SHADER_GROUP_FP32)) {
		shader.enable_group(SHADER_GROUP_FP32_MULTIVIEW);
	}

	if (shader.is_group_enabled(SHADER_GROUP_FP16)) {
		shader.enable_group(SHADER_GROUP_FP16_MULTIVIEW);
	}
}

bool SceneShaderForwardMobile::is_multiview_shader_group_enabled() const {
	return shader.is_group_enabled(SHADER_GROUP_FP32_MULTIVIEW) || shader.is_group_enabled(SHADER_GROUP_FP16_MULTIVIEW);
}

SceneShaderForwardMobile::~SceneShaderForwardMobile() {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	RD::get_singleton()->free_rid(default_vec4_xform_buffer);
	RD::get_singleton()->free_rid(shadow_sampler);

	material_storage->shader_free(overdraw_material_shader);
	material_storage->shader_free(default_shader);
	material_storage->shader_free(debug_shadow_splits_material_shader);

	material_storage->material_free(overdraw_material);
	material_storage->material_free(default_material);
	material_storage->material_free(debug_shadow_splits_material);
}

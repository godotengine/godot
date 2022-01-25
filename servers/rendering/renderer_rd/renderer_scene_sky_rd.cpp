/*************************************************************************/
/*  renderer_scene_sky_rd.cpp                                            */
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

#include "renderer_scene_sky_rd.h"
#include "core/config/project_settings.h"
#include "core/math/math_defs.h"
#include "renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_server_default.h"

////////////////////////////////////////////////////////////////////////////////
// SKY SHADER

void RendererSceneSkyRD::SkyShaderData::set_code(const String &p_code) {
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
	actions.entry_point_stages["sky"] = ShaderCompiler::STAGE_FRAGMENT;

	uses_time = false;
	uses_half_res = false;
	uses_quarter_res = false;
	uses_position = false;
	uses_light = false;

	actions.render_mode_flags["use_half_res_pass"] = &uses_half_res;
	actions.render_mode_flags["use_quarter_res_pass"] = &uses_quarter_res;

	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["POSITION"] = &uses_position;
	actions.usage_flag_pointers["LIGHT0_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_SIZE"] = &uses_light;

	actions.uniforms = &uniforms;

	// !BAS! Contemplate making `SkyShader sky` accessible from this struct or even part of this struct.
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;

	Error err = scene_singleton->sky.sky_shader.compiler.compile(RS::SHADER_SKY, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = scene_singleton->sky.sky_shader.shader.version_create();
	}

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}
	print_line("\n**uniforms:\n" + gen_code.uniforms);
	//	print_line("\n**vertex_globals:\n" + gen_code.vertex_global);
	//	print_line("\n**vertex_code:\n" + gen_code.vertex);
	print_line("\n**fragment_globals:\n" + gen_code.fragment_global);
	print_line("\n**fragment_code:\n" + gen_code.fragment);
	print_line("\n**light_code:\n" + gen_code.light);
#endif

	scene_singleton->sky.sky_shader.shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines);
	ERR_FAIL_COND(!scene_singleton->sky.sky_shader.shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//update pipelines

	for (int i = 0; i < SKY_VERSION_MAX; i++) {
		RD::PipelineDepthStencilState depth_stencil_state;
		depth_stencil_state.enable_depth_test = true;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;

		if (scene_singleton->sky.sky_shader.shader.is_variant_enabled(i)) {
			RID shader_variant = scene_singleton->sky.sky_shader.shader.version_get_shader(version, i);
			pipelines[i].setup(shader_variant, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), depth_stencil_state, RD::PipelineColorBlendState::create_disabled(), 0);
		} else {
			pipelines[i].clear();
		}
	}

	valid = true;
}

void RendererSceneSkyRD::SkyShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
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

void RendererSceneSkyRD::SkyShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
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

void RendererSceneSkyRD::SkyShaderData::get_instance_param_list(List<RendererStorage::InstanceShaderParam> *p_param_list) const {
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

bool RendererSceneSkyRD::SkyShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RendererSceneSkyRD::SkyShaderData::is_animated() const {
	return false;
}

bool RendererSceneSkyRD::SkyShaderData::casts_shadows() const {
	return false;
}

Variant RendererSceneSkyRD::SkyShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode RendererSceneSkyRD::SkyShaderData::get_native_source_code() const {
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;

	return scene_singleton->sky.sky_shader.shader.version_get_native_source_code(version);
}

RendererSceneSkyRD::SkyShaderData::SkyShaderData() {
	valid = false;
}

RendererSceneSkyRD::SkyShaderData::~SkyShaderData() {
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;
	ERR_FAIL_COND(!scene_singleton);
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		scene_singleton->sky.sky_shader.shader.version_free(version);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Sky material

bool RendererSceneSkyRD::SkyMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	RendererSceneRenderRD *scene_singleton = (RendererSceneRenderRD *)RendererSceneRenderRD::singleton;

	uniform_set_updated = true;

	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, scene_singleton->sky.sky_shader.shader.version_get_shader(shader_data->version, 0), SKY_SET_MATERIAL);
}

RendererSceneSkyRD::SkyMaterialData::~SkyMaterialData() {
	free_parameters_uniform_set(uniform_set);
}

////////////////////////////////////////////////////////////////////////////////
// Render sky

static _FORCE_INLINE_ void store_transform_3x3(const Basis &p_basis, float *p_array) {
	p_array[0] = p_basis.elements[0][0];
	p_array[1] = p_basis.elements[1][0];
	p_array[2] = p_basis.elements[2][0];
	p_array[3] = 0;
	p_array[4] = p_basis.elements[0][1];
	p_array[5] = p_basis.elements[1][1];
	p_array[6] = p_basis.elements[2][1];
	p_array[7] = 0;
	p_array[8] = p_basis.elements[0][2];
	p_array[9] = p_basis.elements[1][2];
	p_array[10] = p_basis.elements[2][2];
	p_array[11] = 0;
}

void RendererSceneSkyRD::_render_sky(RD::DrawListID p_list, float p_time, RID p_fb, PipelineCacheRD *p_pipeline, RID p_uniform_set, RID p_texture_set, uint32_t p_view_count, const CameraMatrix *p_projections, const Basis &p_orientation, float p_multiplier, const Vector3 &p_position, float p_luminance_multiplier) {
	SkyPushConstant sky_push_constant;

	memset(&sky_push_constant, 0, sizeof(SkyPushConstant));

	for (uint32_t v = 0; v < p_view_count; v++) {
		// We only need key components of our projection matrix
		sky_push_constant.projections[v][0] = p_projections[v].matrix[2][0];
		sky_push_constant.projections[v][1] = p_projections[v].matrix[0][0];
		sky_push_constant.projections[v][2] = p_projections[v].matrix[2][1];
		sky_push_constant.projections[v][3] = p_projections[v].matrix[1][1];
	}
	sky_push_constant.position[0] = p_position.x;
	sky_push_constant.position[1] = p_position.y;
	sky_push_constant.position[2] = p_position.z;
	sky_push_constant.multiplier = p_multiplier;
	sky_push_constant.time = p_time;
	sky_push_constant.luminance_multiplier = p_luminance_multiplier;
	store_transform_3x3(p_orientation, sky_push_constant.orientation);

	RenderingDevice::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(p_fb);

	RD::DrawListID draw_list = p_list;

	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, p_pipeline->get_render_pipeline(RD::INVALID_ID, fb_format, false, RD::get_singleton()->draw_list_get_current_pass()));

	// Update uniform sets.
	{
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, sky_scene_state.uniform_set, 0);
		if (p_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(p_uniform_set)) { // Material may not have a uniform set.
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_uniform_set, 1);
		}
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_texture_set, 2);
		// Fog uniform set can be invalidated before drawing, so validate at draw time
		if (sky_scene_state.fog_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(sky_scene_state.fog_uniform_set)) {
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, sky_scene_state.fog_uniform_set, 3);
		} else {
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, sky_scene_state.default_fog_uniform_set, 3);
		}
	}

	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &sky_push_constant, sizeof(SkyPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
}

////////////////////////////////////////////////////////////////////////////////
// ReflectionData

void RendererSceneSkyRD::ReflectionData::clear_reflection_data() {
	layers.clear();
	radiance_base_cubemap = RID();
	if (downsampled_radiance_cubemap.is_valid()) {
		RD::get_singleton()->free(downsampled_radiance_cubemap);
	}
	downsampled_radiance_cubemap = RID();
	downsampled_layer.mipmaps.clear();
	coefficient_buffer = RID();
}

void RendererSceneSkyRD::ReflectionData::update_reflection_data(RendererStorageRD *p_storage, int p_size, int p_mipmaps, bool p_use_array, RID p_base_cube, int p_base_layer, bool p_low_quality, int p_roughness_layers, RD::DataFormat p_texture_format) {
	//recreate radiance and all data

	int mipmaps = p_mipmaps;
	uint32_t w = p_size, h = p_size;

	EffectsRD *effects = p_storage->get_effects();
	ERR_FAIL_NULL_MSG(effects, "Effects haven't been initialised");
	bool prefer_raster_effects = effects->get_prefer_raster_effects();

	if (p_use_array) {
		int num_layers = p_low_quality ? 8 : p_roughness_layers;

		for (int i = 0; i < num_layers; i++) {
			ReflectionData::Layer layer;
			uint32_t mmw = w;
			uint32_t mmh = h;
			layer.mipmaps.resize(mipmaps);
			layer.views.resize(mipmaps);
			for (int j = 0; j < mipmaps; j++) {
				ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
				mm.size.width = mmw;
				mm.size.height = mmh;
				for (int k = 0; k < 6; k++) {
					mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + i * 6 + k, j);
					Vector<RID> fbtex;
					fbtex.push_back(mm.views[k]);
					mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
				}

				layer.views.write[j] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + i * 6, j, 1, RD::TEXTURE_SLICE_CUBEMAP);

				mmw = MAX(1, mmw >> 1);
				mmh = MAX(1, mmh >> 1);
			}

			layers.push_back(layer);
		}

	} else {
		mipmaps = p_low_quality ? 8 : mipmaps;
		//regular cubemap, lower quality (aliasing, less memory)
		ReflectionData::Layer layer;
		uint32_t mmw = w;
		uint32_t mmh = h;
		layer.mipmaps.resize(mipmaps);
		layer.views.resize(mipmaps);
		for (int j = 0; j < mipmaps; j++) {
			ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
			mm.size.width = mmw;
			mm.size.height = mmh;
			for (int k = 0; k < 6; k++) {
				mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + k, j);
				Vector<RID> fbtex;
				fbtex.push_back(mm.views[k]);
				mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
			}

			layer.views.write[j] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer, j, 1, RD::TEXTURE_SLICE_CUBEMAP);

			mmw = MAX(1, mmw >> 1);
			mmh = MAX(1, mmh >> 1);
		}

		layers.push_back(layer);
	}

	radiance_base_cubemap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer, 0, 1, RD::TEXTURE_SLICE_CUBEMAP);
	RD::get_singleton()->set_resource_name(radiance_base_cubemap, "radiance base cubemap");
	RD::TextureFormat tf;
	tf.format = p_texture_format;
	tf.width = 64; // Always 64x64
	tf.height = 64;
	tf.texture_type = RD::TEXTURE_TYPE_CUBE;
	tf.array_layers = 6;
	tf.mipmaps = 7;
	tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

	downsampled_radiance_cubemap = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RD::get_singleton()->set_resource_name(downsampled_radiance_cubemap, "downsampled radiance cubemap");
	{
		uint32_t mmw = 64;
		uint32_t mmh = 64;
		downsampled_layer.mipmaps.resize(7);
		for (int j = 0; j < downsampled_layer.mipmaps.size(); j++) {
			ReflectionData::DownsampleLayer::Mipmap &mm = downsampled_layer.mipmaps.write[j];
			mm.size.width = mmw;
			mm.size.height = mmh;
			mm.view = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), downsampled_radiance_cubemap, 0, j, 1, RD::TEXTURE_SLICE_CUBEMAP);
			RD::get_singleton()->set_resource_name(mm.view, "Downsampled Radiance Cubemap Mip " + itos(j) + " ");
			if (prefer_raster_effects) {
				// we need a framebuffer for each side of our cubemap

				for (int k = 0; k < 6; k++) {
					mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), downsampled_radiance_cubemap, k, j);
					RD::get_singleton()->set_resource_name(mm.view, "Downsampled Radiance Cubemap Mip: " + itos(j) + " Face: " + itos(k) + " ");
					Vector<RID> fbtex;
					fbtex.push_back(mm.views[k]);
					mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
				}
			}

			mmw = MAX(1, mmw >> 1);
			mmh = MAX(1, mmh >> 1);
		}
	}
}

void RendererSceneSkyRD::ReflectionData::create_reflection_fast_filter(RendererStorageRD *p_storage, bool p_use_arrays) {
	EffectsRD *effects = p_storage->get_effects();
	ERR_FAIL_NULL_MSG(effects, "Effects haven't been initialised");
	bool prefer_raster_effects = effects->get_prefer_raster_effects();

	if (prefer_raster_effects) {
		RD::get_singleton()->draw_command_begin_label("Downsample radiance map");
		for (int k = 0; k < 6; k++) {
			effects->cubemap_downsample_raster(radiance_base_cubemap, downsampled_layer.mipmaps[0].framebuffers[k], k, downsampled_layer.mipmaps[0].size);
		}

		for (int i = 1; i < downsampled_layer.mipmaps.size(); i++) {
			for (int k = 0; k < 6; k++) {
				effects->cubemap_downsample_raster(downsampled_layer.mipmaps[i - 1].view, downsampled_layer.mipmaps[i].framebuffers[k], k, downsampled_layer.mipmaps[i].size);
			}
		}
		RD::get_singleton()->draw_command_end_label(); // Downsample Radiance

		if (p_use_arrays) {
			RD::get_singleton()->draw_command_begin_label("filter radiance map into array heads");
			for (int i = 0; i < layers.size(); i++) {
				for (int k = 0; k < 6; k++) {
					effects->cubemap_filter_raster(downsampled_radiance_cubemap, layers[i].mipmaps[0].framebuffers[k], k, i);
				}
			}
		} else {
			RD::get_singleton()->draw_command_begin_label("filter radiance map into mipmaps directly");
			for (int j = 0; j < layers[0].mipmaps.size(); j++) {
				for (int k = 0; k < 6; k++) {
					effects->cubemap_filter_raster(downsampled_radiance_cubemap, layers[0].mipmaps[j].framebuffers[k], k, j);
				}
			}
		}
		RD::get_singleton()->draw_command_end_label(); // Filter radiance
	} else {
		effects->cubemap_downsample(radiance_base_cubemap, downsampled_layer.mipmaps[0].view, downsampled_layer.mipmaps[0].size);

		for (int i = 1; i < downsampled_layer.mipmaps.size(); i++) {
			effects->cubemap_downsample(downsampled_layer.mipmaps[i - 1].view, downsampled_layer.mipmaps[i].view, downsampled_layer.mipmaps[i].size);
		}

		Vector<RID> views;
		if (p_use_arrays) {
			for (int i = 1; i < layers.size(); i++) {
				views.push_back(layers[i].views[0]);
			}
		} else {
			for (int i = 1; i < layers[0].views.size(); i++) {
				views.push_back(layers[0].views[i]);
			}
		}

		effects->cubemap_filter(downsampled_radiance_cubemap, views, p_use_arrays);
	}
}

void RendererSceneSkyRD::ReflectionData::create_reflection_importance_sample(RendererStorageRD *p_storage, bool p_use_arrays, int p_cube_side, int p_base_layer, uint32_t p_sky_ggx_samples_quality) {
	EffectsRD *effects = p_storage->get_effects();
	ERR_FAIL_NULL_MSG(effects, "Effects haven't been initialised");
	bool prefer_raster_effects = effects->get_prefer_raster_effects();

	if (prefer_raster_effects) {
		// Need to ask clayjohn but p_cube_side is set to 10, looks like in the compute shader we're doing all 6 sides in one call
		// here we need to do them one by one so ignoring p_cube_side
		if (p_use_arrays) {
			for (int k = 0; k < 6; k++) {
				effects->cubemap_roughness_raster(
						radiance_base_cubemap,
						layers[p_base_layer].mipmaps[0].framebuffers[k],
						k,
						p_sky_ggx_samples_quality,
						float(p_base_layer) / (layers.size() - 1.0),
						layers[p_base_layer].mipmaps[0].size.x);
			}
		} else {
			for (int k = 0; k < 6; k++) {
				effects->cubemap_roughness_raster(
						layers[0].views[p_base_layer - 1],
						layers[0].mipmaps[p_base_layer].framebuffers[k],
						k,
						p_sky_ggx_samples_quality,
						float(p_base_layer) / (layers[0].mipmaps.size() - 1.0),
						layers[0].mipmaps[p_base_layer].size.x);
			}
		}
	} else {
		if (p_use_arrays) {
			//render directly to the layers
			effects->cubemap_roughness(radiance_base_cubemap, layers[p_base_layer].views[0], p_cube_side, p_sky_ggx_samples_quality, float(p_base_layer) / (layers.size() - 1.0), layers[p_base_layer].mipmaps[0].size.x);
		} else {
			effects->cubemap_roughness(
					layers[0].views[p_base_layer - 1],
					layers[0].views[p_base_layer],
					p_cube_side,
					p_sky_ggx_samples_quality,
					float(p_base_layer) / (layers[0].mipmaps.size() - 1.0),
					layers[0].mipmaps[p_base_layer].size.x);
		}
	}
}

void RendererSceneSkyRD::ReflectionData::update_reflection_mipmaps(RendererStorageRD *p_storage, int p_start, int p_end) {
	EffectsRD *effects = p_storage->get_effects();
	ERR_FAIL_NULL_MSG(effects, "Effects haven't been initialised");
	bool prefer_raster_effects = effects->get_prefer_raster_effects();

	RD::get_singleton()->draw_command_begin_label("Update Radiance Cubemap Array Mipmaps");
	for (int i = p_start; i < p_end; i++) {
		for (int j = 0; j < layers[i].views.size() - 1; j++) {
			RID view = layers[i].views[j];
			Size2i size = layers[i].mipmaps[j + 1].size;
			if (prefer_raster_effects) {
				for (int k = 0; k < 6; k++) {
					RID framebuffer = layers[i].mipmaps[j + 1].framebuffers[k];
					effects->cubemap_downsample_raster(view, framebuffer, k, size);
				}
			} else {
				RID texture = layers[i].views[j + 1];
				effects->cubemap_downsample(view, texture, size);
			}
		}
	}
	RD::get_singleton()->draw_command_end_label();
}

////////////////////////////////////////////////////////////////////////////////
// RendererSceneSkyRD::Sky

void RendererSceneSkyRD::Sky::free(RendererStorageRD *p_storage) {
	if (radiance.is_valid()) {
		RD::get_singleton()->free(radiance);
		radiance = RID();
	}
	reflection.clear_reflection_data();

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
		uniform_buffer = RID();
	}

	if (half_res_pass.is_valid()) {
		RD::get_singleton()->free(half_res_pass);
		half_res_pass = RID();
	}

	if (quarter_res_pass.is_valid()) {
		RD::get_singleton()->free(quarter_res_pass);
		quarter_res_pass = RID();
	}

	if (material.is_valid()) {
		p_storage->free(material);
		material = RID();
	}
}

RID RendererSceneSkyRD::Sky::get_textures(RendererStorageRD *p_storage, SkyTextureSetVersion p_version, RID p_default_shader_rd) {
	if (texture_uniform_sets[p_version].is_valid() && RD::get_singleton()->uniform_set_is_valid(texture_uniform_sets[p_version])) {
		return texture_uniform_sets[p_version];
	}
	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 0;
		if (radiance.is_valid() && p_version <= SKY_TEXTURE_SET_QUARTER_RES) {
			u.ids.push_back(radiance);
		} else {
			u.ids.push_back(p_storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
		}
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 1; // half res
		if (half_res_pass.is_valid() && p_version != SKY_TEXTURE_SET_HALF_RES && p_version != SKY_TEXTURE_SET_CUBEMAP_HALF_RES) {
			if (p_version >= SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(reflection.layers[0].views[1]);
			} else {
				u.ids.push_back(half_res_pass);
			}
		} else {
			if (p_version < SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(p_storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			} else {
				u.ids.push_back(p_storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
			}
		}
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 2; // quarter res
		if (quarter_res_pass.is_valid() && p_version != SKY_TEXTURE_SET_QUARTER_RES && p_version != SKY_TEXTURE_SET_CUBEMAP_QUARTER_RES) {
			if (p_version >= SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(reflection.layers[0].views[2]);
			} else {
				u.ids.push_back(quarter_res_pass);
			}
		} else {
			if (p_version < SKY_TEXTURE_SET_CUBEMAP) {
				u.ids.push_back(p_storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			} else {
				u.ids.push_back(p_storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
			}
		}
		uniforms.push_back(u);
	}

	texture_uniform_sets[p_version] = RD::get_singleton()->uniform_set_create(uniforms, p_default_shader_rd, SKY_SET_TEXTURES);
	return texture_uniform_sets[p_version];
}

bool RendererSceneSkyRD::Sky::set_radiance_size(int p_radiance_size) {
	ERR_FAIL_COND_V(p_radiance_size < 32 || p_radiance_size > 2048, false);
	if (radiance_size == p_radiance_size) {
		return false;
	}
	radiance_size = p_radiance_size;

	if (mode == RS::SKY_MODE_REALTIME && radiance_size != 256) {
		WARN_PRINT("Realtime Skies can only use a radiance size of 256. Radiance size will be set to 256 internally.");
		radiance_size = 256;
	}

	if (radiance.is_valid()) {
		RD::get_singleton()->free(radiance);
		radiance = RID();
	}
	reflection.clear_reflection_data();

	return true;
}

bool RendererSceneSkyRD::Sky::set_mode(RS::SkyMode p_mode) {
	if (mode == p_mode) {
		return false;
	}

	mode = p_mode;

	if (mode == RS::SKY_MODE_REALTIME && radiance_size != 256) {
		WARN_PRINT("Realtime Skies can only use a radiance size of 256. Radiance size will be set to 256 internally.");
		set_radiance_size(256);
	}

	if (radiance.is_valid()) {
		RD::get_singleton()->free(radiance);
		radiance = RID();
	}
	reflection.clear_reflection_data();

	return true;
}

bool RendererSceneSkyRD::Sky::set_material(RID p_material) {
	if (material == p_material) {
		return false;
	}

	material = p_material;
	return true;
}

Ref<Image> RendererSceneSkyRD::Sky::bake_panorama(RendererStorageRD *p_storage, float p_energy, int p_roughness_layers, const Size2i &p_size) {
	if (radiance.is_valid()) {
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
		tf.width = p_size.width;
		tf.height = p_size.height;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

		RID rad_tex = RD::get_singleton()->texture_create(tf, RD::TextureView());
		p_storage->get_effects()->copy_cubemap_to_panorama(radiance, rad_tex, p_size, p_roughness_layers, reflection.layers.size() > 1);
		Vector<uint8_t> data = RD::get_singleton()->texture_get_data(rad_tex, 0);
		RD::get_singleton()->free(rad_tex);

		Ref<Image> img;
		img.instantiate();
		img->create(p_size.width, p_size.height, false, Image::FORMAT_RGBAF, data);
		for (int i = 0; i < p_size.width; i++) {
			for (int j = 0; j < p_size.height; j++) {
				Color c = img->get_pixel(i, j);
				c.r *= p_energy;
				c.g *= p_energy;
				c.b *= p_energy;
				img->set_pixel(i, j, c);
			}
		}
		return img;
	}

	return Ref<Image>();
}

////////////////////////////////////////////////////////////////////////////////
// RendererSceneSkyRD

RendererStorageRD::ShaderData *RendererSceneSkyRD::_create_sky_shader_func() {
	SkyShaderData *shader_data = memnew(SkyShaderData);
	return shader_data;
}

RendererStorageRD::ShaderData *RendererSceneSkyRD::_create_sky_shader_funcs() {
	// !BAS! Why isn't _create_sky_shader_func not just static too?
	return static_cast<RendererSceneRenderRD *>(RendererSceneRenderRD::singleton)->sky._create_sky_shader_func();
};

RendererStorageRD::MaterialData *RendererSceneSkyRD::_create_sky_material_func(SkyShaderData *p_shader) {
	SkyMaterialData *material_data = memnew(SkyMaterialData);
	material_data->shader_data = p_shader;
	//update will happen later anyway so do nothing.
	return material_data;
}

RendererStorageRD::MaterialData *RendererSceneSkyRD::_create_sky_material_funcs(RendererStorageRD::ShaderData *p_shader) {
	// !BAS! same here, we could just make _create_sky_material_func static?
	return static_cast<RendererSceneRenderRD *>(RendererSceneRenderRD::singleton)->sky._create_sky_material_func(static_cast<SkyShaderData *>(p_shader));
};

RendererSceneSkyRD::RendererSceneSkyRD() {
	roughness_layers = GLOBAL_GET("rendering/reflections/sky_reflections/roughness_layers");
	sky_ggx_samples_quality = GLOBAL_GET("rendering/reflections/sky_reflections/ggx_samples");
	sky_use_cubemap_array = GLOBAL_GET("rendering/reflections/sky_reflections/texture_array_reflections");
}

void RendererSceneSkyRD::init(RendererStorageRD *p_storage) {
	storage = p_storage;

	{
		// Start with the directional lights for the sky
		sky_scene_state.max_directional_lights = 4;
		uint32_t directional_light_buffer_size = sky_scene_state.max_directional_lights * sizeof(SkyDirectionalLightData);
		sky_scene_state.directional_lights = memnew_arr(SkyDirectionalLightData, sky_scene_state.max_directional_lights);
		sky_scene_state.last_frame_directional_lights = memnew_arr(SkyDirectionalLightData, sky_scene_state.max_directional_lights);
		sky_scene_state.last_frame_directional_light_count = sky_scene_state.max_directional_lights + 1;
		sky_scene_state.directional_light_buffer = RD::get_singleton()->uniform_buffer_create(directional_light_buffer_size);

		String defines = "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(sky_scene_state.max_directional_lights) + "\n";

		// Initialize sky
		Vector<String> sky_modes;
		sky_modes.push_back(""); // Full size
		sky_modes.push_back("\n#define USE_HALF_RES_PASS\n"); // Half Res
		sky_modes.push_back("\n#define USE_QUARTER_RES_PASS\n"); // Quarter res
		sky_modes.push_back("\n#define USE_CUBEMAP_PASS\n"); // Cubemap
		sky_modes.push_back("\n#define USE_CUBEMAP_PASS\n#define USE_HALF_RES_PASS\n"); // Half Res Cubemap
		sky_modes.push_back("\n#define USE_CUBEMAP_PASS\n#define USE_QUARTER_RES_PASS\n"); // Quarter res Cubemap

		sky_modes.push_back("\n#define USE_MULTIVIEW\n"); // Full size multiview
		sky_modes.push_back("\n#define USE_HALF_RES_PASS\n#define USE_MULTIVIEW\n"); // Half Res multiview
		sky_modes.push_back("\n#define USE_QUARTER_RES_PASS\n#define USE_MULTIVIEW\n"); // Quarter res multiview

		sky_shader.shader.initialize(sky_modes, defines);

		if (!RendererCompositorRD::singleton->is_xr_enabled()) {
			sky_shader.shader.set_variant_enabled(SKY_VERSION_BACKGROUND_MULTIVIEW, false);
			sky_shader.shader.set_variant_enabled(SKY_VERSION_HALF_RES_MULTIVIEW, false);
			sky_shader.shader.set_variant_enabled(SKY_VERSION_QUARTER_RES_MULTIVIEW, false);
		}
	}

	// register our shader funds
	storage->shader_set_data_request_function(RendererStorageRD::SHADER_TYPE_SKY, _create_sky_shader_funcs);
	storage->material_set_data_request_function(RendererStorageRD::SHADER_TYPE_SKY, _create_sky_material_funcs);

	{
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "color";
		actions.renames["ALPHA"] = "alpha";
		actions.renames["EYEDIR"] = "cube_normal";
		actions.renames["POSITION"] = "params.position_multiplier.xyz";
		actions.renames["SKY_COORDS"] = "panorama_coords";
		actions.renames["SCREEN_UV"] = "uv";
		actions.renames["TIME"] = "params.time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["HALF_RES_COLOR"] = "half_res_color";
		actions.renames["QUARTER_RES_COLOR"] = "quarter_res_color";
		actions.renames["RADIANCE"] = "radiance";
		actions.renames["FOG"] = "custom_fog";
		actions.renames["LIGHT0_ENABLED"] = "directional_lights.data[0].enabled";
		actions.renames["LIGHT0_DIRECTION"] = "directional_lights.data[0].direction_energy.xyz";
		actions.renames["LIGHT0_ENERGY"] = "directional_lights.data[0].direction_energy.w";
		actions.renames["LIGHT0_COLOR"] = "directional_lights.data[0].color_size.xyz";
		actions.renames["LIGHT0_SIZE"] = "directional_lights.data[0].color_size.w";
		actions.renames["LIGHT1_ENABLED"] = "directional_lights.data[1].enabled";
		actions.renames["LIGHT1_DIRECTION"] = "directional_lights.data[1].direction_energy.xyz";
		actions.renames["LIGHT1_ENERGY"] = "directional_lights.data[1].direction_energy.w";
		actions.renames["LIGHT1_COLOR"] = "directional_lights.data[1].color_size.xyz";
		actions.renames["LIGHT1_SIZE"] = "directional_lights.data[1].color_size.w";
		actions.renames["LIGHT2_ENABLED"] = "directional_lights.data[2].enabled";
		actions.renames["LIGHT2_DIRECTION"] = "directional_lights.data[2].direction_energy.xyz";
		actions.renames["LIGHT2_ENERGY"] = "directional_lights.data[2].direction_energy.w";
		actions.renames["LIGHT2_COLOR"] = "directional_lights.data[2].color_size.xyz";
		actions.renames["LIGHT2_SIZE"] = "directional_lights.data[2].color_size.w";
		actions.renames["LIGHT3_ENABLED"] = "directional_lights.data[3].enabled";
		actions.renames["LIGHT3_DIRECTION"] = "directional_lights.data[3].direction_energy.xyz";
		actions.renames["LIGHT3_ENERGY"] = "directional_lights.data[3].direction_energy.w";
		actions.renames["LIGHT3_COLOR"] = "directional_lights.data[3].color_size.xyz";
		actions.renames["LIGHT3_SIZE"] = "directional_lights.data[3].color_size.w";
		actions.renames["AT_CUBEMAP_PASS"] = "AT_CUBEMAP_PASS";
		actions.renames["AT_HALF_RES_PASS"] = "AT_HALF_RES_PASS";
		actions.renames["AT_QUARTER_RES_PASS"] = "AT_QUARTER_RES_PASS";
		actions.custom_samplers["RADIANCE"] = "material_samplers[3]";
		actions.usage_defines["HALF_RES_COLOR"] = "\n#define USES_HALF_RES_COLOR\n";
		actions.usage_defines["QUARTER_RES_COLOR"] = "\n#define USES_QUARTER_RES_COLOR\n";
		actions.render_mode_defines["disable_fog"] = "#define DISABLE_FOG\n";

		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = 1;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_variables.data";

		sky_shader.compiler.initialize(actions);
	}

	{
		// default material and shader for sky shader
		sky_shader.default_shader = storage->shader_allocate();
		storage->shader_initialize(sky_shader.default_shader);

		storage->shader_set_code(sky_shader.default_shader, R"(
// Default sky shader.

shader_type sky;

void sky() {
	COLOR = vec3(0.0);
}
)");

		sky_shader.default_material = storage->material_allocate();
		storage->material_initialize(sky_shader.default_material);

		storage->material_set_shader(sky_shader.default_material, sky_shader.default_shader);

		SkyMaterialData *md = (SkyMaterialData *)storage->material_get_data(sky_shader.default_material, RendererStorageRD::SHADER_TYPE_SKY);
		sky_shader.default_shader_rd = sky_shader.shader.version_get_shader(md->shader_data->version, SKY_VERSION_BACKGROUND);

		sky_scene_state.uniform_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(SkySceneState::UBO));

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 0;
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
			u.binding = 1;
			u.ids.push_back(storage->global_variables_get_storage_buffer());
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 2;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(sky_scene_state.uniform_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.binding = 3;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.ids.push_back(sky_scene_state.directional_light_buffer);
			uniforms.push_back(u);
		}

		sky_scene_state.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_UNIFORMS);
	}

	{
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			RID vfog = storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);
			u.ids.push_back(vfog);
			uniforms.push_back(u);
		}

		sky_scene_state.default_fog_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_FOG);
	}

	{
		// Need defaults for using fog with clear color
		sky_scene_state.fog_shader = storage->shader_allocate();
		storage->shader_initialize(sky_scene_state.fog_shader);

		storage->shader_set_code(sky_scene_state.fog_shader, R"(
// Default clear color sky shader.

shader_type sky;

uniform vec4 clear_color;

void sky() {
	COLOR = clear_color.rgb;
}
)");
		sky_scene_state.fog_material = storage->material_allocate();
		storage->material_initialize(sky_scene_state.fog_material);

		storage->material_set_shader(sky_scene_state.fog_material, sky_scene_state.fog_shader);

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 0;
			u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;
			u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;
			u.ids.push_back(storage->texture_rd_get_default(RendererStorageRD::DEFAULT_RD_TEXTURE_WHITE));
			uniforms.push_back(u);
		}

		sky_scene_state.fog_only_texture_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, sky_shader.default_shader_rd, SKY_SET_TEXTURES);
	}

	{ //create index array for copy shaders
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
		index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		index_array = RD::get_singleton()->index_array_create(index_buffer, 0, 6);
	}
}

void RendererSceneSkyRD::set_texture_format(RD::DataFormat p_texture_format) {
	texture_format = p_texture_format;
}

RendererSceneSkyRD::~RendererSceneSkyRD() {
	// TODO cleanup anything created in init...

	if (RD::get_singleton()->uniform_set_is_valid(sky_scene_state.uniform_set)) {
		RD::get_singleton()->free(sky_scene_state.uniform_set);
	}

	if (RD::get_singleton()->uniform_set_is_valid(sky_scene_state.default_fog_uniform_set)) {
		RD::get_singleton()->free(sky_scene_state.default_fog_uniform_set);
	}

	if (RD::get_singleton()->uniform_set_is_valid(sky_scene_state.fog_only_texture_uniform_set)) {
		RD::get_singleton()->free(sky_scene_state.fog_only_texture_uniform_set);
	}

	RD::get_singleton()->free(index_buffer); //array gets freed as dependency
}

void RendererSceneSkyRD::setup(RendererSceneEnvironmentRD *p_env, RID p_render_buffers, const CameraMatrix &p_projection, const Transform3D &p_transform, const Size2i p_screen_size, RendererSceneRenderRD *p_scene_render) {
	ERR_FAIL_COND(!p_env); // I guess without an environment we also can't have a sky...

	SkyMaterialData *material = nullptr;
	Sky *sky = get_sky(p_env->sky);

	RID sky_material;

	SkyShaderData *shader_data = nullptr;

	if (sky) {
		sky_material = sky_get_material(p_env->sky);

		if (sky_material.is_valid()) {
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
			if (!material || !material->shader_data->valid) {
				material = nullptr;
			}
		}

		if (!material) {
			sky_material = sky_shader.default_material;
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
		}

		ERR_FAIL_COND(!material);

		shader_data = material->shader_data;

		ERR_FAIL_COND(!shader_data);

		// Invalidate supbass buffers if screen size changes
		if (sky->screen_size != p_screen_size) {
			sky->screen_size = p_screen_size;
			sky->screen_size.x = sky->screen_size.x < 4 ? 4 : sky->screen_size.x;
			sky->screen_size.y = sky->screen_size.y < 4 ? 4 : sky->screen_size.y;
			if (shader_data->uses_half_res) {
				if (sky->half_res_pass.is_valid()) {
					RD::get_singleton()->free(sky->half_res_pass);
					sky->half_res_pass = RID();
				}
				invalidate_sky(sky);
			}
			if (shader_data->uses_quarter_res) {
				if (sky->quarter_res_pass.is_valid()) {
					RD::get_singleton()->free(sky->quarter_res_pass);
					sky->quarter_res_pass = RID();
				}
				invalidate_sky(sky);
			}
		}

		// Create new subpass buffers if necessary
		if ((shader_data->uses_half_res && sky->half_res_pass.is_null()) ||
				(shader_data->uses_quarter_res && sky->quarter_res_pass.is_null()) ||
				sky->radiance.is_null()) {
			invalidate_sky(sky);
			update_dirty_skys();
		}

		if (shader_data->uses_time && p_scene_render->time - sky->prev_time > 0.00001) {
			sky->prev_time = p_scene_render->time;
			sky->reflection.dirty = true;
			RenderingServerDefault::redraw_request();
		}

		if (material != sky->prev_material) {
			sky->prev_material = material;
			sky->reflection.dirty = true;
		}

		if (material->uniform_set_updated) {
			material->uniform_set_updated = false;
			sky->reflection.dirty = true;
		}

		if (!p_transform.origin.is_equal_approx(sky->prev_position) && shader_data->uses_position) {
			sky->prev_position = p_transform.origin;
			sky->reflection.dirty = true;
		}

		if (shader_data->uses_light) {
			// Check whether the directional_light_buffer changes
			bool light_data_dirty = false;

			if (sky_scene_state.ubo.directional_light_count != sky_scene_state.last_frame_directional_light_count) {
				light_data_dirty = true;
				for (uint32_t i = sky_scene_state.ubo.directional_light_count; i < sky_scene_state.max_directional_lights; i++) {
					sky_scene_state.directional_lights[i].enabled = false;
				}
			}
			if (!light_data_dirty) {
				for (uint32_t i = 0; i < sky_scene_state.ubo.directional_light_count; i++) {
					if (sky_scene_state.directional_lights[i].direction[0] != sky_scene_state.last_frame_directional_lights[i].direction[0] ||
							sky_scene_state.directional_lights[i].direction[1] != sky_scene_state.last_frame_directional_lights[i].direction[1] ||
							sky_scene_state.directional_lights[i].direction[2] != sky_scene_state.last_frame_directional_lights[i].direction[2] ||
							sky_scene_state.directional_lights[i].energy != sky_scene_state.last_frame_directional_lights[i].energy ||
							sky_scene_state.directional_lights[i].color[0] != sky_scene_state.last_frame_directional_lights[i].color[0] ||
							sky_scene_state.directional_lights[i].color[1] != sky_scene_state.last_frame_directional_lights[i].color[1] ||
							sky_scene_state.directional_lights[i].color[2] != sky_scene_state.last_frame_directional_lights[i].color[2] ||
							sky_scene_state.directional_lights[i].enabled != sky_scene_state.last_frame_directional_lights[i].enabled ||
							sky_scene_state.directional_lights[i].size != sky_scene_state.last_frame_directional_lights[i].size) {
						light_data_dirty = true;
						break;
					}
				}
			}

			if (light_data_dirty) {
				RD::get_singleton()->buffer_update(sky_scene_state.directional_light_buffer, 0, sizeof(SkyDirectionalLightData) * sky_scene_state.max_directional_lights, sky_scene_state.directional_lights);

				SkyDirectionalLightData *temp = sky_scene_state.last_frame_directional_lights;
				sky_scene_state.last_frame_directional_lights = sky_scene_state.directional_lights;
				sky_scene_state.directional_lights = temp;
				sky_scene_state.last_frame_directional_light_count = sky_scene_state.ubo.directional_light_count;
				sky->reflection.dirty = true;
			}
		}
	}

	//setup fog variables
	sky_scene_state.ubo.volumetric_fog_enabled = false;
	if (p_render_buffers.is_valid()) {
		if (p_scene_render->render_buffers_has_volumetric_fog(p_render_buffers)) {
			sky_scene_state.ubo.volumetric_fog_enabled = true;

			float fog_end = p_scene_render->render_buffers_get_volumetric_fog_end(p_render_buffers);
			if (fog_end > 0.0) {
				sky_scene_state.ubo.volumetric_fog_inv_length = 1.0 / fog_end;
			} else {
				sky_scene_state.ubo.volumetric_fog_inv_length = 1.0;
			}

			float fog_detail_spread = p_scene_render->render_buffers_get_volumetric_fog_detail_spread(p_render_buffers); //reverse lookup
			if (fog_detail_spread > 0.0) {
				sky_scene_state.ubo.volumetric_fog_detail_spread = 1.0 / fog_detail_spread;
			} else {
				sky_scene_state.ubo.volumetric_fog_detail_spread = 1.0;
			}

			sky_scene_state.fog_uniform_set = p_scene_render->render_buffers_get_volumetric_fog_sky_uniform_set(p_render_buffers);
		}
	}

	sky_scene_state.ubo.z_far = p_projection.get_z_far();
	sky_scene_state.ubo.fog_enabled = p_env->fog_enabled;
	sky_scene_state.ubo.fog_density = p_env->fog_density;
	sky_scene_state.ubo.fog_aerial_perspective = p_env->fog_aerial_perspective;
	Color fog_color = p_env->fog_light_color.to_linear();
	float fog_energy = p_env->fog_light_energy;
	sky_scene_state.ubo.fog_light_color[0] = fog_color.r * fog_energy;
	sky_scene_state.ubo.fog_light_color[1] = fog_color.g * fog_energy;
	sky_scene_state.ubo.fog_light_color[2] = fog_color.b * fog_energy;
	sky_scene_state.ubo.fog_sun_scatter = p_env->fog_sun_scatter;

	RD::get_singleton()->buffer_update(sky_scene_state.uniform_buffer, 0, sizeof(SkySceneState::UBO), &sky_scene_state.ubo);
}

void RendererSceneSkyRD::update(RendererSceneEnvironmentRD *p_env, const CameraMatrix &p_projection, const Transform3D &p_transform, double p_time, float p_luminance_multiplier) {
	ERR_FAIL_COND(!p_env);

	Sky *sky = get_sky(p_env->sky);
	ERR_FAIL_COND(!sky);

	RID sky_material = sky_get_material(p_env->sky);

	SkyMaterialData *material = nullptr;

	if (sky_material.is_valid()) {
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
		if (!material || !material->shader_data->valid) {
			material = nullptr;
		}
	}

	if (!material) {
		sky_material = sky_shader.default_material;
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
	}

	ERR_FAIL_COND(!material);

	SkyShaderData *shader_data = material->shader_data;

	ERR_FAIL_COND(!shader_data);

	float multiplier = p_env->bg_energy;

	bool update_single_frame = sky->mode == RS::SKY_MODE_REALTIME || sky->mode == RS::SKY_MODE_QUALITY;
	RS::SkyMode sky_mode = sky->mode;

	if (sky_mode == RS::SKY_MODE_AUTOMATIC) {
		if (shader_data->uses_time || shader_data->uses_position) {
			update_single_frame = true;
			sky_mode = RS::SKY_MODE_REALTIME;
		} else if (shader_data->uses_light || shader_data->ubo_size > 0) {
			update_single_frame = false;
			sky_mode = RS::SKY_MODE_INCREMENTAL;
		} else {
			update_single_frame = true;
			sky_mode = RS::SKY_MODE_QUALITY;
		}
	}

	if (sky->processing_layer == 0 && sky_mode == RS::SKY_MODE_INCREMENTAL) {
		// On the first frame after creating sky, rebuild in single frame
		update_single_frame = true;
		sky_mode = RS::SKY_MODE_QUALITY;
	}

	int max_processing_layer = sky_use_cubemap_array ? sky->reflection.layers.size() : sky->reflection.layers[0].mipmaps.size();

	// Update radiance cubemap
	if (sky->reflection.dirty && (sky->processing_layer >= max_processing_layer || update_single_frame)) {
		static const Vector3 view_normals[6] = {
			Vector3(+1, 0, 0),
			Vector3(-1, 0, 0),
			Vector3(0, +1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, +1),
			Vector3(0, 0, -1)
		};
		static const Vector3 view_up[6] = {
			Vector3(0, -1, 0),
			Vector3(0, -1, 0),
			Vector3(0, 0, +1),
			Vector3(0, 0, -1),
			Vector3(0, -1, 0),
			Vector3(0, -1, 0)
		};

		CameraMatrix cm;
		cm.set_perspective(90, 1, 0.01, 10.0);
		CameraMatrix correction;
		correction.set_depth_correction(true);
		cm = correction * cm;

		if (shader_data->uses_quarter_res) {
			RD::get_singleton()->draw_command_begin_label("Render Sky to Quarter Res Cubemap");
			PipelineCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_CUBEMAP_QUARTER_RES];

			Vector<Color> clear_colors;
			clear_colors.push_back(Color(0.0, 0.0, 0.0));
			RD::DrawListID cubemap_draw_list;

			for (int i = 0; i < 6; i++) {
				Basis local_view = Basis::looking_at(view_normals[i], view_up[i]);
				RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_CUBEMAP_QUARTER_RES, sky_shader.default_shader_rd);

				cubemap_draw_list = RD::get_singleton()->draw_list_begin(sky->reflection.layers[0].mipmaps[2].framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
				_render_sky(cubemap_draw_list, p_time, sky->reflection.layers[0].mipmaps[2].framebuffers[i], pipeline, material->uniform_set, texture_uniform_set, 1, &cm, local_view, multiplier, p_transform.origin, p_luminance_multiplier);
				RD::get_singleton()->draw_list_end();
			}
			RD::get_singleton()->draw_command_end_label();
		}

		if (shader_data->uses_half_res) {
			RD::get_singleton()->draw_command_begin_label("Render Sky to Half Res Cubemap");
			PipelineCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_CUBEMAP_HALF_RES];

			Vector<Color> clear_colors;
			clear_colors.push_back(Color(0.0, 0.0, 0.0));
			RD::DrawListID cubemap_draw_list;

			for (int i = 0; i < 6; i++) {
				Basis local_view = Basis::looking_at(view_normals[i], view_up[i]);
				RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_CUBEMAP_HALF_RES, sky_shader.default_shader_rd);

				cubemap_draw_list = RD::get_singleton()->draw_list_begin(sky->reflection.layers[0].mipmaps[1].framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
				_render_sky(cubemap_draw_list, p_time, sky->reflection.layers[0].mipmaps[1].framebuffers[i], pipeline, material->uniform_set, texture_uniform_set, 1, &cm, local_view, multiplier, p_transform.origin, p_luminance_multiplier);
				RD::get_singleton()->draw_list_end();
			}
			RD::get_singleton()->draw_command_end_label();
		}

		RD::DrawListID cubemap_draw_list;
		PipelineCacheRD *pipeline = &shader_data->pipelines[SKY_VERSION_CUBEMAP];

		RD::get_singleton()->draw_command_begin_label("Render Sky Cubemap");
		for (int i = 0; i < 6; i++) {
			Basis local_view = Basis::looking_at(view_normals[i], view_up[i]);
			RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_CUBEMAP, sky_shader.default_shader_rd);

			cubemap_draw_list = RD::get_singleton()->draw_list_begin(sky->reflection.layers[0].mipmaps[0].framebuffers[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
			_render_sky(cubemap_draw_list, p_time, sky->reflection.layers[0].mipmaps[0].framebuffers[i], pipeline, material->uniform_set, texture_uniform_set, 1, &cm, local_view, multiplier, p_transform.origin, p_luminance_multiplier);
			RD::get_singleton()->draw_list_end();
		}
		RD::get_singleton()->draw_command_end_label();

		if (sky_mode == RS::SKY_MODE_REALTIME) {
			sky->reflection.create_reflection_fast_filter(storage, sky_use_cubemap_array);
			if (sky_use_cubemap_array) {
				sky->reflection.update_reflection_mipmaps(storage, 0, sky->reflection.layers.size());
			}
		} else {
			if (update_single_frame) {
				for (int i = 1; i < max_processing_layer; i++) {
					sky->reflection.create_reflection_importance_sample(storage, sky_use_cubemap_array, 10, i, sky_ggx_samples_quality);
				}
				if (sky_use_cubemap_array) {
					sky->reflection.update_reflection_mipmaps(storage, 0, sky->reflection.layers.size());
				}
			} else {
				if (sky_use_cubemap_array) {
					// Multi-Frame so just update the first array level
					sky->reflection.update_reflection_mipmaps(storage, 0, 1);
				}
			}
			sky->processing_layer = 1;
		}

		sky->reflection.dirty = false;

	} else {
		if (sky_mode == RS::SKY_MODE_INCREMENTAL && sky->processing_layer < max_processing_layer) {
			sky->reflection.create_reflection_importance_sample(storage, sky_use_cubemap_array, 10, sky->processing_layer, sky_ggx_samples_quality);

			if (sky_use_cubemap_array) {
				sky->reflection.update_reflection_mipmaps(storage, sky->processing_layer, sky->processing_layer + 1);
			}

			sky->processing_layer++;
		}
	}
}

void RendererSceneSkyRD::draw(RendererSceneEnvironmentRD *p_env, bool p_can_continue_color, bool p_can_continue_depth, RID p_fb, uint32_t p_view_count, const CameraMatrix *p_projections, const Transform3D &p_transform, double p_time) {
	ERR_FAIL_COND(!p_env);

	ERR_FAIL_COND(p_view_count == 0);
	ERR_FAIL_COND(p_view_count > RendererSceneRender::MAX_RENDER_VIEWS);

	Sky *sky = get_sky(p_env->sky);

	SkyMaterialData *material = nullptr;
	RID sky_material;

	RS::EnvironmentBG background = p_env->background;

	if (!(background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) || sky) {
		ERR_FAIL_COND(!sky);
		sky_material = sky_get_material(p_env->sky);

		if (sky_material.is_valid()) {
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
			if (!material || !material->shader_data->valid) {
				material = nullptr;
			}
		}

		if (!material) {
			sky_material = sky_shader.default_material;
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
		}
	}

	if (background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) {
		sky_material = sky_scene_state.fog_material;
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
	}

	ERR_FAIL_COND(!material);

	SkyShaderData *shader_data = material->shader_data;

	ERR_FAIL_COND(!shader_data);

	Basis sky_transform = p_env->sky_orientation;
	sky_transform.invert();

	float multiplier = p_env->bg_energy;
	float custom_fov = p_env->sky_custom_fov;

	// Camera
	CameraMatrix camera;
	uint32_t view_count = p_view_count;
	const CameraMatrix *projections = p_projections;

	if (custom_fov) {
		// With custom fov we don't support stereo...
		float near_plane = p_projections[0].get_z_near();
		float far_plane = p_projections[0].get_z_far();
		float aspect = p_projections[0].get_aspect();

		camera.set_perspective(custom_fov, aspect, near_plane, far_plane);

		view_count = 1;
		projections = &camera;
	}

	sky_transform = p_transform.basis * sky_transform;

	if (shader_data->uses_quarter_res) {
		PipelineCacheRD *pipeline = &shader_data->pipelines[view_count > 1 ? SKY_VERSION_QUARTER_RES_MULTIVIEW : SKY_VERSION_QUARTER_RES];

		RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_QUARTER_RES, sky_shader.default_shader_rd);

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0.0, 0.0, 0.0));

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(sky->quarter_res_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		_render_sky(draw_list, p_time, sky->quarter_res_framebuffer, pipeline, material->uniform_set, texture_uniform_set, view_count, projections, sky_transform, multiplier, p_transform.origin, 1.0);
		RD::get_singleton()->draw_list_end();
	}

	if (shader_data->uses_half_res) {
		PipelineCacheRD *pipeline = &shader_data->pipelines[view_count > 1 ? SKY_VERSION_HALF_RES_MULTIVIEW : SKY_VERSION_HALF_RES];

		RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_HALF_RES, sky_shader.default_shader_rd);

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0.0, 0.0, 0.0));

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(sky->half_res_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		_render_sky(draw_list, p_time, sky->half_res_framebuffer, pipeline, material->uniform_set, texture_uniform_set, view_count, projections, sky_transform, multiplier, p_transform.origin, 1.0);
		RD::get_singleton()->draw_list_end();
	}

	PipelineCacheRD *pipeline = &shader_data->pipelines[view_count > 1 ? SKY_VERSION_BACKGROUND_MULTIVIEW : SKY_VERSION_BACKGROUND];

	RID texture_uniform_set;
	if (sky) {
		texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_BACKGROUND, sky_shader.default_shader_rd);
	} else {
		texture_uniform_set = sky_scene_state.fog_only_texture_uniform_set;
	}

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_fb, RD::INITIAL_ACTION_CONTINUE, p_can_continue_color ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CONTINUE, p_can_continue_depth ? RD::FINAL_ACTION_CONTINUE : RD::FINAL_ACTION_READ);
	_render_sky(draw_list, p_time, p_fb, pipeline, material->uniform_set, texture_uniform_set, view_count, projections, sky_transform, multiplier, p_transform.origin, 1.0);
	RD::get_singleton()->draw_list_end();
}

void RendererSceneSkyRD::update_res_buffers(RendererSceneEnvironmentRD *p_env, uint32_t p_view_count, const CameraMatrix *p_projections, const Transform3D &p_transform, double p_time, float p_luminance_multiplier) {
	ERR_FAIL_COND(!p_env);

	ERR_FAIL_COND(p_view_count == 0);
	ERR_FAIL_COND(p_view_count > RendererSceneRender::MAX_RENDER_VIEWS);

	Sky *sky = get_sky(p_env->sky);
	ERR_FAIL_COND(!sky);

	SkyMaterialData *material = nullptr;
	RID sky_material;

	sky_material = sky_get_material(p_env->sky);

	if (sky_material.is_valid()) {
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
		if (!material || !material->shader_data->valid) {
			material = nullptr;
		}
	}

	if (!material) {
		sky_material = sky_shader.default_material;
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
	}

	ERR_FAIL_COND(!material);

	SkyShaderData *shader_data = material->shader_data;

	ERR_FAIL_COND(!shader_data);

	Basis sky_transform = p_env->sky_orientation;
	sky_transform.invert();

	float multiplier = p_env->bg_energy;
	float custom_fov = p_env->sky_custom_fov;

	// Camera
	CameraMatrix camera;
	uint32_t view_count = p_view_count;
	const CameraMatrix *projections = p_projections;

	if (custom_fov) {
		// With custom fov we don't support stereo...
		float near_plane = p_projections[0].get_z_near();
		float far_plane = p_projections[0].get_z_far();
		float aspect = p_projections[0].get_aspect();

		camera.set_perspective(custom_fov, aspect, near_plane, far_plane);

		view_count = 1;
		projections = &camera;
	}

	sky_transform = p_transform.basis * sky_transform;

	if (shader_data->uses_quarter_res) {
		PipelineCacheRD *pipeline = &shader_data->pipelines[view_count > 1 ? SKY_VERSION_QUARTER_RES_MULTIVIEW : SKY_VERSION_QUARTER_RES];

		RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_QUARTER_RES, sky_shader.default_shader_rd);

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0.0, 0.0, 0.0));

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(sky->quarter_res_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		_render_sky(draw_list, p_time, sky->quarter_res_framebuffer, pipeline, material->uniform_set, texture_uniform_set, view_count, projections, sky_transform, multiplier, p_transform.origin, p_luminance_multiplier);
		RD::get_singleton()->draw_list_end();
	}

	if (shader_data->uses_half_res) {
		PipelineCacheRD *pipeline = &shader_data->pipelines[view_count > 1 ? SKY_VERSION_HALF_RES_MULTIVIEW : SKY_VERSION_HALF_RES];

		RID texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_HALF_RES, sky_shader.default_shader_rd);

		Vector<Color> clear_colors;
		clear_colors.push_back(Color(0.0, 0.0, 0.0));

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(sky->half_res_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_DISCARD, clear_colors);
		_render_sky(draw_list, p_time, sky->half_res_framebuffer, pipeline, material->uniform_set, texture_uniform_set, view_count, projections, sky_transform, multiplier, p_transform.origin, p_luminance_multiplier);
		RD::get_singleton()->draw_list_end();
	}
}

void RendererSceneSkyRD::draw(RD::DrawListID p_draw_list, RendererSceneEnvironmentRD *p_env, RID p_fb, uint32_t p_view_count, const CameraMatrix *p_projections, const Transform3D &p_transform, double p_time, float p_luminance_multiplier) {
	ERR_FAIL_COND(!p_env);

	ERR_FAIL_COND(p_view_count == 0);
	ERR_FAIL_COND(p_view_count > RendererSceneRender::MAX_RENDER_VIEWS);

	Sky *sky = get_sky(p_env->sky);

	SkyMaterialData *material = nullptr;
	RID sky_material;

	RS::EnvironmentBG background = p_env->background;

	if (!(background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) || sky) {
		ERR_FAIL_COND(!sky);
		sky_material = sky_get_material(p_env->sky);

		if (sky_material.is_valid()) {
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
			if (!material || !material->shader_data->valid) {
				material = nullptr;
			}
		}

		if (!material) {
			sky_material = sky_shader.default_material;
			material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
		}
	}

	if (background == RS::ENV_BG_CLEAR_COLOR || background == RS::ENV_BG_COLOR) {
		sky_material = sky_scene_state.fog_material;
		material = (SkyMaterialData *)storage->material_get_data(sky_material, RendererStorageRD::SHADER_TYPE_SKY);
	}

	ERR_FAIL_COND(!material);

	SkyShaderData *shader_data = material->shader_data;

	ERR_FAIL_COND(!shader_data);

	Basis sky_transform = p_env->sky_orientation;
	sky_transform.invert();

	float multiplier = p_env->bg_energy;
	float custom_fov = p_env->sky_custom_fov;

	// Camera
	CameraMatrix camera;
	uint32_t view_count = p_view_count;
	const CameraMatrix *projections = p_projections;

	if (custom_fov) {
		// With custom fov we don't support stereo...
		float near_plane = p_projections[0].get_z_near();
		float far_plane = p_projections[0].get_z_far();
		float aspect = p_projections[0].get_aspect();

		camera.set_perspective(custom_fov, aspect, near_plane, far_plane);

		view_count = 1;
		projections = &camera;
	}

	sky_transform = p_transform.basis * sky_transform;

	PipelineCacheRD *pipeline = &shader_data->pipelines[view_count > 1 ? SKY_VERSION_BACKGROUND_MULTIVIEW : SKY_VERSION_BACKGROUND];

	RID texture_uniform_set;
	if (sky) {
		texture_uniform_set = sky->get_textures(storage, SKY_TEXTURE_SET_BACKGROUND, sky_shader.default_shader_rd);
	} else {
		texture_uniform_set = sky_scene_state.fog_only_texture_uniform_set;
	}

	_render_sky(p_draw_list, p_time, p_fb, pipeline, material->uniform_set, texture_uniform_set, view_count, projections, sky_transform, multiplier, p_transform.origin, p_luminance_multiplier);
}

void RendererSceneSkyRD::invalidate_sky(Sky *p_sky) {
	if (!p_sky->dirty) {
		p_sky->dirty = true;
		p_sky->dirty_list = dirty_sky_list;
		dirty_sky_list = p_sky;
	}
}

void RendererSceneSkyRD::update_dirty_skys() {
	Sky *sky = dirty_sky_list;

	while (sky) {
		bool texture_set_dirty = false;
		//update sky configuration if texture is missing

		if (sky->radiance.is_null()) {
			int mipmaps = Image::get_image_required_mipmaps(sky->radiance_size, sky->radiance_size, Image::FORMAT_RGBAH) + 1;

			uint32_t w = sky->radiance_size, h = sky->radiance_size;
			int layers = roughness_layers;
			if (sky->mode == RS::SKY_MODE_REALTIME) {
				layers = 8;
				if (roughness_layers != 8) {
					WARN_PRINT("When using REALTIME skies, roughness_layers should be set to 8 in the project settings for best quality reflections");
				}
			}

			if (sky_use_cubemap_array) {
				//array (higher quality, 6 times more memory)
				RD::TextureFormat tf;
				tf.array_layers = layers * 6;
				tf.format = texture_format;
				tf.texture_type = RD::TEXTURE_TYPE_CUBE_ARRAY;
				tf.mipmaps = mipmaps;
				tf.width = w;
				tf.height = h;
				tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

				sky->radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

				sky->reflection.update_reflection_data(storage, sky->radiance_size, mipmaps, true, sky->radiance, 0, sky->mode == RS::SKY_MODE_REALTIME, roughness_layers, texture_format);

			} else {
				//regular cubemap, lower quality (aliasing, less memory)
				RD::TextureFormat tf;
				tf.array_layers = 6;
				tf.format = texture_format;
				tf.texture_type = RD::TEXTURE_TYPE_CUBE;
				tf.mipmaps = MIN(mipmaps, layers);
				tf.width = w;
				tf.height = h;
				tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

				sky->radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

				sky->reflection.update_reflection_data(storage, sky->radiance_size, MIN(mipmaps, layers), false, sky->radiance, 0, sky->mode == RS::SKY_MODE_REALTIME, roughness_layers, texture_format);
			}
			texture_set_dirty = true;
		}

		// Create subpass buffers if they haven't been created already
		if (sky->half_res_pass.is_null() && !RD::get_singleton()->texture_is_valid(sky->half_res_pass) && sky->screen_size.x >= 4 && sky->screen_size.y >= 4) {
			RD::TextureFormat tformat;
			tformat.format = texture_format;
			tformat.width = sky->screen_size.x / 2;
			tformat.height = sky->screen_size.y / 2;
			tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			tformat.texture_type = RD::TEXTURE_TYPE_2D;

			sky->half_res_pass = RD::get_singleton()->texture_create(tformat, RD::TextureView());
			Vector<RID> texs;
			texs.push_back(sky->half_res_pass);
			sky->half_res_framebuffer = RD::get_singleton()->framebuffer_create(texs);
			texture_set_dirty = true;
		}

		if (sky->quarter_res_pass.is_null() && !RD::get_singleton()->texture_is_valid(sky->quarter_res_pass) && sky->screen_size.x >= 4 && sky->screen_size.y >= 4) {
			RD::TextureFormat tformat;
			tformat.format = texture_format;
			tformat.width = sky->screen_size.x / 4;
			tformat.height = sky->screen_size.y / 4;
			tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			tformat.texture_type = RD::TEXTURE_TYPE_2D;

			sky->quarter_res_pass = RD::get_singleton()->texture_create(tformat, RD::TextureView());
			Vector<RID> texs;
			texs.push_back(sky->quarter_res_pass);
			sky->quarter_res_framebuffer = RD::get_singleton()->framebuffer_create(texs);
			texture_set_dirty = true;
		}

		if (texture_set_dirty) {
			for (int i = 0; i < SKY_TEXTURE_SET_MAX; i++) {
				if (sky->texture_uniform_sets[i].is_valid() && RD::get_singleton()->uniform_set_is_valid(sky->texture_uniform_sets[i])) {
					RD::get_singleton()->free(sky->texture_uniform_sets[i]);
					sky->texture_uniform_sets[i] = RID();
				}
			}
		}

		sky->reflection.dirty = true;
		sky->processing_layer = 0;

		Sky *next = sky->dirty_list;
		sky->dirty_list = nullptr;
		sky->dirty = false;
		sky = next;
	}

	dirty_sky_list = nullptr;
}

RID RendererSceneSkyRD::sky_get_material(RID p_sky) const {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	return sky->material;
}

RID RendererSceneSkyRD::allocate_sky_rid() {
	return sky_owner.allocate_rid();
}

void RendererSceneSkyRD::initialize_sky_rid(RID p_rid) {
	sky_owner.initialize_rid(p_rid, Sky());
}

RendererSceneSkyRD::Sky *RendererSceneSkyRD::get_sky(RID p_sky) const {
	return sky_owner.get_or_null(p_sky);
}

void RendererSceneSkyRD::free_sky(RID p_sky) {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND(!sky);

	sky->free(storage);
	sky_owner.free(p_sky);
}

void RendererSceneSkyRD::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->set_radiance_size(p_radiance_size)) {
		invalidate_sky(sky);
	}
}

void RendererSceneSkyRD::sky_set_mode(RID p_sky, RS::SkyMode p_mode) {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->set_mode(p_mode)) {
		invalidate_sky(sky);
	}
}

void RendererSceneSkyRD::sky_set_material(RID p_sky, RID p_material) {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->set_material(p_material)) {
		invalidate_sky(sky);
	}
}

Ref<Image> RendererSceneSkyRD::sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND_V(!sky, Ref<Image>());

	update_dirty_skys();

	return sky->bake_panorama(storage, p_energy, p_bake_irradiance ? roughness_layers : 0, p_size);
}

RID RendererSceneSkyRD::sky_get_radiance_texture_rd(RID p_sky) const {
	Sky *sky = get_sky(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	return sky->radiance;
}

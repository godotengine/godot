/*************************************************************************/
/*  scene_shader_forward_clustered.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RSSR_SCENE_SHADER_FC_H
#define RSSR_SCENE_SHADER_FC_H

#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"
#include "servers/rendering/renderer_rd/shaders/scene_forward_clustered.glsl.gen.h"

namespace RendererSceneRenderImplementation {

class SceneShaderForwardClustered {
private:
	static SceneShaderForwardClustered *singleton;

public:
	RendererStorageRD *storage;

	enum ShaderVersion {
		SHADER_VERSION_DEPTH_PASS,
		SHADER_VERSION_DEPTH_PASS_DP,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_GIPROBE,
		SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL,
		SHADER_VERSION_DEPTH_PASS_WITH_SDF,
		SHADER_VERSION_COLOR_PASS,
		SHADER_VERSION_COLOR_PASS_WITH_FORWARD_GI,
		SHADER_VERSION_COLOR_PASS_WITH_SEPARATE_SPECULAR,
		SHADER_VERSION_LIGHTMAP_COLOR_PASS,
		SHADER_VERSION_LIGHTMAP_COLOR_PASS_WITH_SEPARATE_SPECULAR,
		SHADER_VERSION_MAX
	};

	struct ShaderData : public RendererStorageRD::ShaderData {
		enum BlendMode { //used internally
			BLEND_MODE_MIX,
			BLEND_MODE_ADD,
			BLEND_MODE_SUB,
			BLEND_MODE_MUL,
			BLEND_MODE_ALPHA_TO_COVERAGE
		};

		enum DepthDraw {
			DEPTH_DRAW_DISABLED,
			DEPTH_DRAW_OPAQUE,
			DEPTH_DRAW_ALWAYS
		};

		enum DepthTest {
			DEPTH_TEST_DISABLED,
			DEPTH_TEST_ENABLED
		};

		enum Cull {
			CULL_DISABLED,
			CULL_FRONT,
			CULL_BACK
		};

		enum CullVariant {
			CULL_VARIANT_NORMAL,
			CULL_VARIANT_REVERSED,
			CULL_VARIANT_DOUBLE_SIDED,
			CULL_VARIANT_MAX

		};

		enum AlphaAntiAliasing {
			ALPHA_ANTIALIASING_OFF,
			ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE,
			ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE
		};

		bool valid;
		RID version;
		uint32_t vertex_input_mask;
		PipelineCacheRD pipelines[CULL_VARIANT_MAX][RS::PRIMITIVE_MAX][SHADER_VERSION_MAX];

		String path;

		Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;
		Vector<ShaderCompilerRD::GeneratedCode::Texture> texture_uniforms;

		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size;

		String code;
		Map<StringName, RID> default_texture_params;

		DepthDraw depth_draw;
		DepthTest depth_test;

		bool uses_point_size;
		bool uses_alpha;
		bool uses_blend_alpha;
		bool uses_alpha_clip;
		bool uses_depth_pre_pass;
		bool uses_discard;
		bool uses_roughness;
		bool uses_normal;
		bool uses_particle_trails;

		bool unshaded;
		bool uses_vertex;
		bool uses_sss;
		bool uses_transmittance;
		bool uses_screen_texture;
		bool uses_depth_texture;
		bool uses_normal_texture;
		bool uses_time;
		bool writes_modelview_or_projection;
		bool uses_world_coordinates;

		uint64_t last_pass = 0;
		uint32_t index = 0;

		virtual void set_code(const String &p_Code);
		virtual void set_default_texture_param(const StringName &p_name, RID p_texture);
		virtual void get_param_list(List<PropertyInfo> *p_param_list) const;
		void get_instance_param_list(List<RendererStorage::InstanceShaderParam> *p_param_list) const;

		virtual bool is_param_texture(const StringName &p_param) const;
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual Variant get_default_parameter(const StringName &p_parameter) const;
		virtual RS::ShaderNativeSourceCode get_native_source_code() const;

		ShaderData();
		virtual ~ShaderData();
	};

	RendererStorageRD::ShaderData *_create_shader_func();
	static RendererStorageRD::ShaderData *_create_shader_funcs() {
		return static_cast<SceneShaderForwardClustered *>(singleton)->_create_shader_func();
	}

	struct MaterialData : public RendererStorageRD::MaterialData {
		uint64_t last_frame;
		ShaderData *shader_data;
		RID uniform_buffer;
		RID uniform_set;
		Vector<RID> texture_cache;
		Vector<uint8_t> ubo_data;
		uint64_t last_pass = 0;
		uint32_t index = 0;
		RID next_pass;
		uint8_t priority;
		virtual void set_render_priority(int p_priority);
		virtual void set_next_pass(RID p_pass);
		virtual void update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);
		virtual ~MaterialData();
	};

	RendererStorageRD::MaterialData *_create_material_func(ShaderData *p_shader);
	static RendererStorageRD::MaterialData *_create_material_funcs(RendererStorageRD::ShaderData *p_shader) {
		return static_cast<SceneShaderForwardClustered *>(singleton)->_create_material_func(static_cast<ShaderData *>(p_shader));
	}

	SceneForwardClusteredShaderRD shader;
	ShaderCompilerRD compiler;

	RID default_shader;
	RID default_material;
	RID overdraw_material_shader;
	RID overdraw_material;
	RID wireframe_material_shader;
	RID wireframe_material;
	RID default_shader_rd;
	RID default_shader_sdfgi_rd;

	RID default_vec4_xform_buffer;
	RID default_vec4_xform_uniform_set;

	RID shadow_sampler;

	SceneShaderForwardClustered();
	~SceneShaderForwardClustered();

	void init(RendererStorageRD *p_storage, const String p_defines);
};

} // namespace RendererSceneRenderImplementation
#endif // !RSSR_SCENE_SHADER_FM_H

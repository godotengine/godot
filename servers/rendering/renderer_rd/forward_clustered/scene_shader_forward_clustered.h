/**************************************************************************/
/*  scene_shader_forward_clustered.h                                      */
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

#ifndef SCENE_SHADER_FORWARD_CLUSTERED_H
#define SCENE_SHADER_FORWARD_CLUSTERED_H

#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered.glsl.gen.h"

namespace RendererSceneRenderImplementation {

class SceneShaderForwardClustered {
private:
	static SceneShaderForwardClustered *singleton;

public:
	enum ShaderGroup {
		SHADER_GROUP_BASE, // Always compiled at the beginning.
		SHADER_GROUP_ADVANCED,
		SHADER_GROUP_MULTIVIEW,
		SHADER_GROUP_ADVANCED_MULTIVIEW,
	};

	enum ShaderVersion {
		SHADER_VERSION_DEPTH_PASS,
		SHADER_VERSION_DEPTH_PASS_DP,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI,
		SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL,
		SHADER_VERSION_DEPTH_PASS_WITH_SDF,
		SHADER_VERSION_DEPTH_PASS_MULTIVIEW,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_MULTIVIEW,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI_MULTIVIEW,
		SHADER_VERSION_COLOR_PASS,
		SHADER_VERSION_MAX
	};

	enum ShaderColorPassFlags {
		SHADER_COLOR_PASS_FLAG_SEPARATE_SPECULAR = 1 << 0,
		SHADER_COLOR_PASS_FLAG_LIGHTMAP = 1 << 1,
		SHADER_COLOR_PASS_FLAG_MULTIVIEW = 1 << 2,
		SHADER_COLOR_PASS_FLAG_MOTION_VECTORS = 1 << 3,
		SHADER_COLOR_PASS_FLAG_COUNT = 1 << 4
	};

	enum PipelineVersion {
		PIPELINE_VERSION_DEPTH_PASS,
		PIPELINE_VERSION_DEPTH_PASS_DP,
		PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS,
		PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI,
		PIPELINE_VERSION_DEPTH_PASS_WITH_MATERIAL,
		PIPELINE_VERSION_DEPTH_PASS_WITH_SDF,
		PIPELINE_VERSION_DEPTH_PASS_MULTIVIEW,
		PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_MULTIVIEW,
		PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI_MULTIVIEW,
		PIPELINE_VERSION_COLOR_PASS,
		PIPELINE_VERSION_MAX
	};

	enum PipelineColorPassFlags {
		PIPELINE_COLOR_PASS_FLAG_TRANSPARENT = 1 << 0, // Can't combine with SEPARATE_SPECULAR.
		PIPELINE_COLOR_PASS_FLAG_SEPARATE_SPECULAR = 1 << 1, // Can't combine with TRANSPARENT.
		PIPELINE_COLOR_PASS_FLAG_LIGHTMAP = 1 << 2,
		PIPELINE_COLOR_PASS_FLAG_MULTIVIEW = 1 << 3,
		PIPELINE_COLOR_PASS_FLAG_MOTION_VECTORS = 1 << 4,
		PIPELINE_COLOR_PASS_FLAG_COUNT = 1 << 5,
	};

	enum ShaderSpecializations {
		SHADER_SPECIALIZATION_FORWARD_GI = 1 << 0,
		SHADER_SPECIALIZATION_PROJECTOR = 1 << 1,
		SHADER_SPECIALIZATION_SOFT_SHADOWS = 1 << 2,
		SHADER_SPECIALIZATION_DIRECTIONAL_SOFT_SHADOWS = 1 << 3,
	};

	struct ShaderData : public RendererRD::MaterialStorage::ShaderData {
		enum BlendMode { //used internally
			BLEND_MODE_MIX,
			BLEND_MODE_ADD,
			BLEND_MODE_SUB,
			BLEND_MODE_MUL,
			BLEND_MODE_ALPHA_TO_COVERAGE,
			BLEND_MODE_PREMULT_ALPHA,
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

		bool valid = false;
		RID version;
		uint64_t vertex_input_mask = 0;
		PipelineCacheRD pipelines[CULL_VARIANT_MAX][RS::PRIMITIVE_MAX][PIPELINE_VERSION_MAX];
		PipelineCacheRD color_pipelines[CULL_VARIANT_MAX][RS::PRIMITIVE_MAX][PIPELINE_COLOR_PASS_FLAG_COUNT];

		Vector<ShaderCompiler::GeneratedCode::Texture> texture_uniforms;

		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size = 0;

		String code;

		DepthDraw depth_draw = DEPTH_DRAW_OPAQUE;
		DepthTest depth_test = DEPTH_TEST_ENABLED;

		bool uses_point_size = false;
		bool uses_alpha = false;
		bool uses_blend_alpha = false;
		bool uses_alpha_clip = false;
		bool uses_alpha_antialiasing = false;
		bool uses_depth_prepass_alpha = false;
		bool uses_discard = false;
		bool uses_roughness = false;
		bool uses_normal = false;
		bool uses_tangent = false;
		bool uses_particle_trails = false;

		bool unshaded = false;
		bool uses_vertex = false;
		bool uses_position = false;
		bool uses_sss = false;
		bool uses_transmittance = false;
		bool uses_screen_texture = false;
		bool uses_depth_texture = false;
		bool uses_normal_texture = false;
		bool uses_time = false;
		bool uses_vertex_time = false;
		bool uses_fragment_time = false;
		bool writes_modelview_or_projection = false;
		bool uses_world_coordinates = false;
		bool uses_screen_texture_mipmaps = false;
		Cull cull_mode = CULL_DISABLED;

		uint64_t last_pass = 0;
		uint32_t index = 0;

		virtual void set_code(const String &p_Code);

		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual RS::ShaderNativeSourceCode get_native_source_code() const;

		SelfList<ShaderData> shader_list_element;
		ShaderData();
		virtual ~ShaderData();
	};

	SelfList<ShaderData>::List shader_list;

	RendererRD::MaterialStorage::ShaderData *_create_shader_func();
	static RendererRD::MaterialStorage::ShaderData *_create_shader_funcs() {
		return static_cast<SceneShaderForwardClustered *>(singleton)->_create_shader_func();
	}

	struct MaterialData : public RendererRD::MaterialStorage::MaterialData {
		ShaderData *shader_data = nullptr;
		RID uniform_set;
		uint64_t last_pass = 0;
		uint32_t index = 0;
		RID next_pass;
		uint8_t priority;
		virtual void set_render_priority(int p_priority);
		virtual void set_next_pass(RID p_pass);
		virtual bool update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);
		virtual ~MaterialData();
	};

	RendererRD::MaterialStorage::MaterialData *_create_material_func(ShaderData *p_shader);
	static RendererRD::MaterialStorage::MaterialData *_create_material_funcs(RendererRD::MaterialStorage::ShaderData *p_shader) {
		return static_cast<SceneShaderForwardClustered *>(singleton)->_create_material_func(static_cast<ShaderData *>(p_shader));
	}

	SceneForwardClusteredShaderRD shader;
	ShaderCompiler compiler;

	RID default_shader;
	RID default_material;
	RID overdraw_material_shader;
	RID overdraw_material;
	RID debug_shadow_splits_material_shader;
	RID debug_shadow_splits_material;
	RID default_shader_rd;
	RID default_shader_sdfgi_rd;

	RID default_vec4_xform_buffer;
	RID default_vec4_xform_uniform_set;

	RID shadow_sampler;

	RID default_material_uniform_set;
	ShaderData *default_material_shader_ptr = nullptr;

	RID overdraw_material_uniform_set;
	ShaderData *overdraw_material_shader_ptr = nullptr;

	RID debug_shadow_splits_material_uniform_set;
	ShaderData *debug_shadow_splits_material_shader_ptr = nullptr;

	Vector<RD::PipelineSpecializationConstant> default_specialization_constants;
	bool valid_color_pass_pipelines[PIPELINE_COLOR_PASS_FLAG_COUNT];
	SceneShaderForwardClustered();
	~SceneShaderForwardClustered();

	void init(const String p_defines);
	void set_default_specialization_constants(const Vector<RD::PipelineSpecializationConstant> &p_constants);
	void enable_advanced_shader_group(bool p_needs_multiview = false);
};

} // namespace RendererSceneRenderImplementation

#endif // SCENE_SHADER_FORWARD_CLUSTERED_H

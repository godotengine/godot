/*************************************************************************/
/*  rasterizer_scene_high_end_rd.h                                       */
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

#ifndef RASTERIZER_SCENE_HIGHEND_RD_H
#define RASTERIZER_SCENE_HIGHEND_RD_H

#include "servers/rendering/rasterizer_rd/rasterizer_scene_rd.h"
#include "servers/rendering/rasterizer_rd/rasterizer_storage_rd.h"
#include "servers/rendering/rasterizer_rd/render_pipeline_vertex_format_cache_rd.h"
#include "servers/rendering/rasterizer_rd/shaders/scene_high_end.glsl.gen.h"

class RasterizerSceneHighEndRD : public RasterizerSceneRD {
	enum {
		SCENE_UNIFORM_SET = 0,
		RADIANCE_UNIFORM_SET = 1,
		VIEW_DEPENDANT_UNIFORM_SET = 2,
		RENDER_BUFFERS_UNIFORM_SET = 3,
		TRANSFORMS_UNIFORM_SET = 4,
		MATERIAL_UNIFORM_SET = 5
	};

	enum {
		SDFGI_MAX_CASCADES = 8,
		MAX_GI_PROBES = 8
	};

	/* Scene Shader */

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

	struct {
		SceneHighEndShaderRD scene_shader;
		ShaderCompilerRD compiler;
	} shader;

	RasterizerStorageRD *storage;

	/* Material */

	struct ShaderData : public RasterizerStorageRD::ShaderData {
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
		RenderPipelineVertexFormatCacheRD pipelines[CULL_VARIANT_MAX][RS::PRIMITIVE_MAX][SHADER_VERSION_MAX];

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
		void get_instance_param_list(List<RasterizerStorage::InstanceShaderParam> *p_param_list) const;

		virtual bool is_param_texture(const StringName &p_param) const;
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual Variant get_default_parameter(const StringName &p_parameter) const;
		ShaderData();
		virtual ~ShaderData();
	};

	RasterizerStorageRD::ShaderData *_create_shader_func();
	static RasterizerStorageRD::ShaderData *_create_shader_funcs() {
		return static_cast<RasterizerSceneHighEndRD *>(singleton)->_create_shader_func();
	}

	struct MaterialData : public RasterizerStorageRD::MaterialData {
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

	RasterizerStorageRD::MaterialData *_create_material_func(ShaderData *p_shader);
	static RasterizerStorageRD::MaterialData *_create_material_funcs(RasterizerStorageRD::ShaderData *p_shader) {
		return static_cast<RasterizerSceneHighEndRD *>(singleton)->_create_material_func(static_cast<ShaderData *>(p_shader));
	}

	/* Push Constant */

	struct PushConstant {
		uint32_t index;
		uint32_t pad;
		float bake_uv2_offset[2];
	};

	/* Framebuffer */

	struct RenderBufferDataHighEnd : public RenderBufferData {
		//for rendering, may be MSAAd

		RID color;
		RID depth;
		RID specular;
		RID normal_roughness_buffer;
		RID giprobe_buffer;

		RID ambient_buffer;
		RID reflection_buffer;

		RS::ViewportMSAA msaa;
		RD::TextureSamples texture_samples;

		RID color_msaa;
		RID depth_msaa;
		RID specular_msaa;
		RID normal_roughness_buffer_msaa;
		RID roughness_buffer_msaa;
		RID giprobe_buffer_msaa;

		RID depth_fb;
		RID depth_normal_roughness_fb;
		RID depth_normal_roughness_giprobe_fb;
		RID color_fb;
		RID color_specular_fb;
		RID specular_only_fb;
		int width, height;

		RID render_sdfgi_uniform_set;
		void ensure_specular();
		void ensure_gi();
		void ensure_giprobe();
		void clear();
		virtual void configure(RID p_color_buffer, RID p_depth_buffer, int p_width, int p_height, RS::ViewportMSAA p_msaa);

		RID uniform_set;

		~RenderBufferDataHighEnd();
	};

	virtual RenderBufferData *_create_render_buffer_data();
	void _allocate_normal_roughness_texture(RenderBufferDataHighEnd *rb);

	RID shadow_sampler;
	RID render_base_uniform_set;
	RID view_dependant_uniform_set;

	uint64_t lightmap_texture_array_version = 0xFFFFFFFF;

	virtual void _base_uniforms_changed();
	void _render_buffers_clear_uniform_set(RenderBufferDataHighEnd *rb);
	virtual void _render_buffers_uniform_set_changed(RID p_render_buffers);
	virtual RID _render_buffers_get_normal_texture(RID p_render_buffers);
	virtual RID _render_buffers_get_ambient_texture(RID p_render_buffers);
	virtual RID _render_buffers_get_reflection_texture(RID p_render_buffers);

	void _update_render_base_uniform_set();
	void _setup_view_dependant_uniform_set(RID p_shadow_atlas, RID p_reflection_atlas, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count);
	void _update_render_buffers_uniform_set(RID p_render_buffers);

	struct LightmapData {
		float normal_xform[12];
	};

	struct LightmapCaptureData {
		float sh[9 * 4];
	};

	enum {
		INSTANCE_DATA_FLAG_USE_GI_BUFFERS = 1 << 6,
		INSTANCE_DATA_FLAG_USE_SDFGI = 1 << 7,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP_CAPTURE = 1 << 8,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP = 1 << 9,
		INSTANCE_DATA_FLAG_USE_SH_LIGHTMAP = 1 << 10,
		INSTANCE_DATA_FLAG_USE_GIPROBE = 1 << 11,
		INSTANCE_DATA_FLAG_MULTIMESH = 1 << 12,
		INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D = 1 << 13,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR = 1 << 14,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA = 1 << 15,
		INSTANCE_DATA_FLAGS_MULTIMESH_STRIDE_SHIFT = 16,
		INSTANCE_DATA_FLAGS_MULTIMESH_STRIDE_MASK = 0x7,
		INSTANCE_DATA_FLAG_SKELETON = 1 << 19,
	};

	struct InstanceData {
		float transform[16];
		float normal_transform[16];
		uint32_t flags;
		uint32_t instance_uniforms_ofs; //instance_offset in instancing/skeleton buffer
		uint32_t gi_offset; //GI information when using lightmapping (VCT or lightmap)
		uint32_t mask;
		float lightmap_uv_scale[4];
	};

	struct SceneState {
		struct UBO {
			float projection_matrix[16];
			float inv_projection_matrix[16];

			float camera_matrix[16];
			float inv_camera_matrix[16];

			float viewport_size[2];
			float screen_pixel_size[2];

			float directional_penumbra_shadow_kernel[128]; //32 vec4s
			float directional_soft_shadow_kernel[128];
			float penumbra_shadow_kernel[128];
			float soft_shadow_kernel[128];

			uint32_t directional_penumbra_shadow_samples;
			uint32_t directional_soft_shadow_samples;
			uint32_t penumbra_shadow_samples;
			uint32_t soft_shadow_samples;

			float ambient_light_color_energy[4];

			float ambient_color_sky_mix;
			uint32_t use_ambient_light;
			uint32_t use_ambient_cubemap;
			uint32_t use_reflection_cubemap;

			float radiance_inverse_xform[12];

			float shadow_atlas_pixel_size[2];
			float directional_shadow_pixel_size[2];

			uint32_t directional_light_count;
			float dual_paraboloid_side;
			float z_far;
			float z_near;

			uint32_t ssao_enabled;
			float ssao_light_affect;
			float ssao_ao_affect;
			uint32_t roughness_limiter_enabled;

			float roughness_limiter_amount;
			float roughness_limiter_limit;
			uint32_t roughness_limiter_pad[2];

			float ao_color[4];

			float sdf_to_bounds[16];

			int32_t sdf_offset[3];
			uint32_t material_uv2_mode;

			int32_t sdf_size[3];
			uint32_t gi_upscale_for_msaa;

			uint32_t volumetric_fog_enabled;
			float volumetric_fog_inv_length;
			float volumetric_fog_detail_spread;
			uint32_t volumetric_fog_pad;

			// Fog
			uint32_t fog_enabled;
			float fog_density;
			float fog_height;
			float fog_height_density;

			float fog_light_color[3];
			float fog_sun_scatter;

			float fog_aerial_perspective;

			float time;
			float reflection_multiplier;

			uint32_t pancake_shadows;
		};

		UBO ubo;

		RID uniform_buffer;

		LightmapData *lightmaps;
		uint32_t max_lightmaps;
		RID lightmap_buffer;

		LightmapCaptureData *lightmap_captures;
		uint32_t max_lightmap_captures;
		RID lightmap_capture_buffer;

		RID instance_buffer;
		InstanceData *instances;
		uint32_t max_instances;

		bool used_screen_texture = false;
		bool used_normal_texture = false;
		bool used_depth_texture = false;
		bool used_sss = false;
		uint32_t current_shader_index = 0;
		uint32_t current_material_index = 0;

	} scene_state;

	/* Render List */

	struct RenderList {
		int max_elements;

		struct Element {
			RasterizerScene::InstanceBase *instance;
			MaterialData *material;
			union {
				struct {
					//from least significant to most significant in sort, TODO: should be endian swapped on big endian
					uint64_t geometry_index : 20;
					uint64_t material_index : 15;
					uint64_t shader_index : 12;
					uint64_t uses_instancing : 1;
					uint64_t uses_forward_gi : 1;
					uint64_t uses_lightmap : 1;
					uint64_t depth_layer : 4;
					uint64_t priority : 8;
				};

				uint64_t sort_key;
			};
			uint32_t surface_index;
		};

		Element *base_elements;
		Element **elements;

		int element_count;
		int alpha_element_count;

		void clear() {
			element_count = 0;
			alpha_element_count = 0;
		}

		//should eventually be replaced by radix

		struct SortByKey {
			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {
				return A->sort_key < B->sort_key;
			}
		};

		void sort_by_key(bool p_alpha) {
			SortArray<Element *, SortByKey> sorter;
			if (p_alpha) {
				sorter.sort(&elements[max_elements - alpha_element_count], alpha_element_count);
			} else {
				sorter.sort(elements, element_count);
			}
		}

		struct SortByDepth {
			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {
				return A->instance->depth < B->instance->depth;
			}
		};

		void sort_by_depth(bool p_alpha) { //used for shadows

			SortArray<Element *, SortByDepth> sorter;
			if (p_alpha) {
				sorter.sort(&elements[max_elements - alpha_element_count], alpha_element_count);
			} else {
				sorter.sort(elements, element_count);
			}
		}

		struct SortByReverseDepthAndPriority {
			_FORCE_INLINE_ bool operator()(const Element *A, const Element *B) const {
				uint32_t layer_A = uint32_t(A->priority);
				uint32_t layer_B = uint32_t(B->priority);
				if (layer_A == layer_B) {
					return A->instance->depth > B->instance->depth;
				} else {
					return layer_A < layer_B;
				}
			}
		};

		void sort_by_reverse_depth_and_priority(bool p_alpha) { //used for alpha

			SortArray<Element *, SortByReverseDepthAndPriority> sorter;
			if (p_alpha) {
				sorter.sort(&elements[max_elements - alpha_element_count], alpha_element_count);
			} else {
				sorter.sort(elements, element_count);
			}
		}

		_FORCE_INLINE_ Element *add_element() {
			if (element_count + alpha_element_count >= max_elements) {
				return nullptr;
			}
			elements[element_count] = &base_elements[element_count];
			return elements[element_count++];
		}

		_FORCE_INLINE_ Element *add_alpha_element() {
			if (element_count + alpha_element_count >= max_elements) {
				return nullptr;
			}
			int idx = max_elements - alpha_element_count - 1;
			elements[idx] = &base_elements[idx];
			alpha_element_count++;
			return elements[idx];
		}

		void init() {
			element_count = 0;
			alpha_element_count = 0;
			elements = memnew_arr(Element *, max_elements);
			base_elements = memnew_arr(Element, max_elements);
			for (int i = 0; i < max_elements; i++) {
				elements[i] = &base_elements[i]; // assign elements
			}
		}

		RenderList() {
			max_elements = 0;
		}

		~RenderList() {
			memdelete_arr(elements);
			memdelete_arr(base_elements);
		}
	};

	RenderList render_list;

	static RasterizerSceneHighEndRD *singleton;
	uint64_t render_pass;
	double time;
	RID default_shader;
	RID default_material;
	RID overdraw_material_shader;
	RID overdraw_material;
	RID wireframe_material_shader;
	RID wireframe_material;
	RID default_shader_rd;
	RID default_shader_sdfgi_rd;
	RID default_radiance_uniform_set;
	RID default_render_buffers_uniform_set;

	RID default_vec4_xform_buffer;
	RID default_vec4_xform_uniform_set;

	enum PassMode {
		PASS_MODE_COLOR,
		PASS_MODE_COLOR_SPECULAR,
		PASS_MODE_COLOR_TRANSPARENT,
		PASS_MODE_SHADOW,
		PASS_MODE_SHADOW_DP,
		PASS_MODE_DEPTH,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS_GIPROBE,
		PASS_MODE_DEPTH_MATERIAL,
		PASS_MODE_SDF,
	};

	void _setup_environment(RID p_environment, RID p_render_buffers, const CameraMatrix &p_cam_projection, const Transform &p_cam_transform, RID p_reflection_probe, bool p_no_fog, const Size2 &p_screen_pixel_size, RID p_shadow_atlas, bool p_flip_y, const Color &p_default_bg_color, float p_znear, float p_zfar, bool p_opaque_render_buffers = false, bool p_pancake_shadows = false);
	void _setup_lightmaps(InstanceBase **p_lightmap_cull_result, int p_lightmap_cull_count, const Transform &p_cam_transform);

	void _fill_instances(RenderList::Element **p_elements, int p_element_count, bool p_for_depth, bool p_has_sdfgi = false, bool p_has_opaque_gi = false);
	void _render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderList::Element **p_elements, int p_element_count, bool p_reverse_cull, PassMode p_pass_mode, bool p_no_gi, RID p_radiance_uniform_set, RID p_render_buffers_uniform_set, bool p_force_wireframe = false, const Vector2 &p_uv_offset = Vector2());
	_FORCE_INLINE_ void _add_geometry(InstanceBase *p_instance, uint32_t p_surface, RID p_material, PassMode p_pass_mode, uint32_t p_geometry_index, bool p_using_sdfgi = false);
	_FORCE_INLINE_ void _add_geometry_with_material(InstanceBase *p_instance, uint32_t p_surface, MaterialData *p_material, RID p_material_rid, PassMode p_pass_mode, uint32_t p_geometry_index, bool p_using_sdfgi = false);

	void _fill_render_list(InstanceBase **p_cull_result, int p_cull_count, PassMode p_pass_mode, bool p_using_sdfgi = false);

	Map<Size2i, RID> sdfgi_framebuffer_size_cache;

protected:
	virtual void _render_scene(RID p_render_buffer, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, int p_directional_light_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, InstanceBase **p_lightmap_cull_result, int p_lightmap_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, const Color &p_default_bg_color);
	virtual void _render_shadow(RID p_framebuffer, InstanceBase **p_cull_result, int p_cull_count, const CameraMatrix &p_projection, const Transform &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake);
	virtual void _render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region);
	virtual void _render_uv2(InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region);
	virtual void _render_sdfgi(RID p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, InstanceBase **p_cull_result, int p_cull_count, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture);
	virtual void _render_particle_collider_heightfield(RID p_fb, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, InstanceBase **p_cull_result, int p_cull_count);

public:
	virtual void set_time(double p_time, double p_step);

	virtual bool free(RID p_rid);

	RasterizerSceneHighEndRD(RasterizerStorageRD *p_storage);
	~RasterizerSceneHighEndRD();
};
#endif // RASTERIZER_SCENE_HIGHEND_RD_H

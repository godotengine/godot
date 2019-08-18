/*************************************************************************/
/*  rasterizer_scene_forward_rd.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef RASTERIZER_SCENE_FORWARD_RD_H
#define RASTERIZER_SCENE_FORWARD_RD_H

#include "servers/visual/rasterizer_rd/rasterizer_scene_rd.h"
#include "servers/visual/rasterizer_rd/rasterizer_storage_rd.h"
#include "servers/visual/rasterizer_rd/render_pipeline_vertex_format_cache_rd.h"
#include "servers/visual/rasterizer_rd/shaders/scene_forward.glsl.gen.h"

class RasterizerSceneForwardRD : public RasterizerSceneRD {

	/* Shader */

	enum ShaderVersion {
		SHADER_VERSION_DEPTH_PASS,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL,
		SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS,
		SHADER_VERSION_COLOR_PASS,
		SHADER_VERSION_COLOR_PASS_WITH_SEPARATE_SPECULAR,
		SHADER_VERSION_VCT_COLOR_PASS,
		SHADER_VERSION_VCT_COLOR_PASS_WITH_SEPARATE_SPECULAR,
		SHADER_VERSION_LIGHTMAP_COLOR_PASS,
		SHADER_VERSION_LIGHTMAP_COLOR_PASS_WITH_SEPARATE_SPECULAR,
		SHADER_VERSION_MAX
	};

	struct {
		SceneForwardShaderRD scene_shader;
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

		bool valid;
		RID version;
		uint32_t vertex_input_mask;
		RenderPipelineVertexFormatCacheRD pipelines[CULL_VARIANT_MAX][VS::PRIMITIVE_MAX][SHADER_VERSION_MAX];

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
		bool uses_depth_pre_pass;
		bool uses_discard;
		bool uses_roughness;
		bool uses_normal;

		bool unshaded;
		bool uses_vertex;
		bool uses_sss;
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
		virtual bool is_param_texture(const StringName &p_param) const;
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual Variant get_default_parameter(const StringName &p_parameter) const;
		ShaderData();
		virtual ~ShaderData();
	};

	RasterizerStorageRD::ShaderData *_create_shader_func();
	static RasterizerStorageRD::ShaderData *_create_shader_funcs() {
		return static_cast<RasterizerSceneForwardRD *>(singleton)->_create_shader_func();
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
		return static_cast<RasterizerSceneForwardRD *>(singleton)->_create_material_func(static_cast<ShaderData *>(p_shader));
	}

	/* Instance Custom Data */

	struct InstanceGeometryData : public InstanceCustomData {

		struct UBO {
			float transform[16];
			float normal_transform[12];
			uint32_t flags;
			uint32_t pad[3];
		};

		RID ubo;
		RID uniform_set_base;
		RID uniform_set_gi;

		bool ubo_dirty = true;
		bool using_lightmap_gi = false;
		bool using_vct_gi = false;
	};

	/* Framebuffer */

	struct RenderBufferDataForward : public RenderBufferData {
		//for rendering, may be MSAAd
		RID color;
		RID depth;
		RID color_fb;
		int width, height;

		RID render_target;

		RID uniform_set_opaque;
		RID uniform_set_alpha;

		void clear();
		virtual void configure(RID p_render_target, int p_width, int p_height, VS::ViewportMSAA p_msaa);

		~RenderBufferDataForward();
	};

	virtual RenderBufferData *_create_render_buffer_data();

	RID default_render_buffer_uniform_set;
	/* Instance Data */

	struct InstanceData {
		struct UBO {
		};

		RID state_buffer;
		RID uniform_set;
	};

	RID_Owner<InstanceData> instance_data_owner;

	/* Scene State UBO */

	struct SceneState {
		struct UBO {
			float projection_matrix[16];
			float inv_projection_matrix[16];

			float camera_matrix[16];
			float inv_camera_matrix[16];

			float viewport_size[2];
			float screen_pixel_size[2];

			float shadow_z_offset;
			float shadow_z_slope_scale;

			float time;
			float reflection_multiplier;
		};

		UBO ubo;

		RID uniform_buffer;

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
					uint64_t material_index : 20;
					uint64_t shader_index : 20;
					uint64_t priority : 16;
					uint64_t depth_layer : 8;
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

			if (element_count + alpha_element_count >= max_elements)
				return NULL;
			elements[element_count] = &base_elements[element_count];
			return elements[element_count++];
		}

		_FORCE_INLINE_ Element *add_alpha_element() {

			if (element_count + alpha_element_count >= max_elements)
				return NULL;
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
			for (int i = 0; i < max_elements; i++)
				elements[i] = &base_elements[i]; // assign elements
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

	static RasterizerSceneForwardRD *singleton;
	uint64_t scene_pass;
	uint64_t render_pass;
	double time;
	RID default_shader;
	RID default_material;
	RID default_shader_rd;

	enum PassMode {
		PASS_MODE_COLOR,
		PASS_MODE_COLOR_SPECULAR,
		PASS_MODE_COLOR_TRANSPARENT,
		PASS_MODE_SHADOW,
		PASS_MODE_DEPTH,
		PASS_MODE_DEPTH_NORMAL,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS,
	};

	void _setup_environment(RID p_environment, const CameraMatrix &p_cam_projection, const Transform &p_cam_transform, bool p_no_fog);

	void _render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderList::Element **p_elements, int p_element_count, bool p_reverse_cull, PassMode p_pass_mode, RID p_screen_uniform_set, bool p_no_gi);
	_FORCE_INLINE_ void _add_geometry(InstanceBase *p_instance, uint32_t p_surface, RID p_material, PassMode p_pass_mode);
	_FORCE_INLINE_ void _add_geometry_with_material(InstanceBase *p_instance, uint32_t p_surface, MaterialData *p_material, PassMode p_pass_mode);

	void _fill_render_list(InstanceBase **p_cull_result, int p_cull_count, PassMode p_pass_mode, bool p_no_gi);

protected:
	virtual void _render_scene(RenderBufferData *p_buffer_data, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass);

public:
	virtual void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) {}

	virtual void set_scene_pass(uint64_t p_pass);
	virtual void set_time(double p_time);
	virtual void set_debug_draw_mode(VS::ViewportDebugDraw p_debug_draw) {}

	virtual void instance_create_custom_data(InstanceBase *p_instance);
	virtual void instance_free_custom_data(InstanceBase *p_instance);
	virtual void instance_custom_data_update_lights(InstanceBase *p_instance);
	virtual void instance_custom_data_update_reflection_probes(InstanceBase *p_instance);
	virtual void instance_custom_data_update_gi_probes(InstanceBase *p_instance);
	virtual void instance_custom_data_update_lightmap(InstanceBase *p_instance);
	virtual void instance_custom_data_update_transform(InstanceBase *p_instance);

	virtual bool free(RID p_rid);

	RasterizerSceneForwardRD(RasterizerStorageRD *p_storage);
	~RasterizerSceneForwardRD();
};
#endif // RASTERIZER_SCENE_FORWARD_RD_H

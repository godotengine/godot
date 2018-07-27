/*************************************************************************/
/*  rasterizer_scene_gles2.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RASTERIZERSCENEGLES2_H
#define RASTERIZERSCENEGLES2_H

/* Must come before shaders or the Windows build fails... */
#include "rasterizer_storage_gles2.h"

#include "shaders/cube_to_dp.glsl.gen.h"
#include "shaders/scene.glsl.gen.h"
/*

#include "drivers/gles3/shaders/effect_blur.glsl.gen.h"
#include "drivers/gles3/shaders/exposure.glsl.gen.h"
#include "drivers/gles3/shaders/resolve.glsl.gen.h"
#include "drivers/gles3/shaders/scene.glsl.gen.h"
#include "drivers/gles3/shaders/screen_space_reflection.glsl.gen.h"
#include "drivers/gles3/shaders/ssao.glsl.gen.h"
#include "drivers/gles3/shaders/ssao_blur.glsl.gen.h"
#include "drivers/gles3/shaders/ssao_minify.glsl.gen.h"
#include "drivers/gles3/shaders/subsurf_scattering.glsl.gen.h"
#include "drivers/gles3/shaders/tonemap.glsl.gen.h"

*/

class RasterizerSceneGLES2 : public RasterizerScene {
public:
	RID default_material;
	RID default_material_twosided;
	RID default_shader;
	RID default_shader_twosided;

	uint64_t scene_pass;

	RasterizerStorageGLES2 *storage;
	struct State {

		bool texscreen_copied;
		int current_blend_mode;
		float current_line_width;
		int current_depth_draw;
		bool current_depth_test;
		GLuint current_main_tex;

		SceneShaderGLES2 scene_shader;
		CubeToDpShaderGLES2 cube_to_dp_shader;

		GLuint sky_verts;

		// ResolveShaderGLES3 resolve_shader;
		// ScreenSpaceReflectionShaderGLES3 ssr_shader;
		// EffectBlurShaderGLES3 effect_blur_shader;
		// SubsurfScatteringShaderGLES3 sss_shader;
		// SsaoMinifyShaderGLES3 ssao_minify_shader;
		// SsaoShaderGLES3 ssao_shader;
		// SsaoBlurShaderGLES3 ssao_blur_shader;
		// ExposureShaderGLES3 exposure_shader;
		// TonemapShaderGLES3 tonemap_shader;

		/*
		struct SceneDataUBO {
			//this is a std140 compatible struct. Please read the OpenGL 3.3 Specificaiton spec before doing any changes
			float projection_matrix[16];
			float inv_projection_matrix[16];
			float camera_inverse_matrix[16];
			float camera_matrix[16];
			float ambient_light_color[4];
			float bg_color[4];
			float fog_color_enabled[4];
			float fog_sun_color_amount[4];

			float ambient_energy;
			float bg_energy;
			float z_offset;
			float z_slope_scale;
			float shadow_dual_paraboloid_render_zfar;
			float shadow_dual_paraboloid_render_side;
			float viewport_size[2];
			float screen_pixel_size[2];
			float shadow_atlas_pixel_size[2];
			float shadow_directional_pixel_size[2];

			float time;
			float z_far;
			float reflection_multiplier;
			float subsurface_scatter_width;
			float ambient_occlusion_affect_light;

			uint32_t fog_depth_enabled;
			float fog_depth_begin;
			float fog_depth_curve;
			uint32_t fog_transmit_enabled;
			float fog_transmit_curve;
			uint32_t fog_height_enabled;
			float fog_height_min;
			float fog_height_max;
			float fog_height_curve;
			// make sure this struct is padded to be a multiple of 16 bytes for webgl

		} ubo_data;

		GLuint scene_ubo;

		struct EnvironmentRadianceUBO {

			float transform[16];
			float ambient_contribution;
			uint8_t padding[12];

		} env_radiance_data;

		GLuint env_radiance_ubo;

		GLuint sky_array;

		GLuint directional_ubo;

		GLuint spot_array_ubo;
		GLuint omni_array_ubo;
		GLuint reflection_array_ubo;

		GLuint immediate_buffer;
		GLuint immediate_array;

		uint32_t ubo_light_size;
		uint8_t *spot_array_tmp;
		uint8_t *omni_array_tmp;
		uint8_t *reflection_array_tmp;

		int max_ubo_lights;
		int max_forward_lights_per_object;
		int max_ubo_reflections;
		int max_skeleton_bones;

		bool used_contact_shadows;

		int spot_light_count;
		int omni_light_count;
		int directional_light_count;
		int reflection_probe_count;

		bool cull_front;
		bool cull_disabled;
		bool used_sss;
		bool used_screen_texture;
		bool using_contact_shadows;

		VS::ViewportDebugDraw debug_draw;
		*/
	} state;

	/* SHADOW ATLAS API */

	uint64_t shadow_atlas_realloc_tolerance_msec;

	struct ShadowAtlas : public RID_Data {
		enum {
			QUADRANT_SHIFT = 27,
			SHADOW_INDEX_MASK = (1 << QUADRANT_SHIFT) - 1,
			SHADOW_INVALID = 0xFFFFFFFF,
		};

		struct Quadrant {
			uint32_t subdivision;

			struct Shadow {
				RID owner;
				uint64_t version;
				uint64_t alloc_tick;

				Shadow() {
					version = 0;
					alloc_tick = 0;
				}
			};

			Vector<Shadow> shadows;

			Quadrant() {
				subdivision = 0;
			}
		} quadrants[4];

		int size_order[4];
		uint32_t smallest_subdiv;

		int size;

		GLuint fbo;
		GLuint depth;

		Map<RID, uint32_t> shadow_owners;
	};

	struct ShadowCubeMap {
		GLuint fbo[6];
		GLuint cubemap;
		uint32_t size;
	};

	Vector<ShadowCubeMap> shadow_cubemaps;

	RID_Owner<ShadowAtlas> shadow_atlas_owner;

	RID shadow_atlas_create();
	void shadow_atlas_set_size(RID p_atlas, int p_size);
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision);
	bool _shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow);
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version);

	struct DirectionalShadow {
		GLuint fbo;
		GLuint depth;

		int light_count;
		int size;
		int current_light;
	} directional_shadow;

	virtual int get_directional_light_shadow_size(RID p_light_intance);
	virtual void set_directional_shadow_count(int p_count);

	/* REFLECTION PROBE ATLAS API */

	virtual RID reflection_atlas_create();
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_size);
	virtual void reflection_atlas_set_subdivision(RID p_ref_atlas, int p_subdiv);

	/* REFLECTION CUBEMAPS */

	/* REFLECTION PROBE INSTANCE */

	virtual RID reflection_probe_instance_create(RID p_probe);
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform);
	virtual void reflection_probe_release_atlas_index(RID p_instance);
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance);
	virtual bool reflection_probe_instance_has_reflection(RID p_instance);
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas);
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance);

	/* ENVIRONMENT API */

	struct Environment : public RID_Data {
		VS::EnvironmentBG bg_mode;

		RID sky;
		float sky_custom_fov;

		Color bg_color;
		float bg_energy;
		float sky_ambient;

		Color ambient_color;
		float ambient_energy;
		float ambient_sky_contribution;

		int canvas_max_layer;

		Environment() {
			bg_mode = VS::ENV_BG_CLEAR_COLOR;
			sky_custom_fov = 0.0;
			bg_energy = 1.0;
			sky_ambient = 0;
			ambient_energy = 1.0;
			ambient_sky_contribution = 0.0;
			canvas_max_layer = 0;
		}
	};

	mutable RID_Owner<Environment> environment_owner;

	virtual RID environment_create();

	virtual void environment_set_background(RID p_env, VS::EnvironmentBG p_bg);
	virtual void environment_set_sky(RID p_env, RID p_sky);
	virtual void environment_set_sky_custom_fov(RID p_env, float p_scale);
	virtual void environment_set_bg_color(RID p_env, const Color &p_color);
	virtual void environment_set_bg_energy(RID p_env, float p_energy);
	virtual void environment_set_canvas_max_layer(RID p_env, int p_max_layer);
	virtual void environment_set_ambient_light(RID p_env, const Color &p_color, float p_energy = 1.0, float p_sky_contribution = 0.0);

	virtual void environment_set_dof_blur_near(RID p_env, bool p_enable, float p_distance, float p_transition, float p_amount, VS::EnvironmentDOFBlurQuality p_quality);
	virtual void environment_set_dof_blur_far(RID p_env, bool p_enable, float p_distance, float p_transition, float p_amount, VS::EnvironmentDOFBlurQuality p_quality);
	virtual void environment_set_glow(RID p_env, bool p_enable, int p_level_flags, float p_intensity, float p_strength, float p_bloom_threshold, VS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, bool p_bicubic_upscale);
	virtual void environment_set_fog(RID p_env, bool p_enable, float p_begin, float p_end, RID p_gradient_texture);

	virtual void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_in, float p_fade_out, float p_depth_tolerance, bool p_roughness);
	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_radius2, float p_intensity2, float p_bias, float p_light_affect, float p_ao_channel_affect, const Color &p_color, VS::EnvironmentSSAOQuality p_quality, VS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness);

	virtual void environment_set_tonemap(RID p_env, VS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale);

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp);

	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_color, const Color &p_sun_color, float p_sun_amount);
	virtual void environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_curve, bool p_transmit, float p_transmit_curve);
	virtual void environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve);

	virtual bool is_environment(RID p_env);

	virtual VS::EnvironmentBG environment_get_background(RID p_env);
	virtual int environment_get_canvas_max_layer(RID p_env);

	/* LIGHT INSTANCE */

	struct LightInstance : public RID_Data {

		struct ShadowTransform {
			CameraMatrix camera;
			Transform transform;
			float farplane;
			float split;
			float bias_scale;
		};

		ShadowTransform shadow_transform[4];

		RID self;
		RID light;

		RasterizerStorageGLES2::Light *light_ptr;
		Transform transform;

		Vector3 light_vector;
		Vector3 spot_vector;
		float linear_att;

		// TODO passes and all that stuff ?
		uint64_t last_scene_pass;
		uint64_t last_scene_shadow_pass;

		uint16_t light_index;
		uint16_t light_directional_index;

		Rect2 directional_rect;

		Set<RID> shadow_atlases; // atlases where this light is registered
	};

	mutable RID_Owner<LightInstance> light_instance_owner;

	virtual RID light_instance_create(RID p_light);
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform);
	virtual void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_bias_scale = 1.0);
	virtual void light_instance_mark_visible(RID p_light_instance);

	/* REFLECTION INSTANCE */

	virtual RID gi_probe_instance_create();
	virtual void gi_probe_instance_set_light_data(RID p_probe, RID p_base, RID p_data);
	virtual void gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform);
	virtual void gi_probe_instance_set_bounds(RID p_probe, const Vector3 &p_bounds);

	/* RENDER LIST */

	struct RenderList {
		enum {
			DEFAULT_MAX_ELEMENTS = 65536,
			SORT_FLAG_SKELETON = 1,
			SORT_FLAG_INSTANCING = 2,
			MAX_DIRECTIONAL_LIGHTS = 16,
			MAX_LIGHTS = 4096,
			MAX_REFLECTIONS = 1024,

			SORT_KEY_PRIORITY_SHIFT = 56,
			SORT_KEY_PRIORITY_MASK = 0xFF,
			//depth layer for opaque (56-52)
			SORT_KEY_OPAQUE_DEPTH_LAYER_SHIFT = 52,
			SORT_KEY_OPAQUE_DEPTH_LAYER_MASK = 0xF,
//64 bits unsupported in MSVC
#define SORT_KEY_UNSHADED_FLAG (uint64_t(1) << 49)
#define SORT_KEY_NO_DIRECTIONAL_FLAG (uint64_t(1) << 48)
#define SORT_KEY_LIGHTMAP_CAPTURE_FLAG (uint64_t(1) << 47)
#define SORT_KEY_LIGHTMAP_FLAG (uint64_t(1) << 46)
#define SORT_KEY_GI_PROBES_FLAG (uint64_t(1) << 45)
#define SORT_KEY_VERTEX_LIT_FLAG (uint64_t(1) << 44)
			SORT_KEY_SHADING_SHIFT = 44,
			SORT_KEY_SHADING_MASK = 63,
			//44-28 material index
			SORT_KEY_MATERIAL_INDEX_SHIFT = 28,
			//28-8 geometry index
			SORT_KEY_GEOMETRY_INDEX_SHIFT = 8,
			//bits 5-7 geometry type
			SORT_KEY_GEOMETRY_TYPE_SHIFT = 5,
			//bits 0-5 for flags
			SORT_KEY_OPAQUE_PRE_PASS = 8,
			SORT_KEY_CULL_DISABLED_FLAG = 4,
			SORT_KEY_SKELETON_FLAG = 2,
			SORT_KEY_MIRROR_FLAG = 1
		};

		int max_elements;

		struct Element {
			RasterizerScene::InstanceBase *instance;

			RasterizerStorageGLES2::Geometry *geometry;
			RasterizerStorageGLES2::Material *material;
			RasterizerStorageGLES2::GeometryOwner *owner;

			uint64_t sort_key;
		};

		Element *base_elements;
		Element **elements;

		int element_count;
		int alpha_element_count;

		void clear() {
			element_count = 0;
			alpha_element_count = 0;
		}

		// sorts

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
				uint32_t layer_A = uint32_t(A->sort_key >> SORT_KEY_PRIORITY_SHIFT);
				uint32_t layer_B = uint32_t(B->sort_key >> SORT_KEY_PRIORITY_SHIFT);
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

		// element adding and stuff

		_FORCE_INLINE_ Element *add_element() {
			if (element_count + alpha_element_count >= max_elements)
				return NULL;

			elements[element_count] = &base_elements[element_count];
			return elements[element_count++];
		}

		_FORCE_INLINE_ Element *add_alpha_element() {
			if (element_count + alpha_element_count >= max_elements) {
				return NULL;
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
				elements[i] = &base_elements[i];
			}
		}

		RenderList() {
			max_elements = DEFAULT_MAX_ELEMENTS;
		}

		~RenderList() {
			memdelete_arr(elements);
			memdelete_arr(base_elements);
		}
	};

	RenderList render_list;

	void _add_geometry(RasterizerStorageGLES2::Geometry *p_geometry, InstanceBase *p_instance, RasterizerStorageGLES2::GeometryOwner *p_owner, int p_material, bool p_depth_pass, bool p_shadow_pass);
	void _add_geometry_with_material(RasterizerStorageGLES2::Geometry *p_geometry, InstanceBase *p_instance, RasterizerStorageGLES2::GeometryOwner *p_owner, RasterizerStorageGLES2::Material *p_material, bool p_depth_pass, bool p_shadow_pass);

	void _fill_render_list(InstanceBase **p_cull_result, int p_cull_count, bool p_depth_pass, bool p_shadow_pass);
	void _render_render_list(RenderList::Element **p_elements, int p_element_count, const RID *p_light_cull_result, int p_light_cull_count, const Transform &p_view_transform, const CameraMatrix &p_projection, RID p_shadow_atlas, Environment *p_env, GLuint p_base_env, float p_shadow_bias, float p_shadow_normal_bias, bool p_reverse_cull, bool p_alpha_pass, bool p_shadow, bool p_directional_add, bool p_directional_shadows);

	void _draw_sky(RasterizerStorageGLES2::Sky *p_sky, const CameraMatrix &p_projection, const Transform &p_transform, bool p_vflip, float p_custom_fov, float p_energy);

	void _setup_material(RasterizerStorageGLES2::Material *p_material, bool p_use_radiance_map, bool p_reverse_cull, bool p_shadow_atlas = false, bool p_skeleton_tex = false, Size2i p_skeleton_tex_size = Size2i(0, 0));
	void _setup_geometry(RenderList::Element *p_element, RasterizerStorageGLES2::Skeleton *p_skeleton);
	void _render_geometry(RenderList::Element *p_element);

	virtual void render_scene(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass);
	virtual void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count);
	virtual bool free(RID p_rid);

	virtual void set_scene_pass(uint64_t p_pass);
	virtual void set_debug_draw_mode(VS::ViewportDebugDraw p_debug_draw);

	void iteration();
	void initialize();
	void finalize();
	RasterizerSceneGLES2();
};

#endif // RASTERIZERSCENEGLES2_H

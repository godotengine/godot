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

#include "shaders/scene.glsl.gen.h"
/*

#include "drivers/gles3/shaders/cube_to_dp.glsl.gen.h"
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
	RasterizerStorageGLES2 *storage;
	struct State {

		bool texscreen_copied;
		int current_blend_mode;
		float current_line_width;
		int current_depth_draw;
		bool current_depth_test;
		GLuint current_main_tex;

		SceneShaderGLES2 scene_shader;
		// CubeToDpShaderGLES3 cube_to_dp_shader;
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

		GLuint sky_verts;
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

	RID shadow_atlas_create();
	void shadow_atlas_set_size(RID p_atlas, int p_size);
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision);
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version);

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
	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_radius2, float p_intensity2, float p_bias, float p_light_affect, const Color &p_color, VS::EnvironmentSSAOQuality p_quality, VS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness);

	virtual void environment_set_tonemap(RID p_env, VS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale);

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp);

	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_color, const Color &p_sun_color, float p_sun_amount);
	virtual void environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_curve, bool p_transmit, float p_transmit_curve);
	virtual void environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve);

	virtual bool is_environment(RID p_env);

	virtual VS::EnvironmentBG environment_get_background(RID p_env);
	virtual int environment_get_canvas_max_layer(RID p_env);

	/* LIGHT INSTANCE */
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

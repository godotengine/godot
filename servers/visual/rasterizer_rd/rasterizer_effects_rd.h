/*************************************************************************/
/*  rasterizer_effects_rd.h                                              */
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

#ifndef RASTERIZER_EFFECTS_RD_H
#define RASTERIZER_EFFECTS_RD_H

#include "core/math/camera_matrix.h"
#include "render_pipeline_vertex_format_cache_rd.h"
#include "servers/visual/rasterizer_rd/shaders/blur.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/bokeh_dof.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/copy.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/cubemap_roughness.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/luminance_reduce.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/roughness_limiter.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/sky.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/ssao.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/ssao_blur.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/ssao_minify.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/tonemap.glsl.gen.h"

#include "servers/visual_server.h"

class RasterizerEffectsRD {

	enum BlurMode {
		BLUR_MODE_GAUSSIAN_BLUR,
		BLUR_MODE_GAUSSIAN_GLOW,
		BLUR_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE,
		BLUR_MODE_DOF_NEAR_LOW,
		BLUR_MODE_DOF_NEAR_MEDIUM,
		BLUR_MODE_DOF_NEAR_HIGH,
		BLUR_MODE_DOF_NEAR_MERGE_LOW,
		BLUR_MODE_DOF_NEAR_MERGE_MEDIUM,
		BLUR_MODE_DOF_NEAR_MERGE_HIGH,
		BLUR_MODE_DOF_FAR_LOW,
		BLUR_MODE_DOF_FAR_MEDIUM,
		BLUR_MODE_DOF_FAR_HIGH,
		BLUR_MODE_SSAO_MERGE,
		BLUR_MODE_SIMPLY_COPY,
		BLUR_MODE_MIPMAP,
		BLUR_MODE_MAX,

	};

	enum {
		BLUR_FLAG_HORIZONTAL = (1 << 0),
		BLUR_FLAG_USE_BLUR_SECTION = (1 << 1),
		BLUR_FLAG_USE_ORTHOGONAL_PROJECTION = (1 << 2),
		BLUR_FLAG_DOF_NEAR_FIRST_TAP = (1 << 3),
		BLUR_FLAG_GLOW_FIRST_PASS = (1 << 4),
		BLUR_FLAG_FLIP_Y = (1 << 5),
		BLUR_COPY_FORCE_LUMINANCE = (1 << 6)
	};

	struct BlurPushConstant {
		float section[4];
		float pixel_size[2];
		uint32_t flags;
		uint32_t pad;
		//glow
		float glow_strength;
		float glow_bloom;
		float glow_hdr_threshold;
		float glow_hdr_scale;
		float glow_exposure;
		float glow_white;
		float glow_luminance_cap;
		float glow_auto_exposure_grey;
		//dof
		float dof_begin;
		float dof_end;
		float dof_radius;
		float dof_pad;

		float dof_dir[2];
		float camera_z_far;
		float camera_z_near;

		float ssao_color[4];
	};

	struct Blur {
		BlurPushConstant push_constant;
		BlurShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipelines[BLUR_MODE_MAX];

	} blur;

	enum CubemapRoughnessSource {
		CUBEMAP_ROUGHNESS_SOURCE_PANORAMA,
		CUBEMAP_ROUGHNESS_SOURCE_CUBEMAP,
		CUBEMAP_ROUGHNESS_SOURCE_MAX
	};

	struct CubemapRoughnessPushConstant {
		uint32_t face_id;
		uint32_t sample_count;
		float roughness;
		uint32_t use_direct_write;
	};

	struct CubemapRoughness {

		CubemapRoughnessPushConstant push_constant;
		CubemapRoughnessShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipelines[CUBEMAP_ROUGHNESS_SOURCE_MAX];
	} roughness;

	struct SkyPushConstant {
		float orientation[12];
		float proj[4];
		float multiplier;
		float alpha;
		float depth;
		float pad;
	};

	struct Sky {

		SkyPushConstant push_constant;
		SkyShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipeline;
	} sky;

	enum TonemapMode {
		TONEMAP_MODE_NORMAL,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER,
		TONEMAP_MODE_MAX
	};

	struct TonemapPushConstant {
		float bcs[3];
		uint32_t use_bcs;

		uint32_t use_glow;
		uint32_t use_auto_exposure;
		uint32_t use_color_correction;
		uint32_t tonemapper;

		uint32_t glow_texture_size[2];

		float glow_intensity;
		uint32_t glow_level_flags;
		uint32_t glow_mode;

		float exposure;
		float white;
		float auto_exposure_grey;
	};

	struct Tonemap {

		TonemapPushConstant push_constant;
		TonemapShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipelines[TONEMAP_MODE_MAX];
	} tonemap;

	enum LuminanceReduceMode {
		LUMINANCE_REDUCE_READ,
		LUMINANCE_REDUCE,
		LUMINANCE_REDUCE_WRITE,
		LUMINANCE_REDUCE_MAX
	};

	struct LuminanceReducePushConstant {
		int32_t source_size[2];
		float max_luminance;
		float min_luminance;
		float exposure_adjust;
		float pad[3];
	};

	struct LuminanceReduce {

		LuminanceReducePushConstant push_constant;
		LuminanceReduceShaderRD shader;
		RID shader_version;
		RID pipelines[LUMINANCE_REDUCE_MAX];
	} luminance_reduce;

	struct CopyToDPPushConstant {
		float bias;
		float z_far;
		float z_near;
		uint32_t z_flip;
	};

	enum CopyMode {
		COPY_MODE_CUBE_TO_DP,
		COPY_MODE_MAX
	};

	struct Copy {

		CopyShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipelines[COPY_MODE_MAX];
	} copy;

	struct BokehPushConstant {
		uint32_t size[2];
		float z_far;
		float z_near;

		uint32_t orthogonal;
		float blur_size;
		float blur_scale;
		uint32_t steps;

		uint32_t blur_near_active;
		float blur_near_begin;
		float blur_near_end;
		uint32_t blur_far_active;

		float blur_far_begin;
		float blur_far_end;
		uint32_t second_pass;
		uint32_t half_size;

		uint32_t use_jitter;
		float jitter_seed;
		uint32_t pad[2];
	};

	enum BokehMode {
		BOKEH_GEN_BLUR_SIZE,
		BOKEH_GEN_BOKEH_BOX,
		BOKEH_GEN_BOKEH_HEXAGONAL,
		BOKEH_GEN_BOKEH_CIRCULAR,
		BOKEH_COMPOSITE,
		BOKEH_MAX
	};

	struct Bokeh {

		BokehPushConstant push_constant;
		BokehDofShaderRD shader;
		RID shader_version;
		RID pipelines[BOKEH_MAX];
	} bokeh;

	enum SSAOMode {
		SSAO_MINIFY_FIRST,
		SSAO_MINIFY_MIPMAP,
		SSAO_GATHER_LOW,
		SSAO_GATHER_MEDIUM,
		SSAO_GATHER_HIGH,
		SSAO_GATHER_ULTRA,
		SSAO_GATHER_LOW_HALF,
		SSAO_GATHER_MEDIUM_HALF,
		SSAO_GATHER_HIGH_HALF,
		SSAO_GATHER_ULTRA_HALF,
		SSAO_BLUR_PASS,
		SSAO_BLUR_PASS_HALF,
		SSAO_BLUR_UPSCALE,
		SSAO_MAX
	};

	struct SSAOMinifyPushConstant {
		float pixel_size[2];
		float z_far;
		float z_near;
		int32_t source_size[2];
		uint32_t orthogonal;
		uint32_t pad;
	};

	struct SSAOGatherPushConstant {
		int32_t screen_size[2];
		float z_far;
		float z_near;

		uint32_t orthogonal;
		float intensity_div_r6;
		float radius;
		float bias;

		float proj_info[4];
		float pixel_size[2];
		float proj_scale;
		uint32_t pad;
	};

	struct SSAOBlurPushConstant {
		float edge_sharpness;
		int32_t filter_scale;
		float z_far;
		float z_near;
		uint32_t orthogonal;
		uint32_t pad[3];
		int32_t axis[2];
		int32_t screen_size[2];
	};

	struct SSAO {

		SSAOMinifyPushConstant minify_push_constant;
		SsaoMinifyShaderRD minify_shader;
		RID minify_shader_version;

		SSAOGatherPushConstant gather_push_constant;
		SsaoShaderRD gather_shader;
		RID gather_shader_version;

		SSAOBlurPushConstant blur_push_constant;
		SsaoBlurShaderRD blur_shader;
		RID blur_shader_version;

		RID pipelines[SSAO_MAX];
	} ssao;

	struct RoughnessLimiterPushConstant {
		int32_t screen_size[2];
		float curve;
		uint32_t pad;
	};

	struct RoughnessLimiter {

		RoughnessLimiterPushConstant push_constant;
		RoughnessLimiterShaderRD shader;
		RID shader_version;
		RID pipeline;

	} roughness_limiter;

	RID default_sampler;
	RID default_mipmap_sampler;
	RID index_buffer;
	RID index_array;

	Map<RID, RID> texture_to_uniform_set_cache;

	Map<RID, RID> image_to_uniform_set_cache;
	Map<RID, RID> texture_to_compute_uniform_set_cache;

	RID _get_uniform_set_from_image(RID p_texture);
	RID _get_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);

public:
	//TODO must re-do most of the shaders in compute

	void region_copy(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_region);
	void copy_to_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_rect, bool p_flip_y = false, bool p_force_luminance = false);
	void gaussian_blur(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, const Rect2 &p_region);
	void gaussian_glow(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, float p_strength = 1.0, bool p_first_pass = false, float p_luminance_cap = 16.0, float p_exposure = 1.0, float p_bloom = 0.0, float p_hdr_bleed_treshold = 1.0, float p_hdr_bleed_scale = 1.0, RID p_auto_exposure = RID(), float p_auto_exposure_grey = 1.0);

	void cubemap_roughness(RID p_source_rd_texture, bool p_source_is_panorama, RID p_dest_framebuffer, uint32_t p_face_id, uint32_t p_sample_count, float p_roughness);
	void render_panorama(RD::DrawListID p_list, RenderingDevice::FramebufferFormatID p_fb_format, RID p_panorama, const CameraMatrix &p_camera, const Basis &p_orientation, float p_alpha, float p_multipler);
	void make_mipmap(RID p_source_rd_texture, RID p_framebuffer_half, const Vector2 &p_pixel_size);
	void copy_cubemap_to_dp(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_rect, float p_z_near, float p_z_far, float p_bias, bool p_dp_flip);
	void luminance_reduction(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);
	void bokeh_dof(RID p_base_texture, RID p_depth_texture, const Size2i &p_base_texture_size, RID p_secondary_texture, RID p_bokeh_texture1, RID p_bokeh_texture2, bool p_dof_far, float p_dof_far_begin, float p_dof_far_size, bool p_dof_near, float p_dof_near_begin, float p_dof_near_size, float p_bokeh_size, VS::DOFBokehShape p_bokeh_shape, VS::DOFBlurQuality p_quality, bool p_use_jitter, float p_cam_znear, float p_cam_zfar, bool p_cam_orthogonal);

	struct TonemapSettings {

		bool use_glow = false;
		enum GlowMode {
			GLOW_MODE_ADD,
			GLOW_MODE_SCREEN,
			GLOW_MODE_SOFTLIGHT,
			GLOW_MODE_REPLACE,
			GLOW_MODE_MIX
		};

		GlowMode glow_mode = GLOW_MODE_ADD;
		float glow_intensity = 1.0;
		uint32_t glow_level_flags = 0;
		Vector2i glow_texture_size;
		bool glow_use_bicubic_upscale = false;
		RID glow_texture;

		VS::EnvironmentToneMapper tonemap_mode = VS::ENV_TONE_MAPPER_LINEAR;
		float exposure = 1.0;
		float white = 1.0;

		bool use_auto_exposure = false;
		float auto_exposure_grey = 0.5;
		RID exposure_texture;

		bool use_bcs = false;
		float brightness = 1.0;
		float contrast = 1.0;
		float saturation = 1.0;

		bool use_color_correction = false;
		RID color_correction_texture;
	};

	void tonemapper(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings);

	void generate_ssao(RID p_depth_buffer, RID p_normal_buffer, const Size2i &p_depth_buffer_size, RID p_depth_mipmaps_texture, const Vector<RID> &depth_mipmaps, RID p_ao1, bool p_half_size, RID p_ao2, RID p_upscale_buffer, float p_intensity, float p_radius, float p_bias, const CameraMatrix &p_projection, VS::EnvironmentSSAOQuality p_quality, VS::EnvironmentSSAOBlur p_blur, float p_edge_sharpness);

	void roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve);

	RasterizerEffectsRD();
	~RasterizerEffectsRD();
};

#endif // EFFECTS_RD_H

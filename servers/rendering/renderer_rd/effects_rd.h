/*************************************************************************/
/*  effects_rd.h                                                         */
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

#ifndef EFFECTS_RD_H
#define EFFECTS_RD_H

#include "core/math/camera_matrix.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/blur_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/bokeh_dof.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/bokeh_dof_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/copy.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/copy_to_fb.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cube_to_dp.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cubemap_downsampler.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cubemap_downsampler_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cubemap_filter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cubemap_filter_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cubemap_roughness.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cubemap_roughness_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/fsr_upscale.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/luminance_reduce.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/luminance_reduce_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/resolve.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/roughness_limiter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/screen_space_reflection.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/screen_space_reflection_filter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/screen_space_reflection_scale.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sort.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/specular_merge.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_blur.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_downsample.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_importance_map.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_interleave.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/subsurface_scattering.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/tonemap.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"

#include "servers/rendering_server.h"

class EffectsRD {
private:
	bool prefer_raster_effects;

	enum FSRUpscalePass {
		FSR_UPSCALE_PASS_EASU = 0,
		FSR_UPSCALE_PASS_RCAS = 1
	};

	struct FSRUpscalePushConstant {
		float resolution_width;
		float resolution_height;
		float upscaled_width;
		float upscaled_height;
		float sharpness;
		int pass;
		int _unused0, _unused1;
	};

	struct FSRUpscale {
		FSRUpscalePushConstant push_constant;
		FsrUpscaleShaderRD shader;
		RID shader_version;
		RID pipeline;
	} FSR_upscale;

	enum BlurRasterMode {
		BLUR_MIPMAP,

		BLUR_MODE_GAUSSIAN_BLUR,
		BLUR_MODE_GAUSSIAN_GLOW,
		BLUR_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE,
		BLUR_MODE_COPY,

		BLUR_MODE_MAX
	};

	enum {
		BLUR_FLAG_HORIZONTAL = (1 << 0),
		BLUR_FLAG_USE_ORTHOGONAL_PROJECTION = (1 << 1),
		BLUR_FLAG_GLOW_FIRST_PASS = (1 << 2),
	};

	struct BlurRasterPushConstant {
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
	};

	struct BlurRaster {
		BlurRasterPushConstant push_constant;
		BlurRasterShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[BLUR_MODE_MAX];
	} blur_raster;

	enum CopyMode {
		COPY_MODE_GAUSSIAN_COPY,
		COPY_MODE_GAUSSIAN_COPY_8BIT,
		COPY_MODE_GAUSSIAN_GLOW,
		COPY_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE,
		COPY_MODE_SIMPLY_COPY,
		COPY_MODE_SIMPLY_COPY_8BIT,
		COPY_MODE_SIMPLY_COPY_DEPTH,
		COPY_MODE_SET_COLOR,
		COPY_MODE_SET_COLOR_8BIT,
		COPY_MODE_MIPMAP,
		COPY_MODE_LINEARIZE_DEPTH,
		COPY_MODE_CUBE_TO_PANORAMA,
		COPY_MODE_CUBE_ARRAY_TO_PANORAMA,
		COPY_MODE_MAX,

	};

	enum {
		COPY_FLAG_HORIZONTAL = (1 << 0),
		COPY_FLAG_USE_COPY_SECTION = (1 << 1),
		COPY_FLAG_USE_ORTHOGONAL_PROJECTION = (1 << 2),
		COPY_FLAG_DOF_NEAR_FIRST_TAP = (1 << 3),
		COPY_FLAG_GLOW_FIRST_PASS = (1 << 4),
		COPY_FLAG_FLIP_Y = (1 << 5),
		COPY_FLAG_FORCE_LUMINANCE = (1 << 6),
		COPY_FLAG_ALL_SOURCE = (1 << 7),
		COPY_FLAG_HIGH_QUALITY_GLOW = (1 << 8),
		COPY_FLAG_ALPHA_TO_ONE = (1 << 9),
	};

	struct CopyPushConstant {
		int32_t section[4];
		int32_t target[2];
		uint32_t flags;
		uint32_t pad;
		// Glow.
		float glow_strength;
		float glow_bloom;
		float glow_hdr_threshold;
		float glow_hdr_scale;

		float glow_exposure;
		float glow_white;
		float glow_luminance_cap;
		float glow_auto_exposure_grey;
		// DOF.
		float camera_z_far;
		float camera_z_near;
		uint32_t pad2[2];
		//SET color
		float set_color[4];
	};

	struct Copy {
		CopyPushConstant push_constant;
		CopyShaderRD shader;
		RID shader_version;
		RID pipelines[COPY_MODE_MAX];

	} copy;

	enum CopyToFBMode {
		COPY_TO_FB_COPY,
		COPY_TO_FB_COPY_PANORAMA_TO_DP,
		COPY_TO_FB_COPY2,
		COPY_TO_FB_MAX,

	};

	struct CopyToFbPushConstant {
		float section[4];
		float pixel_size[2];
		uint32_t flip_y;
		uint32_t use_section;

		uint32_t force_luminance;
		uint32_t alpha_to_zero;
		uint32_t srgb;
		uint32_t pad;
	};

	struct CopyToFb {
		CopyToFbPushConstant push_constant;
		CopyToFbShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[COPY_TO_FB_MAX];

	} copy_to_fb;

	struct CubemapRoughnessPushConstant {
		uint32_t face_id;
		uint32_t sample_count;
		float roughness;
		uint32_t use_direct_write;
		float face_size;
		float pad[3];
	};

	struct CubemapRoughness {
		CubemapRoughnessPushConstant push_constant;
		CubemapRoughnessShaderRD compute_shader;
		CubemapRoughnessRasterShaderRD raster_shader;
		RID shader_version;
		RID compute_pipeline;
		PipelineCacheRD raster_pipeline;
	} roughness;

	enum TonemapMode {
		TONEMAP_MODE_NORMAL,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER,
		TONEMAP_MODE_1D_LUT,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER_1D_LUT,
		TONEMAP_MODE_SUBPASS,
		TONEMAP_MODE_SUBPASS_1D_LUT,

		TONEMAP_MODE_NORMAL_MULTIVIEW,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER_MULTIVIEW,
		TONEMAP_MODE_1D_LUT_MULTIVIEW,
		TONEMAP_MODE_BICUBIC_GLOW_FILTER_1D_LUT_MULTIVIEW,
		TONEMAP_MODE_SUBPASS_MULTIVIEW,
		TONEMAP_MODE_SUBPASS_1D_LUT_MULTIVIEW,

		TONEMAP_MODE_MAX
	};

	struct TonemapPushConstant {
		float bcs[3]; // 12 - 12
		uint32_t use_bcs; //  4 - 16

		uint32_t use_glow; //  4 - 20
		uint32_t use_auto_exposure; //  4 - 24
		uint32_t use_color_correction; //  4 - 28
		uint32_t tonemapper; //  4 - 32

		uint32_t glow_texture_size[2]; //  8 - 40
		float glow_intensity; //  4 - 44
		uint32_t pad3; //  4 - 48

		uint32_t glow_mode; //  4 - 52
		float glow_levels[7]; // 28 - 80

		float exposure; //  4 - 84
		float white; //  4 - 88
		float auto_exposure_grey; //  4 - 92
		float luminance_multiplier; //  4 - 96

		float pixel_size[2]; //  8 - 104
		uint32_t use_fxaa; //  4 - 108
		uint32_t use_debanding; //  4 - 112
	};

	/* tonemap actually writes to a framebuffer, which is
	 * better to do using the raster pipeline rather than
	 * compute, as that framebuffer might be in different formats
	 */
	struct Tonemap {
		TonemapPushConstant push_constant;
		TonemapShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[TONEMAP_MODE_MAX];
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

	enum LuminanceReduceRasterMode {
		LUMINANCE_REDUCE_FRAGMENT_FIRST,
		LUMINANCE_REDUCE_FRAGMENT,
		LUMINANCE_REDUCE_FRAGMENT_FINAL,
		LUMINANCE_REDUCE_FRAGMENT_MAX
	};

	struct LuminanceReduceRasterPushConstant {
		int32_t source_size[2];
		int32_t dest_size[2];
		float exposure_adjust;
		float min_luminance;
		float max_luminance;
		uint32_t pad1;
	};

	struct LuminanceReduceFragment {
		LuminanceReduceRasterPushConstant push_constant;
		LuminanceReduceRasterShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[LUMINANCE_REDUCE_FRAGMENT_MAX];
	} luminance_reduce_raster;

	struct CopyToDPPushConstant {
		float z_far;
		float z_near;
		float texel_size[2];
		float screen_rect[4];
	};

	struct CoptToDP {
		CubeToDpShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipeline;
	} cube_to_dp;

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
		BOKEH_GEN_BOKEH_BOX_NOWEIGHT,
		BOKEH_GEN_BOKEH_HEXAGONAL,
		BOKEH_GEN_BOKEH_HEXAGONAL_NOWEIGHT,
		BOKEH_GEN_BOKEH_CIRCULAR,
		BOKEH_COMPOSITE,
		BOKEH_MAX
	};

	struct Bokeh {
		BokehPushConstant push_constant;
		BokehDofShaderRD compute_shader;
		BokehDofRasterShaderRD raster_shader;
		RID shader_version;
		RID compute_pipelines[BOKEH_MAX];
		PipelineCacheRD raster_pipelines[BOKEH_MAX];
	} bokeh;

	enum SSAOMode {
		SSAO_DOWNSAMPLE,
		SSAO_DOWNSAMPLE_HALF_RES,
		SSAO_DOWNSAMPLE_MIPMAP,
		SSAO_DOWNSAMPLE_MIPMAP_HALF_RES,
		SSAO_DOWNSAMPLE_HALF,
		SSAO_DOWNSAMPLE_HALF_RES_HALF,
		SSAO_GATHER,
		SSAO_GATHER_BASE,
		SSAO_GATHER_ADAPTIVE,
		SSAO_GENERATE_IMPORTANCE_MAP,
		SSAO_PROCESS_IMPORTANCE_MAPA,
		SSAO_PROCESS_IMPORTANCE_MAPB,
		SSAO_BLUR_PASS,
		SSAO_BLUR_PASS_SMART,
		SSAO_BLUR_PASS_WIDE,
		SSAO_INTERLEAVE,
		SSAO_INTERLEAVE_SMART,
		SSAO_INTERLEAVE_HALF,
		SSAO_MAX
	};

	struct SSAODownsamplePushConstant {
		float pixel_size[2];
		float z_far;
		float z_near;
		uint32_t orthogonal;
		float radius_sq;
		uint32_t pad[2];
	};

	struct SSAOGatherPushConstant {
		int32_t screen_size[2];
		int pass;
		int quality;

		float half_screen_pixel_size[2];
		int size_multiplier;
		float detail_intensity;

		float NDC_to_view_mul[2];
		float NDC_to_view_add[2];

		float pad[2];
		float half_screen_pixel_size_x025[2];

		float radius;
		float intensity;
		float shadow_power;
		float shadow_clamp;

		float fade_out_mul;
		float fade_out_add;
		float horizon_angle_threshold;
		float inv_radius_near_limit;

		bool is_orthogonal;
		float neg_inv_radius;
		float load_counter_avg_div;
		float adaptive_sample_limit;

		int32_t pass_coord_offset[2];
		float pass_uv_offset[2];
	};

	struct SSAOGatherConstants {
		float rotation_matrices[80]; //5 vec4s * 4
	};

	struct SSAOImportanceMapPushConstant {
		float half_screen_pixel_size[2];
		float intensity;
		float power;
	};

	struct SSAOBlurPushConstant {
		float edge_sharpness;
		float pad;
		float half_screen_pixel_size[2];
	};

	struct SSAOInterleavePushConstant {
		float inv_sharpness;
		uint32_t size_modifier;
		float pixel_size[2];
	};

	struct SSAO {
		SSAODownsamplePushConstant downsample_push_constant;
		SsaoDownsampleShaderRD downsample_shader;
		RID downsample_shader_version;

		SSAOGatherPushConstant gather_push_constant;
		SsaoShaderRD gather_shader;
		RID gather_shader_version;
		RID gather_constants_buffer;
		bool gather_initialized = false;

		SSAOImportanceMapPushConstant importance_map_push_constant;
		SsaoImportanceMapShaderRD importance_map_shader;
		RID importance_map_shader_version;
		RID importance_map_load_counter;
		RID counter_uniform_set;

		SSAOBlurPushConstant blur_push_constant;
		SsaoBlurShaderRD blur_shader;
		RID blur_shader_version;

		SSAOInterleavePushConstant interleave_push_constant;
		SsaoInterleaveShaderRD interleave_shader;
		RID interleave_shader_version;

		RID mirror_sampler;
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

	struct CubemapDownsamplerPushConstant {
		uint32_t face_size;
		uint32_t face_id;
		float pad[2];
	};

	struct CubemapDownsampler {
		CubemapDownsamplerPushConstant push_constant;
		CubemapDownsamplerShaderRD compute_shader;
		CubemapDownsamplerRasterShaderRD raster_shader;
		RID shader_version;
		RID compute_pipeline;
		PipelineCacheRD raster_pipeline;
	} cubemap_downsampler;

	enum CubemapFilterMode {
		FILTER_MODE_HIGH_QUALITY,
		FILTER_MODE_LOW_QUALITY,
		FILTER_MODE_HIGH_QUALITY_ARRAY,
		FILTER_MODE_LOW_QUALITY_ARRAY,
		FILTER_MODE_MAX,
	};

	struct CubemapFilterRasterPushConstant {
		uint32_t mip_level;
		uint32_t face_id;
		float pad[2];
	};

	struct CubemapFilter {
		CubemapFilterShaderRD compute_shader;
		CubemapFilterRasterShaderRD raster_shader;
		RID shader_version;
		RID compute_pipelines[FILTER_MODE_MAX];
		PipelineCacheRD raster_pipelines[FILTER_MODE_MAX];

		RID uniform_set;
		RID image_uniform_set;
		RID coefficient_buffer;
		bool use_high_quality;

	} filter;

	enum SpecularMergeMode {
		SPECULAR_MERGE_ADD,
		SPECULAR_MERGE_SSR,
		SPECULAR_MERGE_ADDITIVE_ADD,
		SPECULAR_MERGE_ADDITIVE_SSR,
		SPECULAR_MERGE_MAX
	};

	/* Specular merge must be done using raster, rather than compute
	 * because it must continue the existing color buffer
	 */

	struct SpecularMerge {
		SpecularMergeShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[SPECULAR_MERGE_MAX];

	} specular_merge;

	enum ScreenSpaceReflectionMode {
		SCREEN_SPACE_REFLECTION_NORMAL,
		SCREEN_SPACE_REFLECTION_ROUGH,
		SCREEN_SPACE_REFLECTION_MAX,
	};

	struct ScreenSpaceReflectionPushConstant {
		float proj_info[4];

		int32_t screen_size[2];
		float camera_z_near;
		float camera_z_far;

		int32_t num_steps;
		float depth_tolerance;
		float distance_fade;
		float curve_fade_in;

		uint32_t orthogonal;
		float filter_mipmap_levels;
		uint32_t use_half_res;
		uint8_t metallic_mask[4];

		float projection[16];
	};

	struct ScreenSpaceReflection {
		ScreenSpaceReflectionPushConstant push_constant;
		ScreenSpaceReflectionShaderRD shader;
		RID shader_version;
		RID pipelines[SCREEN_SPACE_REFLECTION_MAX];

	} ssr;

	struct ScreenSpaceReflectionFilterPushConstant {
		float proj_info[4];

		uint32_t orthogonal;
		float edge_tolerance;
		int32_t increment;
		uint32_t pad;

		int32_t screen_size[2];
		uint32_t vertical;
		uint32_t steps;
	};
	enum {
		SCREEN_SPACE_REFLECTION_FILTER_HORIZONTAL,
		SCREEN_SPACE_REFLECTION_FILTER_VERTICAL,
		SCREEN_SPACE_REFLECTION_FILTER_MAX,
	};

	struct ScreenSpaceReflectionFilter {
		ScreenSpaceReflectionFilterPushConstant push_constant;
		ScreenSpaceReflectionFilterShaderRD shader;
		RID shader_version;
		RID pipelines[SCREEN_SPACE_REFLECTION_FILTER_MAX];
	} ssr_filter;

	struct ScreenSpaceReflectionScalePushConstant {
		int32_t screen_size[2];
		float camera_z_near;
		float camera_z_far;

		uint32_t orthogonal;
		uint32_t filter;
		uint32_t pad[2];
	};

	struct ScreenSpaceReflectionScale {
		ScreenSpaceReflectionScalePushConstant push_constant;
		ScreenSpaceReflectionScaleShaderRD shader;
		RID shader_version;
		RID pipeline;
	} ssr_scale;

	struct SubSurfaceScatteringPushConstant {
		int32_t screen_size[2];
		float camera_z_far;
		float camera_z_near;

		uint32_t vertical;
		uint32_t orthogonal;
		float unit_size;
		float scale;

		float depth_scale;
		uint32_t pad[3];
	};

	struct SubSurfaceScattering {
		SubSurfaceScatteringPushConstant push_constant;
		SubsurfaceScatteringShaderRD shader;
		RID shader_version;
		RID pipelines[3]; //3 quality levels
	} sss;

	struct ResolvePushConstant {
		int32_t screen_size[2];
		int32_t samples;
		uint32_t pad;
	};

	enum ResolveMode {
		RESOLVE_MODE_GI,
		RESOLVE_MODE_GI_VOXEL_GI,
		RESOLVE_MODE_DEPTH,
		RESOLVE_MODE_MAX
	};

	struct Resolve {
		ResolvePushConstant push_constant;
		ResolveShaderRD shader;
		RID shader_version;
		RID pipelines[RESOLVE_MODE_MAX]; //3 quality levels
	} resolve;

	enum SortMode {
		SORT_MODE_BLOCK,
		SORT_MODE_STEP,
		SORT_MODE_INNER,
		SORT_MODE_MAX
	};

	struct Sort {
		struct PushConstant {
			uint32_t total_elements;
			uint32_t pad[3];
			int32_t job_params[4];
		};

		SortShaderRD shader;
		RID shader_version;
		RID pipelines[SORT_MODE_MAX];
	} sort;

	RID default_sampler;
	RID default_mipmap_sampler;
	RID index_buffer;
	RID index_array;

	Map<RID, RID> texture_to_uniform_set_cache;
	Map<RID, RID> input_to_uniform_set_cache;

	Map<RID, RID> image_to_uniform_set_cache;

	struct TexturePair {
		RID texture1;
		RID texture2;
		_FORCE_INLINE_ bool operator<(const TexturePair &p_pair) const {
			if (texture1 == p_pair.texture1) {
				return texture2 < p_pair.texture2;
			} else {
				return texture1 < p_pair.texture1;
			}
		}
	};

	struct TextureSamplerPair {
		RID texture;
		RID sampler;
		_FORCE_INLINE_ bool operator<(const TextureSamplerPair &p_pair) const {
			if (texture == p_pair.texture) {
				return sampler < p_pair.sampler;
			} else {
				return texture < p_pair.texture;
			}
		}
	};

	Map<RID, RID> texture_to_compute_uniform_set_cache;
	Map<TexturePair, RID> texture_pair_to_compute_uniform_set_cache;
	Map<TexturePair, RID> image_pair_to_compute_uniform_set_cache;
	Map<TextureSamplerPair, RID> texture_sampler_to_compute_uniform_set_cache;

	RID _get_uniform_set_from_image(RID p_texture);
	RID _get_uniform_set_for_input(RID p_texture);
	RID _get_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_texture_and_sampler(RID p_texture, RID p_sampler);
	RID _get_compute_uniform_set_from_texture_pair(RID p_texture, RID p_texture2, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_image_pair(RID p_texture, RID p_texture2);

public:
	bool get_prefer_raster_effects();

	void fsr_upscale(RID p_source_rd_texture, RID p_secondary_texture, RID p_destination_texture, const Size2i &p_internal_size, const Size2i &p_size, float p_fsr_upscale_sharpness);
	void copy_to_fb_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2i &p_rect, bool p_flip_y = false, bool p_force_luminance = false, bool p_alpha_to_zero = false, bool p_srgb = false, RID p_secondary = RID());
	void copy_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y = false, bool p_force_luminance = false, bool p_all_source = false, bool p_8_bit_dst = false, bool p_alpha_to_one = false);
	void copy_cubemap_to_panorama(RID p_source_cube, RID p_dest_panorama, const Size2i &p_panorama_size, float p_lod, bool p_is_array);
	void copy_depth_to_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2i &p_rect, bool p_flip_y = false);
	void copy_depth_to_rect_and_linearize(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, float p_z_near, float p_z_far);
	void copy_to_atlas_fb(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_uv_rect, RD::DrawListID p_draw_list, bool p_flip_y = false, bool p_panorama = false);
	void gaussian_blur(RID p_source_rd_texture, RID p_texture, RID p_back_texture, const Rect2i &p_region, bool p_8bit_dst = false);
	void set_color(RID p_dest_texture, const Color &p_color, const Rect2i &p_region, bool p_8bit_dst = false);
	void gaussian_glow(RID p_source_rd_texture, RID p_back_texture, const Size2i &p_size, float p_strength = 1.0, bool p_high_quality = false, bool p_first_pass = false, float p_luminance_cap = 16.0, float p_exposure = 1.0, float p_bloom = 0.0, float p_hdr_bleed_threshold = 1.0, float p_hdr_bleed_scale = 1.0, RID p_auto_exposure = RID(), float p_auto_exposure_grey = 1.0);
	void gaussian_glow_raster(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, float p_strength = 1.0, bool p_high_quality = false, bool p_first_pass = false, float p_luminance_cap = 16.0, float p_exposure = 1.0, float p_bloom = 0.0, float p_hdr_bleed_threshold = 1.0, float p_hdr_bleed_scale = 1.0, RID p_auto_exposure = RID(), float p_auto_exposure_grey = 1.0);

	void cubemap_roughness(RID p_source_rd_texture, RID p_dest_texture, uint32_t p_face_id, uint32_t p_sample_count, float p_roughness, float p_size);
	void cubemap_roughness_raster(RID p_source_rd_texture, RID p_dest_framebuffer, uint32_t p_face_id, uint32_t p_sample_count, float p_roughness, float p_size);
	void make_mipmap(RID p_source_rd_texture, RID p_dest_texture, const Size2i &p_size);
	void make_mipmap_raster(RID p_source_rd_texture, RID p_dest_framebuffer, const Size2i &p_size);
	void copy_cubemap_to_dp(RID p_source_rd_texture, RID p_dst_framebuffer, const Rect2 &p_rect, const Vector2 &p_dst_size, float p_z_near, float p_z_far, bool p_dp_flip);
	void luminance_reduction(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);
	void luminance_reduction_raster(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, Vector<RID> p_fb, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);

	struct BokehBuffers {
		// bokeh buffers

		// textures
		Size2i base_texture_size;
		RID base_texture;
		RID depth_texture;
		RID secondary_texture;
		RID half_texture[2];

		// raster only
		RID base_fb;
		RID secondary_fb; // with weights
		RID half_fb[2]; // with weights
		RID base_weight_fb;
		RID weight_texture[4];
	};

	void bokeh_dof(const BokehBuffers &p_buffers, bool p_dof_far, float p_dof_far_begin, float p_dof_far_size, bool p_dof_near, float p_dof_near_begin, float p_dof_near_size, float p_bokeh_size, RS::DOFBokehShape p_bokeh_shape, RS::DOFBlurQuality p_quality, bool p_use_jitter, float p_cam_znear, float p_cam_zfar, bool p_cam_orthogonal);
	void bokeh_dof_raster(const BokehBuffers &p_buffers, bool p_dof_far, float p_dof_far_begin, float p_dof_far_size, bool p_dof_near, float p_dof_near_begin, float p_dof_near_size, float p_dof_blur_amount, RenderingServer::DOFBokehShape p_bokeh_shape, RS::DOFBlurQuality p_quality, float p_cam_znear, float p_cam_zfar, bool p_cam_orthogonal);

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
		float glow_levels[7] = { 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0 };
		Vector2i glow_texture_size;
		bool glow_use_bicubic_upscale = false;
		RID glow_texture;

		RS::EnvironmentToneMapper tonemap_mode = RS::ENV_TONE_MAPPER_LINEAR;
		float exposure = 1.0;
		float white = 1.0;

		bool use_auto_exposure = false;
		float auto_exposure_grey = 0.5;
		RID exposure_texture;
		float luminance_multiplier = 1.0;

		bool use_bcs = false;
		float brightness = 1.0;
		float contrast = 1.0;
		float saturation = 1.0;

		bool use_color_correction = false;
		bool use_1d_color_correction = false;
		RID color_correction_texture;

		bool use_fxaa = false;
		bool use_debanding = false;
		Vector2i texture_size;
		uint32_t view_count = 1;
	};

	struct SSAOSettings {
		float radius = 1.0;
		float intensity = 2.0;
		float power = 1.5;
		float detail = 0.5;
		float horizon = 0.06;
		float sharpness = 0.98;

		RS::EnvironmentSSAOQuality quality = RS::ENV_SSAO_QUALITY_MEDIUM;
		bool half_size = false;
		float adaptive_target = 0.5;
		int blur_passes = 2;
		float fadeout_from = 50.0;
		float fadeout_to = 300.0;

		Size2i full_screen_size = Size2i();
		Size2i half_screen_size = Size2i();
		Size2i quarter_screen_size = Size2i();
	};

	void tonemapper(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings);
	void tonemapper(RD::DrawListID p_subpass_draw_list, RID p_source_color, RD::FramebufferFormatID p_dst_format_id, const TonemapSettings &p_settings);

	void gather_ssao(RD::ComputeListID p_compute_list, const Vector<RID> p_ao_slices, const SSAOSettings &p_settings, bool p_adaptive_base_pass, RID p_gather_uniform_set, RID p_importance_map_uniform_set);
	void generate_ssao(RID p_depth_buffer, RID p_normal_buffer, RID p_depth_mipmaps_texture, const Vector<RID> &depth_mipmaps, RID p_ao, const Vector<RID> p_ao_slices, RID p_ao_pong, const Vector<RID> p_ao_pong_slices, RID p_upscale_buffer, RID p_importance_map, RID p_importance_map_pong, const CameraMatrix &p_projection, const SSAOSettings &p_settings, bool p_invalidate_uniform_sets, RID &r_downsample_uniform_set, RID &r_gather_uniform_set, RID &r_importance_map_uniform_set);

	void roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve);
	void cubemap_downsample(RID p_source_cubemap, RID p_dest_cubemap, const Size2i &p_size);
	void cubemap_downsample_raster(RID p_source_cubemap, RID p_dest_framebuffer, uint32_t p_face_id, const Size2i &p_size);
	void cubemap_filter(RID p_source_cubemap, Vector<RID> p_dest_cubemap, bool p_use_array);
	void cubemap_filter_raster(RID p_source_cubemap, RID p_dest_framebuffer, uint32_t p_face_id, uint32_t p_mip_level);

	void screen_space_reflection(RID p_diffuse, RID p_normal_roughness, RS::EnvironmentSSRRoughnessQuality p_roughness_quality, RID p_blur_radius, RID p_blur_radius2, RID p_metallic, const Color &p_metallic_mask, RID p_depth, RID p_scale_depth, RID p_scale_normal, RID p_output, RID p_output_blur, const Size2i &p_screen_size, int p_max_steps, float p_fade_in, float p_fade_out, float p_tolerance, const CameraMatrix &p_camera);
	void merge_specular(RID p_dest_framebuffer, RID p_specular, RID p_base, RID p_reflection);
	void sub_surface_scattering(RID p_diffuse, RID p_diffuse2, RID p_depth, const CameraMatrix &p_camera, const Size2i &p_screen_size, float p_scale, float p_depth_scale, RS::SubSurfaceScatteringQuality p_quality);

	void resolve_gi(RID p_source_depth, RID p_source_normal_roughness, RID p_source_voxel_gi, RID p_dest_depth, RID p_dest_normal_roughness, RID p_dest_voxel_gi, Vector2i p_screen_size, int p_samples, uint32_t p_barrier = RD::BARRIER_MASK_ALL);
	void resolve_depth(RID p_source_depth, RID p_dest_depth, Vector2i p_screen_size, int p_samples, uint32_t p_barrier = RD::BARRIER_MASK_ALL);

	void sort_buffer(RID p_uniform_set, int p_size);

	EffectsRD(bool p_prefer_raster_effects);
	~EffectsRD();
};

#endif // !RASTERIZER_EFFECTS_RD_H

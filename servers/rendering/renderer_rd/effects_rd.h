/*************************************************************************/
/*  effects_rd.h                                                         */
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

#ifndef EFFECTS_RD_H
#define EFFECTS_RD_H

#include "core/math/camera_matrix.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/fsr_upscale.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/luminance_reduce.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/luminance_reduce_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/roughness_limiter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/screen_space_reflection.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/screen_space_reflection_filter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/screen_space_reflection_scale.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sort.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/specular_merge.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ss_effects_downsample.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_blur.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_importance_map.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssao_interleave.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssil.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssil_blur.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssil_importance_map.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/ssil_interleave.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/subsurface_scattering.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/taa_resolve.glsl.gen.h"
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

	struct TAAResolvePushConstant {
		float resolution_width;
		float resolution_height;
		float disocclusion_threshold;
		float disocclusion_scale;
	};

	struct TAAResolve {
		TAAResolvePushConstant push_constant;
		TaaResolveShaderRD shader;
		RID shader_version;
		RID pipeline;
	} TAA_resolve;

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

	struct SSEffectsDownsamplePushConstant {
		float pixel_size[2];
		float z_far;
		float z_near;
		uint32_t orthogonal;
		float radius_sq;
		uint32_t pad[2];
	};

	enum SSEffectsMode {
		SS_EFFECTS_DOWNSAMPLE,
		SS_EFFECTS_DOWNSAMPLE_HALF_RES,
		SS_EFFECTS_DOWNSAMPLE_MIPMAP,
		SS_EFFECTS_DOWNSAMPLE_MIPMAP_HALF_RES,
		SS_EFFECTS_DOWNSAMPLE_HALF,
		SS_EFFECTS_DOWNSAMPLE_HALF_RES_HALF,
		SS_EFFECTS_DOWNSAMPLE_FULL_MIPS,
		SS_EFFECTS_MAX
	};

	struct SSEffectsGatherConstants {
		float rotation_matrices[80]; //5 vec4s * 4
	};

	struct SSEffects {
		SSEffectsDownsamplePushConstant downsample_push_constant;
		SsEffectsDownsampleShaderRD downsample_shader;
		RID downsample_shader_version;
		RID downsample_uniform_set;
		bool used_half_size_last_frame = false;
		bool used_mips_last_frame = false;
		bool used_full_mips_last_frame = false;

		RID gather_constants_buffer;

		RID mirror_sampler;

		RID pipelines[SS_EFFECTS_MAX];
	} ss_effects;

	enum SSAOMode {
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

		uint32_t is_orthogonal;
		float neg_inv_radius;
		float load_counter_avg_div;
		float adaptive_sample_limit;

		int32_t pass_coord_offset[2];
		float pass_uv_offset[2];
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
		SSAOGatherPushConstant gather_push_constant;
		SsaoShaderRD gather_shader;
		RID gather_shader_version;

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

		RID pipelines[SSAO_MAX];
	} ssao;

	enum SSILMode {
		SSIL_GATHER,
		SSIL_GATHER_BASE,
		SSIL_GATHER_ADAPTIVE,
		SSIL_GENERATE_IMPORTANCE_MAP,
		SSIL_PROCESS_IMPORTANCE_MAPA,
		SSIL_PROCESS_IMPORTANCE_MAPB,
		SSIL_BLUR_PASS,
		SSIL_BLUR_PASS_SMART,
		SSIL_BLUR_PASS_WIDE,
		SSIL_INTERLEAVE,
		SSIL_INTERLEAVE_SMART,
		SSIL_INTERLEAVE_HALF,
		SSIL_MAX
	};

	struct SSILGatherPushConstant {
		int32_t screen_size[2];
		int pass;
		int quality;

		float half_screen_pixel_size[2];
		float half_screen_pixel_size_x025[2];

		float NDC_to_view_mul[2];
		float NDC_to_view_add[2];

		float pad2[2];
		float z_near;
		float z_far;

		float radius;
		float intensity;
		int size_multiplier;
		int pad;

		float fade_out_mul;
		float fade_out_add;
		float normal_rejection_amount;
		float inv_radius_near_limit;

		uint32_t is_orthogonal;
		float neg_inv_radius;
		float load_counter_avg_div;
		float adaptive_sample_limit;

		int32_t pass_coord_offset[2];
		float pass_uv_offset[2];
	};

	struct SSILImportanceMapPushConstant {
		float half_screen_pixel_size[2];
		float intensity;
		float pad;
	};

	struct SSILBlurPushConstant {
		float edge_sharpness;
		float pad;
		float half_screen_pixel_size[2];
	};

	struct SSILInterleavePushConstant {
		float inv_sharpness;
		uint32_t size_modifier;
		float pixel_size[2];
	};

	struct SSILProjectionUniforms {
		float inv_last_frame_projection_matrix[16];
	};

	struct SSIL {
		SSILGatherPushConstant gather_push_constant;
		SsilShaderRD gather_shader;
		RID gather_shader_version;
		RID projection_uniform_buffer;

		SSILImportanceMapPushConstant importance_map_push_constant;
		SsilImportanceMapShaderRD importance_map_shader;
		RID importance_map_shader_version;
		RID importance_map_load_counter;
		RID counter_uniform_set;

		SSILBlurPushConstant blur_push_constant;
		SsilBlurShaderRD blur_shader;
		RID blur_shader_version;

		SSILInterleavePushConstant interleave_push_constant;
		SsilInterleaveShaderRD interleave_shader;
		RID interleave_shader_version;

		RID pipelines[SSIL_MAX];
	} ssil;

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

	HashMap<RID, RID> texture_to_uniform_set_cache;
	HashMap<RID, RID> input_to_uniform_set_cache;

	HashMap<RID, RID> image_to_uniform_set_cache;

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

	RBMap<TexturePair, RID> texture_pair_to_uniform_set_cache;
	RBMap<RID, RID> texture_to_compute_uniform_set_cache;
	RBMap<TexturePair, RID> texture_pair_to_compute_uniform_set_cache;
	RBMap<TexturePair, RID> image_pair_to_compute_uniform_set_cache;
	RBMap<TextureSamplerPair, RID> texture_sampler_to_compute_uniform_set_cache;

	RID _get_uniform_set_from_image(RID p_texture);
	RID _get_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_texture_and_sampler(RID p_texture, RID p_sampler);
	RID _get_compute_uniform_set_from_texture_pair(RID p_texture, RID p_texture2, bool p_use_mipmaps = false);
	RID _get_compute_uniform_set_from_image_pair(RID p_texture, RID p_texture2);

public:
	bool get_prefer_raster_effects();

	void fsr_upscale(RID p_source_rd_texture, RID p_secondary_texture, RID p_destination_texture, const Size2i &p_internal_size, const Size2i &p_size, float p_fsr_upscale_sharpness);
	void taa_resolve(RID p_frame, RID p_temp, RID p_depth, RID p_velocity, RID p_prev_velocity, RID p_history, Size2 p_resolution, float p_z_near, float p_z_far);

	void luminance_reduction(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);
	void luminance_reduction_raster(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, Vector<RID> p_fb, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);

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

	struct SSILSettings {
		float radius = 1.0;
		float intensity = 2.0;
		float sharpness = 0.98;
		float normal_rejection = 1.0;

		RS::EnvironmentSSILQuality quality = RS::ENV_SSIL_QUALITY_MEDIUM;
		bool half_size = true;
		float adaptive_target = 0.5;
		int blur_passes = 4;
		float fadeout_from = 50.0;
		float fadeout_to = 300.0;

		Size2i full_screen_size = Size2i();
		Size2i half_screen_size = Size2i();
		Size2i quarter_screen_size = Size2i();
	};

	void downsample_depth(RID p_depth_buffer, const Vector<RID> &p_depth_mipmaps, RS::EnvironmentSSAOQuality p_ssao_quality, RS::EnvironmentSSILQuality p_ssil_quality, bool p_invalidate_uniform_set, bool p_ssao_half_size, bool p_ssil_half_size, Size2i p_full_screen_size, const CameraMatrix &p_projection);

	void gather_ssao(RD::ComputeListID p_compute_list, const Vector<RID> p_ao_slices, const SSAOSettings &p_settings, bool p_adaptive_base_pass, RID p_gather_uniform_set, RID p_importance_map_uniform_set);
	void generate_ssao(RID p_normal_buffer, RID p_depth_mipmaps_texture, RID p_ao, const Vector<RID> p_ao_slices, RID p_ao_pong, const Vector<RID> p_ao_pong_slices, RID p_upscale_buffer, RID p_importance_map, RID p_importance_map_pong, const CameraMatrix &p_projection, const SSAOSettings &p_settings, bool p_invalidate_uniform_sets, RID &r_gather_uniform_set, RID &r_importance_map_uniform_set);

	void gather_ssil(RD::ComputeListID p_compute_list, const Vector<RID> p_ssil_slices, const Vector<RID> p_edges_slices, const SSILSettings &p_settings, bool p_adaptive_base_pass, RID p_gather_uniform_set, RID p_importance_map_uniform_set, RID p_projection_uniform_set);
	void screen_space_indirect_lighting(RID p_diffuse, RID p_destination, RID p_normal_buffer, RID p_depth_mipmaps_texture, RID p_ao, const Vector<RID> p_ao_slices, RID p_ao_pong, const Vector<RID> p_ao_pong_slices, RID p_importance_map, RID p_importance_map_pong, RID p_edges, const Vector<RID> p_edges_slices, const CameraMatrix &p_projection, const CameraMatrix &p_last_projection, const SSILSettings &p_settings, bool p_invalidate_uniform_sets, RID &r_gather_uniform_set, RID &r_importance_map_uniform_set, RID &r_projection_uniform_set);

	void roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve);

	void screen_space_reflection(RID p_diffuse, RID p_normal_roughness, RS::EnvironmentSSRRoughnessQuality p_roughness_quality, RID p_blur_radius, RID p_blur_radius2, RID p_metallic, const Color &p_metallic_mask, RID p_depth, RID p_scale_depth, RID p_scale_normal, RID p_output, RID p_output_blur, const Size2i &p_screen_size, int p_max_steps, float p_fade_in, float p_fade_out, float p_tolerance, const CameraMatrix &p_camera);
	void merge_specular(RID p_dest_framebuffer, RID p_specular, RID p_base, RID p_reflection);
	void sub_surface_scattering(RID p_diffuse, RID p_diffuse2, RID p_depth, const CameraMatrix &p_camera, const Size2i &p_screen_size, float p_scale, float p_depth_scale, RS::SubSurfaceScatteringQuality p_quality);

	void sort_buffer(RID p_uniform_set, int p_size);

	EffectsRD(bool p_prefer_raster_effects);
	~EffectsRD();
};

#endif // !RASTERIZER_EFFECTS_RD_H

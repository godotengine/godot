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

#include "core/math/projection.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/fsr_upscale.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/luminance_reduce.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/luminance_reduce_raster.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/roughness_limiter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sort.glsl.gen.h"
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

public:
	bool get_prefer_raster_effects();

	void fsr_upscale(RID p_source_rd_texture, RID p_secondary_texture, RID p_destination_texture, const Size2i &p_internal_size, const Size2i &p_size, float p_fsr_upscale_sharpness);
	void taa_resolve(RID p_frame, RID p_temp, RID p_depth, RID p_velocity, RID p_prev_velocity, RID p_history, Size2 p_resolution, float p_z_near, float p_z_far);

	void luminance_reduction(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);
	void luminance_reduction_raster(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, Vector<RID> p_fb, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set = false);

	void roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve);

	void sub_surface_scattering(RID p_diffuse, RID p_diffuse2, RID p_depth, const Projection &p_camera, const Size2i &p_screen_size, float p_scale, float p_depth_scale, RS::SubSurfaceScatteringQuality p_quality);

	void sort_buffer(RID p_uniform_set, int p_size);

	EffectsRD(bool p_prefer_raster_effects);
	~EffectsRD();
};

#endif // EFFECTS_RD_H

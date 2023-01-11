/**************************************************************************/
/*  effects_rd.h                                                          */
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

#ifndef EFFECTS_RD_H
#define EFFECTS_RD_H

#include "core/math/projection.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/roughness_limiter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sort.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"

#include "servers/rendering_server.h"

class EffectsRD {
private:
	bool prefer_raster_effects;

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
	RID _get_compute_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps = false);

public:
	bool get_prefer_raster_effects();

	void roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve);

	void sort_buffer(RID p_uniform_set, int p_size);

	EffectsRD(bool p_prefer_raster_effects);
	~EffectsRD();
};

#endif // EFFECTS_RD_H

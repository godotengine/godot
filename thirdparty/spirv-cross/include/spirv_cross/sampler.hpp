/*
 * Copyright 2015-2017 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_SAMPLER_HPP
#define SPIRV_CROSS_SAMPLER_HPP

#include <vector>

namespace spirv_cross
{
struct spirv_cross_sampler_2d
{
	inline virtual ~spirv_cross_sampler_2d()
	{
	}
};

template <typename T>
struct sampler2DBase : spirv_cross_sampler_2d
{
	sampler2DBase(const spirv_cross_sampler_info *info)
	{
		mips.insert(mips.end(), info->mipmaps, info->mipmaps + info->num_mipmaps);
		format = info->format;
		wrap_s = info->wrap_s;
		wrap_t = info->wrap_t;
		min_filter = info->min_filter;
		mag_filter = info->mag_filter;
		mip_filter = info->mip_filter;
	}

	inline virtual T sample(glm::vec2 uv, float bias)
	{
		return sampleLod(uv, bias);
	}

	inline virtual T sampleLod(glm::vec2 uv, float lod)
	{
		if (mag_filter == SPIRV_CROSS_FILTER_NEAREST)
		{
			uv.x = wrap(uv.x, wrap_s, mips[0].width);
			uv.y = wrap(uv.y, wrap_t, mips[0].height);
			glm::vec2 uv_full = uv * glm::vec2(mips[0].width, mips[0].height);

			int x = int(uv_full.x);
			int y = int(uv_full.y);
			return sample(x, y, 0);
		}
		else
		{
			return T(0, 0, 0, 1);
		}
	}

	inline float wrap(float v, spirv_cross_wrap wrap, unsigned size)
	{
		switch (wrap)
		{
		case SPIRV_CROSS_WRAP_REPEAT:
			return v - glm::floor(v);
		case SPIRV_CROSS_WRAP_CLAMP_TO_EDGE:
		{
			float half = 0.5f / size;
			return glm::clamp(v, half, 1.0f - half);
		}

		default:
			return 0.0f;
		}
	}

	std::vector<spirv_cross_miplevel> mips;
	spirv_cross_format format;
	spirv_cross_wrap wrap_s;
	spirv_cross_wrap wrap_t;
	spirv_cross_filter min_filter;
	spirv_cross_filter mag_filter;
	spirv_cross_mipfilter mip_filter;
};

typedef sampler2DBase<glm::vec4> sampler2D;
typedef sampler2DBase<glm::ivec4> isampler2D;
typedef sampler2DBase<glm::uvec4> usampler2D;

template <typename T>
inline T texture(const sampler2DBase<T> &samp, const glm::vec2 &uv, float bias = 0.0f)
{
	return samp.sample(uv, bias);
}
}

#endif

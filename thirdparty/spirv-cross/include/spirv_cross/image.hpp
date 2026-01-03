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

#ifndef SPIRV_CROSS_IMAGE_HPP
#define SPIRV_CROSS_IMAGE_HPP

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE
#endif

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/glm.hpp>

namespace spirv_cross
{
template <typename T>
struct image2DBase
{
	virtual ~image2DBase() = default;
	inline virtual T load(glm::ivec2 coord) const
	{
		return T(0, 0, 0, 1);
	}
	inline virtual void store(glm::ivec2 coord, const T &v)
	{
	}
};

typedef image2DBase<glm::vec4> image2D;
typedef image2DBase<glm::ivec4> iimage2D;
typedef image2DBase<glm::uvec4> uimage2D;

template <typename T>
inline T imageLoad(const image2DBase<T> &image, glm::ivec2 coord)
{
	return image.load(coord);
}

template <typename T>
void imageStore(image2DBase<T> &image, glm::ivec2 coord, const T &value)
{
	image.store(coord, value);
}
}

#endif

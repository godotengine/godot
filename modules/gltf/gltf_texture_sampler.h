/*************************************************************************/
/*  gltf_texture_sampler.h                                               */
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

#ifndef GLTF_TEXTURE_SAMPLER_H
#define GLTF_TEXTURE_SAMPLER_H

#include "core/resource.h"

class GLTFTextureSampler : public Resource {
	GDCLASS(GLTFTextureSampler, Resource);

public:
	enum class MagFilter {
		NEAREST = 9728,
		LINEAR = 9729
	};

	enum class MinFilter {
		NEAREST = 9728,
		LINEAR = 9729,
		NEAREST_MIPMAP_NEAREST = 9984,
		LINEAR_MIPMAP_NEAREST = 9985,
		NEAREST_MIPMAP_LINEAR = 9986,
		LINEAR_MIPMAP_LINEAR = 9987
	};

	enum class WrapMode {
		CLAMP_TO_EDGE = 33071,
		MIRRORED_REPEAT = 33648,
		REPEAT = 10497,
		DEFAULT = REPEAT
	};

	int get_mag_filter() const {
		return (int)mag_filter;
	};

	void set_mag_filter(const int filter_mode) {
		mag_filter = (MagFilter)filter_mode;
	};

	int get_min_filter() const {
		return (int)min_filter;
	};

	void set_min_filter(const int filter_mode) {
		min_filter = (MinFilter)filter_mode;
	};

	int get_wrap_s() const {
		return (int)wrap_s;
	};

	void set_wrap_s(const int wrap_mode) {
		wrap_s = (WrapMode)wrap_mode;
	};

	int get_wrap_t() const {
		return (int)wrap_t;
	};

	void set_wrap_t(const int wrap_mode) {
		wrap_s = (WrapMode)wrap_mode;
	};

protected:
	static void _bind_methods();

private:
	MagFilter mag_filter;
	MinFilter min_filter;
	WrapMode wrap_s;
	WrapMode wrap_t;
};

#endif
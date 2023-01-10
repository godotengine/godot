/**************************************************************************/
/*  gltf_texture_sampler.h                                                */
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

#ifndef GLTF_TEXTURE_SAMPLER_H
#define GLTF_TEXTURE_SAMPLER_H

#include "core/resource.h"
#include "scene/resources/texture.h"

class GLTFTextureSampler : public Resource {
	GDCLASS(GLTFTextureSampler, Resource);

public:
	enum FilterMode {
		NEAREST = 9728,
		LINEAR = 9729,
		NEAREST_MIPMAP_NEAREST = 9984,
		LINEAR_MIPMAP_NEAREST = 9985,
		NEAREST_MIPMAP_LINEAR = 9986,
		LINEAR_MIPMAP_LINEAR = 9987
	};

	enum WrapMode {
		CLAMP_TO_EDGE = 33071,
		MIRRORED_REPEAT = 33648,
		REPEAT = 10497
	};

	int get_mag_filter() const {
		return mag_filter;
	}

	void set_mag_filter(const int filter_mode) {
		mag_filter = (FilterMode)filter_mode;
	}

	int get_min_filter() const {
		return min_filter;
	}

	void set_min_filter(const int filter_mode) {
		min_filter = (FilterMode)filter_mode;
	}

	int get_wrap_s() const {
		return wrap_s;
	}

	void set_wrap_s(const int wrap_mode) {
		wrap_s = (WrapMode)wrap_mode;
	}

	int get_wrap_t() const {
		return wrap_t;
	}

	void set_wrap_t(const int wrap_mode) {
		wrap_s = (WrapMode)wrap_mode;
	}

	Texture::Flags get_filter_mode() const {
		switch (min_filter) {
			case NEAREST:
				return (Texture::Flags)0;
			case LINEAR:
				return Texture::Flags::FLAG_FILTER;
			case NEAREST_MIPMAP_NEAREST:
			case NEAREST_MIPMAP_LINEAR:
				return Texture::Flags::FLAG_MIPMAPS;
			case LINEAR_MIPMAP_NEAREST:
			case LINEAR_MIPMAP_LINEAR:
			default:
				return (Texture::Flags)(Texture::Flags::FLAG_FILTER | Texture::Flags::FLAG_MIPMAPS);
		}
	}

	void set_filter_mode(uint32_t flags) {
		const bool filter = (flags & Texture::Flags::FLAG_FILTER);
		const bool mipmaps = (flags & Texture::Flags::FLAG_MIPMAPS);

		if (filter && mipmaps) {
			min_filter = FilterMode::LINEAR_MIPMAP_LINEAR;
			mag_filter = FilterMode::LINEAR;
		} else if (filter) {
			min_filter = FilterMode::LINEAR;
			mag_filter = FilterMode::LINEAR;
		} else if (mipmaps) {
			min_filter = FilterMode::NEAREST_MIPMAP_LINEAR;
			mag_filter = FilterMode::NEAREST;
		} else {
			min_filter = FilterMode::NEAREST;
			mag_filter = FilterMode::NEAREST;
		}
	}

	Texture::Flags get_wrap_mode() const {
		if ((wrap_s == WrapMode::MIRRORED_REPEAT) && (wrap_t == WrapMode::MIRRORED_REPEAT)) {
			return Texture::Flags::FLAG_MIRRORED_REPEAT;
		} else if ((wrap_s == WrapMode::REPEAT) && (wrap_t == WrapMode::REPEAT)) {
			return Texture::Flags::FLAG_REPEAT;
		} else {
			return (Texture::Flags)0;
		}
	}

	void set_wrap_mode(uint32_t flags) {
		if (flags & Texture::Flags::FLAG_MIRRORED_REPEAT) {
			wrap_s = WrapMode::MIRRORED_REPEAT;
			wrap_t = WrapMode::MIRRORED_REPEAT;
		} else if (flags & Texture::Flags::FLAG_REPEAT) {
			wrap_s = WrapMode::REPEAT;
			wrap_t = WrapMode::REPEAT;
		} else {
			wrap_s = WrapMode::CLAMP_TO_EDGE;
			wrap_t = WrapMode::CLAMP_TO_EDGE;
		}
	}

	Texture::Flags get_texture_flags() const {
		return (Texture::Flags)(get_filter_mode() | get_wrap_mode());
	}

	void set_texture_flags(uint32_t flags) {
		set_filter_mode(flags);
		set_wrap_mode(flags);
	}

protected:
	static void _bind_methods();

private:
	FilterMode mag_filter = FilterMode::LINEAR;
	FilterMode min_filter = FilterMode::LINEAR_MIPMAP_LINEAR;
	WrapMode wrap_s = WrapMode::REPEAT;
	WrapMode wrap_t = WrapMode::REPEAT;
};

#endif // GLTF_TEXTURE_SAMPLER_H

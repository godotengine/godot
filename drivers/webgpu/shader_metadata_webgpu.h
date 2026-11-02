/**************************************************************************/
/*  shader_metadata_webgpu.h                                            */
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

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Inspector records travel with WGSL through disk caches and baked containers. Only this
// versioned prefix is parsed; layout creation does not depend on WGSL declaration formatting.
namespace webgpu_metadata {

enum Kind : uint32_t {
	UNIFORM_BUFFER,
	STORAGE_BUFFER,
	READ_ONLY_STORAGE_BUFFER,
	SAMPLER,
	COMPARISON_SAMPLER,
	TEXTURE,
	MULTISAMPLED_TEXTURE,
	DEPTH_TEXTURE,
	DEPTH_MULTISAMPLED_TEXTURE,
	WRITE_ONLY_STORAGE_TEXTURE,
	READ_ONLY_STORAGE_TEXTURE,
	READ_WRITE_STORAGE_TEXTURE,
	KIND_COUNT,
};

enum Dimension : uint32_t { NONE,
	D1,
	D2,
	D2_ARRAY,
	D3,
	CUBE,
	CUBE_ARRAY,
	DIMENSION_COUNT };
enum SampleType : uint32_t { FLOAT,
	UINT,
	SINT,
	UNFILTERABLE_FLOAT, // A float texture no sampler reads.
	SAMPLE_TYPE_COUNT };

struct Resource {
	uint32_t group = 0;
	uint32_t binding = 0;
	Kind kind = UNIFORM_BUFFER;
	Dimension dimension = NONE;
	SampleType sample_type = FLOAT;
	uint32_t depth_alias = UINT32_MAX;
	bool used = true;
	std::string format;
};

struct Shader {
	std::vector<Resource> resources;
	std::vector<uint32_t> overrides;
	bool has_overrides = false;
};

inline bool read_uint(const char *&r_text, uint32_t &r_value, char p_delimiter) {
	if (*r_text < '0' || *r_text > '9') {
		return false;
	}
	uint32_t value = 0;
	while (*r_text >= '0' && *r_text <= '9') {
		const uint32_t digit = *r_text++ - '0';
		if (value > (UINT32_MAX - digit) / 10) {
			return false;
		}
		value = value * 10 + digit;
	}
	if (*r_text++ != p_delimiter) {
		return false;
	}
	r_value = value;
	return true;
}

inline bool parse(const char *p_text, Shader &r_shader) {
	r_shader = {};
	if (!p_text || std::strncmp(p_text, "//RESOURCE_USAGE\n", 17) != 0) {
		return false;
	}
	p_text += 17;
	while (std::strncmp(p_text, "//RESOURCE_USED:", 16) == 0) {
		p_text += 16;
		uint32_t group, binding;
		if (!read_uint(p_text, group, ',') || !read_uint(p_text, binding, '\n') || group > 65535 || binding > 65535) {
			return false;
		}
	}
	constexpr const char *PREFIX = "//WEBGPU_METADATA:4,";
	if (std::strncmp(p_text, PREFIX, std::strlen(PREFIX)) != 0) {
		return false;
	}
	p_text += std::strlen(PREFIX);
	uint32_t has_overrides;
	if (!read_uint(p_text, has_overrides, '\n') || has_overrides > 1) {
		return false;
	}
	r_shader.has_overrides = has_overrides != 0;
	while (std::strncmp(p_text, "//BINDING:", 10) == 0) {
		p_text += 10;
		Resource resource;
		uint32_t kind, dimension, sample, used;
		if (!read_uint(p_text, resource.group, ',') || !read_uint(p_text, resource.binding, ',') ||
				!read_uint(p_text, kind, ',') || !read_uint(p_text, dimension, ',') ||
				!read_uint(p_text, sample, ',') || !read_uint(p_text, resource.depth_alias, ',') || !read_uint(p_text, used, ',') || used > 1) {
			return false;
		}
		if (resource.group > 65535 || resource.binding > 65535 || kind >= KIND_COUNT || dimension >= DIMENSION_COUNT || sample >= SAMPLE_TYPE_COUNT) {
			return false;
		}
		resource.kind = Kind(kind);
		resource.dimension = Dimension(dimension);
		resource.sample_type = SampleType(sample);
		resource.used = used != 0;
		while ((*p_text >= 'a' && *p_text <= 'z') || (*p_text >= '0' && *p_text <= '9')) {
			resource.format += *p_text++;
			if (resource.format.size() > 32) {
				return false;
			}
		}
		if (*p_text++ != '\n' || r_shader.resources.size() >= 65536) {
			return false;
		}
		r_shader.resources.push_back(resource);
	}
	while (std::strncmp(p_text, "//OVERRIDE_USED:", 16) == 0) {
		p_text += 16;
		uint32_t id;
		if (!read_uint(p_text, id, '\n') || id > 65535 || r_shader.overrides.size() >= 65536) {
			return false;
		}
		r_shader.overrides.push_back(id);
	}
	constexpr const char *END = "//WEBGPU_METADATA_END\n";
	return std::strncmp(p_text, END, std::strlen(END)) == 0;
}

inline std::string encode(const Shader &p_shader) {
	std::string result = "//RESOURCE_USAGE\n";
	for (const Resource &r : p_shader.resources) {
		if (r.used) {
			result += "//RESOURCE_USED:" + std::to_string(r.group) + "," + std::to_string(r.binding) + "\n";
		}
	}
	result += "//WEBGPU_METADATA:4," + std::to_string(p_shader.has_overrides) + "\n";
	for (const Resource &r : p_shader.resources) {
		result += "//BINDING:" + std::to_string(r.group) + "," + std::to_string(r.binding) + "," +
				std::to_string(r.kind) + "," + std::to_string(r.dimension) + "," + std::to_string(r.sample_type) + "," +
				std::to_string(r.depth_alias) + "," + std::to_string(r.used) + "," + r.format + "\n";
	}
	for (uint32_t id : p_shader.overrides) {
		result += "//OVERRIDE_USED:" + std::to_string(id) + "\n";
	}
	return result + "//WEBGPU_METADATA_END\n";
}

} // namespace webgpu_metadata

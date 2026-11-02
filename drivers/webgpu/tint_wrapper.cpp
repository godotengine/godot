/**************************************************************************/
/*  tint_wrapper.cpp                                                      */
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

#include "tint_wrapper.h"

#include "shader_metadata_webgpu.h"
#include "src/tint/api/tint.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/wgsl/program/program.h"
#include "src/tint/lang/wgsl/writer/common/options.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static const char *_storage_format(tint::inspector::ResourceBinding::TexelFormat p_format) {
	using I = tint::inspector::ResourceBinding;
	switch (p_format) {
		case I::TexelFormat::kR8Snorm:
			return "r8snorm";
		case I::TexelFormat::kR8Uint:
			return "r8uint";
		case I::TexelFormat::kR8Sint:
			return "r8sint";
		case I::TexelFormat::kRg8Unorm:
			return "rg8unorm";
		case I::TexelFormat::kRg8Snorm:
			return "rg8snorm";
		case I::TexelFormat::kRg8Uint:
			return "rg8uint";
		case I::TexelFormat::kRg8Sint:
			return "rg8sint";
		case I::TexelFormat::kR16Unorm:
			return "r16unorm";
		case I::TexelFormat::kR16Snorm:
			return "r16snorm";
		case I::TexelFormat::kR16Uint:
			return "r16uint";
		case I::TexelFormat::kR16Sint:
			return "r16sint";
		case I::TexelFormat::kR16Float:
			return "r16float";
		case I::TexelFormat::kRg16Unorm:
			return "rg16unorm";
		case I::TexelFormat::kRg16Snorm:
			return "rg16snorm";
		case I::TexelFormat::kRg16Uint:
			return "rg16uint";
		case I::TexelFormat::kRg16Sint:
			return "rg16sint";
		case I::TexelFormat::kRg16Float:
			return "rg16float";
		case I::TexelFormat::kBgra8Unorm:
			return "bgra8unorm";
		case I::TexelFormat::kRgba8Unorm:
			return "rgba8unorm";
		case I::TexelFormat::kRgba8Snorm:
			return "rgba8snorm";
		case I::TexelFormat::kRgba8Uint:
			return "rgba8uint";
		case I::TexelFormat::kRgba8Sint:
			return "rgba8sint";
		case I::TexelFormat::kRgba16Unorm:
			return "rgba16unorm";
		case I::TexelFormat::kRgba16Snorm:
			return "rgba16snorm";
		case I::TexelFormat::kRgba16Uint:
			return "rgba16uint";
		case I::TexelFormat::kRgba16Sint:
			return "rgba16sint";
		case I::TexelFormat::kRgba16Float:
			return "rgba16float";
		case I::TexelFormat::kR32Uint:
			return "r32uint";
		case I::TexelFormat::kR32Sint:
			return "r32sint";
		case I::TexelFormat::kR32Float:
			return "r32float";
		case I::TexelFormat::kRg32Uint:
			return "rg32uint";
		case I::TexelFormat::kRg32Sint:
			return "rg32sint";
		case I::TexelFormat::kRg32Float:
			return "rg32float";
		case I::TexelFormat::kRgba32Uint:
			return "rgba32uint";
		case I::TexelFormat::kRgba32Sint:
			return "rgba32sint";
		case I::TexelFormat::kRgba32Float:
			return "rgba32float";
		case I::TexelFormat::kR8Unorm:
			return "r8unorm";
		case I::TexelFormat::kRgb10A2Uint:
			return "rgb10a2uint";
		case I::TexelFormat::kRgb10A2Unorm:
			return "rgb10a2unorm";
		case I::TexelFormat::kRg11B10Ufloat:
			return "rg11b10ufloat";
		default:
			return "";
	}
}

static webgpu_metadata::Resource _resource(const tint::inspector::ResourceBinding &p_binding) {
	using I = tint::inspector::ResourceBinding;
	webgpu_metadata::Resource r;
	r.group = p_binding.bind_group;
	r.binding = p_binding.binding;
	switch (p_binding.resource_type) {
		case I::ResourceType::kUniformBuffer:
			r.kind = webgpu_metadata::UNIFORM_BUFFER;
			break;
		case I::ResourceType::kStorageBuffer:
			r.kind = webgpu_metadata::STORAGE_BUFFER;
			break;
		case I::ResourceType::kReadOnlyStorageBuffer:
			r.kind = webgpu_metadata::READ_ONLY_STORAGE_BUFFER;
			break;
		case I::ResourceType::kSampler:
			r.kind = webgpu_metadata::SAMPLER;
			break;
		case I::ResourceType::kSampledTexture:
			r.kind = webgpu_metadata::TEXTURE;
			break;
		case I::ResourceType::kMultisampledTexture:
			r.kind = webgpu_metadata::MULTISAMPLED_TEXTURE;
			break;
		case I::ResourceType::kDepthTexture:
			r.kind = webgpu_metadata::DEPTH_TEXTURE;
			break;
		case I::ResourceType::kDepthMultisampledTexture:
			r.kind = webgpu_metadata::DEPTH_MULTISAMPLED_TEXTURE;
			break;
		case I::ResourceType::kWriteOnlyStorageTexture:
			r.kind = webgpu_metadata::WRITE_ONLY_STORAGE_TEXTURE;
			break;
		case I::ResourceType::kReadOnlyStorageTexture:
			r.kind = webgpu_metadata::READ_ONLY_STORAGE_TEXTURE;
			break;
		case I::ResourceType::kReadWriteStorageTexture:
			r.kind = webgpu_metadata::READ_WRITE_STORAGE_TEXTURE;
			break;
		case I::ResourceType::kInputAttachment:
			r.kind = webgpu_metadata::TEXTURE;
			break;
		default:
			r.kind = webgpu_metadata::KIND_COUNT;
			break;
	}
	if (r.kind == webgpu_metadata::SAMPLER && p_binding.sampler_type == I::SamplerType::kComparison) {
		r.kind = webgpu_metadata::COMPARISON_SAMPLER;
	}
	if (r.kind >= webgpu_metadata::TEXTURE && r.kind < webgpu_metadata::KIND_COUNT) {
		switch (p_binding.dim) {
			case I::TextureDimension::k1d:
				r.dimension = webgpu_metadata::D1;
				break;
			case I::TextureDimension::k2d:
				r.dimension = webgpu_metadata::D2;
				break;
			case I::TextureDimension::k2dArray:
				r.dimension = webgpu_metadata::D2_ARRAY;
				break;
			case I::TextureDimension::k3d:
				r.dimension = webgpu_metadata::D3;
				break;
			case I::TextureDimension::kCube:
				r.dimension = webgpu_metadata::CUBE;
				break;
			case I::TextureDimension::kCubeArray:
				r.dimension = webgpu_metadata::CUBE_ARRAY;
				break;
			case I::TextureDimension::kNone:
				r.dimension = webgpu_metadata::NONE;
				break;
		}
		if (r.kind != webgpu_metadata::DEPTH_TEXTURE && r.kind != webgpu_metadata::DEPTH_MULTISAMPLED_TEXTURE) {
			r.sample_type = p_binding.sampled_kind == I::SampledKind::kUInt ? webgpu_metadata::UINT : (p_binding.sampled_kind == I::SampledKind::kSInt ? webgpu_metadata::SINT : webgpu_metadata::FLOAT);
		}
		if (r.kind >= webgpu_metadata::WRITE_ONLY_STORAGE_TEXTURE) {
			r.format = _storage_format(p_binding.image_format);
		}
		const std::string &name = p_binding.variable_name;
		if (name.size() >= 12 && name.compare(name.size() - 12, 12, "_depth_alias") == 0 && r.binding > 0) {
			r.depth_alias = r.binding - 1;
		}
	}
	return r;
}

void tint_wrapper_initialize() {
	static std::once_flag init;
	std::call_once(init, [] { tint::Initialize(); });
}

char *tint_wrapper_spirv_to_wgsl(const uint32_t *p_spirv_words, size_t p_word_count, char **r_error) {
	tint_wrapper_initialize();
	// Opaque calls were inlined by spirv_preprocess::freeze_and_inline().
	std::vector<uint32_t> words(p_spirv_words, p_spirv_words + p_word_count);

	// Allow all WGSL extensions and language features so Tint can emit
	// constructs like readonly storage textures without validation errors.
	tint::wgsl::writer::Options wgsl_options;
	wgsl_options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
	// Godot's GLSL shaders use textureSample/dpdx in non-uniform control flow
	// (valid in Vulkan, but WGSL requires uniform control flow for derivatives).
	// This inserts `diagnostic(off, derivative_uniformity)` in the output.
	wgsl_options.allow_non_uniform_derivatives = true;

	// Keep the resolved Program so layout visibility can follow Tint's call
	// graph, including bindings introduced by texture/sampler lowering.
	auto convert = [&]() -> tint::Result<std::string> {
		TINT_CHECK_RESULT_UNWRAP(ir, tint::spirv::reader::ReadIR(words));
		TINT_CHECK_RESULT_UNWRAP(program, tint::wgsl::writer::ProgramFromIR(ir, wgsl_options));
		tint::inspector::Inspector inspector(program);
		webgpu_metadata::Shader metadata;
		metadata.has_overrides = !inspector.Overrides().empty();
		std::set<uint64_t> used_bindings;
		std::set<uint64_t> sampled_textures;
		for (const auto &entry : inspector.GetEntryPoints()) {
			for (const auto &binding : inspector.GetResourceBindings(entry.name)) {
				used_bindings.insert((uint64_t(binding.bind_group) << 32) | binding.binding);
			}
			for (const auto &use : inspector.GetSamplerTextureUses(entry.name)) {
				sampled_textures.insert((uint64_t(use.texture_binding_point.group) << 32) | use.texture_binding_point.binding);
			}
			for (const auto &constant : entry.overrides) {
				if (constant.is_id_specified) {
					metadata.overrides.push_back(constant.id.value);
				}
			}
		}
		for (const auto &binding : inspector.GetModuleResourceBindings()) {
			auto resource = _resource(binding);
			if (resource.kind == webgpu_metadata::KIND_COUNT) {
				return tint::Failure{ "WebGPU: unsupported reflected resource type" };
			}
			const uint64_t key = (uint64_t(resource.group) << 32) | resource.binding;
			resource.used = used_bindings.count(key) != 0;
			// A float texture that is only loaded binds as unfilterable, which also admits depth textures.
			if (resource.kind == webgpu_metadata::TEXTURE && resource.sample_type == webgpu_metadata::FLOAT && resource.used && !sampled_textures.count(key)) {
				resource.sample_type = webgpu_metadata::UNFILTERABLE_FLOAT;
			}
			metadata.resources.push_back(resource);
		}
		if (inspector.has_error()) {
			return tint::Failure{ inspector.error() };
		}
		TINT_CHECK_RESULT_UNWRAP(output, tint::wgsl::writer::Generate(program, wgsl_options));
		return webgpu_metadata::encode(metadata) + output.wgsl;
	};
	try {
		auto result = convert();
		if (result != tint::Success) {
			throw std::runtime_error(result.Failure().reason);
		}

		const std::string &wgsl = result.Get();
		char *out = (char *)malloc(wgsl.size() + 1);
		if (!out) {
			return nullptr;
		}
		memcpy(out, wgsl.c_str(), wgsl.size() + 1);
		return out;
	} catch (const std::exception &error) {
		if (r_error) {
			const size_t size = strlen(error.what()) + 1;
			*r_error = static_cast<char *>(malloc(size));
			if (*r_error) {
				memcpy(*r_error, error.what(), size);
			}
		}
		return nullptr;
	}
}

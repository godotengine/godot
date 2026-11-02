/**************************************************************************/
/*  rendering_shader_container_webgpu.cpp                                 */
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

#if defined(WEBGPU_ENABLED) || defined(WEBGPU_SHADER_BAKER_ENABLED)

#include "rendering_shader_container_webgpu.h"

#include "shader_metadata_webgpu.h"
#include "tint_wrapper.h"
#include "webgpu_varying_locations.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/compression.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/templates/hashfuncs.h"
#include "core/version.h"

#ifdef WEB_ENABLED
#include <emscripten/emscripten.h>
#endif
#include <cstdlib>
#include <cstring>
#include <vector>
// =============================================================================
// SPIR-V → WGSL conversion (Tint + SPIR-V preprocessing)
// =============================================================================
//
// emdawnwebgpu only accepts WGSL, so every shader stage must be converted from
// SPIR-V. Tint (C++, linked directly into the engine WASM) handles the
// conversion after spirv_preprocess::run_pipeline() (push constant rewriting,
// sampler splitting, Y-flip, etc.).
//
// CPU shader code survives device restarts; bound the cache for large editor exports.
// Uses a 64-bit key (two MurmurHash3 passes with different seeds) to avoid
// birthday-paradox collisions that a 32-bit hash would risk at ~1k entries.
static HashMap<uint64_t, String> _spv_to_wgsl_cache;
static Mutex _spv_to_wgsl_cache_mutex;

// Run SPIR-V preprocessing passes and translate to WGSL via Tint.
// Returns a malloc'd null-terminated WGSL string, or nullptr on failure.
static char *_translate_spirv_to_wgsl(const uint8_t *p_spv_ptr, int p_spv_size) {
	if (p_spv_size < 20 || (p_spv_size % 4) != 0) {
		return nullptr;
	}

	// Wrap raw bytes in a Vector for the preprocessing API.
	Vector<uint8_t> spv;
	spv.resize(p_spv_size);
	memcpy(spv.ptrw(), p_spv_ptr, p_spv_size);

	spv = spirv_preprocess::run_pipeline(spv);
#ifdef WEB_ENABLED
	EM_ASM({
		if (globalThis.webgpuDump) {
			globalThis.webgpuLastSpirv = Array.from(HEAPU8.subarray($0, $0 + $1));
		} }, spv.ptr(), (int)spv.size());
#endif

	// Convert to uint32_t words for Tint.
	int word_count = spv.size() / 4;
	std::vector<uint32_t> spirv_words(word_count);
	memcpy(spirv_words.data(), spv.ptr(), spv.size());

	// Call Tint SPIR-V → WGSL via the wrapper (isolates C++20 from Godot's build).
	char *error_msg = nullptr;
	char *out = tint_wrapper_spirv_to_wgsl(spirv_words.data(), spirv_words.size(), &error_msg);
	if (!out) {
		if (error_msg) {
			// Truncate the error message: Tint includes the full SPIR-V disassembly
			// which floods the console. Keep first 1500 chars for diagnostics.
			String err_str = String::utf8(error_msg);
			// Replace newlines with pipes to prevent console splitting.
			err_str = err_str.replace("\n", " | ");
			if (err_str.length() > 1500) {
				err_str = err_str.left(1500) + "... [truncated]";
			}
			ERR_PRINT(vformat("Tint SPIR-V→WGSL failed: %s", err_str));
			free(error_msg);
		} else {
			ERR_PRINT("Tint SPIR-V→WGSL failed (unknown error)");
		}
		return nullptr;
	}
	return out;
}

// Returns a malloc'd null-terminated WGSL string (caller must free), or nullptr on
// failure. Reuses translated shaders from memory or the persistent shader cache before invoking Tint.
char *RenderingShaderContainerWebGPU::compile_spirv(const uint8_t *p_spv_ptr, int p_spv_size) {
	MutexLock lock(_spv_to_wgsl_cache_mutex);
	uint32_t hash_lo = hash_murmur3_buffer(p_spv_ptr, p_spv_size);
	uint32_t hash_hi = hash_murmur3_buffer(p_spv_ptr, p_spv_size, 0x9E3779B9);
	uint64_t spv_hash = ((uint64_t)hash_hi << 32) | hash_lo;

	const String *cached = _spv_to_wgsl_cache.getptr(spv_hash);
	if (!cached && _spv_to_wgsl_cache.size() >= 1024) {
		_spv_to_wgsl_cache.clear();
	}
	String cache_path;
	if (!cached && GLOBAL_GET("rendering/shader_compiler/shader_cache/enabled")) {
		String cache_dir = Engine::get_singleton()->get_shader_cache_path();
		if (cache_dir.is_empty()) {
			cache_dir = "user://";
		}
		cache_dir = cache_dir.path_join("shader_cache/webgpu_wgsl_v4").path_join(GODOT_VERSION_HASH);
		if (DirAccess::make_dir_recursive_absolute(cache_dir) == OK) {
			cache_path = cache_dir.path_join(String::num_uint64(spv_hash, 16) + ".wgsl");
		}
	}
	if (!cache_path.is_empty() && FileAccess::exists(cache_path)) {
		String code = FileAccess::get_file_as_string(cache_path);
		webgpu_metadata::Shader reflection_info;
		if (webgpu_metadata::parse(code.utf8().get_data(), reflection_info)) {
			_spv_to_wgsl_cache[spv_hash] = code;
			cached = _spv_to_wgsl_cache.getptr(spv_hash);
		}
	}
	if (cached) {
		CharString cs = cached->utf8();
		size_t len = (size_t)cs.length() + 1;
		char *out = (char *)malloc(len);
		if (!out) {
			return nullptr;
		}
		memcpy(out, cs.get_data(), len);
		return out;
	}

	char *wgsl_str = _translate_spirv_to_wgsl(p_spv_ptr, p_spv_size);

	if (wgsl_str) {
		_spv_to_wgsl_cache[spv_hash] = String(wgsl_str);
		DEV_ASSERT(_spv_to_wgsl_cache.size() <= 1024);
		if (!cache_path.is_empty()) {
			// Other host processes must never observe a partially written shader.
			String temporary_path = cache_path + "." + itos(OS::get_singleton()->get_process_id()) + ".tmp";
			Ref<FileAccess> file = FileAccess::open(temporary_path, FileAccess::WRITE);
			if (file.is_valid()) {
				file->store_string(String(wgsl_str));
				file.unref();
				DirAccess::rename_absolute(temporary_path, cache_path);
			}
		}
	}

	return wgsl_str;
}

static bool _is_handle_uniform(RenderingDeviceCommons::UniformType p_type) {
	using RDC = RenderingDeviceCommons;
	return p_type == RDC::UNIFORM_TYPE_SAMPLER || p_type == RDC::UNIFORM_TYPE_TEXTURE ||
			p_type == RDC::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE || p_type == RDC::UNIFORM_TYPE_IMAGE ||
			p_type == RDC::UNIFORM_TYPE_INPUT_ATTACHMENT;
}

bool RenderingShaderContainerWebGPU::expand_bindings(RDC::ShaderReflection &r_reflection, spirv_preprocess::BindingMap &r_bindings) {
	for (int set = 0; set < r_reflection.uniform_sets.size(); set++) {
		Vector<RDC::ShaderUniform> expanded;
		uint32_t next_binding = 0;
		for (const auto &source : r_reflection.uniform_sets[set]) {
			next_binding = MAX(next_binding, source.binding);
			r_bindings[(uint64_t(set) << 32) | source.binding] = next_binding;
			const uint32_t count = _is_handle_uniform(source.type) ? MAX(source.length, 1u) : 1;
			ERR_FAIL_COND_V_MSG(count > spirv_preprocess::PUSH_CONSTANT_BINDING / 2 || next_binding > spirv_preprocess::PUSH_CONSTANT_BINDING / 2 - count,
					false, "WebGPU descriptor arrays exceed the binding range.");
			for (uint32_t element = 0; element < count; element++) {
				auto uniform = source;
				uniform.binding = next_binding++;
				if (_is_handle_uniform(source.type)) {
					uniform.length = 1;
				}
				expanded.push_back(uniform);
			}
		}
		r_reflection.uniform_sets.write[set] = expanded;
	}
	return true;
}

bool RenderingShaderContainerWebGPU::_set_code_from_spirv(const ReflectShader &p_shader) {
	const uint32_t stage_count = p_shader.shader_stages.size();
	shaders.resize(stage_count);
	std::vector<std::vector<uint32_t>> stages(stage_count);
	for (uint32_t i = 0; i < stage_count; i++) {
		const auto words = p_shader.shader_stages[i].spirv();
		stages[i].assign(words.ptr(), words.ptr() + words.size());
	}
	ERR_FAIL_COND_V_MSG(!webgpu_varyings::compact(stages), false, "WebGPU: malformed shader varying locations.");
	auto reflection = get_shader_reflection();
	spirv_preprocess::BindingMap bindings;
	ERR_FAIL_COND_V(!expand_bindings(reflection, bindings), false);
	for (uint32_t i = 0; i < stage_count; i++) {
		Vector<uint8_t> spirv;
		spirv.resize(stages[i].size() * 4);
		memcpy(spirv.ptrw(), stages[i].data(), spirv.size());
		spirv = spirv_preprocess::remap_bindings(spirv, bindings);
		if (const char *feature = spirv_preprocess::unsupported_feature(spirv)) {
			// A baking editor enables variants for its own device; WebGPU never selects these.
			if (bake_skips_unsupported) {
				print_verbose(vformat("WebGPU: not baking a %s variant, which uses %s.", String::utf8(shader_name.get_data()), feature));
				return false;
			}
			ERR_FAIL_V_MSG(false, vformat("WebGPU: shader '%s' uses %s, which WebGPU does not support.", String::utf8(shader_name.get_data()), feature));
		}
		char *wgsl = compile_spirv(spirv.ptr(), spirv.size());
		ERR_FAIL_NULL_V_MSG(wgsl, false, vformat("WebGPU: failed to translate stage %d of shader '%s' to WGSL.", (int)p_shader.shader_stages[i].shader_stage, String::utf8(shader_name.get_data())));
		const size_t wgsl_size = strlen(wgsl) + 1;
		PackedByteArray payload;
		payload.resize(sizeof(uint32_t) + spirv.size() + wgsl_size);
		encode_uint32(spirv.size(), payload.ptrw());
		memcpy(payload.ptrw() + sizeof(uint32_t), spirv.ptr(), spirv.size());
		memcpy(payload.ptrw() + sizeof(uint32_t) + spirv.size(), wgsl, wgsl_size);
		free(wgsl);
		Shader &stage = shaders.write[i];
		stage.shader_stage = p_shader.shader_stages[i].shader_stage;
		stage.code_decompressed_size = payload.size();
		stage.code_compressed_bytes.resize(payload.size());
		uint32_t compressed_size = 0;
		ERR_FAIL_COND_V(!compress_code(payload.ptr(), payload.size(), stage.code_compressed_bytes.ptrw(), &compressed_size, &stage.code_compression_flags), false);
		stage.code_compressed_bytes.resize(compressed_size);
	}
	header_data.push_constant_bind_group = p_shader.push_constant_size > 0 ? spirv_preprocess::PUSH_CONSTANT_SET : NO_PUSH_CONSTANTS;
	return true;
}

bool RenderingShaderContainerWebGPU::get_stage_code(uint32_t p_index, Vector<uint8_t> &r_spirv, CharString &r_wgsl) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_index, (uint32_t)shaders.size(), false);
	const Shader &stage = shaders[p_index];
	// Old SPIR-V-only containers remain usable; recompilation also supplies current metadata.
	if (stage.code_decompressed_size == 0) {
		std::vector<std::vector<uint32_t>> stages(shaders.size());
		for (int i = 0; i < shaders.size(); i++) {
			const auto &bytes = shaders[i].code_compressed_bytes;
			ERR_FAIL_COND_V(bytes.size() < 20 || bytes.size() % 4 != 0, false);
			stages[i].resize(bytes.size() / 4);
			memcpy(stages[i].data(), bytes.ptr(), bytes.size());
		}
		ERR_FAIL_COND_V(!webgpu_varyings::compact(stages), false);
		auto reflection = get_shader_reflection();
		spirv_preprocess::BindingMap bindings;
		ERR_FAIL_COND_V(!expand_bindings(reflection, bindings), false);
		r_spirv.resize(stages[p_index].size() * 4);
		memcpy(r_spirv.ptrw(), stages[p_index].data(), r_spirv.size());
		r_spirv = spirv_preprocess::remap_bindings(r_spirv, bindings);
		char *wgsl = compile_spirv(r_spirv.ptr(), r_spirv.size());
		ERR_FAIL_NULL_V(wgsl, false);
		r_wgsl = CharString(wgsl);
		free(wgsl);
		return true;
	}
	ERR_FAIL_COND_V_MSG(stage.code_decompressed_size < 25 || stage.code_decompressed_size > 64 * 1024 * 1024,
			false, "WebGPU: invalid baked shader size.");
	PackedByteArray payload;
	payload.resize(stage.code_decompressed_size);
	ERR_FAIL_COND_V(stage.code_compression_flags & ~COMPRESSION_FLAG_ZSTD, false);
	if (stage.code_compression_flags & COMPRESSION_FLAG_ZSTD) {
		ERR_FAIL_COND_V_MSG(Compression::decompress(payload.ptrw(), payload.size(), stage.code_compressed_bytes.ptr(), stage.code_compressed_bytes.size(), Compression::MODE_ZSTD) != payload.size(),
				false, "WebGPU: malformed compressed shader.");
	} else {
		ERR_FAIL_COND_V(stage.code_compressed_bytes.size() != payload.size(), false);
		memcpy(payload.ptrw(), stage.code_compressed_bytes.ptr(), payload.size());
	}
	const uint32_t spirv_size = decode_uint32(payload.ptr());
	ERR_FAIL_COND_V_MSG(spirv_size < 20 || spirv_size % 4 != 0 || spirv_size >= payload.size() - sizeof(uint32_t) || payload[payload.size() - 1] != 0,
			false, "WebGPU: malformed baked shader payload.");
	r_spirv.resize(spirv_size);
	memcpy(r_spirv.ptrw(), payload.ptr() + sizeof(uint32_t), spirv_size);
	ERR_FAIL_COND_V(decode_uint32(r_spirv.ptr()) != 0x07230203, false);
	const char *wgsl = (const char *)(payload.ptr() + sizeof(uint32_t) + spirv_size);
	webgpu_metadata::Shader reflection_info;
	if (!webgpu_metadata::parse(wgsl, reflection_info)) {
		// Retained SPIR-V upgrades containers written by older metadata producers.
		char *updated = compile_spirv(r_spirv.ptr(), r_spirv.size());
		ERR_FAIL_NULL_V(updated, false);
		r_wgsl = CharString(updated);
		free(updated);
		return true;
	}
	r_wgsl = CharString(wgsl);
	return true;
}

// =========================================================================
// Serialization
// =========================================================================

uint32_t RenderingShaderContainerWebGPU::_from_bytes_header_extra_data(const uint8_t *p_bytes) {
	if (p_bytes) {
		memcpy(&header_data, p_bytes, sizeof(HeaderData));
	}
	return sizeof(HeaderData);
}

uint32_t RenderingShaderContainerWebGPU::_to_bytes_header_extra_data(uint8_t *p_bytes) const {
	if (p_bytes) {
		memcpy(p_bytes, &header_data, sizeof(HeaderData));
	}
	return sizeof(HeaderData);
}

// =========================================================================
// Public API
// =========================================================================

RenderingShaderContainerWebGPU::RenderingShaderContainerWebGPU() {
}

RenderingShaderContainerWebGPU::~RenderingShaderContainerWebGPU() {
}

#endif // WEBGPU_ENABLED

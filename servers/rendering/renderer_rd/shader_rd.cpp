/*************************************************************************/
/*  shader_rd.cpp                                                        */
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

#include "shader_rd.h"

#include "core/io/compression.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"
#include "thirdparty/misc/smolv.h"

void ShaderRD::_add_stage(const char *p_code, StageType p_stage_type) {
	Vector<String> lines = String(p_code).split("\n");

	String text;

	for (int i = 0; i < lines.size(); i++) {
		String l = lines[i];
		bool push_chunk = false;

		StageTemplate::Chunk chunk;

		if (l.begins_with("#VERSION_DEFINES")) {
			chunk.type = StageTemplate::Chunk::TYPE_VERSION_DEFINES;
			push_chunk = true;
		} else if (l.begins_with("#GLOBALS")) {
			switch (p_stage_type) {
				case STAGE_TYPE_VERTEX:
					chunk.type = StageTemplate::Chunk::TYPE_VERTEX_GLOBALS;
					break;
				case STAGE_TYPE_FRAGMENT:
					chunk.type = StageTemplate::Chunk::TYPE_FRAGMENT_GLOBALS;
					break;
				case STAGE_TYPE_COMPUTE:
					chunk.type = StageTemplate::Chunk::TYPE_COMPUTE_GLOBALS;
					break;
				default: {
				}
			}

			push_chunk = true;
		} else if (l.begins_with("#MATERIAL_UNIFORMS")) {
			chunk.type = StageTemplate::Chunk::TYPE_MATERIAL_UNIFORMS;
			push_chunk = true;
		} else if (l.begins_with("#CODE")) {
			chunk.type = StageTemplate::Chunk::TYPE_CODE;
			push_chunk = true;
			chunk.code = l.replace_first("#CODE", String()).replace(":", "").strip_edges().to_upper();
		} else {
			text += l + "\n";
		}

		if (push_chunk) {
			if (!text.is_empty()) {
				StageTemplate::Chunk text_chunk;
				text_chunk.type = StageTemplate::Chunk::TYPE_TEXT;
				text_chunk.text = text.utf8();
				stage_templates[p_stage_type].chunks.push_back(text_chunk);
				text = String();
			}
			stage_templates[p_stage_type].chunks.push_back(chunk);
		}
	}

	if (!text.is_empty()) {
		StageTemplate::Chunk text_chunk;
		text_chunk.type = StageTemplate::Chunk::TYPE_TEXT;
		text_chunk.text = text.utf8();
		stage_templates[p_stage_type].chunks.push_back(text_chunk);
		text = String();
	}
}

void ShaderRD::setup(const char *p_vertex_code, const char *p_fragment_code, const char *p_compute_code, const char *p_name) {
	name = p_name;

	if (p_compute_code) {
		_add_stage(p_compute_code, STAGE_TYPE_COMPUTE);
		is_compute = true;
	} else {
		is_compute = false;
		if (p_vertex_code) {
			_add_stage(p_vertex_code, STAGE_TYPE_VERTEX);
		}
		if (p_fragment_code) {
			_add_stage(p_fragment_code, STAGE_TYPE_FRAGMENT);
		}
	}

	StringBuilder tohash;
	tohash.append("[SpirvCacheKey]");
	tohash.append(RenderingDevice::get_singleton()->shader_get_spirv_cache_key());
	tohash.append("[BinaryCacheKey]");
	tohash.append(RenderingDevice::get_singleton()->shader_get_binary_cache_key());
	tohash.append("[Vertex]");
	tohash.append(p_vertex_code ? p_vertex_code : "");
	tohash.append("[Fragment]");
	tohash.append(p_fragment_code ? p_fragment_code : "");
	tohash.append("[Compute]");
	tohash.append(p_compute_code ? p_compute_code : "");

	base_sha256 = tohash.as_string().sha256_text();
}

RID ShaderRD::version_create() {
	//initialize() was never called
	ERR_FAIL_COND_V(variant_defines.size() == 0, RID());

	Version version;
	version.dirty = true;
	version.valid = false;
	version.initialize_needed = true;
	version.variants = nullptr;
	return version_owner.make_rid(version);
}

void ShaderRD::_clear_version(Version *p_version) {
	//clear versions if they exist
	if (p_version->variants) {
		for (int i = 0; i < variant_defines.size(); i++) {
			if (variants_enabled[i]) {
				RD::get_singleton()->free(p_version->variants[i]);
			}
		}

		memdelete_arr(p_version->variants);
		if (p_version->variant_data) {
			memdelete_arr(p_version->variant_data);
		}
		p_version->variants = nullptr;
	}
}

void ShaderRD::_build_variant_code(StringBuilder &builder, uint32_t p_variant, const Version *p_version, const StageTemplate &p_template) {
	for (uint32_t i = 0; i < p_template.chunks.size(); i++) {
		const StageTemplate::Chunk &chunk = p_template.chunks[i];
		switch (chunk.type) {
			case StageTemplate::Chunk::TYPE_VERSION_DEFINES: {
				builder.append("\n"); //make sure defines begin at newline
				builder.append(general_defines.get_data());
				builder.append(variant_defines[p_variant].get_data());
				for (int j = 0; j < p_version->custom_defines.size(); j++) {
					builder.append(p_version->custom_defines[j].get_data());
				}
				builder.append("\n"); //make sure defines begin at newline
				if (p_version->uniforms.size()) {
					builder.append("#define MATERIAL_UNIFORMS_USED\n");
				}
				for (const KeyValue<StringName, CharString> &E : p_version->code_sections) {
					builder.append(String("#define ") + String(E.key) + "_CODE_USED\n");
				}
#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
				builder.append("#define MOLTENVK_USED\n");
#endif
			} break;
			case StageTemplate::Chunk::TYPE_MATERIAL_UNIFORMS: {
				builder.append(p_version->uniforms.get_data()); //uniforms (same for vertex and fragment)
			} break;
			case StageTemplate::Chunk::TYPE_VERTEX_GLOBALS: {
				builder.append(p_version->vertex_globals.get_data()); // vertex globals
			} break;
			case StageTemplate::Chunk::TYPE_FRAGMENT_GLOBALS: {
				builder.append(p_version->fragment_globals.get_data()); // fragment globals
			} break;
			case StageTemplate::Chunk::TYPE_COMPUTE_GLOBALS: {
				builder.append(p_version->compute_globals.get_data()); // compute globals
			} break;
			case StageTemplate::Chunk::TYPE_CODE: {
				if (p_version->code_sections.has(chunk.code)) {
					builder.append(p_version->code_sections[chunk.code].get_data());
				}
			} break;
			case StageTemplate::Chunk::TYPE_TEXT: {
				builder.append(chunk.text.get_data());
			} break;
		}
	}
}

void ShaderRD::_compile_variant(uint32_t p_variant, Version *p_version) {
	if (!variants_enabled[p_variant]) {
		return; //variant is disabled, return
	}

	Vector<RD::ShaderStageSPIRVData> stages;

	String error;
	String current_source;
	RD::ShaderStage current_stage = RD::SHADER_STAGE_VERTEX;
	bool build_ok = true;

	if (!is_compute) {
		//vertex stage

		StringBuilder builder;
		_build_variant_code(builder, p_variant, p_version, stage_templates[STAGE_TYPE_VERTEX]);

		current_source = builder.as_string();
		RD::ShaderStageSPIRVData stage;
		stage.spir_v = RD::get_singleton()->shader_compile_spirv_from_source(RD::SHADER_STAGE_VERTEX, current_source, RD::SHADER_LANGUAGE_GLSL, &error);
		if (stage.spir_v.size() == 0) {
			build_ok = false;
		} else {
			stage.shader_stage = RD::SHADER_STAGE_VERTEX;
			stages.push_back(stage);
		}
	}

	if (!is_compute && build_ok) {
		//fragment stage
		current_stage = RD::SHADER_STAGE_FRAGMENT;

		StringBuilder builder;
		_build_variant_code(builder, p_variant, p_version, stage_templates[STAGE_TYPE_FRAGMENT]);

		current_source = builder.as_string();
		RD::ShaderStageSPIRVData stage;
		stage.spir_v = RD::get_singleton()->shader_compile_spirv_from_source(RD::SHADER_STAGE_FRAGMENT, current_source, RD::SHADER_LANGUAGE_GLSL, &error);
		if (stage.spir_v.size() == 0) {
			build_ok = false;
		} else {
			stage.shader_stage = RD::SHADER_STAGE_FRAGMENT;
			stages.push_back(stage);
		}
	}

	if (is_compute) {
		//compute stage
		current_stage = RD::SHADER_STAGE_COMPUTE;

		StringBuilder builder;
		_build_variant_code(builder, p_variant, p_version, stage_templates[STAGE_TYPE_COMPUTE]);

		current_source = builder.as_string();

		RD::ShaderStageSPIRVData stage;
		stage.spir_v = RD::get_singleton()->shader_compile_spirv_from_source(RD::SHADER_STAGE_COMPUTE, current_source, RD::SHADER_LANGUAGE_GLSL, &error);
		if (stage.spir_v.size() == 0) {
			build_ok = false;
		} else {
			stage.shader_stage = RD::SHADER_STAGE_COMPUTE;
			stages.push_back(stage);
		}
	}

	if (!build_ok) {
		MutexLock lock(variant_set_mutex); //properly print the errors
		ERR_PRINT("Error compiling " + String(current_stage == RD::SHADER_STAGE_COMPUTE ? "Compute " : (current_stage == RD::SHADER_STAGE_VERTEX ? "Vertex" : "Fragment")) + " shader, variant #" + itos(p_variant) + " (" + variant_defines[p_variant].get_data() + ").");
		ERR_PRINT(error);

#ifdef DEBUG_ENABLED
		ERR_PRINT("code:\n" + current_source.get_with_code_lines());
#endif
		return;
	}

	Vector<uint8_t> shader_data = RD::get_singleton()->shader_compile_binary_from_spirv(stages, name + ":" + itos(p_variant));

	ERR_FAIL_COND(shader_data.size() == 0);

	RID shader = RD::get_singleton()->shader_create_from_bytecode(shader_data);
	{
		MutexLock lock(variant_set_mutex);
		p_version->variants[p_variant] = shader;
		p_version->variant_data[p_variant] = shader_data;
	}
}

RS::ShaderNativeSourceCode ShaderRD::version_get_native_source_code(RID p_version) {
	Version *version = version_owner.get_or_null(p_version);
	RS::ShaderNativeSourceCode source_code;
	ERR_FAIL_COND_V(!version, source_code);

	source_code.versions.resize(variant_defines.size());

	for (int i = 0; i < source_code.versions.size(); i++) {
		if (!is_compute) {
			//vertex stage

			StringBuilder builder;
			_build_variant_code(builder, i, version, stage_templates[STAGE_TYPE_VERTEX]);

			RS::ShaderNativeSourceCode::Version::Stage stage;
			stage.name = "vertex";
			stage.code = builder.as_string();

			source_code.versions.write[i].stages.push_back(stage);
		}

		if (!is_compute) {
			//fragment stage

			StringBuilder builder;
			_build_variant_code(builder, i, version, stage_templates[STAGE_TYPE_FRAGMENT]);

			RS::ShaderNativeSourceCode::Version::Stage stage;
			stage.name = "fragment";
			stage.code = builder.as_string();

			source_code.versions.write[i].stages.push_back(stage);
		}

		if (is_compute) {
			//compute stage

			StringBuilder builder;
			_build_variant_code(builder, i, version, stage_templates[STAGE_TYPE_COMPUTE]);

			RS::ShaderNativeSourceCode::Version::Stage stage;
			stage.name = "compute";
			stage.code = builder.as_string();

			source_code.versions.write[i].stages.push_back(stage);
		}
	}

	return source_code;
}

String ShaderRD::_version_get_sha1(Version *p_version) const {
	StringBuilder hash_build;

	hash_build.append("[uniforms]");
	hash_build.append(p_version->uniforms.get_data());
	hash_build.append("[vertex_globals]");
	hash_build.append(p_version->vertex_globals.get_data());
	hash_build.append("[fragment_globals]");
	hash_build.append(p_version->fragment_globals.get_data());
	hash_build.append("[compute_globals]");
	hash_build.append(p_version->compute_globals.get_data());

	Vector<StringName> code_sections;
	for (const KeyValue<StringName, CharString> &E : p_version->code_sections) {
		code_sections.push_back(E.key);
	}
	code_sections.sort_custom<StringName::AlphCompare>();

	for (int i = 0; i < code_sections.size(); i++) {
		hash_build.append(String("[code:") + String(code_sections[i]) + "]");
		hash_build.append(p_version->code_sections[code_sections[i]].get_data());
	}
	for (int i = 0; i < p_version->custom_defines.size(); i++) {
		hash_build.append("[custom_defines:" + itos(i) + "]");
		hash_build.append(p_version->custom_defines[i].get_data());
	}

	return hash_build.as_string().sha1_text();
}

static const char *shader_file_header = "GDSC";
static const uint32_t cache_file_version = 2;

bool ShaderRD::_load_from_cache(Version *p_version) {
	String sha1 = _version_get_sha1(p_version);
	String path = shader_cache_dir.plus_file(name).plus_file(base_sha256).plus_file(sha1) + ".cache";

	FileAccessRef f = FileAccess::open(path, FileAccess::READ);
	if (!f) {
		return false;
	}

	char header[5] = { 0, 0, 0, 0, 0 };
	f->get_buffer((uint8_t *)header, 4);
	ERR_FAIL_COND_V(header != String(shader_file_header), false);

	uint32_t file_version = f->get_32();
	if (file_version != cache_file_version) {
		return false; // wrong version
	}

	uint32_t variant_count = f->get_32();

	ERR_FAIL_COND_V(variant_count != (uint32_t)variant_defines.size(), false); //should not happen but check

	for (uint32_t i = 0; i < variant_count; i++) {
		uint32_t variant_size = f->get_32();
		ERR_FAIL_COND_V(variant_size == 0 && variants_enabled[i], false);
		if (!variants_enabled[i]) {
			continue;
		}
		Vector<uint8_t> variant_bytes;
		variant_bytes.resize(variant_size);

		uint32_t br = f->get_buffer(variant_bytes.ptrw(), variant_size);

		ERR_FAIL_COND_V(br != variant_size, false);

		p_version->variant_data[i] = variant_bytes;
	}

	for (uint32_t i = 0; i < variant_count; i++) {
		if (!variants_enabled[i]) {
			MutexLock lock(variant_set_mutex);
			p_version->variants[i] = RID();
			continue;
		}
		RID shader = RD::get_singleton()->shader_create_from_bytecode(p_version->variant_data[i]);
		if (shader.is_null()) {
			for (uint32_t j = 0; j < i; j++) {
				RD::get_singleton()->free(p_version->variants[i]);
			}
			ERR_FAIL_COND_V(shader.is_null(), false);
		}
		{
			MutexLock lock(variant_set_mutex);
			p_version->variants[i] = shader;
		}
	}

	memdelete_arr(p_version->variant_data); //clear stages
	p_version->variant_data = nullptr;
	p_version->valid = true;
	return true;
}

void ShaderRD::_save_to_cache(Version *p_version) {
	String sha1 = _version_get_sha1(p_version);
	String path = shader_cache_dir.plus_file(name).plus_file(base_sha256).plus_file(sha1) + ".cache";

	FileAccessRef f = FileAccess::open(path, FileAccess::WRITE);
	ERR_FAIL_COND(!f);
	f->store_buffer((const uint8_t *)shader_file_header, 4);
	f->store_32(cache_file_version); //file version
	uint32_t variant_count = variant_defines.size();
	f->store_32(variant_count); //variant count

	for (uint32_t i = 0; i < variant_count; i++) {
		f->store_32(p_version->variant_data[i].size()); //stage count
		f->store_buffer(p_version->variant_data[i].ptr(), p_version->variant_data[i].size());
	}

	f->close();
}

void ShaderRD::_compile_version(Version *p_version) {
	_clear_version(p_version);

	p_version->valid = false;
	p_version->dirty = false;

	p_version->variants = memnew_arr(RID, variant_defines.size());
	typedef Vector<uint8_t> ShaderStageData;
	p_version->variant_data = memnew_arr(ShaderStageData, variant_defines.size());

	if (shader_cache_dir_valid) {
		if (_load_from_cache(p_version)) {
			return;
		}
	}

#if 1

	RendererThreadPool::singleton->thread_work_pool.do_work(variant_defines.size(), this, &ShaderRD::_compile_variant, p_version);
#else
	for (int i = 0; i < variant_defines.size(); i++) {
		_compile_variant(i, p_version);
	}
#endif

	bool all_valid = true;
	for (int i = 0; i < variant_defines.size(); i++) {
		if (!variants_enabled[i]) {
			continue; //disabled
		}
		if (p_version->variants[i].is_null()) {
			all_valid = false;
			break;
		}
	}

	if (!all_valid) {
		//clear versions if they exist
		for (int i = 0; i < variant_defines.size(); i++) {
			if (!variants_enabled[i]) {
				continue; //disabled
			}
			if (!p_version->variants[i].is_null()) {
				RD::get_singleton()->free(p_version->variants[i]);
			}
		}
		memdelete_arr(p_version->variants);
		if (p_version->variant_data) {
			memdelete_arr(p_version->variant_data);
		}
		p_version->variants = nullptr;
		p_version->variant_data = nullptr;
		return;
	} else if (shader_cache_dir_valid) {
		//save shader cache
		_save_to_cache(p_version);
	}

	memdelete_arr(p_version->variant_data); //clear stages
	p_version->variant_data = nullptr;

	p_version->valid = true;
}

void ShaderRD::version_set_code(RID p_version, const Map<String, String> &p_code, const String &p_uniforms, const String &p_vertex_globals, const String &p_fragment_globals, const Vector<String> &p_custom_defines) {
	ERR_FAIL_COND(is_compute);

	Version *version = version_owner.get_or_null(p_version);
	ERR_FAIL_COND(!version);
	version->vertex_globals = p_vertex_globals.utf8();
	version->fragment_globals = p_fragment_globals.utf8();
	version->uniforms = p_uniforms.utf8();
	version->code_sections.clear();
	for (const KeyValue<String, String> &E : p_code) {
		version->code_sections[StringName(E.key.to_upper())] = E.value.utf8();
	}

	version->custom_defines.clear();
	for (int i = 0; i < p_custom_defines.size(); i++) {
		version->custom_defines.push_back(p_custom_defines[i].utf8());
	}

	version->dirty = true;
	if (version->initialize_needed) {
		_compile_version(version);
		version->initialize_needed = false;
	}
}

void ShaderRD::version_set_compute_code(RID p_version, const Map<String, String> &p_code, const String &p_uniforms, const String &p_compute_globals, const Vector<String> &p_custom_defines) {
	ERR_FAIL_COND(!is_compute);

	Version *version = version_owner.get_or_null(p_version);
	ERR_FAIL_COND(!version);

	version->compute_globals = p_compute_globals.utf8();
	version->uniforms = p_uniforms.utf8();

	version->code_sections.clear();
	for (const KeyValue<String, String> &E : p_code) {
		version->code_sections[StringName(E.key.to_upper())] = E.value.utf8();
	}

	version->custom_defines.clear();
	for (int i = 0; i < p_custom_defines.size(); i++) {
		version->custom_defines.push_back(p_custom_defines[i].utf8());
	}

	version->dirty = true;
	if (version->initialize_needed) {
		_compile_version(version);
		version->initialize_needed = false;
	}
}

bool ShaderRD::version_is_valid(RID p_version) {
	Version *version = version_owner.get_or_null(p_version);
	ERR_FAIL_COND_V(!version, false);

	if (version->dirty) {
		_compile_version(version);
	}

	return version->valid;
}

bool ShaderRD::version_free(RID p_version) {
	if (version_owner.owns(p_version)) {
		Version *version = version_owner.get_or_null(p_version);
		_clear_version(version);
		version_owner.free(p_version);
	} else {
		return false;
	}

	return true;
}

void ShaderRD::set_variant_enabled(int p_variant, bool p_enabled) {
	ERR_FAIL_COND(version_owner.get_rid_count() > 0); //versions exist
	ERR_FAIL_INDEX(p_variant, variants_enabled.size());
	variants_enabled.write[p_variant] = p_enabled;
}

bool ShaderRD::is_variant_enabled(int p_variant) const {
	ERR_FAIL_INDEX_V(p_variant, variants_enabled.size(), false);
	return variants_enabled[p_variant];
}

bool ShaderRD::shader_cache_cleanup_on_start = false;

ShaderRD::ShaderRD() {
	// Do not feel forced to use this, in most cases it makes little to no difference.
	bool use_32_threads = false;
	if (RD::get_singleton()->get_device_vendor_name() == "NVIDIA") {
		use_32_threads = true;
	}
	String base_compute_define_text;
	if (use_32_threads) {
		base_compute_define_text = "\n#define NATIVE_LOCAL_GROUP_SIZE 32\n#define NATIVE_LOCAL_SIZE_2D_X 8\n#define NATIVE_LOCAL_SIZE_2D_Y 4\n";
	} else {
		base_compute_define_text = "\n#define NATIVE_LOCAL_GROUP_SIZE 64\n#define NATIVE_LOCAL_SIZE_2D_X 8\n#define NATIVE_LOCAL_SIZE_2D_Y 8\n";
	}

	base_compute_defines = base_compute_define_text.ascii();
}

void ShaderRD::initialize(const Vector<String> &p_variant_defines, const String &p_general_defines) {
	ERR_FAIL_COND(variant_defines.size());
	ERR_FAIL_COND(p_variant_defines.size() == 0);

	general_defines = p_general_defines.utf8();

	for (int i = 0; i < p_variant_defines.size(); i++) {
		variant_defines.push_back(p_variant_defines[i].utf8());
		variants_enabled.push_back(true);
	}

	if (!shader_cache_dir.is_empty()) {
		StringBuilder hash_build;

		hash_build.append("[base_hash]");
		hash_build.append(base_sha256);
		hash_build.append("[general_defines]");
		hash_build.append(general_defines.get_data());
		for (int i = 0; i < variant_defines.size(); i++) {
			hash_build.append("[variant_defines:" + itos(i) + "]");
			hash_build.append(variant_defines[i].get_data());
		}

		base_sha256 = hash_build.as_string().sha256_text();

		DirAccessRef d = DirAccess::open(shader_cache_dir);
		ERR_FAIL_COND(!d);
		if (d->change_dir(name) != OK) {
			Error err = d->make_dir(name);
			ERR_FAIL_COND(err != OK);
			d->change_dir(name);
		}

		//erase other versions?
		if (shader_cache_cleanup_on_start) {
		}
		//
		if (d->change_dir(base_sha256) != OK) {
			Error err = d->make_dir(base_sha256);
			ERR_FAIL_COND(err != OK);
		}
		shader_cache_dir_valid = true;

		print_verbose("Shader '" + name + "' SHA256: " + base_sha256);
	}
}

void ShaderRD::set_shader_cache_dir(const String &p_dir) {
	shader_cache_dir = p_dir;
}

void ShaderRD::set_shader_cache_save_compressed(bool p_enable) {
	shader_cache_save_compressed = p_enable;
}

void ShaderRD::set_shader_cache_save_compressed_zstd(bool p_enable) {
	shader_cache_save_compressed_zstd = p_enable;
}

void ShaderRD::set_shader_cache_save_debug(bool p_enable) {
	shader_cache_save_debug = p_enable;
}

String ShaderRD::shader_cache_dir;
bool ShaderRD::shader_cache_save_compressed = true;
bool ShaderRD::shader_cache_save_compressed_zstd = true;
bool ShaderRD::shader_cache_save_debug = true;

ShaderRD::~ShaderRD() {
	List<RID> remaining;
	version_owner.get_owned_list(&remaining);
	if (remaining.size()) {
		ERR_PRINT(itos(remaining.size()) + " shaders of type " + name + " were never freed");
		while (remaining.size()) {
			version_free(remaining.front()->get());
			remaining.pop_front();
		}
	}
}

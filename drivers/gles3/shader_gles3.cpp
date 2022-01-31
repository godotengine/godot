/*************************************************************************/
/*  shader_gles3.cpp                                                     */
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

#include "shader_gles3.h"
#ifdef GLES3_ENABLED

#include "core/io/compression.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"

void ShaderGLES3::_add_stage(const char *p_code, StageType p_stage_type) {
	Vector<String> lines = String(p_code).split("\n");

	String text;

	for (int i = 0; i < lines.size(); i++) {
		String l = lines[i];
		bool push_chunk = false;

		StageTemplate::Chunk chunk;

		if (l.begins_with("#GLOBALS")) {
			switch (p_stage_type) {
				case STAGE_TYPE_VERTEX:
					chunk.type = StageTemplate::Chunk::TYPE_VERTEX_GLOBALS;
					break;
				case STAGE_TYPE_FRAGMENT:
					chunk.type = StageTemplate::Chunk::TYPE_FRAGMENT_GLOBALS;
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
			if (text != String()) {
				StageTemplate::Chunk text_chunk;
				text_chunk.type = StageTemplate::Chunk::TYPE_TEXT;
				text_chunk.text = text.utf8();
				stage_templates[p_stage_type].chunks.push_back(text_chunk);
				text = String();
			}
			stage_templates[p_stage_type].chunks.push_back(chunk);
		}

		if (text != String()) {
			StageTemplate::Chunk text_chunk;
			text_chunk.type = StageTemplate::Chunk::TYPE_TEXT;
			text_chunk.text = text.utf8();
			stage_templates[p_stage_type].chunks.push_back(text_chunk);
			text = String();
		}
	}
}

void ShaderGLES3::_setup(const char *p_vertex_code, const char *p_fragment_code, const char *p_name, int p_uniform_count, const char **p_uniform_names, int p_ubo_count, const UBOPair *p_ubos, int p_texture_count, const TexUnitPair *p_tex_units, int p_specialization_count, const Specialization *p_specializations, int p_variant_count, const char **p_variants) {
	name = p_name;

	if (p_vertex_code) {
		_add_stage(p_vertex_code, STAGE_TYPE_VERTEX);
	}
	if (p_fragment_code) {
		_add_stage(p_fragment_code, STAGE_TYPE_FRAGMENT);
	}

	uniform_names = p_uniform_names;
	uniform_count = p_uniform_count;
	ubo_pairs = p_ubos;
	ubo_count = p_ubo_count;
	texunit_pairs = p_tex_units;
	texunit_pair_count = p_texture_count;
	specializations = p_specializations;
	specialization_count = p_specialization_count;
	specialization_default_mask = 0;
	for (int i = 0; i < specialization_count; i++) {
		if (specializations[i].default_value) {
			specialization_default_mask |= (uint64_t(1) << uint64_t(i));
		}
	}
	variant_defines = p_variants;
	variant_count = p_variant_count;

	StringBuilder tohash;
	/*
	tohash.append("[SpirvCacheKey]");
	tohash.append(RenderingDevice::get_singleton()->shader_get_spirv_cache_key());
	tohash.append("[BinaryCacheKey]");
	tohash.append(RenderingDevice::get_singleton()->shader_get_binary_cache_key());
	*/
	tohash.append("[Vertex]");
	tohash.append(p_vertex_code ? p_vertex_code : "");
	tohash.append("[Fragment]");
	tohash.append(p_fragment_code ? p_fragment_code : "");

	base_sha256 = tohash.as_string().sha256_text();
}

RID ShaderGLES3::version_create() {
	//initialize() was never called
	ERR_FAIL_COND_V(variant_count == 0, RID());

	Version version;
	return version_owner.make_rid(version);
}

void ShaderGLES3::_build_variant_code(StringBuilder &builder, uint32_t p_variant, const Version *p_version, const StageTemplate &p_template, uint64_t p_specialization) {
#ifdef GLES_OVER_GL
	builder.append("#version 330\n");
	builder.append("#define USE_GLES_OVER_GL\n");
#else
	builder.append("#version 300 es\n");
#endif

	for (int i = 0; i < specialization_count; i++) {
		if (p_specialization & (uint64_t(1) << uint64_t(i))) {
			builder.append("#define " + String(specializations[i].name) + "\n");
		}
	}
	if (p_version->uniforms.size()) {
		builder.append("#define MATERIAL_UNIFORMS_USED\n");
	}
	for (const KeyValue<StringName, CharString> &E : p_version->code_sections) {
		builder.append(String("#define ") + String(E.key) + "_CODE_USED\n");
	}

	builder.append("\n"); //make sure defines begin at newline
	builder.append(general_defines.get_data());
	builder.append(variant_defines[p_variant]);
	for (int j = 0; j < p_version->custom_defines.size(); j++) {
		builder.append(p_version->custom_defines[j].get_data());
	}
	builder.append("\n"); //make sure defines begin at newline

	for (uint32_t i = 0; i < p_template.chunks.size(); i++) {
		const StageTemplate::Chunk &chunk = p_template.chunks[i];
		switch (chunk.type) {
			case StageTemplate::Chunk::TYPE_MATERIAL_UNIFORMS: {
				builder.append(p_version->uniforms.get_data()); //uniforms (same for vertex and fragment)
			} break;
			case StageTemplate::Chunk::TYPE_VERTEX_GLOBALS: {
				builder.append(p_version->vertex_globals.get_data()); // vertex globals
			} break;
			case StageTemplate::Chunk::TYPE_FRAGMENT_GLOBALS: {
				builder.append(p_version->fragment_globals.get_data()); // fragment globals
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

static void _display_error_with_code(const String &p_error, const String &p_code) {
	int line = 1;
	Vector<String> lines = p_code.split("\n");

	for (int j = 0; j < lines.size(); j++) {
		print_line(itos(line) + ": " + lines[j]);
		line++;
	}

	ERR_PRINT(p_error);
}

void ShaderGLES3::_compile_specialization(Version::Specialization &spec, uint32_t p_variant, Version *p_version, uint64_t p_specialization) {
	spec.id = glCreateProgram();
	spec.ok = false;
	GLint status;

	//vertex stage
	{
		StringBuilder builder;
		_build_variant_code(builder, p_variant, p_version, stage_templates[STAGE_TYPE_VERTEX], p_specialization);

		spec.vert_id = glCreateShader(GL_VERTEX_SHADER);
		String builder_string = builder.as_string();
		CharString cs = builder_string.utf8();
		const char *cstr = cs.ptr();
		glShaderSource(spec.vert_id, 1, &cstr, nullptr);
		glCompileShader(spec.vert_id);

		glGetShaderiv(spec.vert_id, GL_COMPILE_STATUS, &status);
		if (status == GL_FALSE) {
			GLsizei iloglen;
			glGetShaderiv(spec.vert_id, GL_INFO_LOG_LENGTH, &iloglen);

			if (iloglen < 0) {
				glDeleteShader(spec.vert_id);
				glDeleteProgram(spec.id);
				spec.id = 0;

				ERR_PRINT("No OpenGL vertex shader compiler log.");
			} else {
				if (iloglen == 0) {
					iloglen = 4096; // buggy driver (Adreno 220+)
				}

				char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
				ilogmem[iloglen] = '\0';
				glGetShaderInfoLog(spec.vert_id, iloglen, &iloglen, ilogmem);

				String err_string = name + ": Vertex shader compilation failed:\n";

				err_string += ilogmem;

				_display_error_with_code(err_string, builder_string);

				Memory::free_static(ilogmem);
				glDeleteShader(spec.vert_id);
				glDeleteProgram(spec.id);
				spec.id = 0;
			}

			ERR_FAIL();
		}
	}

	//fragment stage
	{
		StringBuilder builder;
		_build_variant_code(builder, p_variant, p_version, stage_templates[STAGE_TYPE_FRAGMENT], p_specialization);

		spec.frag_id = glCreateShader(GL_FRAGMENT_SHADER);
		String builder_string = builder.as_string();
		CharString cs = builder_string.utf8();
		const char *cstr = cs.ptr();
		glShaderSource(spec.frag_id, 1, &cstr, nullptr);
		glCompileShader(spec.frag_id);

		glGetShaderiv(spec.frag_id, GL_COMPILE_STATUS, &status);
		if (status == GL_FALSE) {
			GLsizei iloglen;
			glGetShaderiv(spec.frag_id, GL_INFO_LOG_LENGTH, &iloglen);

			if (iloglen < 0) {
				glDeleteShader(spec.frag_id);
				glDeleteProgram(spec.id);
				spec.id = 0;

				ERR_PRINT("No OpenGL fragment shader compiler log.");
			} else {
				if (iloglen == 0) {
					iloglen = 4096; // buggy driver (Adreno 220+)
				}

				char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
				ilogmem[iloglen] = '\0';
				glGetShaderInfoLog(spec.frag_id, iloglen, &iloglen, ilogmem);

				String err_string = name + ": Fragment shader compilation failed:\n";

				err_string += ilogmem;

				_display_error_with_code(err_string, builder_string);

				Memory::free_static(ilogmem);
				glDeleteShader(spec.frag_id);
				glDeleteProgram(spec.id);
				spec.id = 0;
			}

			ERR_FAIL();
		}
	}

	glAttachShader(spec.id, spec.frag_id);
	glAttachShader(spec.id, spec.vert_id);

	//for (int i = 0; i < attribute_pair_count; i++) {
	//	glBindAttribLocation(v.id, attribute_pairs[i].index, attribute_pairs[i].name);
	//}

	glLinkProgram(spec.id);

	glGetProgramiv(spec.id, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLsizei iloglen;
		glGetProgramiv(spec.id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {
			glDeleteShader(spec.frag_id);
			glDeleteShader(spec.vert_id);
			glDeleteProgram(spec.id);
			spec.id = 0;

			ERR_PRINT("No OpenGL program link log. What the frick?");
			ERR_FAIL();
		}

		if (iloglen == 0) {
			iloglen = 4096; // buggy driver (Adreno 220+)
		}

		char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
		ilogmem[iloglen] = '\0';
		glGetProgramInfoLog(spec.id, iloglen, &iloglen, ilogmem);

		String err_string = name + ": Program linking failed:\n";

		err_string += ilogmem;

		_display_error_with_code(err_string, String());

		Memory::free_static(ilogmem);
		glDeleteShader(spec.frag_id);
		glDeleteShader(spec.vert_id);
		glDeleteProgram(spec.id);
		spec.id = 0;

		ERR_FAIL();
	}

	// get uniform locations

	glUseProgram(spec.id);

	spec.uniform_location.resize(uniform_count);
	for (int i = 0; i < uniform_count; i++) {
		spec.uniform_location[i] = glGetUniformLocation(spec.id, uniform_names[i]);
	}

	for (int i = 0; i < texunit_pair_count; i++) {
		GLint loc = glGetUniformLocation(spec.id, texunit_pairs[i].name);
		if (loc >= 0) {
			if (texunit_pairs[i].index < 0) {
				glUniform1i(loc, max_image_units + texunit_pairs[i].index);
			} else {
				glUniform1i(loc, texunit_pairs[i].index);
			}
		}
	}

	for (int i = 0; i < ubo_count; i++) {
		GLint loc = glGetUniformBlockIndex(spec.id, ubo_pairs[i].name);
		if (loc >= 0) {
			glUniformBlockBinding(spec.id, loc, ubo_pairs[i].index);
		}
	}
	// textures
	for (int i = 0; i < p_version->texture_uniforms.size(); i++) {
		String native_uniform_name = p_version->texture_uniforms[i];
		GLint location = glGetUniformLocation(spec.id, (native_uniform_name).ascii().get_data());
		glUniform1i(location, i + base_texture_index);
	}

	glUseProgram(0);
	spec.ok = true;
}

RS::ShaderNativeSourceCode ShaderGLES3::version_get_native_source_code(RID p_version) {
	Version *version = version_owner.get_or_null(p_version);
	RS::ShaderNativeSourceCode source_code;
	ERR_FAIL_COND_V(!version, source_code);

	source_code.versions.resize(variant_count);

	for (int i = 0; i < source_code.versions.size(); i++) {
		//vertex stage

		{
			StringBuilder builder;
			_build_variant_code(builder, i, version, stage_templates[STAGE_TYPE_VERTEX], specialization_default_mask);

			RS::ShaderNativeSourceCode::Version::Stage stage;
			stage.name = "vertex";
			stage.code = builder.as_string();

			source_code.versions.write[i].stages.push_back(stage);
		}

		//fragment stage
		{
			StringBuilder builder;
			_build_variant_code(builder, i, version, stage_templates[STAGE_TYPE_FRAGMENT], specialization_default_mask);

			RS::ShaderNativeSourceCode::Version::Stage stage;
			stage.name = "fragment";
			stage.code = builder.as_string();

			source_code.versions.write[i].stages.push_back(stage);
		}
	}

	return source_code;
}

String ShaderGLES3::_version_get_sha1(Version *p_version) const {
	StringBuilder hash_build;

	hash_build.append("[uniforms]");
	hash_build.append(p_version->uniforms.get_data());
	hash_build.append("[vertex_globals]");
	hash_build.append(p_version->vertex_globals.get_data());
	hash_build.append("[fragment_globals]");
	hash_build.append(p_version->fragment_globals.get_data());

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

//static const char *shader_file_header = "GLSC";
//static const uint32_t cache_file_version = 2;

bool ShaderGLES3::_load_from_cache(Version *p_version) {
#if 0
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

	ERR_FAIL_COND_V(variant_count != (uint32_t)variant_count, false); //should not happen but check

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
		RID shader = GLES3::get_singleton()->shader_create_from_bytecode(p_version->variant_data[i]);
		if (shader.is_null()) {
			for (uint32_t j = 0; j < i; j++) {
				GLES3::get_singleton()->free(p_version->variants[i]);
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
#endif
	return false;
}

void ShaderGLES3::_save_to_cache(Version *p_version) {
#if 0
	String sha1 = _version_get_sha1(p_version);
	String path = shader_cache_dir.plus_file(name).plus_file(base_sha256).plus_file(sha1) + ".cache";

	FileAccessRef f = FileAccess::open(path, FileAccess::WRITE);
	ERR_FAIL_COND(!f);
	f->store_buffer((const uint8_t *)shader_file_header, 4);
	f->store_32(cache_file_version); //file version
	uint32_t variant_count = variant_count;
	f->store_32(variant_count); //variant count

	for (uint32_t i = 0; i < variant_count; i++) {
		f->store_32(p_version->variant_data[i].size()); //stage count
		f->store_buffer(p_version->variant_data[i].ptr(), p_version->variant_data[i].size());
	}

	f->close();
#endif
}

void ShaderGLES3::_clear_version(Version *p_version) {
	// Variants not compiled yet, just return
	if (p_version->variants.size() == 0) {
		return;
	}

	for (int i = 0; i < variant_count; i++) {
		for (OAHashMap<uint64_t, Version::Specialization>::Iterator it = p_version->variants[i].iter(); it.valid; it = p_version->variants[i].next_iter(it)) {
			if (it.valid) {
				glDeleteShader(it.value->vert_id);
				glDeleteShader(it.value->frag_id);
				glDeleteProgram(it.value->id);
			}
		}
	}

	p_version->variants.clear();
}

void ShaderGLES3::_initialize_version(Version *p_version) {
	ERR_FAIL_COND(p_version->variants.size() > 0);
	p_version->variants.reserve(variant_count);
	for (int i = 0; i < variant_count; i++) {
		OAHashMap<uint64_t, Version::Specialization> variant;
		p_version->variants.push_back(variant);
		Version::Specialization spec;
		_compile_specialization(spec, i, p_version, specialization_default_mask);
		p_version->variants[i].insert(specialization_default_mask, spec);
	}
}

void ShaderGLES3::version_set_code(RID p_version, const Map<String, String> &p_code, const String &p_uniforms, const String &p_vertex_globals, const String &p_fragment_globals, const Vector<String> &p_custom_defines, const Vector<StringName> &p_texture_uniforms, bool p_initialize) {
	Version *version = version_owner.get_or_null(p_version);
	ERR_FAIL_COND(!version);

	_clear_version(version); //clear if existing

	version->vertex_globals = p_vertex_globals.utf8();
	version->fragment_globals = p_fragment_globals.utf8();
	version->uniforms = p_uniforms.utf8();
	version->code_sections.clear();
	version->texture_uniforms = p_texture_uniforms;
	for (const KeyValue<String, String> &E : p_code) {
		version->code_sections[StringName(E.key.to_upper())] = E.value.utf8();
	}

	version->custom_defines.clear();
	for (int i = 0; i < p_custom_defines.size(); i++) {
		version->custom_defines.push_back(p_custom_defines[i].utf8());
	}

	if (p_initialize) {
		_initialize_version(version);
	}
}

bool ShaderGLES3::version_is_valid(RID p_version) {
	Version *version = version_owner.get_or_null(p_version);
	return version != nullptr;
}

bool ShaderGLES3::version_free(RID p_version) {
	if (version_owner.owns(p_version)) {
		Version *version = version_owner.get_or_null(p_version);
		_clear_version(version);
		version_owner.free(p_version);
	} else {
		return false;
	}

	return true;
}

bool ShaderGLES3::shader_cache_cleanup_on_start = false;

ShaderGLES3::ShaderGLES3() {
}

void ShaderGLES3::initialize(const String &p_general_defines, int p_base_texture_index) {
	general_defines = p_general_defines.utf8();
	base_texture_index = p_base_texture_index;

	_init();

	if (shader_cache_dir != String()) {
		StringBuilder hash_build;

		hash_build.append("[base_hash]");
		hash_build.append(base_sha256);
		hash_build.append("[general_defines]");
		hash_build.append(general_defines.get_data());
		for (int i = 0; i < variant_count; i++) {
			hash_build.append("[variant_defines:" + itos(i) + "]");
			hash_build.append(variant_defines[i]);
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

	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_image_units);
}

void ShaderGLES3::set_shader_cache_dir(const String &p_dir) {
	shader_cache_dir = p_dir;
}

void ShaderGLES3::set_shader_cache_save_compressed(bool p_enable) {
	shader_cache_save_compressed = p_enable;
}

void ShaderGLES3::set_shader_cache_save_compressed_zstd(bool p_enable) {
	shader_cache_save_compressed_zstd = p_enable;
}

void ShaderGLES3::set_shader_cache_save_debug(bool p_enable) {
	shader_cache_save_debug = p_enable;
}

String ShaderGLES3::shader_cache_dir;
bool ShaderGLES3::shader_cache_save_compressed = true;
bool ShaderGLES3::shader_cache_save_compressed_zstd = true;
bool ShaderGLES3::shader_cache_save_debug = true;

ShaderGLES3::~ShaderGLES3() {
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
#endif

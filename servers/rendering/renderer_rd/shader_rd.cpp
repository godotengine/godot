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

#include "renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"

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
			if (text != String()) {
				StageTemplate::Chunk text_chunk;
				text_chunk.type = StageTemplate::Chunk::TYPE_TEXT;
				text_chunk.text = text.utf8();
				stage_templates[p_stage_type].chunks.push_back(text_chunk);
				text = String();
			}
			stage_templates[p_stage_type].chunks.push_back(chunk);
		}
	}

	if (text != String()) {
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
			RD::get_singleton()->free(p_version->variants[i]);
		}

		memdelete_arr(p_version->variants);
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
				for (Map<StringName, CharString>::Element *E = p_version->code_sections.front(); E; E = E->next()) {
					builder.append(String("#define ") + String(E->key()) + "_CODE_USED\n");
				}
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

	Vector<RD::ShaderStageData> stages;

	String error;
	String current_source;
	RD::ShaderStage current_stage = RD::SHADER_STAGE_VERTEX;
	bool build_ok = true;

	if (!is_compute) {
		//vertex stage

		StringBuilder builder;
		_build_variant_code(builder, p_variant, p_version, stage_templates[STAGE_TYPE_VERTEX]);

		current_source = builder.as_string();
		RD::ShaderStageData stage;
		stage.spir_v = RD::get_singleton()->shader_compile_from_source(RD::SHADER_STAGE_VERTEX, current_source, RD::SHADER_LANGUAGE_GLSL, &error);
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
		RD::ShaderStageData stage;
		stage.spir_v = RD::get_singleton()->shader_compile_from_source(RD::SHADER_STAGE_FRAGMENT, current_source, RD::SHADER_LANGUAGE_GLSL, &error);
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

		RD::ShaderStageData stage;
		stage.spir_v = RD::get_singleton()->shader_compile_from_source(RD::SHADER_STAGE_COMPUTE, current_source, RD::SHADER_LANGUAGE_GLSL, &error);
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

	RID shader = RD::get_singleton()->shader_create(stages);
	{
		MutexLock lock(variant_set_mutex);
		p_version->variants[p_variant] = shader;
	}
}

RS::ShaderNativeSourceCode ShaderRD::version_get_native_source_code(RID p_version) {
	Version *version = version_owner.getornull(p_version);
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

void ShaderRD::_compile_version(Version *p_version) {
	_clear_version(p_version);

	p_version->valid = false;
	p_version->dirty = false;

	p_version->variants = memnew_arr(RID, variant_defines.size());
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
		p_version->variants = nullptr;
		return;
	}

	p_version->valid = true;
}

void ShaderRD::version_set_code(RID p_version, const Map<String, String> &p_code, const String &p_uniforms, const String &p_vertex_globals, const String &p_fragment_globals, const Vector<String> &p_custom_defines) {
	ERR_FAIL_COND(is_compute);

	Version *version = version_owner.getornull(p_version);
	ERR_FAIL_COND(!version);
	version->vertex_globals = p_vertex_globals.utf8();
	version->fragment_globals = p_fragment_globals.utf8();
	version->uniforms = p_uniforms.utf8();
	version->code_sections.clear();
	for (Map<String, String>::Element *E = p_code.front(); E; E = E->next()) {
		version->code_sections[StringName(E->key().to_upper())] = E->get().utf8();
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

	Version *version = version_owner.getornull(p_version);
	ERR_FAIL_COND(!version);

	version->compute_globals = p_compute_globals.utf8();
	version->uniforms = p_uniforms.utf8();

	version->code_sections.clear();
	for (Map<String, String>::Element *E = p_code.front(); E; E = E->next()) {
		version->code_sections[StringName(E->key().to_upper())] = E->get().utf8();
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
	Version *version = version_owner.getornull(p_version);
	ERR_FAIL_COND_V(!version, false);

	if (version->dirty) {
		_compile_version(version);
	}

	return version->valid;
}

bool ShaderRD::version_free(RID p_version) {
	if (version_owner.owns(p_version)) {
		Version *version = version_owner.getornull(p_version);
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
}

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

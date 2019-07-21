/*************************************************************************/
/*  shader_rd.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core/string_builder.h"
#include "servers/visual/rendering_device.h"

void ShaderRD::setup(const char *p_vertex_code, const char *p_fragment_code, const char *p_name) {

	name = p_name;
	//split vertex and shader code (thank you, shader compiler programmers from you know what company).
	{
		String defines_tag = "\nVERSION_DEFINES";
		String globals_tag = "\nVERTEX_SHADER_GLOBALS";
		String material_tag = "\nMATERIAL_UNIFORMS";
		String code_tag = "\nVERTEX_SHADER_CODE";
		String code = p_vertex_code;

		int cpos = code.find(defines_tag);
		if (cpos != -1) {
			vertex_codev = code.substr(0, cpos).ascii();
			code = code.substr(cpos + defines_tag.length(), code.length());
		}

		cpos = code.find(material_tag);

		if (cpos == -1) {
			vertex_code0 = code.ascii();
		} else {
			vertex_code0 = code.substr(0, cpos).ascii();
			code = code.substr(cpos + material_tag.length(), code.length());

			cpos = code.find(globals_tag);

			if (cpos == -1) {
				vertex_code1 = code.ascii();
			} else {

				vertex_code1 = code.substr(0, cpos).ascii();
				String code2 = code.substr(cpos + globals_tag.length(), code.length());

				cpos = code2.find(code_tag);
				if (cpos == -1) {
					vertex_code2 = code2.ascii();
				} else {

					vertex_code2 = code2.substr(0, cpos).ascii();
					vertex_code3 = code2.substr(cpos + code_tag.length(), code2.length()).ascii();
				}
			}
		}
	}

	{
		String defines_tag = "\nVERSION_DEFINES";
		String globals_tag = "\nFRAGMENT_SHADER_GLOBALS";
		String material_tag = "\nMATERIAL_UNIFORMS";
		String code_tag = "\nFRAGMENT_SHADER_CODE";
		String light_code_tag = "\nLIGHT_SHADER_CODE";
		String code = p_fragment_code;

		int cpos = code.find(defines_tag);
		if (cpos != -1) {
			fragment_codev = code.substr(0, cpos).ascii();
			code = code.substr(cpos + defines_tag.length(), code.length());
		}

		cpos = code.find(material_tag);
		if (cpos == -1) {
			fragment_code0 = code.ascii();
		} else {
			fragment_code0 = code.substr(0, cpos).ascii();
			//print_line("CODE0:\n"+String(fragment_code0.get_data()));
			code = code.substr(cpos + material_tag.length(), code.length());
			cpos = code.find(globals_tag);

			if (cpos == -1) {
				fragment_code1 = code.ascii();
			} else {

				fragment_code1 = code.substr(0, cpos).ascii();
				//print_line("CODE1:\n"+String(fragment_code1.get_data()));

				String code2 = code.substr(cpos + globals_tag.length(), code.length());
				cpos = code2.find(light_code_tag);

				if (cpos == -1) {
					fragment_code2 = code2.ascii();
				} else {

					fragment_code2 = code2.substr(0, cpos).ascii();
					//print_line("CODE2:\n"+String(fragment_code2.get_data()));

					String code3 = code2.substr(cpos + light_code_tag.length(), code2.length());

					cpos = code3.find(code_tag);
					if (cpos == -1) {
						fragment_code3 = code3.ascii();
					} else {

						fragment_code3 = code3.substr(0, cpos).ascii();
						//print_line("CODE3:\n"+String(fragment_code3.get_data()));
						fragment_code4 = code3.substr(cpos + code_tag.length(), code3.length()).ascii();
						//print_line("CODE4:\n"+String(fragment_code4.get_data()));
					}
				}
			}
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
	version.variants = NULL;
	return version_owner.make_rid(version);
}

void ShaderRD::_clear_version(Version *p_version) {
	//clear versions if they exist
	if (p_version->variants) {
		for (int i = 0; i < variant_defines.size(); i++) {
			RD::get_singleton()->free(p_version->variants[i]);
		}

		memdelete_arr(p_version->variants);
		p_version->variants = NULL;
	}
}
void ShaderRD::_compile_version(Version *p_version) {

	_clear_version(p_version);

	p_version->valid = false;
	p_version->dirty = false;

	p_version->variants = memnew_arr(RID, variant_defines.size());

	for (int i = 0; i < variant_defines.size(); i++) {

		Vector<RD::ShaderStageSource> stages;

		{
			//vertex stage

			StringBuilder builder;

			builder.append(vertex_codev.get_data()); // version info (if exists)
			builder.append("\n"); //make sure defines begin at newline
			builder.append(general_defines.get_data());
			builder.append(variant_defines[i].get_data());

			for (int j = 0; j < p_version->custom_defines.size(); j++) {
				builder.append(p_version->custom_defines[j].get_data());
			}

			builder.append(vertex_code0.get_data()); //first part of vertex

			builder.append(p_version->uniforms.get_data()); //uniforms (same for vertex and fragment)

			builder.append(vertex_code1.get_data()); //second part of vertex

			builder.append(p_version->vertex_globals.get_data()); // vertex globals

			builder.append(vertex_code2.get_data()); //third part of vertex

			builder.append(p_version->vertex_code.get_data()); // code

			builder.append(vertex_code3.get_data()); //fourth of vertex

			RD::ShaderStageSource stage;
			stage.shader_source = builder.as_string();
			stage.shader_stage = RD::SHADER_STAGE_VERTEX;

			stages.push_back(stage);
		}

		{
			//fragment stage

			StringBuilder builder;

			builder.append(fragment_codev.get_data()); // version info (if exists)
			builder.append("\n"); //make sure defines begin at newline

			builder.append(general_defines.get_data());
			builder.append(variant_defines[i].get_data());
			for (int j = 0; j < p_version->custom_defines.size(); j++) {
				builder.append(p_version->custom_defines[j].get_data());
			}

			builder.append(fragment_code0.get_data()); //first part of fragment

			builder.append(p_version->uniforms.get_data()); //uniforms (same for fragment and fragment)

			builder.append(fragment_code1.get_data()); //first part of fragment

			builder.append(p_version->fragment_globals.get_data()); // fragment globals

			builder.append(fragment_code2.get_data()); //third part of fragment

			builder.append(p_version->fragment_light.get_data()); // fragment light

			builder.append(fragment_code3.get_data()); //fourth part of fragment

			builder.append(p_version->fragment_code.get_data()); // fragment code

			builder.append(fragment_code4.get_data()); //fourth part of fragment

			RD::ShaderStageSource stage;
			stage.shader_source = builder.as_string();
			stage.shader_stage = RD::SHADER_STAGE_FRAGMENT;
#if 0
			if (stage.shader_stage == RD::SHADER_STAGE_FRAGMENT && p_version->uniforms.length()) {
				print_line(stage.shader_source.get_with_code_lines());
			}
#endif
			stages.push_back(stage);
		}

		String error;
		RD::ShaderStage error_stage;
		RID shader = RD::get_singleton()->shader_create_from_source(stages, &error, &error_stage);

		if (shader.is_null() && error != String()) {
			ERR_PRINTS("Error compiling shader, variant #" + itos(i) + " (" + variant_defines[i].get_data() + ").");
			ERR_PRINTS(error);

#ifdef DEBUG_ENABLED
			if (error_stage < RD::SHADER_STAGE_MAX) {
				ERR_PRINTS("code:\n" + stages[error_stage].shader_source.get_with_code_lines());
			}
#endif
			//clear versions if they exist
			for (int j = 0; j < i; j++) {
				RD::get_singleton()->free(p_version->variants[j]);
			}

			memdelete_arr(p_version->variants);
			p_version->variants = NULL;
			return;
		}

		p_version->variants[i] = shader;
	}

	p_version->valid = true;
}

void ShaderRD::version_set_code(RID p_version, const String &p_uniforms, const String &p_vertex_globals, const String &p_vertex_code, const String &p_fragment_globals, const String &p_fragment_light, const String &p_fragment_code, const Vector<String> &p_custom_defines) {

	Version *version = version_owner.getornull(p_version);
	ERR_FAIL_COND(!version);
	version->vertex_globals = p_vertex_globals.utf8();
	version->vertex_code = p_vertex_code.utf8();
	version->fragment_light = p_fragment_light.utf8();
	version->fragment_globals = p_fragment_globals.utf8();
	version->fragment_code = p_fragment_code.utf8();
	version->uniforms = p_uniforms.utf8();

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

void ShaderRD::initialize(const Vector<String> &p_variant_defines, const String &p_general_defines) {
	ERR_FAIL_COND(variant_defines.size());
	ERR_FAIL_COND(p_variant_defines.size() == 0);
	general_defines = p_general_defines.utf8();
	for (int i = 0; i < p_variant_defines.size(); i++) {

		variant_defines.push_back(p_variant_defines[i].utf8());
	}
}

ShaderRD::~ShaderRD() {
	List<RID> remaining;
	version_owner.get_owned_list(&remaining);
	if (remaining.size()) {
		ERR_PRINTS(itos(remaining.size()) + " shaders of type " + name + " were never freed");
		while (remaining.size()) {
			version_free(remaining.front()->get());
			remaining.pop_front();
		}
	}
}

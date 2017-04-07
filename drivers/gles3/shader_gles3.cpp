/*************************************************************************/
/*  shader_gles3.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "print_string.h"

//#define DEBUG_OPENGL

#ifdef DEBUG_OPENGL

#define DEBUG_TEST_ERROR(m_section)                                         \
	{                                                                       \
		uint32_t err = glGetError();                                        \
		if (err) {                                                          \
			print_line("OpenGL Error #" + itos(err) + " at: " + m_section); \
		}                                                                   \
	}
#else

#define DEBUG_TEST_ERROR(m_section)

#endif

ShaderGLES3 *ShaderGLES3::active = NULL;

//#define DEBUG_SHADER

#ifdef DEBUG_SHADER

#define DEBUG_PRINT(m_text) print_line(m_text);

#else

#define DEBUG_PRINT(m_text)

#endif

void ShaderGLES3::bind_uniforms() {

	if (!uniforms_dirty) {
		return;
	};

	// upload default uniforms
	const Map<uint32_t, Variant>::Element *E = uniform_defaults.front();

	while (E) {
		int idx = E->key();
		int location = version->uniform_location[idx];

		if (location < 0) {
			E = E->next();
			continue;
		}

		const Variant &v = E->value();
		_set_uniform_variant(location, v);
		//print_line("uniform "+itos(location)+" value "+v+ " type "+Variant::get_type_name(v.get_type()));
		E = E->next();
	};

	const Map<uint32_t, CameraMatrix>::Element *C = uniform_cameras.front();
	while (C) {

		int location = version->uniform_location[C->key()];
		if (location < 0) {
			C = C->next();
			continue;
		}

		glUniformMatrix4fv(location, 1, false, &(C->get().matrix[0][0]));
		C = C->next();
	};

	uniforms_dirty = false;
};

GLint ShaderGLES3::get_uniform_location(int p_idx) const {

	ERR_FAIL_COND_V(!version, -1);

	return version->uniform_location[p_idx];
};

bool ShaderGLES3::bind() {

	if (active != this || !version || new_conditional_version.key != conditional_version.key) {
		conditional_version = new_conditional_version;
		version = get_current_version();
	} else {

		return false;
	}

	ERR_FAIL_COND_V(!version, false);

	glUseProgram(version->id);

	DEBUG_TEST_ERROR("Use Program");

	active = this;
	uniforms_dirty = true;
	/*
 *	why on earth is this code here?
	for (int i=0;i<texunit_pair_count;i++) {

		glUniform1i(texunit_pairs[i].location, texunit_pairs[i].index);
		DEBUG_TEST_ERROR("Uniform 1 i");
	}

*/
	return true;
}

void ShaderGLES3::unbind() {

	version = NULL;
	glUseProgram(0);
	uniforms_dirty = true;
	active = NULL;
}

static void _display_error_with_code(const String &p_error, const Vector<const char *> &p_code) {

	int line = 1;
	String total_code;

	for (int i = 0; i < p_code.size(); i++) {
		total_code += String(p_code[i]);
	}

	Vector<String> lines = String(total_code).split("\n");

	for (int j = 0; j < lines.size(); j++) {

		print_line(itos(line) + ": " + lines[j]);
		line++;
	}

	ERR_PRINTS(p_error);
}

ShaderGLES3::Version *ShaderGLES3::get_current_version() {

	Version *_v = version_map.getptr(conditional_version);

	if (_v) {

		if (conditional_version.code_version != 0) {
			CustomCode *cc = custom_code_map.getptr(conditional_version.code_version);
			ERR_FAIL_COND_V(!cc, _v);
			if (cc->version == _v->code_version)
				return _v;
		} else {
			return _v;
		}
	}

	if (!_v)
		version_map[conditional_version] = Version();

	Version &v = version_map[conditional_version];

	if (!_v) {

		v.uniform_location = memnew_arr(GLint, uniform_count);

	} else {
		if (v.ok) {
			//bye bye shaders
			glDeleteShader(v.vert_id);
			glDeleteShader(v.frag_id);
			glDeleteProgram(v.id);
			v.id = 0;
		}
	}

	v.ok = false;
	/* SETUP CONDITIONALS */

	Vector<const char *> strings;
#ifdef GLES_OVER_GL
	strings.push_back("#version 330\n");
#else
	strings.push_back("#version 300 es\n");
#endif

	int define_line_ofs = 1;

	for (int i = 0; i < custom_defines.size(); i++) {

		strings.push_back(custom_defines[i].get_data());
		define_line_ofs++;
	}

	for (int j = 0; j < conditional_count; j++) {

		bool enable = ((1 << j) & conditional_version.version);
		strings.push_back(enable ? conditional_defines[j] : "");
		if (enable)
			define_line_ofs++;

		if (enable) {
			DEBUG_PRINT(conditional_defines[j]);
		}
	}

	//keep them around during the function
	CharString code_string;
	CharString code_string2;
	CharString code_globals;
	CharString material_string;

	//print_line("code version? "+itos(conditional_version.code_version));

	CustomCode *cc = NULL;

	if (conditional_version.code_version > 0) {
		//do custom code related stuff

		ERR_FAIL_COND_V(!custom_code_map.has(conditional_version.code_version), NULL);
		cc = &custom_code_map[conditional_version.code_version];
		v.code_version = cc->version;
		define_line_ofs += 2;
	}

	/* CREATE PROGRAM */

	v.id = glCreateProgram();

	ERR_FAIL_COND_V(v.id == 0, NULL);

	/* VERTEX SHADER */

	if (cc) {
		for (int i = 0; i < cc->custom_defines.size(); i++) {

			strings.push_back(cc->custom_defines[i].get_data());
			DEBUG_PRINT("CD #" + itos(i) + ": " + String(cc->custom_defines[i]));
		}
	}

	int strings_base_size = strings.size();

	//vertex precision is high
	strings.push_back("precision highp float;\n");
	strings.push_back("precision highp int;\n");

#if 0
	if (cc) {

		String _code_string = "#define VERTEX_SHADER_CODE "+cc->vertex+"\n";
		String _code_globals = "#define VERTEX_SHADER_GLOBALS "+cc->vertex_globals+"\n";

		code_string=_code_string.ascii();
		code_globals=_code_globals.ascii();
		DEBUG_PRINT( code_globals.get_data() );
		DEBUG_PRINT( code_string.get_data() );
		strings.push_back(code_globals);
		strings.push_back(code_string);
	}
#endif

	strings.push_back(vertex_code0.get_data());
	if (cc) {
		code_globals = cc->vertex_globals.ascii();
		strings.push_back(code_globals.get_data());
	}

	strings.push_back(vertex_code1.get_data());

	if (cc) {
		material_string = cc->uniforms.ascii();
		strings.push_back(material_string.get_data());
	}

	strings.push_back(vertex_code2.get_data());

	if (cc) {
		code_string = cc->vertex.ascii();
		strings.push_back(code_string.get_data());
	}

	strings.push_back(vertex_code3.get_data());
#ifdef DEBUG_SHADER

	DEBUG_PRINT("\nVertex Code:\n\n" + String(code_string.get_data()));
	for (int i = 0; i < strings.size(); i++) {

		//print_line("vert strings "+itos(i)+":"+String(strings[i]));
	}
#endif

	v.vert_id = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(v.vert_id, strings.size(), &strings[0], NULL);
	glCompileShader(v.vert_id);

	GLint status;

	glGetShaderiv(v.vert_id, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		// error compiling
		GLsizei iloglen;
		glGetShaderiv(v.vert_id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {

			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;

			ERR_PRINT("NO LOG, WTF");
		} else {

			if (iloglen == 0) {

				iloglen = 4096; //buggy driver (Adreno 220+....)
			}

			char *ilogmem = (char *)memalloc(iloglen + 1);
			ilogmem[iloglen] = 0;
			glGetShaderInfoLog(v.vert_id, iloglen, &iloglen, ilogmem);

			String err_string = get_shader_name() + ": Vertex Program Compilation Failed:\n";

			err_string += ilogmem;
			_display_error_with_code(err_string, strings);
			memfree(ilogmem);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;
		}

		ERR_FAIL_V(NULL);
	}

	/* FRAGMENT SHADER */

	strings.resize(strings_base_size);
	//fragment precision is medium
	strings.push_back("precision highp float;\n");
	strings.push_back("precision highp int;\n");

#if 0
	if (cc) {

		String _code_string = "#define FRAGMENT_SHADER_CODE "+cc->fragment+"\n";
		String _code_globals = "#define FRAGMENT_SHADER_GLOBALS "+cc->fragment_globals+"\n";

		code_string=_code_string.ascii();
		code_globals=_code_globals.ascii();
		DEBUG_PRINT( code_globals.get_data() );
		DEBUG_PRINT( code_string.get_data() );
		strings.push_back(code_globals);
		strings.push_back(code_string);
	}
#endif

	strings.push_back(fragment_code0.get_data());
	if (cc) {
		code_globals = cc->fragment_globals.ascii();
		strings.push_back(code_globals.get_data());
	}

	strings.push_back(fragment_code1.get_data());

	if (cc) {
		material_string = cc->uniforms.ascii();
		strings.push_back(material_string.get_data());
	}

	strings.push_back(fragment_code2.get_data());

	if (cc) {
		code_string = cc->fragment.ascii();
		strings.push_back(code_string.get_data());
	}

	strings.push_back(fragment_code3.get_data());

	if (cc) {
		code_string2 = cc->light.ascii();
		strings.push_back(code_string2.get_data());
	}

	strings.push_back(fragment_code4.get_data());

#ifdef DEBUG_SHADER
	DEBUG_PRINT("\nFragment Code:\n\n" + String(code_string.get_data()));
	for (int i = 0; i < strings.size(); i++) {

		//print_line("frag strings "+itos(i)+":"+String(strings[i]));
	}
#endif

	v.frag_id = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(v.frag_id, strings.size(), &strings[0], NULL);
	glCompileShader(v.frag_id);

	glGetShaderiv(v.frag_id, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		// error compiling
		GLsizei iloglen;
		glGetShaderiv(v.frag_id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {

			glDeleteShader(v.frag_id);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;
			ERR_PRINT("NO LOG, WTF");
		} else {

			if (iloglen == 0) {

				iloglen = 4096; //buggy driver (Adreno 220+....)
			}

			char *ilogmem = (char *)memalloc(iloglen + 1);
			ilogmem[iloglen] = 0;
			glGetShaderInfoLog(v.frag_id, iloglen, &iloglen, ilogmem);

			String err_string = get_shader_name() + ": Fragment Program Compilation Failed:\n";

			err_string += ilogmem;
			_display_error_with_code(err_string, strings);
			ERR_PRINT(err_string.ascii().get_data());
			memfree(ilogmem);
			glDeleteShader(v.frag_id);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;
		}

		ERR_FAIL_V(NULL);
	}

	glAttachShader(v.id, v.frag_id);
	glAttachShader(v.id, v.vert_id);

	// bind attributes before linking
	for (int i = 0; i < attribute_pair_count; i++) {

		glBindAttribLocation(v.id, attribute_pairs[i].index, attribute_pairs[i].name);
	}

	//if feedback exists, set it up

	if (feedback_count) {
		Vector<const char *> feedback;
		for (int i = 0; i < feedback_count; i++) {

			if (feedbacks[i].conditional == -1 || (1 << feedbacks[i].conditional) & conditional_version.version) {
				//conditional for this feedback is enabled
				print_line("tf varying: " + itos(feedback.size()) + " " + String(feedbacks[i].name));
				feedback.push_back(feedbacks[i].name);
			}
		}

		if (feedback.size()) {
			glTransformFeedbackVaryings(v.id, feedback.size(), feedback.ptr(), GL_INTERLEAVED_ATTRIBS);
		}
	}

	glLinkProgram(v.id);

	glGetProgramiv(v.id, GL_LINK_STATUS, &status);

	if (status == GL_FALSE) {
		// error linking
		GLsizei iloglen;
		glGetProgramiv(v.id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {

			glDeleteShader(v.frag_id);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;
			ERR_FAIL_COND_V(iloglen <= 0, NULL);
		}

		if (iloglen == 0) {

			iloglen = 4096; //buggy driver (Adreno 220+....)
		}

		char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
		ilogmem[iloglen] = 0;
		glGetProgramInfoLog(v.id, iloglen, &iloglen, ilogmem);

		String err_string = get_shader_name() + ": Program LINK FAILED:\n";

		err_string += ilogmem;
		_display_error_with_code(err_string, strings);
		ERR_PRINT(err_string.ascii().get_data());
		Memory::free_static(ilogmem);
		glDeleteShader(v.frag_id);
		glDeleteShader(v.vert_id);
		glDeleteProgram(v.id);
		v.id = 0;

		ERR_FAIL_V(NULL);
	}

	/* UNIFORMS */

	glUseProgram(v.id);

	//print_line("uniforms:  ");
	for (int j = 0; j < uniform_count; j++) {

		v.uniform_location[j] = glGetUniformLocation(v.id, uniform_names[j]);
		//print_line("uniform "+String(uniform_names[j])+" location "+itos(v.uniform_location[j]));
	}

	// set texture uniforms
	for (int i = 0; i < texunit_pair_count; i++) {

		GLint loc = glGetUniformLocation(v.id, texunit_pairs[i].name);
		if (loc >= 0) {
			if (texunit_pairs[i].index < 0) {
				glUniform1i(loc, max_image_units + texunit_pairs[i].index); //negative, goes down
			} else {

				glUniform1i(loc, texunit_pairs[i].index);
			}
		}
	}

	// assign uniform block bind points
	for (int i = 0; i < ubo_count; i++) {

		GLint loc = glGetUniformBlockIndex(v.id, ubo_pairs[i].name);
		if (loc >= 0)
			glUniformBlockBinding(v.id, loc, ubo_pairs[i].index);
	}

	if (cc) {

		v.texture_uniform_locations.resize(cc->texture_uniforms.size());
		for (int i = 0; i < cc->texture_uniforms.size(); i++) {

			v.texture_uniform_locations[i] = glGetUniformLocation(v.id, String(cc->texture_uniforms[i]).ascii().get_data());
			glUniform1i(v.texture_uniform_locations[i], i + base_material_tex_index);
		}
	}

	glUseProgram(0);

	v.ok = true;

	return &v;
}

GLint ShaderGLES3::get_uniform_location(const String &p_name) const {

	ERR_FAIL_COND_V(!version, -1);
	return glGetUniformLocation(version->id, p_name.ascii().get_data());
}

void ShaderGLES3::setup(const char **p_conditional_defines, int p_conditional_count, const char **p_uniform_names, int p_uniform_count, const AttributePair *p_attribute_pairs, int p_attribute_count, const TexUnitPair *p_texunit_pairs, int p_texunit_pair_count, const UBOPair *p_ubo_pairs, int p_ubo_pair_count, const Feedback *p_feedback, int p_feedback_count, const char *p_vertex_code, const char *p_fragment_code, int p_vertex_code_start, int p_fragment_code_start) {

	ERR_FAIL_COND(version);
	conditional_version.key = 0;
	new_conditional_version.key = 0;
	uniform_count = p_uniform_count;
	conditional_count = p_conditional_count;
	conditional_defines = p_conditional_defines;
	uniform_names = p_uniform_names;
	vertex_code = p_vertex_code;
	fragment_code = p_fragment_code;
	texunit_pairs = p_texunit_pairs;
	texunit_pair_count = p_texunit_pair_count;
	vertex_code_start = p_vertex_code_start;
	fragment_code_start = p_fragment_code_start;
	attribute_pairs = p_attribute_pairs;
	attribute_pair_count = p_attribute_count;
	ubo_pairs = p_ubo_pairs;
	ubo_count = p_ubo_pair_count;
	feedbacks = p_feedback;
	feedback_count = p_feedback_count;

	//split vertex and shader code (thank you, retarded shader compiler programmers from you know what company).
	{
		String globals_tag = "\nVERTEX_SHADER_GLOBALS";
		String material_tag = "\nMATERIAL_UNIFORMS";
		String code_tag = "\nVERTEX_SHADER_CODE";
		String code = vertex_code;
		int cpos = code.find(globals_tag);
		if (cpos == -1) {
			vertex_code0 = code.ascii();
		} else {
			vertex_code0 = code.substr(0, cpos).ascii();
			code = code.substr(cpos + globals_tag.length(), code.length());

			cpos = code.find(material_tag);

			if (cpos == -1) {
				vertex_code1 = code.ascii();
			} else {

				vertex_code1 = code.substr(0, cpos).ascii();
				String code2 = code.substr(cpos + material_tag.length(), code.length());

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
		String globals_tag = "\nFRAGMENT_SHADER_GLOBALS";
		String material_tag = "\nMATERIAL_UNIFORMS";
		String code_tag = "\nFRAGMENT_SHADER_CODE";
		String light_code_tag = "\nLIGHT_SHADER_CODE";
		String code = fragment_code;
		int cpos = code.find(globals_tag);
		if (cpos == -1) {
			fragment_code0 = code.ascii();
		} else {
			fragment_code0 = code.substr(0, cpos).ascii();
			//print_line("CODE0:\n"+String(fragment_code0.get_data()));
			code = code.substr(cpos + globals_tag.length(), code.length());
			cpos = code.find(material_tag);

			if (cpos == -1) {
				fragment_code1 = code.ascii();
			} else {

				fragment_code1 = code.substr(0, cpos).ascii();
				//print_line("CODE1:\n"+String(fragment_code1.get_data()));

				String code2 = code.substr(cpos + material_tag.length(), code.length());
				cpos = code2.find(code_tag);

				if (cpos == -1) {
					fragment_code2 = code2.ascii();
				} else {

					fragment_code2 = code2.substr(0, cpos).ascii();
					//print_line("CODE2:\n"+String(fragment_code2.get_data()));

					String code3 = code2.substr(cpos + code_tag.length(), code2.length());

					cpos = code3.find(light_code_tag);
					if (cpos == -1) {
						fragment_code3 = code3.ascii();
					} else {

						fragment_code3 = code3.substr(0, cpos).ascii();
						//print_line("CODE3:\n"+String(fragment_code3.get_data()));
						fragment_code4 = code3.substr(cpos + light_code_tag.length(), code3.length()).ascii();
						//print_line("CODE4:\n"+String(fragment_code4.get_data()));
					}
				}
			}
		}
	}

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &max_image_units);
}

void ShaderGLES3::finish() {

	const VersionKey *V = NULL;
	while ((V = version_map.next(V))) {

		Version &v = version_map[*V];
		glDeleteShader(v.vert_id);
		glDeleteShader(v.frag_id);
		glDeleteProgram(v.id);
		memdelete_arr(v.uniform_location);
	}
}

void ShaderGLES3::clear_caches() {

	const VersionKey *V = NULL;
	while ((V = version_map.next(V))) {

		Version &v = version_map[*V];
		glDeleteShader(v.vert_id);
		glDeleteShader(v.frag_id);
		glDeleteProgram(v.id);
		memdelete_arr(v.uniform_location);
	}

	version_map.clear();

	custom_code_map.clear();
	version = NULL;
	last_custom_code = 1;
	uniforms_dirty = true;
}

uint32_t ShaderGLES3::create_custom_shader() {

	custom_code_map[last_custom_code] = CustomCode();
	custom_code_map[last_custom_code].version = 1;
	return last_custom_code++;
}

void ShaderGLES3::set_custom_shader_code(uint32_t p_code_id, const String &p_vertex, const String &p_vertex_globals, const String &p_fragment, const String &p_light, const String &p_fragment_globals, const String &p_uniforms, const Vector<StringName> &p_texture_uniforms, const Vector<CharString> &p_custom_defines) {

	ERR_FAIL_COND(!custom_code_map.has(p_code_id));
	CustomCode *cc = &custom_code_map[p_code_id];

	cc->vertex = p_vertex;
	cc->vertex_globals = p_vertex_globals;
	cc->fragment = p_fragment;
	cc->fragment_globals = p_fragment_globals;
	cc->light = p_light;
	cc->texture_uniforms = p_texture_uniforms;
	cc->uniforms = p_uniforms;
	cc->custom_defines = p_custom_defines;
	cc->version++;
}

void ShaderGLES3::set_custom_shader(uint32_t p_code_id) {

	new_conditional_version.code_version = p_code_id;
}

void ShaderGLES3::free_custom_shader(uint32_t p_code_id) {

	/*  if (! custom_code_map.has( p_code_id )) {
        print_line("no code id "+itos(p_code_id));
    } else {
        print_line("freed code id "+itos(p_code_id));

    }*/

	ERR_FAIL_COND(!custom_code_map.has(p_code_id));
	if (conditional_version.code_version == p_code_id)
		conditional_version.code_version = 0; //bye

	custom_code_map.erase(p_code_id);
}

void ShaderGLES3::set_base_material_tex_index(int p_idx) {

	base_material_tex_index = p_idx;
}

ShaderGLES3::ShaderGLES3() {
	version = NULL;
	last_custom_code = 1;
	uniforms_dirty = true;
	base_material_tex_index = 0;
}

ShaderGLES3::~ShaderGLES3() {

	finish();
}

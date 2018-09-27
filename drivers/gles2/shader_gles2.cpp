/*************************************************************************/
/*  shader_gles2.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "shader_gles2.h"

#include "core/os/memory.h"
#include "core/print_string.h"
#include "core/string_builder.h"
#include "rasterizer_gles2.h"
#include "rasterizer_storage_gles2.h"

// #define DEBUG_OPENGL

// #include "shaders/copy.glsl.gen.h"

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

ShaderGLES2 *ShaderGLES2::active = NULL;

//#define DEBUG_SHADER

#ifdef DEBUG_SHADER

#define DEBUG_PRINT(m_text) print_line(m_text);

#else

#define DEBUG_PRINT(m_text)

#endif

void ShaderGLES2::bind_uniforms() {
	if (!uniforms_dirty)
		return;

	// regular uniforms

	const Map<uint32_t, Variant>::Element *E = uniform_defaults.front();

	while (E) {
		int idx = E->key();
		int location = version->uniform_location[idx];

		if (location < 0) {
			E = E->next();
			continue;
		}

		Variant v;

		v = E->value();

		_set_uniform_variant(location, v);
		E = E->next();
	}

	// camera uniforms

	const Map<uint32_t, CameraMatrix>::Element *C = uniform_cameras.front();

	while (C) {
		int idx = E->key();
		int location = version->uniform_location[idx];

		if (location < 0) {
			C = C->next();
			continue;
		}

		glUniformMatrix4fv(location, 1, GL_FALSE, &(C->get().matrix[0][0]));
		C = C->next();
	}

	uniforms_dirty = false;
}

GLint ShaderGLES2::get_uniform_location(int p_index) const {

	ERR_FAIL_COND_V(!version, -1);

	return version->uniform_location[p_index];
}

bool ShaderGLES2::bind() {

	if (active != this || !version || new_conditional_version.key != conditional_version.key) {
		conditional_version = new_conditional_version;
		version = get_current_version();
	} else {
		return false;
	}

	ERR_FAIL_COND_V(!version, false);

	if (!version->ok) { //broken, unable to bind (do not throw error, you saw it before already when it failed compilation).
		glUseProgram(0);
		return false;
	}

	glUseProgram(version->id);

	// find out uniform names and locations

	int count;
	glGetProgramiv(version->id, GL_ACTIVE_UNIFORMS, &count);
	version->uniform_names.resize(count);

	for (int i = 0; i < count; i++) {
		GLchar uniform_name[1024];
		int len = 0;
		GLint size = 0;
		GLenum type;

		glGetActiveUniform(version->id, i, 1024, &len, &size, &type, uniform_name);

		uniform_name[len] = '\0';
		String name = String((const char *)uniform_name);

		version->uniform_names.write[i] = name;
	}

	bind_uniforms();

	DEBUG_TEST_ERROR("use program");

	active = this;
	uniforms_dirty = true;

	return true;
}

void ShaderGLES2::unbind() {
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

ShaderGLES2::Version *ShaderGLES2::get_current_version() {

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
			glDeleteShader(v.vert_id);
			glDeleteShader(v.frag_id);
			glDeleteProgram(v.id);
			v.id = 0;
		}
	}

	v.ok = false;

	Vector<const char *> strings;

#ifdef GLES_OVER_GL
	strings.push_back("#version 120\n");
	strings.push_back("#define USE_GLES_OVER_GL\n");
#else
	strings.push_back("#version 100\n");
#endif

	int define_line_ofs = 1;

	for (int j = 0; j < conditional_count; j++) {
		bool enable = (conditional_version.version & (1 << j)) > 0;

		if (enable) {
			strings.push_back(conditional_defines[j]);
			define_line_ofs++;
			DEBUG_PRINT(conditional_defines[j]);
		}
	}

	// keep them around during the function
	CharString code_string;
	CharString code_string2;
	CharString code_globals;

	CustomCode *cc = NULL;

	if (conditional_version.code_version > 0) {
		cc = custom_code_map.getptr(conditional_version.code_version);

		ERR_FAIL_COND_V(!cc, NULL);
		v.code_version = cc->version;
		define_line_ofs += 2;
	}

	// program

	v.id = glCreateProgram();
	ERR_FAIL_COND_V(v.id == 0, NULL);

	if (cc) {
		for (int i = 0; i < cc->custom_defines.size(); i++) {
			strings.push_back(cc->custom_defines.write[i]);
			DEBUG_PRINT("CD #" + itos(i) + ": " + String(cc->custom_defines[i].get_data()));
		}
	}

	// vertex shader

	int string_base_size = strings.size();

	strings.push_back(vertex_code0.get_data());

	if (cc) {
		code_globals = cc->vertex_globals.ascii();
		strings.push_back(code_globals.get_data());
	}

	strings.push_back(vertex_code1.get_data());

	if (cc) {
		code_string = cc->vertex.ascii();
		strings.push_back(code_string.get_data());
	}

	strings.push_back(vertex_code2.get_data());

#ifdef DEBUG_SHADER

	DEBUG_PRINT("\nVertex Code:\n\n" + String(code_string.get_data()));

#endif

	v.vert_id = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(v.vert_id, strings.size(), &strings[0], NULL);
	glCompileShader(v.vert_id);

	GLint status;

	glGetShaderiv(v.vert_id, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLsizei iloglen;
		glGetShaderiv(v.vert_id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;

			ERR_PRINT("No OpenGL vertex shader compiler log. What the frick?");
		} else {
			if (iloglen == 0) {
				iloglen = 4096; // buggy driver (Adreno 220+)
			}

			char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
			ilogmem[iloglen] = '\0';
			glGetShaderInfoLog(v.vert_id, iloglen, &iloglen, ilogmem);

			String err_string = get_shader_name() + ": Vertex shader compilation failed:\n";

			err_string += ilogmem;

			_display_error_with_code(err_string, strings);

			Memory::free_static(ilogmem);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;
		}

		ERR_FAIL_V(NULL);
	}

	strings.resize(string_base_size);

	// fragment shader

	strings.push_back(fragment_code0.get_data());

	if (cc) {
		code_globals = cc->fragment_globals.ascii();
		strings.push_back(code_globals.get_data());
	}

	strings.push_back(fragment_code1.get_data());

	if (cc) {
		code_string = cc->fragment.ascii();
		strings.push_back(code_string.get_data());
	}

	strings.push_back(fragment_code2.get_data());

	if (cc) {
		code_string2 = cc->light.ascii();
		strings.push_back(code_string2.get_data());
	}

	strings.push_back(fragment_code3.get_data());

#ifdef DEBUG_SHADER

	if (cc) {
		DEBUG_PRINT("\nFragment Code:\n\n" + String(cc->fragment_globals));
	}
	DEBUG_PRINT("\nFragment Code:\n\n" + String(code_string.get_data()));
#endif

	v.frag_id = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(v.frag_id, strings.size(), &strings[0], NULL);
	glCompileShader(v.frag_id);

	glGetShaderiv(v.frag_id, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLsizei iloglen;
		glGetShaderiv(v.frag_id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {
			glDeleteShader(v.frag_id);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;

			ERR_PRINT("No OpenGL fragment shader compiler log. What the frick?");
		} else {
			if (iloglen == 0) {
				iloglen = 4096; // buggy driver (Adreno 220+)
			}

			char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
			ilogmem[iloglen] = '\0';
			glGetShaderInfoLog(v.frag_id, iloglen, &iloglen, ilogmem);

			String err_string = get_shader_name() + ": Fragment shader compilation failed:\n";

			err_string += ilogmem;

			_display_error_with_code(err_string, strings);

			Memory::free_static(ilogmem);
			glDeleteShader(v.frag_id);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;
		}

		ERR_FAIL_V(NULL);
	}

	glAttachShader(v.id, v.frag_id);
	glAttachShader(v.id, v.vert_id);

	// bind the attribute locations. This has to be done before linking so that the
	// linker doesn't assign some random indices

	for (int i = 0; i < attribute_pair_count; i++) {
		glBindAttribLocation(v.id, attribute_pairs[i].index, attribute_pairs[i].name);
	}

	glLinkProgram(v.id);

	glGetProgramiv(v.id, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLsizei iloglen;
		glGetProgramiv(v.id, GL_INFO_LOG_LENGTH, &iloglen);

		if (iloglen < 0) {
			glDeleteShader(v.frag_id);
			glDeleteShader(v.vert_id);
			glDeleteProgram(v.id);
			v.id = 0;

			ERR_PRINT("No OpenGL program link log. What the frick?");
			ERR_FAIL_V(NULL);
		}

		if (iloglen == 0) {
			iloglen = 4096; // buggy driver (Adreno 220+)
		}

		char *ilogmem = (char *)Memory::alloc_static(iloglen + 1);
		ilogmem[iloglen] = '\0';
		glGetProgramInfoLog(v.id, iloglen, &iloglen, ilogmem);

		String err_string = get_shader_name() + ": Program linking failed:\n";

		err_string += ilogmem;

		_display_error_with_code(err_string, strings);

		Memory::free_static(ilogmem);
		glDeleteShader(v.frag_id);
		glDeleteShader(v.vert_id);
		glDeleteProgram(v.id);
		v.id = 0;

		ERR_FAIL_V(NULL);
	}

	// get uniform locations

	glUseProgram(v.id);

	for (int i = 0; i < uniform_count; i++) {
		v.uniform_location[i] = glGetUniformLocation(v.id, uniform_names[i]);
	}

	for (int i = 0; i < texunit_pair_count; i++) {
		GLint loc = glGetUniformLocation(v.id, texunit_pairs[i].name);
		if (loc >= 0) {
			if (texunit_pairs[i].index < 0) {
				glUniform1i(loc, max_image_units + texunit_pairs[i].index);
			} else {
				glUniform1i(loc, texunit_pairs[i].index);
			}
		}
	}

	if (cc) {
		// uniforms
		for (int i = 0; i < cc->custom_uniforms.size(); i++) {
			StringName native_uniform_name = "m_" + cc->custom_uniforms[i];
			GLint location = glGetUniformLocation(v.id, ((String)native_uniform_name).ascii().get_data());
			v.custom_uniform_locations[cc->custom_uniforms[i]] = location;
		}

		// textures
		for (int i = 0; i < cc->texture_uniforms.size(); i++) {
			StringName native_uniform_name = "m_" + cc->texture_uniforms[i];
			GLint location = glGetUniformLocation(v.id, ((String)native_uniform_name).ascii().get_data());
			v.custom_uniform_locations[cc->texture_uniforms[i]] = location;
		}
	}

	glUseProgram(0);
	v.ok = true;

	return &v;
}

GLint ShaderGLES2::get_uniform_location(const String &p_name) const {

	ERR_FAIL_COND_V(!version, -1);
	return glGetUniformLocation(version->id, p_name.ascii().get_data());
}

void ShaderGLES2::setup(
		const char **p_conditional_defines,
		int p_conditional_count,
		const char **p_uniform_names,
		int p_uniform_count,
		const AttributePair *p_attribute_pairs,
		int p_attribute_count,
		const TexUnitPair *p_texunit_pairs,
		int p_texunit_pair_count,
		const char *p_vertex_code,
		const char *p_fragment_code,
		int p_vertex_code_start,
		int p_fragment_code_start) {

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

	{
		String globals_tag = "\nVERTEX_SHADER_GLOBALS";
		String code_tag = "\nVERTEX_SHADER_CODE";
		String code = vertex_code;
		int cpos = code.find(globals_tag);
		if (cpos == -1) {
			vertex_code0 = code.ascii();
		} else {
			vertex_code0 = code.substr(0, cpos).ascii();
			code = code.substr(cpos + globals_tag.length(), code.length());

			cpos = code.find(code_tag);

			if (cpos == -1) {
				vertex_code1 = code.ascii();
			} else {
				vertex_code1 = code.substr(0, cpos).ascii();
				vertex_code2 = code.substr(cpos + code_tag.length(), code.length()).ascii();
			}
		}
	}

	{
		String globals_tag = "\nFRAGMENT_SHADER_GLOBALS";
		String code_tag = "\nFRAGMENT_SHADER_CODE";
		String light_code_tag = "\nLIGHT_SHADER_CODE";
		String code = fragment_code;
		int cpos = code.find(globals_tag);
		if (cpos == -1) {
			fragment_code0 = code.ascii();
		} else {
			fragment_code0 = code.substr(0, cpos).ascii();
			code = code.substr(cpos + globals_tag.length(), code.length());

			cpos = code.find(code_tag);

			if (cpos == -1) {
				fragment_code1 = code.ascii();
			} else {

				fragment_code1 = code.substr(0, cpos).ascii();
				String code2 = code.substr(cpos + code_tag.length(), code.length());

				cpos = code2.find(light_code_tag);
				if (cpos == -1) {
					fragment_code2 = code2.ascii();
				} else {
					fragment_code2 = code2.substr(0, cpos).ascii();
					fragment_code3 = code2.substr(cpos + light_code_tag.length(), code2.length()).ascii();
				}
			}
		}
	}

	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_image_units);
}

void ShaderGLES2::finish() {
	const VersionKey *V = NULL;

	while ((V = version_map.next(V))) {
		Version &v = version_map[*V];
		glDeleteShader(v.vert_id);
		glDeleteShader(v.frag_id);
		glDeleteProgram(v.id);
		memdelete_arr(v.uniform_location);
	}
}

void ShaderGLES2::clear_caches() {
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

uint32_t ShaderGLES2::create_custom_shader() {
	custom_code_map[last_custom_code] = CustomCode();
	custom_code_map[last_custom_code].version = 1;
	return last_custom_code++;
}

void ShaderGLES2::set_custom_shader_code(uint32_t p_code_id,
		const String &p_vertex,
		const String &p_vertex_globals,
		const String &p_fragment,
		const String &p_light,
		const String &p_fragment_globals,
		const Vector<StringName> &p_uniforms,
		const Vector<StringName> &p_texture_uniforms,
		const Vector<CharString> &p_custom_defines) {
	CustomCode *cc = custom_code_map.getptr(p_code_id);
	ERR_FAIL_COND(!cc);

	cc->vertex = p_vertex;
	cc->vertex_globals = p_vertex_globals;
	cc->fragment = p_fragment;
	cc->fragment_globals = p_fragment_globals;
	cc->light = p_light;
	cc->custom_uniforms = p_uniforms;
	cc->custom_defines = p_custom_defines;
	cc->texture_uniforms = p_texture_uniforms;
	cc->version++;
}

void ShaderGLES2::set_custom_shader(uint32_t p_code_id) {
	new_conditional_version.code_version = p_code_id;
}

void ShaderGLES2::free_custom_shader(uint32_t p_code_id) {
	ERR_FAIL_COND(!custom_code_map.has(p_code_id));
	if (conditional_version.code_version == p_code_id)
		conditional_version.code_version = 0;

	custom_code_map.erase(p_code_id);
}

void ShaderGLES2::use_material(void *p_material) {
	RasterizerStorageGLES2::Material *material = (RasterizerStorageGLES2::Material *)p_material;

	if (!material) {
		return;
	}

	if (!material->shader) {
		return;
	}

	Version *v = version_map.getptr(conditional_version);

	// bind uniforms
	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = material->shader->uniforms.front(); E; E = E->next()) {

		if (E->get().texture_order >= 0)
			continue; // this is a texture, doesn't go here

		Map<StringName, Variant>::Element *V = material->params.find(E->key());

		Pair<ShaderLanguage::DataType, Vector<ShaderLanguage::ConstantNode::Value> > value;

		value.first = E->get().type;
		value.second = E->get().default_value;

		if (V) {
			value.second = Vector<ShaderLanguage::ConstantNode::Value>();
			value.second.resize(E->get().default_value.size());
			switch (E->get().type) {
				case ShaderLanguage::TYPE_BOOL: {
					if (value.second.size() < 1)
						value.second.resize(1);
					value.second.write[0].boolean = V->get();
				} break;

				case ShaderLanguage::TYPE_BVEC2: {
					if (value.second.size() < 2)
						value.second.resize(2);
					int flags = V->get();
					value.second.write[0].boolean = flags & 1;
					value.second.write[1].boolean = flags & 2;
				} break;

				case ShaderLanguage::TYPE_BVEC3: {
					if (value.second.size() < 3)
						value.second.resize(3);
					int flags = V->get();
					value.second.write[0].boolean = flags & 1;
					value.second.write[1].boolean = flags & 2;
					value.second.write[2].boolean = flags & 4;

				} break;

				case ShaderLanguage::TYPE_BVEC4: {
					if (value.second.size() < 4)
						value.second.resize(4);
					int flags = V->get();
					value.second.write[0].boolean = flags & 1;
					value.second.write[1].boolean = flags & 2;
					value.second.write[2].boolean = flags & 4;
					value.second.write[3].boolean = flags & 8;

				} break;

				case ShaderLanguage::TYPE_INT: {
					if (value.second.size() < 1)
						value.second.resize(1);
					int val = V->get();
					value.second.write[0].sint = val;
				} break;

				case ShaderLanguage::TYPE_IVEC2: {
					if (value.second.size() < 2)
						value.second.resize(2);
					PoolIntArray val = V->get();
					for (int i = 0; i < val.size(); i++) {
						value.second.write[i].sint = val[i];
					}
				} break;

				case ShaderLanguage::TYPE_IVEC3: {
					if (value.second.size() < 3)
						value.second.resize(3);
					PoolIntArray val = V->get();
					for (int i = 0; i < val.size(); i++) {
						value.second.write[i].sint = val[i];
					}

				} break;

				case ShaderLanguage::TYPE_IVEC4: {
					if (value.second.size() < 4)
						value.second.resize(4);
					PoolIntArray val = V->get();
					for (int i = 0; i < val.size(); i++) {
						value.second.write[i].sint = val[i];
					}

				} break;

				case ShaderLanguage::TYPE_UINT: {
					if (value.second.size() < 1)
						value.second.resize(1);
					uint32_t val = V->get();
					value.second.write[0].uint = val;
				} break;

				case ShaderLanguage::TYPE_UVEC2: {
					if (value.second.size() < 2)
						value.second.resize(2);
					PoolIntArray val = V->get();
					for (int i = 0; i < val.size(); i++) {
						value.second.write[i].uint = val[i];
					}

				} break;

				case ShaderLanguage::TYPE_UVEC3: {
					if (value.second.size() < 3)
						value.second.resize(3);
					PoolIntArray val = V->get();
					for (int i = 0; i < val.size(); i++) {
						value.second.write[i].uint = val[i];
					}

				} break;

				case ShaderLanguage::TYPE_UVEC4: {
					if (value.second.size() < 4)
						value.second.resize(4);
					PoolIntArray val = V->get();
					for (int i = 0; i < val.size(); i++) {
						value.second.write[i].uint = val[i];
					}

				} break;

				case ShaderLanguage::TYPE_FLOAT: {
					if (value.second.size() < 1)
						value.second.resize(1);
					value.second.write[0].real = V->get();

				} break;

				case ShaderLanguage::TYPE_VEC2: {
					if (value.second.size() < 2)
						value.second.resize(2);
					Vector2 val = V->get();
					value.second.write[0].real = val.x;
					value.second.write[1].real = val.y;
				} break;

				case ShaderLanguage::TYPE_VEC3: {
					if (value.second.size() < 3)
						value.second.resize(3);
					Vector3 val = V->get();
					value.second.write[0].real = val.x;
					value.second.write[1].real = val.y;
					value.second.write[2].real = val.z;
				} break;

				case ShaderLanguage::TYPE_VEC4: {
					if (value.second.size() < 4)
						value.second.resize(4);
					if (V->get().get_type() == Variant::PLANE) {
						Plane val = V->get();
						value.second.write[0].real = val.normal.x;
						value.second.write[1].real = val.normal.y;
						value.second.write[2].real = val.normal.z;
						value.second.write[3].real = val.d;
					} else {
						Color val = V->get();
						value.second.write[0].real = val.r;
						value.second.write[1].real = val.g;
						value.second.write[2].real = val.b;
						value.second.write[3].real = val.a;
					}

				} break;

				case ShaderLanguage::TYPE_MAT2: {
					Transform2D val = V->get();

					if (value.second.size() < 4) {
						value.second.resize(4);
					}

					value.second.write[0].real = val.elements[0][0];
					value.second.write[1].real = val.elements[0][1];
					value.second.write[2].real = val.elements[1][0];
					value.second.write[3].real = val.elements[1][1];

				} break;

				case ShaderLanguage::TYPE_MAT3: {
					Basis val = V->get();

					if (value.second.size() < 9) {
						value.second.resize(9);
					}

					value.second.write[0].real = val.elements[0][0];
					value.second.write[1].real = val.elements[0][1];
					value.second.write[2].real = val.elements[0][2];
					value.second.write[3].real = val.elements[1][0];
					value.second.write[4].real = val.elements[1][1];
					value.second.write[5].real = val.elements[1][2];
					value.second.write[6].real = val.elements[2][0];
					value.second.write[7].real = val.elements[2][1];
					value.second.write[8].real = val.elements[2][2];
				} break;

				case ShaderLanguage::TYPE_MAT4: {
					Transform val = V->get();

					if (value.second.size() < 16) {
						value.second.resize(16);
					}

					value.second.write[0].real = val.basis.elements[0][0];
					value.second.write[1].real = val.basis.elements[0][1];
					value.second.write[2].real = val.basis.elements[0][2];
					value.second.write[3].real = 0;
					value.second.write[4].real = val.basis.elements[1][0];
					value.second.write[5].real = val.basis.elements[1][1];
					value.second.write[6].real = val.basis.elements[1][2];
					value.second.write[7].real = 0;
					value.second.write[8].real = val.basis.elements[2][0];
					value.second.write[9].real = val.basis.elements[2][1];
					value.second.write[10].real = val.basis.elements[2][2];
					value.second.write[11].real = 0;
					value.second.write[12].real = val.origin[0];
					value.second.write[13].real = val.origin[1];
					value.second.write[14].real = val.origin[2];
					value.second.write[15].real = 1;
				} break;

				default: {

				} break;
			}
		} else {
			if (value.second.size() == 0) {
				// No default value set... weird, let's just use zero for everything
				size_t default_arg_size = 1;
				bool is_float = false;
				switch (E->get().type) {
					case ShaderLanguage::TYPE_BOOL:
					case ShaderLanguage::TYPE_INT:
					case ShaderLanguage::TYPE_UINT: {
						default_arg_size = 1;
					} break;

					case ShaderLanguage::TYPE_FLOAT: {
						default_arg_size = 1;
						is_float = true;
					} break;

					case ShaderLanguage::TYPE_BVEC2:
					case ShaderLanguage::TYPE_IVEC2:
					case ShaderLanguage::TYPE_UVEC2: {
						default_arg_size = 2;
					} break;

					case ShaderLanguage::TYPE_VEC2: {
						default_arg_size = 2;
						is_float = true;
					} break;

					case ShaderLanguage::TYPE_BVEC3:
					case ShaderLanguage::TYPE_IVEC3:
					case ShaderLanguage::TYPE_UVEC3: {
						default_arg_size = 3;
					} break;

					case ShaderLanguage::TYPE_VEC3: {
						default_arg_size = 3;
						is_float = true;
					} break;

					case ShaderLanguage::TYPE_BVEC4:
					case ShaderLanguage::TYPE_IVEC4:
					case ShaderLanguage::TYPE_UVEC4: {
						default_arg_size = 4;
					} break;

					case ShaderLanguage::TYPE_VEC4: {
						default_arg_size = 4;
						is_float = true;
					} break;

					default: {
						// TODO matricies and all that stuff
						default_arg_size = 1;
					} break;
				}

				value.second.resize(default_arg_size);

				for (size_t i = 0; i < default_arg_size; i++) {
					if (is_float) {
						value.second.write[i].real = 0.0;
					} else {
						value.second.write[i].uint = 0;
					}
				}
			}
		}

		GLint location;
		if (v->custom_uniform_locations.has(E->key())) {
			location = v->custom_uniform_locations[E->key()];
		} else {
			int idx = v->uniform_names.find(E->key()); // TODO maybe put those in a Map?
			if (idx < 0) {
				location = -1;
			} else {
				location = v->uniform_location[idx];
			}
		}

		_set_uniform_value(location, value);
	}

	// bind textures
	int tc = material->textures.size();
	Pair<StringName, RID> *textures = material->textures.ptrw();

	for (int i = 0; i < tc; i++) {

		Pair<ShaderLanguage::DataType, Vector<ShaderLanguage::ConstantNode::Value> > value;
		value.first = ShaderLanguage::TYPE_INT;
		value.second.resize(1);
		value.second.write[0].sint = i;

		// GLint location = get_uniform_location(textures[i].first);

		// if (location < 0) {
		//	location = material->shader->uniform_locations[textures[i].first];
		// }
		GLint location = -1;
		if (v->custom_uniform_locations.has(textures[i].first)) {
			location = v->custom_uniform_locations[textures[i].first];
		} else {
			location = get_uniform_location(textures[i].first);
		}

		_set_uniform_value(location, value);
	}
}

void ShaderGLES2::set_base_material_tex_index(int p_idx) {
}

ShaderGLES2::ShaderGLES2() {
	version = NULL;
	last_custom_code = 1;
	uniforms_dirty = true;
}

ShaderGLES2::~ShaderGLES2() {
	finish();
}

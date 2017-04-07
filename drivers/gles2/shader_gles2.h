/*************************************************************************/
/*  shader_gles2.h                                                       */
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
#ifndef SHADER_GLES2_H
#define SHADER_GLES2_H

#ifdef GLES2_ENABLED

#include "platform_config.h"
#ifndef GLES2_INCLUDE_H
#include <GLES2/gl2.h>
#else
#include GLES2_INCLUDE_H
#endif

#include "camera_matrix.h"
#include "hash_map.h"
#include "map.h"
#include "variant.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ShaderGLES2 {
protected:
	struct Enum {

		uint64_t mask;
		uint64_t shift;
		const char *defines[16];
	};

	struct EnumValue {

		uint64_t set_mask;
		uint64_t clear_mask;
	};

	struct AttributePair {

		const char *name;
		int index;
	};

	struct UniformPair {
		const char *name;
		Variant::Type type_hint;
	};

	struct TexUnitPair {

		const char *name;
		int index;
	};

	bool uniforms_dirty;

private:
	//@TODO Optimize to a fixed set of shader pools and use a LRU
	int uniform_count;
	int texunit_pair_count;
	int conditional_count;
	int vertex_code_start;
	int fragment_code_start;
	int attribute_pair_count;

	struct CustomCode {

		String vertex;
		String vertex_globals;
		String fragment;
		String fragment_globals;
		String light;
		uint32_t version;
		Vector<StringName> custom_uniforms;
		Vector<const char *> custom_defines;
	};

	struct Version {

		GLuint id;
		GLuint vert_id;
		GLuint frag_id;
		GLint *uniform_location;
		Vector<GLint> custom_uniform_locations;
		uint32_t code_version;
		bool ok;
		Version() {
			code_version = 0;
			ok = false;
			uniform_location = NULL;
		}
	};

	Version *version;

	union VersionKey {

		struct {
			uint32_t version;
			uint32_t code_version;
		};
		uint64_t key;
		bool operator==(const VersionKey &p_key) const { return key == p_key.key; }
		bool operator<(const VersionKey &p_key) const { return key < p_key.key; }
	};

	struct VersionKeyHash {

		static _FORCE_INLINE_ uint32_t hash(const VersionKey &p_key) { return HashMapHasherDefault::hash(p_key.key); };
	};

	//this should use a way more cachefriendly version..
	HashMap<VersionKey, Version, VersionKeyHash> version_map;

	HashMap<uint32_t, CustomCode> custom_code_map;
	uint32_t last_custom_code;

	VersionKey conditional_version;
	VersionKey new_conditional_version;

	virtual String get_shader_name() const = 0;

	const char **conditional_defines;
	const char **uniform_names;
	const AttributePair *attribute_pairs;
	const TexUnitPair *texunit_pairs;
	const char *vertex_code;
	const char *fragment_code;
	CharString fragment_code0;
	CharString fragment_code1;
	CharString fragment_code2;
	CharString fragment_code3;

	CharString vertex_code0;
	CharString vertex_code1;
	CharString vertex_code2;

	Version *get_current_version();

	static ShaderGLES2 *active;

	_FORCE_INLINE_ void _set_uniform_variant(GLint p_uniform, const Variant &p_value) {

		if (p_uniform < 0)
			return; // do none
		switch (p_value.get_type()) {

			case Variant::BOOL:
			case Variant::INT: /* {

			int val=p_value;
			glUniform1i( p_uniform, val );
		} break; */
			case Variant::REAL: {

				real_t val = p_value;
				glUniform1f(p_uniform, val);
			} break;
			case Variant::COLOR: {

				Color val = p_value;
				glUniform4f(p_uniform, val.r, val.g, val.b, val.a);
			} break;
			case Variant::VECTOR2: {

				Vector2 val = p_value;
				glUniform2f(p_uniform, val.x, val.y);
			} break;
			case Variant::VECTOR3: {

				Vector3 val = p_value;
				glUniform3f(p_uniform, val.x, val.y, val.z);
			} break;
			case Variant::PLANE: {

				Plane val = p_value;
				glUniform4f(p_uniform, val.normal.x, val.normal.y, val.normal.z, val.d);
			} break;
			case Variant::QUAT: {

				Quat val = p_value;
				glUniform4f(p_uniform, val.x, val.y, val.z, val.w);
			} break;

			case Variant::MATRIX32: {

				Transform2D tr = p_value;
				GLfloat matrix[16] = { /* build a 16x16 matrix */
					tr.elements[0][0],
					tr.elements[0][1],
					0,
					0,
					tr.elements[1][0],
					tr.elements[1][1],
					0,
					0,
					0,
					0,
					1,
					0,
					tr.elements[2][0],
					tr.elements[2][1],
					0,
					1
				};

				glUniformMatrix4fv(p_uniform, 1, false, matrix);

			} break;
			case Variant::MATRIX3:
			case Variant::TRANSFORM: {

				Transform tr = p_value;
				GLfloat matrix[16] = { /* build a 16x16 matrix */
					tr.basis.elements[0][0],
					tr.basis.elements[1][0],
					tr.basis.elements[2][0],
					0,
					tr.basis.elements[0][1],
					tr.basis.elements[1][1],
					tr.basis.elements[2][1],
					0,
					tr.basis.elements[0][2],
					tr.basis.elements[1][2],
					tr.basis.elements[2][2],
					0,
					tr.origin.x,
					tr.origin.y,
					tr.origin.z,
					1
				};

				glUniformMatrix4fv(p_uniform, 1, false, matrix);
			} break;
			default: { ERR_FAIL(); } // do nothing
		}
	}

	Map<uint32_t, Variant> uniform_defaults;
	Map<uint32_t, CameraMatrix> uniform_cameras;

protected:
	_FORCE_INLINE_ int _get_uniform(int p_which) const;
	_FORCE_INLINE_ void _set_conditional(int p_which, bool p_value);

	void setup(const char **p_conditional_defines, int p_conditional_count, const char **p_uniform_names, int p_uniform_count, const AttributePair *p_attribute_pairs, int p_attribute_count, const TexUnitPair *p_texunit_pairs, int p_texunit_pair_count, const char *p_vertex_code, const char *p_fragment_code, int p_vertex_code_start, int p_fragment_code_start);

	ShaderGLES2();

public:
	enum {
		CUSTOM_SHADER_DISABLED = 0
	};

	GLint get_uniform_location(const String &p_name) const;
	GLint get_uniform_location(int p_uniform) const;

	static _FORCE_INLINE_ ShaderGLES2 *get_active() { return active; };
	bool bind();
	void unbind();
	void bind_uniforms();

	inline GLuint get_program() const { return version ? version->id : 0; }

	void clear_caches();

	uint32_t create_custom_shader();
	void set_custom_shader_code(uint32_t p_id, const String &p_vertex, const String &p_vertex_globals, const String &p_fragment, const String &p_p_light, const String &p_fragment_globals, const Vector<StringName> &p_uniforms, const Vector<const char *> &p_custom_defines);
	void set_custom_shader(uint32_t p_id);
	void free_custom_shader(uint32_t p_id);

	void set_uniform_default(int p_idx, const Variant &p_value) {

		if (p_value.get_type() == Variant::NIL) {

			uniform_defaults.erase(p_idx);
		} else {

			uniform_defaults[p_idx] = p_value;
		}
		uniforms_dirty = true;
	};

	uint32_t get_version() const { return new_conditional_version.version; }

	void set_uniform_camera(int p_idx, const CameraMatrix &p_mat) {

		uniform_cameras[p_idx] = p_mat;
		uniforms_dirty = true;
	};

	_FORCE_INLINE_ void set_custom_uniform(int p_idx, const Variant &p_value) {

		ERR_FAIL_COND(!version);
		ERR_FAIL_INDEX(p_idx, version->custom_uniform_locations.size());
		_set_uniform_variant(version->custom_uniform_locations[p_idx], p_value);
	}

	_FORCE_INLINE_ GLint get_custom_uniform_location(int p_idx) {

		ERR_FAIL_COND_V(!version, -1);
		ERR_FAIL_INDEX_V(p_idx, version->custom_uniform_locations.size(), -1);
		return version->custom_uniform_locations[p_idx];
	}

	virtual void init() = 0;
	void finish();

	virtual ~ShaderGLES2();
};

// called a lot, made inline

int ShaderGLES2::_get_uniform(int p_which) const {

	ERR_FAIL_INDEX_V(p_which, uniform_count, -1);
	ERR_FAIL_COND_V(!version, -1);
	return version->uniform_location[p_which];
}

void ShaderGLES2::_set_conditional(int p_which, bool p_value) {

	ERR_FAIL_INDEX(p_which, conditional_count);
	if (p_value)
		new_conditional_version.version |= (1 << p_which);
	else
		new_conditional_version.version &= ~(1 << p_which);
}

#endif
#endif

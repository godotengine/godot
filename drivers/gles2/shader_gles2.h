/*************************************************************************/
/*  shader_gles2.h                                                       */
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

#ifndef SHADER_GLES2_H
#define SHADER_GLES2_H

// This must come first to avoid windows.h mess
#include "platform_config.h"
#ifndef GLES2_INCLUDE_H
#include <GLES2/gl2.h>
#else
#include GLES2_INCLUDE_H
#endif

#include "core/hash_map.h"
#include "core/map.h"
#include "core/math/camera_matrix.h"
#include "core/pair.h"
#include "core/variant.h"
#include "servers/visual/shader_language.h"

#include <stdio.h>

class RasterizerStorageGLES2;

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
		Vector<StringName> texture_uniforms;
		Vector<StringName> custom_uniforms;
		Vector<CharString> custom_defines;
		Set<uint64_t> versions;
	};

	struct Version {
		GLuint id;
		GLuint vert_id;
		GLuint frag_id;
		GLint *uniform_location;
		Vector<GLint> texture_uniform_locations;
		Map<StringName, GLint> custom_uniform_locations;
		uint32_t code_version;
		bool ok;
		Version() {
			id = 0;
			vert_id = 0;
			frag_id = 0;
			uniform_location = nullptr;
			code_version = 0;
			ok = false;
		}
	};

	Version *version;

	union VersionKey {
		struct {
			uint64_t version;
			uint32_t code_version;
		};
		unsigned char key[12];
		bool operator==(const VersionKey &p_key) const { return version == p_key.version && code_version == p_key.code_version; }
		bool operator<(const VersionKey &p_key) const { return version < p_key.version || (version == p_key.version && code_version < p_key.code_version); }
	};

	struct VersionKeyHash {
		static _FORCE_INLINE_ uint32_t hash(const VersionKey &p_key) { return hash_djb2_buffer(p_key.key, sizeof(p_key.key)); }
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

	Vector<CharString> custom_defines;

	Version *get_current_version();

	static ShaderGLES2 *active;

	int max_image_units;

	Map<StringName, Pair<ShaderLanguage::DataType, Vector<ShaderLanguage::ConstantNode::Value>>> uniform_values;

protected:
	_FORCE_INLINE_ int _get_uniform(int p_which) const;
	_FORCE_INLINE_ void _set_conditional(int p_which, bool p_value);

	void setup(const char **p_conditional_defines,
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
			int p_fragment_code_start);

	ShaderGLES2();

public:
	enum {
		CUSTOM_SHADER_DISABLED = 0
	};

	GLint get_uniform_location(const String &p_name) const;
	GLint get_uniform_location(int p_index) const;

	static _FORCE_INLINE_ ShaderGLES2 *get_active() { return active; }
	bool bind();
	void unbind();

	inline GLuint get_program() const { return version ? version->id : 0; }

	void clear_caches();

	uint32_t create_custom_shader();
	void set_custom_shader_code(uint32_t p_code_id,
			const String &p_vertex,
			const String &p_vertex_globals,
			const String &p_fragment,
			const String &p_light,
			const String &p_fragment_globals,
			const Vector<StringName> &p_uniforms,
			const Vector<StringName> &p_texture_uniforms,
			const Vector<CharString> &p_custom_defines);

	void set_custom_shader(uint32_t p_code_id);
	void free_custom_shader(uint32_t p_code_id);

	uint64_t get_version_key() const { return conditional_version.version; }

	// this void* is actually a RasterizerStorageGLES2::Material, but C++ doesn't
	// like forward declared nested classes.
	void use_material(void *p_material);

	_FORCE_INLINE_ uint64_t get_version() const { return new_conditional_version.version; }
	_FORCE_INLINE_ bool is_version_valid() const { return version && version->ok; }

	virtual void init() = 0;
	void finish();

	void add_custom_define(const String &p_define) {
		custom_defines.push_back(p_define.utf8());
	}

	void get_custom_defines(Vector<String> *p_defines) {
		for (int i = 0; i < custom_defines.size(); i++) {
			p_defines->push_back(custom_defines[i].get_data());
		}
	}

	void remove_custom_define(const String &p_define) {
		custom_defines.erase(p_define.utf8());
	}

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
	ERR_FAIL_INDEX(p_which, (int)sizeof(new_conditional_version.version) * 8);

	if (p_value) {
		new_conditional_version.version |= (uint64_t(1) << p_which);
	} else {
		new_conditional_version.version &= ~(uint64_t(1) << p_which);
	}
}

#endif

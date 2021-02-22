/*************************************************************************/
/*  shader_gles3.h                                                       */
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

#ifndef SHADER_GLES3_H
#define SHADER_GLES3_H

#include "core/hash_map.h"
#include "core/local_vector.h"
#include "core/map.h"
#include "core/math/camera_matrix.h"
#include "core/safe_refcount.h"
#include "core/self_list.h"
#include "core/variant.h"

#include "platform_config.h"
#ifndef GLES3_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include GLES3_INCLUDE_H
#endif

#include <stdio.h>

template <class K>
class ThreadedCallableQueue;
class ShaderCacheGLES3;

class ShaderGLES3 {
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

	struct UBOPair {
		const char *name;
		int index;
	};

	struct Feedback {
		const char *name;
		int conditional;
	};

	bool uniforms_dirty;

private:
	//@TODO Optimize to a fixed set of shader pools and use a LRU
	int uniform_count;
	int texunit_pair_count;
	int conditional_count;
	int ubo_count;
	int feedback_count;
	int vertex_code_start;
	int fragment_code_start;
	int attribute_pair_count;

	struct CustomCode {
		String vertex;
		String vertex_globals;
		String fragment;
		String fragment_globals;
		String light;
		String uniforms;
		int fallback_mode_hint;
		uint32_t version;
		Vector<StringName> texture_uniforms;
		Vector<CharString> custom_defines;
		Set<uint32_t> versions;
	};

public:
	static ShaderCacheGLES3 *shader_cache;
	static ThreadedCallableQueue<GLuint> *cache_write_queue;

	enum FallbackMode {
		FALLBACK_MODE_NONE,
		FALLBACK_MODE_NO_RENDER,
		FALLBACK_MODE_SIMPLE,
	};

	using DecideFallbackModeFunc = void (*)(uint32_t p_conditionals, int p_fallback_mode_hint, int *r_fallback_mode, uint32_t *r_simple_fallback_version);

	static bool force_no_render_fallback;
#ifdef DEBUG_ENABLED
	static bool force_use_fallbacks;
#endif
	static bool force_sync_compile;
	static int *shader_compiles_started_this_frame;
	static int max_shaders_compiling;
	static uint64_t current_frame;

	static void advance_async_shaders_compilation();

private:
	static int versions_compiling_count;

	struct Version {
		uint64_t version_key;

		// Set by the render thread upfront; the compile thread (for queued async.) reads them
		struct Ids {
			GLuint main;
			GLuint vert;
			GLuint frag;
		} ids;

		ShaderGLES3 *shader;
		uint32_t code_version;

		int fallback_mode;
		uint32_t simple_fallback_version; // Conditionals for the simple fallback version if fallback mode is simple
		GLint *uniform_location;
		Vector<GLint> texture_uniform_locations;
		bool uniforms_ready;
		uint64_t last_frame_processed;

		enum CompileStatus {
			COMPILE_STATUS_PENDING,
			COMPILE_STATUS_SOURCE_PROVIDED,
			COMPILE_STATUS_COMPILING,
			COMPILE_STATUS_PROCESSING_AT_QUEUE,
			COMPILE_STATUS_BINARY_READY,
			COMPILE_STATUS_BINARY_READY_FROM_CACHE,
			COMPILE_STATUS_LINKING,
			COMPILE_STATUS_ERROR,
			COMPILE_STATUS_RESTART_NEEDED,
			COMPILE_STATUS_OK,
		};
		CompileStatus compile_status;
		SelfList<Version> compiling_list;

		struct ProgramBinary {
			String cache_hash;
			enum Source {
				SOURCE_NONE,
				SOURCE_LOCAL, // Binary data will only be available if cache enabled
				SOURCE_QUEUE,
				SOURCE_CACHE,
			} source;
			// Shared with the compile thread (for queued async.); otherwise render thread only
			GLenum format;
			PoolByteArray data;
			SafeNumeric<int> result_from_queue;
		} program_binary;

		Version() :
				version_key(0),
				ids(),
				shader(nullptr),
				code_version(0),
				fallback_mode(FALLBACK_MODE_NONE),
				simple_fallback_version(0),
				uniform_location(nullptr),
				uniforms_ready(false),
				last_frame_processed(UINT64_MAX),
				compile_status(COMPILE_STATUS_PENDING),
				compiling_list(this),
				program_binary() {}
	};
	static SelfList<Version>::List versions_compiling;

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
	const UBOPair *ubo_pairs;
	const Feedback *feedbacks;
	const char *vertex_code;
	const char *fragment_code;
	CharString fragment_code0;
	CharString fragment_code1;
	CharString fragment_code2;
	CharString fragment_code3;
	CharString fragment_code4;

	CharString vertex_code0;
	CharString vertex_code1;
	CharString vertex_code2;
	CharString vertex_code3;

	Vector<CharString> custom_defines;

	int base_material_tex_index;

	ThreadedCallableQueue<GLuint> *compile_queue; // Non-null if using queued asynchronous compilation (via seconday context)
	bool parallel_compile_supported; // True if using natively supported asyncrhonous compilation
	uint32_t fallback_code_id;
	DecideFallbackModeFunc decide_fallback_mode_func;

	Version *get_current_version();
	// These will run on the shader compile thread if using que compile queue approach to async.
	void _set_source(Version::Ids p_ids, const LocalVector<const char *> &p_vertex_strings, const LocalVector<const char *> &p_fragment_strings) const;
	void _compile(Version::Ids p_ids) const;
	bool _complete_compile(Version::Ids p_ids, bool p_retrievable) const;
	bool _complete_link(Version::Ids p_ids, GLenum *r_program_format = nullptr, PoolByteArray *r_program_binary = nullptr) const;
	// ---
	bool _process_program_state();
	void _setup_uniforms(CustomCode *p_cc) const;
	void _dispose_program(Version *p_version);

	static ShaderGLES3 *active;

	int max_image_units;

	_FORCE_INLINE_ void _set_uniform_variant(GLint p_uniform, const Variant &p_value) {
		if (p_uniform < 0) {
			return; // do none
		}
		switch (p_value.get_type()) {
			case Variant::BOOL:
			case Variant::INT: {
				int val = p_value;
				glUniform1i(p_uniform, val);
			} break;
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

			case Variant::TRANSFORM2D: {
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
			case Variant::BASIS:
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
			default: {
				ERR_FAIL();
			} // do nothing
		}
	}

	Map<uint32_t, Variant> uniform_defaults;
	Map<uint32_t, CameraMatrix> uniform_cameras;

	bool _bind(bool p_fallback_forbidden, bool *r_fallen_back = nullptr);

protected:
	_FORCE_INLINE_ int _get_uniform(int p_which) const;
	_FORCE_INLINE_ void _set_conditional(int p_which, bool p_value);

	void setup(const char **p_conditional_defines, int p_conditional_count, const char **p_uniform_names, int p_uniform_count, const AttributePair *p_attribute_pairs, int p_attribute_count, const TexUnitPair *p_texunit_pairs, int p_texunit_pair_count, const UBOPair *p_ubo_pairs, int p_ubo_pair_count, const Feedback *p_feedback, int p_feedback_count, const char *p_vertex_code, const char *p_fragment_code, int p_vertex_code_start, int p_fragment_code_start);

	ShaderGLES3();

public:
	enum {
		CUSTOM_SHADER_DISABLED = 0
	};

	GLint get_uniform_location(const String &p_name) const;
	GLint get_uniform_location(int p_index) const;

	static _FORCE_INLINE_ ShaderGLES3 *get_active() { return active; };
	bool bind(bool *r_fallen_back = nullptr);
	void unbind();
	void bind_uniforms();

	void clear_caches();

	uint32_t create_custom_shader();
	void set_custom_shader_code(uint32_t p_code_id, const String &p_vertex, const String &p_vertex_globals, const String &p_fragment, const String &p_light, const String &p_fragment_globals, const String &p_uniforms, const Vector<StringName> &p_texture_uniforms, const Vector<CharString> &p_custom_defines, int p_fallback_mode_hint);
	void set_custom_shader(uint32_t p_code_id);
	void free_custom_shader(uint32_t p_code_id);

	void set_uniform_default(int p_idx, const Variant &p_value) {
		if (p_value.get_type() == Variant::NIL) {
			uniform_defaults.erase(p_idx);
		} else {
			uniform_defaults[p_idx] = p_value;
		}
		uniforms_dirty = true;
	}

	uint32_t get_version() const { return new_conditional_version.version; }
	void set_version(uint32_t p_version) { new_conditional_version.version = p_version; }
	_FORCE_INLINE_ bool is_version_valid() const { return version && version->compile_status == Version::COMPILE_STATUS_OK; }

	void set_uniform_camera(int p_idx, const CameraMatrix &p_mat) {
		uniform_cameras[p_idx] = p_mat;
		uniforms_dirty = true;
	};

	_FORCE_INLINE_ void set_texture_uniform(int p_idx, const Variant &p_value) {
		ERR_FAIL_COND(!version);
		ERR_FAIL_INDEX(p_idx, version->texture_uniform_locations.size());
		_set_uniform_variant(version->texture_uniform_locations[p_idx], p_value);
	}

	_FORCE_INLINE_ GLint get_texture_uniform_location(int p_idx) {
		ERR_FAIL_COND_V(!version, -1);
		ERR_FAIL_INDEX_V(p_idx, version->texture_uniform_locations.size(), -1);
		return version->texture_uniform_locations[p_idx];
	}

	virtual void init() = 0;
	void setup_async_compilation(ThreadedCallableQueue<GLuint> *p_compile_queue, bool p_parallel_compile_supported, uint32_t p_code_id, DecideFallbackModeFunc p_decide_fallback_mode_func);
	void finish();

	void set_base_material_tex_index(int p_idx);

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

	virtual ~ShaderGLES3();
};

// called a lot, made inline

int ShaderGLES3::_get_uniform(int p_which) const {
	ERR_FAIL_INDEX_V(p_which, uniform_count, -1);
	ERR_FAIL_COND_V(!version, -1);
	return version->uniform_location[p_which];
}

void ShaderGLES3::_set_conditional(int p_which, bool p_value) {
	ERR_FAIL_INDEX(p_which, conditional_count);
	if (p_value) {
		new_conditional_version.version |= (1 << p_which);
	} else {
		new_conditional_version.version &= ~(1 << p_which);
	}
}

#endif

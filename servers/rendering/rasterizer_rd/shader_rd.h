/*************************************************************************/
/*  shader_rd.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SHADER_RD_H
#define SHADER_RD_H

#include "core/hash_map.h"
#include "core/map.h"
#include "core/os/mutex.h"
#include "core/rid_owner.h"
#include "core/variant.h"

#include <stdio.h>
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ShaderRD {
	//versions
	CharString general_defines;
	Vector<CharString> variant_defines;

	struct Version {
		CharString uniforms;
		CharString vertex_globals;
		CharString vertex_code;
		CharString compute_globals;
		CharString compute_code;
		CharString fragment_light;
		CharString fragment_globals;
		CharString fragment_code;
		Vector<CharString> custom_defines;

		RID *variants; //same size as version defines

		bool valid;
		bool dirty;
		bool initialize_needed;
	};

	Mutex variant_set_mutex;

	void _compile_variant(uint32_t p_variant, Version *p_version);

	void _clear_version(Version *p_version);
	void _compile_version(Version *p_version);

	RID_Owner<Version> version_owner;

	CharString fragment_codev; //for version and extensions
	CharString fragment_code0;
	CharString fragment_code1;
	CharString fragment_code2;
	CharString fragment_code3;
	CharString fragment_code4;

	CharString vertex_codev; //for version and extensions
	CharString vertex_code0;
	CharString vertex_code1;
	CharString vertex_code2;
	CharString vertex_code3;

	bool is_compute = false;

	CharString compute_codev; //for version and extensions
	CharString compute_code0;
	CharString compute_code1;
	CharString compute_code2;
	CharString compute_code3;

	const char *name;

protected:
	ShaderRD() {}
	void setup(const char *p_vertex_code, const char *p_fragment_code, const char *p_compute_code, const char *p_name);

public:
	RID version_create();

	void version_set_code(RID p_version, const String &p_uniforms, const String &p_vertex_globals, const String &p_vertex_code, const String &p_fragment_globals, const String &p_fragment_light, const String &p_fragment_code, const Vector<String> &p_custom_defines);
	void version_set_compute_code(RID p_version, const String &p_uniforms, const String &p_compute_globals, const String &p_compute_code, const Vector<String> &p_custom_defines);

	_FORCE_INLINE_ RID version_get_shader(RID p_version, int p_variant) {
		ERR_FAIL_INDEX_V(p_variant, variant_defines.size(), RID());

		Version *version = version_owner.getornull(p_version);
		ERR_FAIL_COND_V(!version, RID());

		if (version->dirty) {
			_compile_version(version);
		}

		if (!version->valid) {
			return RID();
		}

		return version->variants[p_variant];
	}

	bool version_is_valid(RID p_version);

	bool version_free(RID p_version);

	void initialize(const Vector<String> &p_variant_defines, const String &p_general_defines = "");
	virtual ~ShaderRD();
};

#endif

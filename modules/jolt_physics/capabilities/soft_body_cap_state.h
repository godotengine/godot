/**************************************************************************/
/*  soft_body_cap_state.h                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/math/transform_3d.h"
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

#ifdef TESTS_ENABLED
#include "../tests/soft_body_cap_probe.h"
#endif

// Per-body configuration survives mesh/space changes for rebuilds.
// soft_body_create() clears it to prevent state leaking through RID reuse.
class JoltSoftBodyCapState {
	HashMap<StringName, Variant> values;

public:
	Transform3D bind_transform;
	mutable bool target_initialized = false;
	mutable int lra_skipped = 0;
	mutable PackedInt32Array volume_tetrahedra;
	mutable PackedInt32Array volume_faces;
	mutable LocalVector<int> skin_tuples;
	mutable PackedVector3Array skin_bind_points;
	mutable Array skin_inv_bind;
#ifdef TESTS_ENABLED
	mutable JoltSoftBodyCapProbe::BuildCounts test_counts;
#endif
	bool has(const StringName &p_name) const { return values.has(p_name); }

	Variant get(const StringName &p_name) const {
		const Variant *value = values.getptr(p_name);
		return value != nullptr ? *value : Variant();
	}

	void set(const StringName &p_name, const Variant &p_value) { values[p_name] = p_value; }

	void erase(const StringName &p_name) { values.erase(p_name); }

	void clear() { values.clear(); }

	PackedVector3Array v3(const StringName &p_name) const {
		const Variant *value = values.getptr(p_name);
		return value != nullptr ? PackedVector3Array(*value) : PackedVector3Array();
	}

	PackedFloat32Array f32(const StringName &p_name) const {
		const Variant *value = values.getptr(p_name);
		return value != nullptr ? PackedFloat32Array(*value) : PackedFloat32Array();
	}

	int i(const StringName &p_name, int p_default = 0) const {
		const Variant *value = values.getptr(p_name);
		return value != nullptr ? (int)*value : p_default;
	}
};

// Invalid known writes report offending values; unknown prefixes fail silently.
enum class CapStoreResult {
	OK,
	UNCHANGED,
	UNKNOWN_PREFIX,
	REJECTED,
};

// Type-check and validate a scratch candidate; commit only on success.
CapStoreResult cap_state_store(JoltSoftBodyCapState &r_state, const StringName &p_name, const Variant &p_value, String *r_err);

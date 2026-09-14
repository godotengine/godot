/**************************************************************************/
/*  soft_body_capability.cpp                                              */
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

#include "soft_body_capability.h"

#include "soft_body_cap_state.h"
#include "soft_body_cap_validation.h"
#include "soft_body_capabilities.h"

void cap_collect_active(const JoltSoftBodyCapState &p_state, LocalVector<const CapabilitySpec *> &r_active) {
	r_active.clear();

	for (const CapabilitySpec *cap : all_capabilities()) {
		if (p_state.has(StringName(cap->required_key))) {
			r_active.push_back(cap);
		}
	}
}

int cap_replace_count(const LocalVector<const CapabilitySpec *> &p_active) {
	int count = 0;

	for (const CapabilitySpec *cap : p_active) {
		if (cap->geometry == GeometryRole::REPLACE) {
			++count;
		}
	}

	return count;
}

void cap_build_settings(const JoltSoftBodyCapState &p_state, const LocalVector<const CapabilitySpec *> &p_active, JPH::SoftBodySharedSettings &r_settings, const LocalVector<int> &p_mesh_to_physics, float p_mass) {
	for (const CapabilitySpec *cap : p_active) {
		if (cap->contribute != nullptr) {
			cap->contribute(p_state, r_settings, p_mesh_to_physics, p_mass);
		}
	}

	for (const CapabilitySpec *cap : p_active) {
		if (cap->derive != nullptr) {
			cap->derive(r_settings);
		}
	}

#ifdef TESTS_ENABLED
	++p_state.test_counts.optimize;
#endif
	r_settings.Optimize();
}

void cap_apply_body(const JoltSoftBodyCapState &p_state, const LocalVector<const CapabilitySpec *> &p_active, JPH::SoftBodyCreationSettings &r_settings) {
	for (const CapabilitySpec *cap : p_active) {
		if (cap->apply_body != nullptr) {
			cap->apply_body(p_state, r_settings);
		}
	}
}

CapStoreResult cap_state_store(JoltSoftBodyCapState &r_state, const StringName &p_name, const Variant &p_value, String *r_err) {
	const CapabilitySpec *cap = find_capability(p_name);
	if (cap == nullptr) {
		return CapStoreResult::UNKNOWN_PREFIX;
	}

	const CapProperty *prop = nullptr;
	for (const CapProperty &candidate : cap->props) {
		if (p_name == StringName(candidate.name)) {
			prop = &candidate;
			break;
		}
	}

	const bool remaining = String(cap->required_key).ends_with("/config");
	if (prop == nullptr) {
		*r_err = (remaining ? String("SBREM-KEY ") : String()) + vformat("'%s' is not a valid key of the '%s' capability.", String(p_name), cap->prefix);
		return CapStoreResult::REJECTED;
	}

	if (prop->live_read) {
		*r_err = (remaining ? String("SBREM-KEY ") : String()) + vformat("'%s' is read-only and cannot be written.", String(p_name));
		return CapStoreResult::REJECTED;
	}

	if (p_value.get_type() == Variant::NIL && prop->clearable) {
		if (!r_state.has(p_name)) {
			return CapStoreResult::UNCHANGED;
		}
		r_state.erase(p_name);
		if (p_name == StringName("skin/config")) {
			r_state.erase("skin/pose");
			r_state.skin_tuples.clear();
			r_state.skin_bind_points = PackedVector3Array();
			r_state.skin_inv_bind = Array();
			r_state.target_initialized = false;
		}
		if (p_name == StringName("volume/config")) {
			r_state.volume_tetrahedra = PackedInt32Array();
			r_state.volume_faces = PackedInt32Array();
		}
		return CapStoreResult::OK;
	}

	if (p_value.get_type() != prop->type) {
		*r_err = (remaining ? String("SBREM-TYPE ") : String()) + vformat("'%s' expects a value of type %s, got %s.", String(p_name), Variant::get_type_name(prop->type), Variant::get_type_name(p_value.get_type()));
		return CapStoreResult::REJECTED;
	}

	// Validate the complete candidate on a scratch copy. Rejection preserves
	// author data; configurations never expose partial topology updates.
	JoltSoftBodyCapState scratch = r_state;
	scratch.set(p_name, remaining ? SoftBodyCapValidation::snapshot(p_value) : p_value);
	if (p_name == StringName("skin/pose")) {
		if (!scratch.has("skin/config")) {
			*r_err = "SBREM-POSE skin/pose: skin/config is required";
			return CapStoreResult::REJECTED;
		}
		const Dictionary config = scratch.get("skin/config");
		const Array pose = p_value;
		const Array binds = config["inv_bind"];
		if (pose.size() != binds.size()) {
			*r_err = vformat("SBREM-POSE skin/pose: count %d requires %d", pose.size(), binds.size());
			return CapStoreResult::REJECTED;
		}
		for (int i = 0; i < pose.size(); ++i) {
			if (!SoftBodyCapValidation::matrix(pose[i], r_err, vformat("skin/pose[%d]", i))) {
				return CapStoreResult::REJECTED;
			}
		}
	}

	if (cap->validate != nullptr && !cap->validate(scratch, r_err)) {
		return CapStoreResult::REJECTED;
	}

	if (prop->clearable && r_state.has(p_name) && SoftBodyCapValidation::equal(r_state.get(p_name), p_value)) {
		return CapStoreResult::UNCHANGED;
	}
	if (p_name == StringName("skin/config")) {
		const Dictionary config = scratch.get(p_name);
		scratch.set("skin/pose", SoftBodyCapValidation::snapshot(config["initial_pose"]));
	}
	r_state = scratch;
	return CapStoreResult::OK;
}

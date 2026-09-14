/**************************************************************************/
/*  soft_body_capabilities.h                                              */
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

#include "soft_body_capability.h"

#include "core/string/string_name.h"
#include "core/templates/span.h"

extern const CapabilitySpec CAP_ROD;
extern const CapabilitySpec CAP_SCALAR;
extern const CapabilitySpec CAP_BEND;
extern const CapabilitySpec CAP_LRA;
extern const CapabilitySpec CAP_VERTEX;
extern const CapabilitySpec CAP_VOLUME;
extern const CapabilitySpec CAP_SKIN;

#ifdef TESTS_ENABLED
// Test-only geometry provider for replace_count > 1 arbitration.
extern const CapabilitySpec CAP_TEST_REPLACE;

#include "cap_files.gen.h"
#endif

// Shared registry for lookup and enumeration. External spec linkage prevents
// static archives from discarding capability objects.
inline Span<const CapabilitySpec *const> all_capabilities() {
	static const CapabilitySpec *const kAll[] = {
		&CAP_ROD,
		&CAP_SCALAR,
		&CAP_BEND,
		&CAP_LRA,
		&CAP_VERTEX,
		&CAP_VOLUME,
		&CAP_SKIN,
#ifdef TESTS_ENABLED
		&CAP_TEST_REPLACE,
#endif
	};
	return kAll;
}

// Resolve a key's prefix, or return nullptr if unclaimed.
inline const CapabilitySpec *find_capability(const StringName &p_name) {
	const String name = String(p_name);
	for (const CapabilitySpec *cap : all_capabilities()) {
		if (name.begins_with(cap->prefix)) {
			return cap;
		}
	}
	return nullptr;
}

// Activation checks required_key presence; schema validation runs before storage.
void cap_collect_active(const JoltSoftBodyCapState &p_state, LocalVector<const CapabilitySpec *> &r_active);

int cap_replace_count(const LocalVector<const CapabilitySpec *> &p_active);

// contribute -> derive -> Optimize(), exactly once. Only the framework may
// call Optimize(): repeated global reordering corrupts other capabilities.
void cap_build_settings(const JoltSoftBodyCapState &p_state, const LocalVector<const CapabilitySpec *> &p_active, JPH::SoftBodySharedSettings &r_settings, const LocalVector<int> &p_mesh_to_physics, float p_mass);

// Apply scalar/body settings in _add_to_space(); reset mUpdatePosition each rebuild.
void cap_apply_body(const JoltSoftBodyCapState &p_state, const LocalVector<const CapabilitySpec *> &p_active, JPH::SoftBodyCreationSettings &r_settings);

// Re-pin both settings and motion-property vertices through their mInvMass field.
template <typename TJoltVertex>
void cap_pin_vertices(const JoltSoftBodyCapState &p_state, JPH::Array<TJoltVertex> &r_vertices) {
	LocalVector<const CapabilitySpec *> active;
	cap_collect_active(p_state, active);

	LocalVector<int> indices;
	for (const CapabilitySpec *cap : active) {
		if (cap->pinned_vertices == nullptr) {
			continue;
		}
		indices.clear();
		cap->pinned_vertices(p_state, indices);
		for (int index : indices) {
			if (index >= 0 && index < (int)r_vertices.size()) {
				r_vertices[index].mInvMass = 0.0f;
			}
		}
	}
}

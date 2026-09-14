/**************************************************************************/
/*  cap_rod.cpp                                                           */
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

#include "soft_body_cap_state.h"
#include "soft_body_cap_validation.h"
#include "soft_body_capabilities.h"

#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

namespace {

const char *KEY_CONFIG = "rod/config";
const char *KEY_JOINTS = "joints";
const char *KEY_COMPLIANCE = "compliance";
const char *KEY_BEND = "bend";
const char *KEY_FIXED = "fixed";
const char *KEY_STATE = "rod/state";

// Segment/bone indices require an identity `Optimize()` remap.
// Measurements cover up to two orders of magnitude below this enforced limit.
constexpr int MAX_JOINTS = 1024;

// Guard Jolt's division by segment length; its zero-length assert is absent
// in release builds. Squaring keeps the check to one multiply.
constexpr double MIN_SEGMENT_LENGTH_SQUARED = 1e-12;

bool validate_finite_array(const PackedFloat32Array &p_values, const char *p_key, String *r_err) {
	for (int i = 0; i < p_values.size(); ++i) {
		const float value = p_values[i];

		if (!Math::is_finite(value)) {
			*r_err = vformat("SBREM-VALUE rod/config.%s entry %d is not a finite number (%f).", p_key, i, value);
			return false;
		}

		// Negative compliance invalidates the solver denominator.
		if (value < 0.0f) {
			*r_err = vformat("SBREM-VALUE rod/config.%s entry %d must not be negative, got %f.", p_key, i, value);
			return false;
		}
	}

	return true;
}

bool rod_validate(const JoltSoftBodyCapState &p_state, String *r_err) {
	if (!SoftBodyCapValidation::schema(p_state, KEY_CONFIG, r_err)) {
		return false;
	}
	const Dictionary config = p_state.get(KEY_CONFIG);
	const PackedVector3Array joints = config[KEY_JOINTS];
	const int joint_count = joints.size();
	if (joint_count < 2 || joint_count > MAX_JOINTS) {
		return SoftBodyCapValidation::fail(r_err, "SBREM-SIZE", KEY_CONFIG, vformat("joint count %d requires 2..%d", joint_count, MAX_JOINTS));
	}
	for (int i = 0; i + 1 < joint_count; ++i) {
		const double length_squared = (double)joints[i + 1].distance_squared_to(joints[i]);
		if (length_squared < MIN_SEGMENT_LENGTH_SQUARED) {
			return SoftBodyCapValidation::fail(r_err, "SBREM-VALUE", KEY_CONFIG, vformat("Segment %d squared length %s is below %s", i, length_squared, MIN_SEGMENT_LENGTH_SQUARED));
		}
	}
	const int rods = joint_count - 1;

	if (config.has(KEY_COMPLIANCE)) {
		const PackedFloat32Array compliance = config[KEY_COMPLIANCE];

		if (!validate_finite_array(compliance, KEY_COMPLIANCE, r_err)) {
			return false;
		}

		if (compliance.size() != rods) {
			*r_err = vformat("SBREM-SIZE rod/config.%s has %d entries but a rod with %d joints has %d segments.", KEY_COMPLIANCE, compliance.size(), joint_count, rods);
			return false;
		}
	}

	if (config.has(KEY_BEND)) {
		const PackedFloat32Array bend = config[KEY_BEND];

		if (!validate_finite_array(bend, KEY_BEND, r_err)) {
			return false;
		}

		// One bend-twist constraint per adjacent segment pair.
		const int expected_bend = rods > 0 ? rods - 1 : 0;

		if (bend.size() != expected_bend) {
			*r_err = vformat("SBREM-SIZE rod/config.%s has %d entries but a rod with %d segments has %d bend-twist constraints.", KEY_BEND, bend.size(), rods, expected_bend);
			return false;
		}
	}

	if (config.has(KEY_FIXED)) {
		const int64_t fixed = config[KEY_FIXED];

		if (fixed < 0 || fixed > joint_count) {
			return SoftBodyCapValidation::fail(r_err, "SBREM-VALUE", "rod/config.fixed", vformat("must be in the range [0, %d], got %d", joint_count, fixed));
		}
	}

	return true;
}

// Shared pins for initial build and mass updates.
void rod_pinned_vertices(const JoltSoftBodyCapState &p_state, LocalVector<int> &r_indices) {
	const Dictionary config = p_state.get(KEY_CONFIG);
	const int fixed = config.get(KEY_FIXED, 0);

	for (int i = 0; i < fixed; ++i) {
		r_indices.push_back(i);
	}
}

void rod_contribute(const JoltSoftBodyCapState &p_state, JPH::SoftBodySharedSettings &r_settings, const LocalVector<int> &p_mesh_to_physics, float p_mass) {
	const Dictionary config = p_state.get(KEY_CONFIG);
	const PackedVector3Array joints = config[KEY_JOINTS];
	const int joint_count = joints.size();
	ERR_FAIL_COND(joint_count < 2);

	const int rods = joint_count - 1;
	const PackedFloat32Array compliance = config.get(KEY_COMPLIANCE, PackedFloat32Array());
	const PackedFloat32Array bend = config.get(KEY_BEND, PackedFloat32Array());

	JPH::Array<JPH::SoftBodySharedSettings::Vertex> &vertices = r_settings.mVertices;
	const int vertex_base = (int)vertices.size();

	for (int i = 0; i < joint_count; ++i) {
		const Vector3 joint = joints[i];
		vertices.emplace_back(JPH::Float3((float)joint.x, (float)joint.y, (float)joint.z));
	}

	// Match `_update_mass()` before deriving rod inverse masses.
	const float inverse_vertex_mass = (float)vertices.size() / p_mass;
	for (int i = 0; i < joint_count; ++i) {
		vertices[vertex_base + i].mInvMass = inverse_vertex_mass;
	}

	JPH::Array<JPH::SoftBodySharedSettings::RodStretchShear> &stretch_shear = r_settings.mRodStretchShearConstraints;
	const int rod_base = (int)stretch_shear.size();

	// Stretch-shear indexes particles.
	for (int i = 0; i < rods; ++i) {
		const float value = i < compliance.size() ? compliance[i] : 0.0f;
		stretch_shear.emplace_back((JPH::uint32)(vertex_base + i), (JPH::uint32)(vertex_base + i + 1), value);
	}

	// Bend-twist alone indexes segments, not particles; wrong indices silently
	// change the stiffness distribution.
	JPH::Array<JPH::SoftBodySharedSettings::RodBendTwist> &bend_twist = r_settings.mRodBendTwistConstraints;
	for (int i = 0; i + 1 < rods; ++i) {
		const float value = i < bend.size() ? bend[i] : 0.0f;
		bend_twist.emplace_back((JPH::uint32)(rod_base + i), (JPH::uint32)(rod_base + i + 1), value);
	}

	LocalVector<int> pinned;
	rod_pinned_vertices(p_state, pinned);
	for (int index : pinned) {
		if (index >= 0 && index < joint_count) {
			vertices[vertex_base + index].mInvMass = 0.0f;
		}
	}
}

void rod_derive(JPH::SoftBodySharedSettings &r_settings) {
	r_settings.CalculateRodProperties();
}

void rod_apply_body(const JoltSoftBodyCapState &p_state, JPH::SoftBodyCreationSettings &r_settings) {
	// Keep the origin fixed so rod state stays in its authored coordinate frame.
	r_settings.mUpdatePosition = false;
}

Variant rod_get(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name) {
	if (p_name != StringName(KEY_STATE)) {
		return p_state.get(p_name);
	}

	ERR_FAIL_NULL_V(p_body, Variant());

	const JPH::SoftBodyMotionProperties &motion_properties = static_cast<const JPH::SoftBodyMotionProperties &>(*p_body->GetMotionPropertiesUnchecked());
	const int rods = (int)motion_properties.GetSettings()->mRodStretchShearConstraints.size();

	// Flattened, not boxed: one allocation instead of N `Variant`s.
	PackedFloat32Array out;
	out.resize(rods * 4);
	float *write = out.ptrw();

	for (int i = 0; i < rods; ++i) {
		const JPH::Quat rotation = motion_properties.GetRodRotation((JPH::uint)i);
		write[i * 4 + 0] = rotation.GetX();
		write[i * 4 + 1] = rotation.GetY();
		write[i * 4 + 2] = rotation.GetZ();
		write[i * 4 + 3] = rotation.GetW();
	}

	return out;
}

const CapProperty ROD_PROPS[] = {
	{ KEY_CONFIG, Variant::DICTIONARY, false, true, true },
	{ KEY_STATE, Variant::PACKED_FLOAT32_ARRAY, true },
};

} // namespace

const CapabilitySpec CAP_ROD = {
	/*prefix*/ "rod/",
	/*required_key*/ KEY_CONFIG,
	/*props*/ Span<CapProperty>(ROD_PROPS, 2),
	/*geometry*/ GeometryRole::REPLACE,
	/*validate*/ rod_validate,
	/*contribute*/ rod_contribute,
	/*pinned_vertices*/ rod_pinned_vertices,
	/*apply_body*/ rod_apply_body,
	/*derive*/ rod_derive,
	/*get*/ rod_get,
};

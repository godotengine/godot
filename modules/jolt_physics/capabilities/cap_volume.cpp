/**************************************************************************/
/*  cap_volume.cpp                                                        */
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

#include "../misc/jolt_type_conversions.h"
#include "soft_body_cap_state.h"
#include "soft_body_cap_validation.h"
#include "soft_body_capability.h"

#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

namespace {
const CapProperty properties[] = {
	{ "volume/config", Variant::DICTIONARY, false, true, true, true },
	{ "volume/count", Variant::INT, true },
	{ "volume/current", Variant::FLOAT, true },
	{ "volume/positions", Variant::PACKED_VECTOR3_ARRAY, true },
	{ "volume/faces", Variant::PACKED_INT32_ARRAY, true },
};

bool validate(const JoltSoftBodyCapState &p_state, String *r_error) {
	return SoftBodyCapValidation::schema(p_state, "volume/config", r_error) &&
			SoftBodyCapValidation::volume_topology(p_state.get("volume/config"), p_state.volume_tetrahedra, p_state.volume_faces, r_error);
}

void pins(const JoltSoftBodyCapState &p_state, LocalVector<int> &r_indices) {
	const Dictionary config = p_state.get("volume/config");
	const PackedInt32Array fixed = config.get("fixed", PackedInt32Array());
	for (int index : fixed) {
		r_indices.push_back(index);
	}
}

void contribute(const JoltSoftBodyCapState &p_state, JPH::SoftBodySharedSettings &r_settings, const LocalVector<int> &, float p_mass) {
	const Dictionary config = p_state.get("volume/config");
	const PackedVector3Array vertices = config["vertices"];
	const float inverse_mass = float(double(vertices.size()) / p_mass);
	for (const Vector3 &vertex : vertices) {
		r_settings.mVertices.emplace_back(JPH::Float3(vertex.x, vertex.y, vertex.z), JPH::Float3(0, 0, 0), inverse_mass);
	}
	LocalVector<int> fixed;
	pins(p_state, fixed);
	for (int index : fixed) {
		r_settings.mVertices[index].mInvMass = 0;
	}
	for (int i = 0; i < p_state.volume_faces.size(); i += 3) {
		r_settings.mFaces.emplace_back(p_state.volume_faces[i], p_state.volume_faces[i + 2], p_state.volume_faces[i + 1]);
	}
	const float compliance = config.get("compliance", 0.0);
	for (int i = 0; i < p_state.volume_tetrahedra.size(); i += 4) {
		r_settings.mVolumeConstraints.emplace_back(p_state.volume_tetrahedra[i], p_state.volume_tetrahedra[i + 1], p_state.volume_tetrahedra[i + 2], p_state.volume_tetrahedra[i + 3], compliance);
	}
}

void derive(JPH::SoftBodySharedSettings &r_settings) {
	r_settings.CalculateVolumeConstraintVolumes();
}

Variant get(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name) {
	if (p_name == StringName("volume/config")) {
		return p_state.get(p_name);
	}
	const auto &motion = static_cast<const JPH::SoftBodyMotionProperties &>(*p_body->GetMotionPropertiesUnchecked());
	if (p_name == StringName("volume/count")) {
		return int64_t(motion.GetSettings()->mVolumeConstraints.size());
	}
	if (p_name == StringName("volume/faces")) {
		return p_state.has("volume/config") ? p_state.volume_faces : PackedInt32Array();
	}
	if (p_name == StringName("volume/positions")) {
		PackedVector3Array positions;
		if (p_state.has("volume/config")) {
			const auto frame = p_body->GetCenterOfMassTransform();
			for (const auto &vertex : motion.GetVertices()) {
				positions.push_back(to_godot(frame * vertex.mPosition));
			}
		}
		return positions;
	}
	double volume = 0;
	for (const auto &tetra : motion.GetSettings()->mVolumeConstraints) {
		const JPH::Vec3 a = motion.GetVertex(tetra.mVertex[0]).mPosition;
		const JPH::Vec3 b = motion.GetVertex(tetra.mVertex[1]).mPosition;
		const JPH::Vec3 c = motion.GetVertex(tetra.mVertex[2]).mPosition;
		const JPH::Vec3 d = motion.GetVertex(tetra.mVertex[3]).mPosition;
		const double x[3] = { double(b.GetX()) - a.GetX(), double(b.GetY()) - a.GetY(), double(b.GetZ()) - a.GetZ() };
		const double y[3] = { double(c.GetX()) - a.GetX(), double(c.GetY()) - a.GetY(), double(c.GetZ()) - a.GetZ() };
		const double z[3] = { double(d.GetX()) - a.GetX(), double(d.GetY()) - a.GetY(), double(d.GetZ()) - a.GetZ() };
		volume += Math::abs((x[1] * y[2] - x[2] * y[1]) * z[0] + (x[2] * y[0] - x[0] * y[2]) * z[1] + (x[0] * y[1] - x[1] * y[0]) * z[2]) / 6.0;
	}
	return volume;
}
} // namespace

extern const CapabilitySpec CAP_VOLUME = {
	"volume/",
	"volume/config",
	properties,
	GeometryRole::REPLACE,
	validate,
	contribute,
	pins,
	nullptr,
	derive,
	get,
};

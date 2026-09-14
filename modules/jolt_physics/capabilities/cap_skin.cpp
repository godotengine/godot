/**************************************************************************/
/*  cap_skin.cpp                                                          */
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

#include <cfloat>

namespace {
const CapProperty properties[] = {
	{ "skin/config", Variant::DICTIONARY, false, true, true, true },
	{ "skin/pose", Variant::ARRAY, false, false, false, false },
	{ "skin/count", Variant::INT, true },
	{ "skin/frame", Variant::TRANSFORM3D, true },
};

bool validate(const JoltSoftBodyCapState &p_state, String *r_error) {
	return SoftBodyCapValidation::schema(p_state, "skin/config", r_error);
}

void pins(const JoltSoftBodyCapState &p_state, const LocalVector<int> &p_map, LocalVector<int> &r_indices) {
	const Dictionary config = p_state.get("skin/config");
	if (float(config.get("max_distance", 0.05)) != 0) {
		return;
	}
	const PackedInt32Array vertices = config["vertices"];
	HashSet<int> unique;
	for (int source : vertices) {
		// Defer invalid-map diagnostics to attributes; do not index it for hard pins.
		if (source >= 0 && source < int(p_map.size()) && p_map[source] >= 0 && !unique.has(p_map[source])) {
			unique.insert(p_map[source]);
			r_indices.push_back(p_map[source]);
		}
	}
}

bool attributes(const JoltSoftBodyCapState &p_state, const LocalVector<int> &p_map, LocalVector<JPH::SoftBodySharedSettings::VertexAttributes> &r_attributes, JPH::SoftBodySharedSettings::EBendType &, String *r_error) {
	using namespace SoftBodyCapValidation;
	const Dictionary config = p_state.get("skin/config");
	const PackedInt32Array selected = config["vertices"];
	const PackedInt32Array indices = config["joint_indices"];
	const PackedFloat32Array weights = config["joint_weights"];
	LocalVector<int> tuples;
	tuples.resize(p_map.size());
	for (int &tuple : tuples) {
		tuple = -1;
	}
	for (int i = 0; i < selected.size(); ++i) {
		const int source = selected[i];
		if (source < 0 || source >= int(p_map.size()) || p_map[source] < 0) {
			return fail(r_error, "SBREM-INDEX", vformat("skin/config.vertices[%d]", i), vformat("source %d is not a referenced mesh vertex", source));
		}
		tuples[source] = i;
	}
	p_state.skin_tuples.resize(r_attributes.size());
	LocalVector<int> owners;
	owners.resize(r_attributes.size());
	for (int &owner : owners) {
		owner = -1;
	}
	for (uint32_t source = 0; source < p_map.size(); ++source) {
		const int physics = p_map[source];
		if (physics < 0) {
			continue;
		}
		const int tuple = tuples[source];
		if (owners[physics] >= 0) {
			const int other = p_state.skin_tuples[physics];
			bool same = (other < 0) == (tuple < 0);
			if (same && tuple >= 0) {
				for (int slot = 0; slot < 4; ++slot) {
					same &= indices[4 * tuple + slot] == indices[4 * other + slot] && weights[4 * tuple + slot] == weights[4 * other + slot];
				}
			}
			if (!same) {
				return fail(r_error, "SBREM-ALIAS", "skin/config.vertices", vformat("source vertices %d and %d have different skin tuples or selection", owners[physics], source));
			}
		} else {
			owners[physics] = source;
			p_state.skin_tuples[physics] = tuple;
		}
	}
	return true;
}

void contribute(const JoltSoftBodyCapState &p_state, JPH::SoftBodySharedSettings &r_settings, const LocalVector<int> &, float) {
	const Dictionary config = p_state.get("skin/config");
	const PackedInt32Array indices = config["joint_indices"];
	const PackedFloat32Array weights = config["joint_weights"];
	const Array inv_bind = config["inv_bind"];
	p_state.skin_bind_points = PackedVector3Array();
	p_state.skin_inv_bind = Array();
	for (const auto &vertex : r_settings.mVertices) {
		p_state.skin_bind_points.push_back(to_godot(JPH::Vec3(vertex.mPosition)));
	}
	for (int joint = 0; joint < inv_bind.size(); ++joint) {
		const Transform3D converted = Transform3D(inv_bind[joint]) * p_state.bind_transform;
		p_state.skin_inv_bind.push_back(converted);
		r_settings.mInvBindMatrices.emplace_back(joint, to_jolt(converted));
	}
	for (uint32_t vertex = 0; vertex < p_state.skin_tuples.size(); ++vertex) {
		const int tuple = p_state.skin_tuples[vertex];
		if (tuple < 0) {
			continue;
		}
		JPH::SoftBodySharedSettings::Skinned constraint(vertex, config.get("max_distance", 0.05), config.get("back_stop_distance", double(FLT_MAX)), config.get("back_stop_radius", 40.0));
		for (int slot = 0; slot < 4; ++slot) {
			constraint.mWeights[slot] = JPH::SoftBodySharedSettings::SkinWeight(indices[tuple * 4 + slot], weights[tuple * 4 + slot]);
		}
		r_settings.mSkinnedConstraints.push_back(constraint);
	}
}

bool validate_build(const JoltSoftBodyCapState &p_state, const JPH::SoftBodySharedSettings &p_settings, const LocalVector<int> &, String *r_error) {
	using namespace SoftBodyCapValidation;
	const Dictionary config = p_state.get("skin/config");
	for (int joint = 0; joint < p_state.skin_inv_bind.size(); ++joint) {
		if (!Transform3D(p_state.skin_inv_bind[joint]).is_finite()) {
			return fail(r_error, "SBREM-VALUE", vformat("skin/config.inv_bind[%d]", joint), "binding-frame product is nonfinite");
		}
	}
	if (!skin_targets(p_state, config["initial_pose"], p_state.bind_transform, "initial", r_error) || !skin_targets(p_state, p_state.get("skin/pose"), p_state.bind_transform, "build", r_error)) {
		return false;
	}
	LocalVector<uint32_t> counts;
	counts.resize(p_settings.mVertices.size());
	for (uint32_t &count : counts) {
		count = 0;
	}
	for (const auto &face : p_settings.mFaces) {
		if (p_state.skin_tuples[face.mVertex[0]] >= 0 && p_state.skin_tuples[face.mVertex[1]] >= 0 && p_state.skin_tuples[face.mVertex[2]] >= 0) {
			for (uint32_t vertex : face.mVertex) {
				++counts[vertex];
			}
		}
	}
	uint64_t start = 0;
	for (const auto &constraint : p_settings.mSkinnedConstraints) {
		const uint32_t count = counts[constraint.mVertex];
		uint32_t packed;
		if (!normal_info(start, count, packed, r_error)) {
			return false;
		}
		if (count == 0 && constraint.mBackStopDistance < constraint.mMaxDistance) {
			return fail(r_error, "SBREM-SKIN-NORMAL", vformat("skin/config.vertices[%d]", constraint.mVertex), "back-stop requires a fully skinned adjacent face");
		}
		start += count;
	}
	return true;
}

void derive(JPH::SoftBodySharedSettings &r_settings) {
	r_settings.CalculateSkinnedConstraintNormals();
}

bool pre_step(const JoltSoftBodyCapState &p_state, JPH::Body &r_body, float, JPH::TempAllocator &r_allocator, String *r_error) {
	using namespace SoftBodyCapValidation;
	const auto com = r_body.GetCenterOfMassTransform();
	const Array pose = p_state.get("skin/pose");
	if (!skin_targets(p_state, pose, to_godot(com), "pre_step", r_error)) {
		return false;
	}
	LocalVector<JPH::Mat44> local;
	const auto inverse_com = com.InversedRotationTranslation();
	for (int joint = 0; joint < pose.size(); ++joint) {
		local.push_back(inverse_com * to_jolt(Transform3D(pose[joint])));
		if (!to_godot(local[joint]).is_finite()) {
			return fail(r_error, "SBREM-POSE", vformat("skin/pose.pre_step.joint[%d]", joint), "Jolt COM-local matrix is nonfinite");
		}
	}
	auto &motion = static_cast<JPH::SoftBodyMotionProperties &>(*r_body.GetMotionPropertiesUnchecked());
	// Validate float32 products in scratch before mutating targets or live vertices.
	LocalVector<JPH::Mat44> transforms;
	for (const auto &bind : motion.GetSettings()->mInvBindMatrices) {
		const auto transform = local[bind.mJointIndex] * bind.mInvBind;
		if (!to_godot(transform).is_finite()) {
			return fail(r_error, "SBREM-POSE", vformat("skin/pose.pre_step.joint[%d]", bind.mJointIndex), "Jolt skin matrix is nonfinite");
		}
		transforms.push_back(transform);
	}
	for (const auto &skin : motion.GetSettings()->mSkinnedConstraints) {
		JPH::Vec3 target = JPH::Vec3::sZero();
		const JPH::Vec3 point(motion.GetSettings()->mVertices[skin.mVertex].mPosition);
		for (const auto &weight : skin.mWeights) {
			if (weight.mWeight == 0) {
				break;
			}
			target += weight.mWeight * (transforms[weight.mInvBindIndex] * point);
			if (!to_godot(target).is_finite()) {
				return fail(r_error, "SBREM-POSE", vformat("skin/pose.pre_step.vertex[%d]", skin.mVertex), "Jolt weighted target is nonfinite");
			}
		}
	}
	const bool fresh = !p_state.target_initialized;
#ifdef TESTS_ENABLED
	if (fresh) {
		++p_state.test_counts.skin_init;
	} else {
		++p_state.test_counts.skin_recurring;
	}
#endif
	motion.SkinVertices(com, local.ptr(), local.size(), fresh, r_allocator);
	p_state.target_initialized = true;
	return true;
}

Variant get(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name) {
	if (p_name == StringName("skin/config") || p_name == StringName("skin/pose")) {
		return p_state.get(p_name);
	}
	if (p_name == StringName("skin/frame")) {
		return to_godot(p_body->GetCenterOfMassTransform());
	}
	const auto &motion = static_cast<const JPH::SoftBodyMotionProperties &>(*p_body->GetMotionPropertiesUnchecked());
	return int64_t(motion.GetSettings()->mSkinnedConstraints.size());
}
} // namespace

extern const CapabilitySpec CAP_SKIN = {
	"skin/",
	"skin/config",
	properties,
	GeometryRole::NONE,
	validate,
	contribute,
	nullptr,
	nullptr,
	derive,
	get,
	attributes,
	pins,
	validate_build,
	pre_step,
};

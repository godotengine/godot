/**************************************************************************/
/*  cap_lra.cpp                                                           */
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
#include "soft_body_capability.h"

#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

namespace {
const CapProperty properties[] = {
	{ "lra/config", Variant::DICTIONARY, false, true, true, true },
	{ "lra/count", Variant::INT, true },
};

bool validate(const JoltSoftBodyCapState &p_state, String *r_error) {
	return SoftBodyCapValidation::schema(p_state, "lra/config", r_error);
}

bool attributes(const JoltSoftBodyCapState &p_state, const LocalVector<int> &p_map, LocalVector<JPH::SoftBodySharedSettings::VertexAttributes> &r_attributes, JPH::SoftBodySharedSettings::EBendType &, String *r_error) {
	const Dictionary config = p_state.get("lra/config");
	LocalVector<float> types;
	LocalVector<float> multipliers;
	if (!SoftBodyCapValidation::column(config, "type", 0, p_map.size(), p_map, types, r_error) ||
			!SoftBodyCapValidation::column(config, "max_distance_multiplier", 1, p_map.size(), p_map, multipliers, r_error)) {
		return false;
	}
	for (uint32_t i = 0; i < r_attributes.size(); ++i) {
		switch (int(types[i])) {
			case 0:
				r_attributes[i].mLRAType = JPH::SoftBodySharedSettings::ELRAType::None;
				break;
			case 1:
				r_attributes[i].mLRAType = JPH::SoftBodySharedSettings::ELRAType::EuclideanDistance;
				break;
			case 2:
				r_attributes[i].mLRAType = JPH::SoftBodySharedSettings::ELRAType::GeodesicDistance;
				break;
		}
		r_attributes[i].mLRAMaxDistanceMultiplier = multipliers[i];
	}
	return true;
}

bool validate_build(const JoltSoftBodyCapState &p_state, const JPH::SoftBodySharedSettings &p_settings, const LocalVector<int> &p_map, String *r_error) {
	LocalVector<float> types;
	if (!SoftBodyCapValidation::column(p_state.get("lra/config"), "type", 0, p_map.size(), p_map, types, r_error)) {
		return false;
	}
	int enabled = 0;
	for (uint32_t i = 0; i < types.size(); ++i) {
		enabled += types[i] != 0 && p_settings.mVertices[i].mInvMass > 0;
	}
	for (const auto &constraint : p_settings.mLRAConstraints) {
		if (!Math::is_finite(constraint.mMaxDistance)) {
			return SoftBodyCapValidation::fail(r_error, "SBREM-VALUE", "lra/config.max_distance_multiplier", vformat("derived distance for vertex %d is %s", constraint.mVertex[1], constraint.mMaxDistance));
		}
	}
	p_state.lra_skipped = enabled - p_settings.mLRAConstraints.size();
	return true;
}

void apply(const JoltSoftBodyCapState &p_state, JPH::SoftBodyCreationSettings &) {
	// Warn only on successful builds, never during preflight or steps.
	// All-pinned builds stay silent.
	if (p_state.lra_skipped > 0) {
		WARN_PRINT(vformat("SBREM-LRA-NO-ANCHOR lra/config.type: skipped %d enabled dynamic vertices without a reachable pin", p_state.lra_skipped));
	}
}

Variant get(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name) {
	if (p_name == StringName("lra/config")) {
		return p_state.get(p_name);
	}
	const auto &motion = static_cast<const JPH::SoftBodyMotionProperties &>(*p_body->GetMotionPropertiesUnchecked());
	return int64_t(motion.GetSettings()->mLRAConstraints.size());
}
} // namespace

extern const CapabilitySpec CAP_LRA = {
	"lra/",
	"lra/config",
	properties,
	GeometryRole::NONE,
	validate,
	nullptr,
	nullptr,
	apply,
	nullptr,
	get,
	attributes,
	nullptr,
	validate_build,
};

/**************************************************************************/
/*  cap_bend.cpp                                                          */
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
	{ "bend/config", Variant::DICTIONARY, false, true, true, true },
	{ "bend/count", Variant::INT, true },
};

bool validate(const JoltSoftBodyCapState &p_state, String *r_error) {
	return SoftBodyCapValidation::schema(p_state, "bend/config", r_error);
}

bool attributes(const JoltSoftBodyCapState &p_state, const LocalVector<int> &p_map, LocalVector<JPH::SoftBodySharedSettings::VertexAttributes> &r_attributes, JPH::SoftBodySharedSettings::EBendType &r_type, String *r_error) {
	const Dictionary config = p_state.get("bend/config");
	LocalVector<float> compliance;
	if (!SoftBodyCapValidation::column(config, "compliance", 0, p_map.size(), p_map, compliance, r_error)) {
		return false;
	}
	for (uint32_t i = 0; i < r_attributes.size(); ++i) {
		r_attributes[i].mBendCompliance = compliance[i];
	}
	// Public enum intentionally differs from Jolt's enum (which includes None).
	r_type = int(config.get("type", 1)) == 0 ? JPH::SoftBodySharedSettings::EBendType::Distance : JPH::SoftBodySharedSettings::EBendType::Dihedral;
	return true;
}

Variant get(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name) {
	if (p_name == StringName("bend/config")) {
		return p_state.get(p_name);
	}
	const auto &motion = static_cast<const JPH::SoftBodyMotionProperties &>(*p_body->GetMotionPropertiesUnchecked());
	return int64_t(motion.GetSettings()->mDihedralBendConstraints.size());
}
} // namespace

extern const CapabilitySpec CAP_BEND = {
	"bend/",
	"bend/config",
	properties,
	GeometryRole::NONE,
	validate,
	nullptr,
	nullptr,
	nullptr,
	nullptr,
	get,
	attributes,
};

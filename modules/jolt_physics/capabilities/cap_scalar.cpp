/**************************************************************************/
/*  cap_scalar.cpp                                                        */
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
CapProperty properties[] = {
	{ "scalar/config", Variant::DICTIONARY, false, true, true, true },
	{ "scalar/current", Variant::DICTIONARY, true },
};

bool validate(const JoltSoftBodyCapState &p_state, String *r_error) {
	return SoftBodyCapValidation::schema(p_state, "scalar/config", r_error);
}

void apply(const JoltSoftBodyCapState &p_state, JPH::SoftBodyCreationSettings &r_settings) {
	const Dictionary config = p_state.get("scalar/config");
	if (config.has("friction")) {
		r_settings.mFriction = config["friction"];
	}
	if (config.has("restitution")) {
		r_settings.mRestitution = config["restitution"];
	}
	if (config.has("gravity_factor")) {
		r_settings.mGravityFactor = config["gravity_factor"];
	}
	if (config.has("faces_double_sided")) {
		r_settings.mFacesDoubleSided = config["faces_double_sided"];
	}
	if (config.has("vertex_radius")) {
		r_settings.mVertexRadius = config["vertex_radius"];
	}
}

Variant get(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name) {
	if (p_name == StringName("scalar/config")) {
		return p_state.get(p_name);
	}
	const auto &motion = static_cast<const JPH::SoftBodyMotionProperties &>(*p_body->GetMotionPropertiesUnchecked());
	Dictionary current;
	current["friction"] = p_body->GetFriction();
	current["restitution"] = p_body->GetRestitution();
	current["gravity_factor"] = motion.GetGravityFactor();
	current["faces_double_sided"] = motion.GetFacesDoubleSided();
	current["vertex_radius"] = motion.GetVertexRadius();
	return current;
}
} // namespace

extern const CapabilitySpec CAP_SCALAR = {
	"scalar/",
	"scalar/config",
	properties,
	GeometryRole::NONE,
	validate,
	nullptr,
	nullptr,
	apply,
	nullptr,
	get,
};

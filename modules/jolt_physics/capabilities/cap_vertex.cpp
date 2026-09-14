/**************************************************************************/
/*  cap_vertex.cpp                                                        */
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

namespace {
const CapProperty properties[] = {
	{ "vertex/config", Variant::DICTIONARY, false, true, true, true },
};

bool validate(const JoltSoftBodyCapState &p_state, String *r_error) {
	return SoftBodyCapValidation::schema(p_state, "vertex/config", r_error);
}

bool attributes(const JoltSoftBodyCapState &p_state, const LocalVector<int> &p_map, LocalVector<JPH::SoftBodySharedSettings::VertexAttributes> &r_attributes, JPH::SoftBodySharedSettings::EBendType &, String *r_error) {
	const Dictionary config = p_state.get("vertex/config");
	LocalVector<float> edge;
	LocalVector<float> shear;
	// Compare authored scales before multiplying: a zero base must not hide
	// conflicting UV-seam aliases that would disagree at a later stiffness.
	if (!SoftBodyCapValidation::column(config, "edge_scale", 1, p_map.size(), p_map, edge, r_error) ||
			!SoftBodyCapValidation::column(config, "shear_scale", 1, p_map.size(), p_map, shear, r_error)) {
		return false;
	}
	for (uint32_t i = 0; i < r_attributes.size(); ++i) {
		String error;
		if (!SoftBodyCapValidation::finite_float(double(r_attributes[i].mCompliance) * edge[i], r_attributes[i].mCompliance, &error, vformat("vertex/config.edge_scale[%d]", i)) ||
				!SoftBodyCapValidation::finite_float(double(r_attributes[i].mShearCompliance) * shear[i], r_attributes[i].mShearCompliance, &error, vformat("vertex/config.shear_scale[%d]", i))) {
			return SoftBodyCapValidation::fail(r_error, "SBREM-CONTEXT", "typed.compliance", error);
		}
	}
	return true;
}

Variant get(const JoltSoftBodyCapState &p_state, const JPH::Body *, const StringName &p_name) {
	return p_state.get(p_name);
}
} // namespace

extern const CapabilitySpec CAP_VERTEX = {
	"vertex/",
	"vertex/config",
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

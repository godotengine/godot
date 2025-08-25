/**************************************************************************/
/*  jolt_tapered_capsule_shape_3d.cpp                                     */
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

#include "jolt_tapered_capsule_shape_3d.h"

#include "../misc/jolt_type_conversions.h"

#include "Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h"

JPH::ShapeRefC JoltTaperedCapsuleShape3D::_build() const {
	ERR_FAIL_COND_V_MSG(radius1 <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its radius1 must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(radius2 <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its radius2 must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(height <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its height must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(height < radius1 + radius2, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its height must be at least the sum of its radii. This shape belongs to %s.", to_string(), _owners_to_string()));

	const float half_height = height / 2.0f;

	const JPH::TaperedCapsuleShapeSettings shape_settings(half_height, radius1, radius2);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
	ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

	return shape_result.Get();
}

Variant JoltTaperedCapsuleShape3D::get_data() const {
	Dictionary data;
	data["radius1"] = radius1;
	data["radius2"] = radius2;
	data["height"] = height;
	return data;
}

void JoltTaperedCapsuleShape3D::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_radius1 = data.get("radius1", Variant());
	ERR_FAIL_COND(maybe_radius1.get_type() != Variant::FLOAT);

	const Variant maybe_radius2 = data.get("radius2", Variant());
	ERR_FAIL_COND(maybe_radius2.get_type() != Variant::FLOAT);

	const Variant maybe_height = data.get("height", Variant());
	ERR_FAIL_COND(maybe_height.get_type() != Variant::FLOAT);

	const float new_radius1 = maybe_radius1;
	const float new_radius2 = maybe_radius2;
	const float new_height = maybe_height;

	if (unlikely(new_radius1 == radius1 && new_radius2 == radius2 && new_height == height)) {
		return;
	}

	radius1 = new_radius1;
	radius2 = new_radius2;
	height = new_height;

	destroy();
}

AABB JoltTaperedCapsuleShape3D::get_aabb() const {
	const float max_radius = MAX(radius1, radius2);
	const Vector3 half_extents(max_radius, height / 2.0f + max_radius, max_radius);
	return AABB(-half_extents, half_extents * 2.0f);
}

String JoltTaperedCapsuleShape3D::to_string() const {
	return vformat("{radius1=%f radius2=%f height=%f}", radius1, radius2, height);
}

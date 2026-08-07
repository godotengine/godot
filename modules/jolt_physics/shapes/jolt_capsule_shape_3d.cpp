/**************************************************************************/
/*  jolt_capsule_shape_3d.cpp                                             */
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

#include "jolt_capsule_shape_3d.h"

#include "../misc/jolt_type_conversions.h"

#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h>

JPH::ShapeRefC JoltCapsuleShape3D::_build() const {
	if (radius_top != radius_bottom) {
		ERR_FAIL_COND_V_MSG(radius_top <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its radius_top must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
		ERR_FAIL_COND_V_MSG(radius_bottom <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its radius_bottom must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
		ERR_FAIL_COND_V_MSG(mid_height <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its mid_height must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
		// This condition isn't necessary, as Jolt handles the case where the capsule becomes a sphere
		//ERR_FAIL_COND_V_MSG(mid_height < Math::abs(radius_top - radius_bottom), nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its mid_height must be at least the absolute value of the difference of its radii. This shape belongs to %s.", to_string(), _owners_to_string()));

		const float half_height = mid_height / 2.0f;

		const JPH::TaperedCapsuleShapeSettings shape_settings(half_height, radius_top, radius_bottom);
		const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
		ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

		return shape_result.Get();
	} else {
		const float radius = (radius_top + radius_bottom) / 2.f;
		ERR_FAIL_COND_V_MSG(radius <= 0.0f, nullptr, vformat("Failed to build Jolt Physics capsule shape with %s. Its radius must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
		ERR_FAIL_COND_V_MSG(mid_height <= 0.0f, nullptr, vformat("Failed to build Jolt Physics capsule shape with %s. Its height must be at least double that of its radius. This shape belongs to %s.", to_string(), _owners_to_string()));

		const float half_height = mid_height / 2.0f;

		const JPH::CapsuleShapeSettings shape_settings(half_height, radius);
		const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
		ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics capsule shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));
		return shape_result.Get();
	}
}

Variant JoltCapsuleShape3D::get_data() const {
	Dictionary data;
	data["radius_top"] = radius_top;
	data["radius_bottom"] = radius_bottom;
	data["mid_height"] = mid_height;
	data["height"] = radius_top + radius_bottom + mid_height;
	data["radius"] = (radius_top + radius_bottom) / 2.f;
	return data;
}

void JoltCapsuleShape3D::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);
	// Handle SHAPE_CAPSULE data
	const Dictionary data = p_data;
	//tapered capsule
	const Variant maybe_radius_top = data.get("radius_top", Variant());
	bool has_top_radius = maybe_radius_top.get_type() == Variant::FLOAT;
	const Variant maybe_radius_bottom = data.get("radius_bottom", Variant());
	bool has_bottom_radius = maybe_radius_bottom.get_type() == Variant::FLOAT;
	//capsule
	const Variant maybe_radius = data.get("radius", Variant());
	bool has_radius = maybe_radius.get_type() == Variant::FLOAT;

	//tapered capsule handling
	const Variant maybe_mid_height = data.get("mid_height", Variant());
	bool has_mid_height = maybe_mid_height.get_type() == Variant::FLOAT;
	//capsule handling
	const Variant maybe_height = data.get("height", Variant());
	bool has_height = maybe_height.get_type() == Variant::FLOAT;
	float new_radius_top;
	float new_radius_bottom;
	float new_mid_height;
	if (has_bottom_radius && has_top_radius) {
		new_radius_bottom = maybe_radius_bottom;
		new_radius_top = maybe_radius_top;
	} else if (has_radius) {
		new_radius_bottom = maybe_radius;
		new_radius_top = maybe_radius;
	} else {
		ERR_FAIL_MSG("Failed to create capsule: Missing radius parameters");
	}
	if (has_mid_height) {
		new_mid_height = maybe_mid_height;
	} else if (has_height) {
		new_mid_height = maybe_height;
		new_mid_height = new_mid_height - new_radius_top - new_radius_bottom;
	} else {
		ERR_FAIL_MSG("Failed to create capsule: Missing height parameters");
	}

	if (unlikely(new_radius_top == radius_top && new_radius_bottom == radius_bottom && new_mid_height == mid_height)) {
		return;
	}
	radius_top = new_radius_top;
	radius_bottom = new_radius_bottom;
	mid_height = new_mid_height;

	destroy();
}

AABB JoltCapsuleShape3D::get_aabb() const {
	const float max_radius = MAX(radius_top, radius_bottom);
	const Vector3 extents(max_radius * 2, mid_height + radius_bottom + radius_top, max_radius * 2);
	const Vector3 origin(-max_radius, -mid_height / 2 - radius_bottom, -max_radius);
	return AABB(origin, extents);
}

String JoltCapsuleShape3D::to_string() const {
	return vformat("{height=%f radius=%f radius_top=%f radius_bottom=%f mid_height=%f}", mid_height + radius_bottom + radius_top, (radius_top + radius_bottom) / 2.f, radius_top, radius_bottom, mid_height);
}

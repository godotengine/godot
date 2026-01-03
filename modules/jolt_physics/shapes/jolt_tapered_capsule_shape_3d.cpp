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
	ERR_FAIL_COND_V_MSG(radius_top <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its radius_top must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(radius_bottom <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its radius_bottom must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(mid_height <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its mid_height must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(mid_height < radius_top + radius_bottom, nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. Its mid_height must be at least the sum of its radii. This shape belongs to %s.", to_string(), _owners_to_string()));

	const float half_height = mid_height / 2.0f;

	const JPH::TaperedCapsuleShapeSettings shape_settings(half_height, radius_top, radius_bottom);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
	ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics tapered capsule shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

	return shape_result.Get();
}

Variant JoltTaperedCapsuleShape3D::get_data() const {
	Vector<double> data;
	data.resize(3);
	data.write[0] = radius_top;
	data.write[1] = radius_bottom;
	data.write[2] = mid_height;
	return data;
}

void JoltTaperedCapsuleShape3D::set_data(const Variant &p_data) {
	if (p_data.get_type() == Variant::DICTIONARY) {
		// Handle SHAPE_CAPSULE data
		const Dictionary data = p_data;
		const Variant maybe_radius = data.get("radius", Variant());
		const Variant maybe_height = data.get("height", Variant());
		if (maybe_radius.get_type() == Variant::FLOAT && maybe_height.get_type() == Variant::FLOAT) {
			real_t radius = maybe_radius;
			real_t height = maybe_height;
			if (radius == radius_top && radius == radius_bottom && height == mid_height) {
				return;
			}
			radius_top = radius;
			radius_bottom = radius;
			mid_height = height;
			destroy();
			return;
		}
	}

	if (p_data.get_type() == Variant::PACKED_FLOAT32_ARRAY || p_data.get_type() == Variant::PACKED_FLOAT64_ARRAY) {
		const Vector<real_t> data = p_data;
		if (data.size() == 3) {
			real_t new_radius_top = data[0];
			real_t new_radius_bottom = data[1];
			real_t new_mid_height = data[2];
			if (new_radius_top == radius_top && new_radius_bottom == radius_bottom && new_mid_height == mid_height) {
				return;
			}
			radius_top = new_radius_top;
			radius_bottom = new_radius_bottom;
			mid_height = new_mid_height;
			destroy();
			return;
		}
	}

	ERR_FAIL_MSG("Invalid data for JoltTaperedCapsuleShape3D");
}

AABB JoltTaperedCapsuleShape3D::get_aabb() const {
	const float max_radius = MAX(radius_top, radius_bottom);
	const Vector3 half_extents(max_radius, mid_height / 2.0f + max_radius, max_radius);
	return AABB(-half_extents, half_extents * 2.0f);
}

String JoltTaperedCapsuleShape3D::to_string() const {
	return vformat("{radius_top=%f radius_bottom=%f mid_height=%f}", radius_top, radius_bottom, mid_height);
}

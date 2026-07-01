/**************************************************************************/
/*  jolt_tapered_cylinder_shape_3d.cpp                                    */
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

#include "jolt_tapered_cylinder_shape_3d.h"

#include "../misc/jolt_type_conversions.h"
#include "Jolt/Physics/Collision/Shape/TaperedCylinderShape.h"

JPH::ShapeRefC JoltTaperedCylinderShape3D::_build() const {
	ERR_FAIL_COND_V_MSG(radius_top < 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. Its radius_top cannot be negative. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(radius_bottom < 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. Its radius_bottom cannot be negative. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(height <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. Its height must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));

	const float half_height = height / 2.0f;

	const JPH::TaperedCylinderShapeSettings shape_settings(half_height, radius_top, radius_bottom);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
	ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

	return shape_result.Get();
}

Variant JoltTaperedCylinderShape3D::get_data() const {
	Dictionary data;
	data["radius_top"] = radius_top;
	data["radius_bottom"] = radius_bottom;
	data["height"] = height;
	return data;
}

void JoltTaperedCylinderShape3D::set_data(const Variant &p_data) {
	if (p_data.get_type() == Variant::DICTIONARY) {
		const Dictionary data = p_data;
		const Variant maybe_radius_top = data.get("radius_top", Variant());
		ERR_FAIL_COND_MSG(maybe_radius_top.get_type() != Variant::FLOAT, "radius_top is not float");
		const Variant maybe_radius_bottom = data.get("radius_bottom", Variant());
		ERR_FAIL_COND_MSG(maybe_radius_bottom.get_type() != Variant::FLOAT, "radius_bottom is not float");
		const Variant maybe_height = data.get("height", Variant());
		ERR_FAIL_COND_MSG(maybe_height.get_type() != Variant::FLOAT, "height is not float");
		float mradius_top = maybe_radius_top;
		float mradius_bottom = maybe_radius_bottom;
		float mheight = maybe_height;
		if (unlikely(mradius_top == radius_top && mradius_bottom == radius_bottom && mheight == height)) {
			return;
		}
		radius_top = mradius_top;
		radius_bottom = mradius_bottom;
		height = mheight;
		destroy();
		return;
	}

	if (p_data.get_type() == Variant::PACKED_FLOAT32_ARRAY || p_data.get_type() == Variant::PACKED_FLOAT64_ARRAY) {
		const Vector<float> data = p_data;
		if (data.size() == 3) {
			float new_radius_top = data[0];
			float new_radius_bottom = data[1];
			float new_height = data[2];
			if (new_radius_top == radius_top && new_radius_bottom == radius_bottom && new_height == height) {
				return;
			}
			radius_top = new_radius_top;
			radius_bottom = new_radius_bottom;
			height = new_height;
			destroy();
			return;
		}
	}

	ERR_FAIL_MSG("Invalid data for JoltTaperedCylinderShape3D, expecting a dictionary or an array of floats");
}

AABB JoltTaperedCylinderShape3D::get_aabb() const {
	const float max_radius = MAX(radius_top, radius_bottom);
	const Vector3 extents(max_radius * 2, height, max_radius * 2);
	const Vector3 origin(-max_radius, -height / 2, -max_radius);
	return AABB(origin, extents);
}

String JoltTaperedCylinderShape3D::to_string() const {
	return vformat("{radius_top=%f radius_bottom=%f height=%f}", radius_top, radius_bottom, height);
}

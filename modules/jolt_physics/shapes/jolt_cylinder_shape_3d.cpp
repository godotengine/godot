/**************************************************************************/
/*  jolt_cylinder_shape_3d.cpp                                            */
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

#include "jolt_cylinder_shape_3d.h"

#include "../jolt_project_settings.h"
#include "../misc/jolt_type_conversions.h"

#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/TaperedCylinderShape.h>

JPH::ShapeRefC JoltCylinderShape3D::_build() const {
	if (radius_top != radius_bottom) {
		ERR_FAIL_COND_V_MSG(radius_top < 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. Its radius_top cannot be negative. This shape belongs to %s.", to_string(), _owners_to_string()));
		ERR_FAIL_COND_V_MSG(radius_bottom < 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. Its radius_bottom cannot be negative. This shape belongs to %s.", to_string(), _owners_to_string()));
		ERR_FAIL_COND_V_MSG(height <= 0.0f, nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. Its height must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));

		const float half_height = height / 2.0f;
		const float min_half_extent = MIN(MIN(half_height, radius_top), radius_bottom);
		const float actual_margin = MIN(margin, min_half_extent * JoltProjectSettings::collision_margin_fraction);

		const JPH::TaperedCylinderShapeSettings shape_settings(half_height, radius_top, radius_bottom, actual_margin);
		const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
		ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics tapered cylinder shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

		return shape_result.Get();
	} else {
		const float half_height = height / 2.0f;
		const float radius = (radius_top + radius_bottom) / 2.f;
		const float min_half_extent = MIN(half_height, radius);
		const float actual_margin = MIN(margin, min_half_extent * JoltProjectSettings::collision_margin_fraction);

		const JPH::CylinderShapeSettings shape_settings(half_height, radius, actual_margin);
		const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
		ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics cylinder shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

		return shape_result.Get();
	}
}

Variant JoltCylinderShape3D::get_data() const {
	Dictionary data;
	data["height"] = height;
	data["radius_top"] = radius_top;
	data["radius_bottom"] = radius_bottom;
	data["radius"] = (radius_top + radius_bottom) / 2.f;
	return data;
}

void JoltCylinderShape3D::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;
	//tapered capsule
	const Variant maybe_radius_top = data.get("radius_top", Variant());
	bool has_top_radius = maybe_radius_top.get_type() == Variant::FLOAT;
	const Variant maybe_radius_bottom = data.get("radius_bottom", Variant());
	bool has_bottom_radius = maybe_radius_bottom.get_type() == Variant::FLOAT;
	//capsule
	const Variant maybe_radius = data.get("radius", Variant());
	bool has_radius = maybe_radius.get_type() == Variant::FLOAT;

	const Variant maybe_height = data.get("height", Variant());
	bool has_height = maybe_height.get_type() == Variant::FLOAT;
	float new_radius_top;
	float new_radius_bottom;
	float new_height;
	if (has_bottom_radius && has_top_radius) {
		new_radius_bottom = maybe_radius_bottom;
		new_radius_top = maybe_radius_top;
	} else if (has_radius) {
		new_radius_bottom = maybe_radius;
		new_radius_top = maybe_radius;
	} else {
		ERR_FAIL_MSG("Failed to create capsule: Missing radius parameters");
	}
	if (has_height) {
		new_height = maybe_height;
	} else {
		ERR_FAIL_MSG("Failed to create capsule: Missing height parameters");
	}

	if (unlikely(new_radius_top == radius_top && new_radius_bottom == radius_bottom && new_height == height)) {
		return;
	}
	radius_top = new_radius_top;
	radius_bottom = new_radius_bottom;
	height = new_height;

	destroy();
}

void JoltCylinderShape3D::set_margin(float p_margin) {
	if (unlikely(margin == p_margin)) {
		return;
	}

	margin = p_margin;

	destroy();
}

AABB JoltCylinderShape3D::get_aabb() const {
	const float max_radius = MAX(radius_top, radius_bottom);
	const Vector3 extents(max_radius * 2, height, max_radius * 2);
	return AABB(-extents / 2.0f, extents);
}

String JoltCylinderShape3D::to_string() const {
	return vformat("{height=%f radius=%f margin=%f radius_top=%f radius_bottom=%f}", height, (radius_top + radius_bottom) / 2.f, margin, radius_top, radius_bottom);
}

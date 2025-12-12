/**************************************************************************/
/*  jolt_separation_ray_shape_3d.cpp                                      */
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

#include "jolt_separation_ray_shape_3d.h"

#include "../misc/jolt_type_conversions.h"
#include "jolt_custom_ray_shape.h"

JPH::ShapeRefC JoltSeparationRayShape3D::_build() const {
	ERR_FAIL_COND_V_MSG(length <= 0.0f, nullptr, vformat("Failed to build Jolt Physics separation ray shape with %s. Its length must be greater than 0. This shape belongs to %s.", to_string(), _owners_to_string()));

	const JoltCustomRayShapeSettings shape_settings(length, stops_motion, separate_along_ray);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
	ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics separation ray shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

	return shape_result.Get();
}

Variant JoltSeparationRayShape3D::get_data() const {
	Dictionary data;
	data["length"] = length;
	data["stops_motion"] = stops_motion;
	data["separate_along_ray"] = separate_along_ray;
	return data;
}

void JoltSeparationRayShape3D::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_length = data.get("length", Variant());
	ERR_FAIL_COND(maybe_length.get_type() != Variant::FLOAT);

	const Variant maybe_stops_motion = data.get("stops_motion", Variant());
	ERR_FAIL_COND(maybe_stops_motion.get_type() != Variant::BOOL);

	const Variant maybe_separate_along_ray = data.get("separate_along_ray", Variant());
	ERR_FAIL_COND(maybe_separate_along_ray.get_type() != Variant::BOOL);

	const float new_length = maybe_length;
	const bool new_stops_motion = maybe_stops_motion;
	const bool new_separate_along_ray = maybe_separate_along_ray;

	if (unlikely(new_length == length && new_stops_motion == stops_motion && new_separate_along_ray == separate_along_ray)) {
		return;
	}

	length = new_length;
	stops_motion = new_stops_motion;
	separate_along_ray = new_separate_along_ray;

	destroy();
}

AABB JoltSeparationRayShape3D::get_aabb() const {
	constexpr float size_xy = 0.1f;
	constexpr float half_size_xy = size_xy / 2.0f;
	return AABB(Vector3(-half_size_xy, -half_size_xy, 0.0f), Vector3(size_xy, size_xy, length));
}

String JoltSeparationRayShape3D::to_string() const {
	return vformat("{length=%f stops_motion=%s separate_along_ray=%s}", length, stops_motion, separate_along_ray);
}

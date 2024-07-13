/**************************************************************************/
/*  renik_helper.cpp                                                      */
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

#include "renik_helper.h"

float RenIKHelper::safe_acos(float p_value) {
	if (p_value > 1) {
		p_value = 1;
	} else if (p_value < -1) {
		p_value = -1;
	}
	return acos(p_value);
}

Vector3 RenIKHelper::get_perpendicular_vector(Vector3 p_v) {
	Vector3 perpendicular;
	if (p_v[0] != 0 && p_v[1] != 0) {
		perpendicular = Vector3(0, 0, 1).cross(p_v).normalized();
	} else {
		perpendicular = Vector3(1, 0, 0);
	}
	return perpendicular;
}

Vector3 RenIKHelper::vector_rejection(Vector3 p_vector, Vector3 p_normal) {
	if (p_vector.length_squared() == 0 || p_normal.length_squared() == 0) {
		return Vector3();
	}
	float normalLength = p_normal.length();
	Vector3 proj = (p_normal.dot(p_vector) / normalLength) * (p_normal / normalLength);
	return p_vector - proj;
}

Quaternion RenIKHelper::align_vectors(Vector3 p_a, Vector3 p_b, float p_influence) {
	if (p_a.length_squared() == 0 || p_b.length_squared() == 0) {
		return Quaternion();
	}
	p_a.normalize();
	p_b.normalize();
	if (p_a.length_squared() != 0 && p_b.length_squared() != 0) {
		// Find the axis perpendicular to both vectors and rotate along it by the
		// angular difference
		Vector3 perpendicular = p_a.cross(p_b);
		float angleDiff = p_a.angle_to(p_b) * p_influence;
		if (perpendicular.length_squared() == 0) {
			perpendicular = get_perpendicular_vector(p_a);
		}
		return Quaternion(perpendicular.normalized().normalized(), angleDiff)
				.normalized(); // lmao look at this double normalization bullshit
	} else {
		return Quaternion();
	}
}

Vector3 RenIKHelper::log_clamp(Vector3 p_vector, Vector3 p_target,
		float p_looseness) {
	p_vector.x = log_clamp(p_vector.x, p_target.x, p_looseness);
	p_vector.y = log_clamp(p_vector.y, p_target.y, p_looseness);
	p_vector.z = log_clamp(p_vector.z, p_target.z, p_looseness);
	return p_vector;
}
float RenIKHelper::log_clamp(float p_value, float p_target, float p_looseness) {
	float difference = p_value - p_target;
	float effectiveLooseness = difference >= 0 ? p_looseness : p_looseness * -1;
	return p_target +
			effectiveLooseness * log(1 + (difference / effectiveLooseness));
}

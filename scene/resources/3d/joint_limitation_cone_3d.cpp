/**************************************************************************/
/*  joint_limitation_cone_3d.cpp                                          */
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

#include "joint_limitation_cone_3d.h"

#ifndef DISABLE_DEPRECATED
bool JointLimitationCone3D::_set(const StringName &p_path, const Variant &p_value) {
	// To keep compatibility between 4.6.beta2 and beta3.
	if (p_path == SNAME("radius_range")) {
		set_angle((float)p_value * Math::TAU);
	} else {
		return false;
	}
	return true;
}
#endif // DISABLE_DEPRECATED

void JointLimitationCone3D::set_angle(real_t p_angle) {
	angle = p_angle;
	emit_changed();
}

real_t JointLimitationCone3D::get_angle() const {
	return angle;
}

void JointLimitationCone3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_angle", "angle"), &JointLimitationCone3D::set_angle);
	ClassDB::bind_method(D_METHOD("get_angle"), &JointLimitationCone3D::get_angle);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angle", PROPERTY_HINT_RANGE, "0,360,0.01,radians_as_degrees"), "set_angle", "get_angle");
}

Vector3 JointLimitationCone3D::_solve(const Vector3 &p_direction) const {
	// Assume the central (forward of the cone) axis is the +Y.
	// This is based on the coordinate system set by JointLimitation3D::_make_space().
	Vector3 center_axis = Vector3(0, 1, 0);

	// Apply the limitation if the angle exceeds radius_range * PI.
	real_t current_angle = p_direction.angle_to(center_axis);
	real_t max_angle = angle * 0.5;

	if (current_angle <= max_angle) {
		// If within the limitation range, return the new direction as is.
		return p_direction;
	}

	// If outside the limitation range, calculate the closest direction within the range.
	// Define a plane using the central axis and the new direction vector.
	Vector3 plane_normal;

	// Special handling for when the new direction vector is completely opposite to the central axis.
	if (Math::is_equal_approx((double)current_angle, Math::PI)) {
		// Select an arbitrary perpendicular axis
		plane_normal = center_axis.get_any_perpendicular();
	} else {
		plane_normal = center_axis.cross(p_direction).normalized();
	}

	// Calculate a vector rotated by the maximum angle from the central axis on the plane.
	Quaternion rotation = Quaternion(plane_normal, max_angle);
	Vector3 limited_dir = rotation.xform(center_axis);

	// Return the vector within the limitation range that is closest to p_direction.
	// This preserves the directionality of p_direction as much as possible.
	Vector3 projection = p_direction - center_axis * p_direction.dot(center_axis);
	if (projection.length_squared() > CMP_EPSILON) {
		Vector3 side_dir = projection.normalized();
		Quaternion side_rotation = Quaternion(center_axis.cross(side_dir).normalized(), max_angle);
		limited_dir = side_rotation.xform(center_axis);
	}

	return limited_dir.normalized();
}

#ifdef TOOLS_ENABLED
void JointLimitationCone3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	static const int N = 16;
	static const real_t DP = Math::TAU / (real_t)N;

	real_t sphere_r = p_bone_length * (real_t)0.25;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}
	real_t alpha = CLAMP((real_t)angle, (real_t)0.0, (real_t)Math::TAU) * 0.5;
	real_t y_cap = sphere_r * Math::cos(alpha);
	real_t r_cap = sphere_r * Math::sin(alpha);

	LocalVector<Vector3> vts;

	// Cone bottom.
	if (r_cap > CMP_EPSILON) {
		for (int i = 0; i < N; i++) {
			real_t a0 = (real_t)i * DP;
			real_t a1 = (real_t)((i + 1) % N) * DP;
			Vector3 p0 = Vector3(r_cap * Math::cos(a0), y_cap, r_cap * Math::sin(a0));
			Vector3 p1 = Vector3(r_cap * Math::cos(a1), y_cap, r_cap * Math::sin(a1));
			vts.push_back(p0);
			vts.push_back(p1);
		}
	}

	// Rotate arcs around Y-axis.
	real_t t_start;
	real_t arc_len;
	if (alpha <= (real_t)1e-6) {
		t_start = (real_t)0.5 * Math::PI;
		arc_len = Math::PI;
	} else {
		t_start = (real_t)0.5 * Math::PI + alpha;
		arc_len = Math::PI - alpha;
	}
	real_t dt = arc_len / (real_t)N;

	for (int k = 0; k < N; k++) {
		Basis ry(Vector3(0, 1, 0), (real_t)k * DP);

		Vector3 prev = ry.xform(Vector3(sphere_r * Math::cos(t_start), sphere_r * Math::sin(t_start), 0));

		for (int s = 1; s <= N; s++) {
			real_t t = t_start + dt * (real_t)s;
			Vector3 cur = ry.xform(Vector3(sphere_r * Math::cos(t), sphere_r * Math::sin(t), 0));

			vts.push_back(prev);
			vts.push_back(cur);

			prev = cur;
		}

		Vector3 mouth = ry.xform(Vector3(sphere_r * Math::cos(t_start), sphere_r * Math::sin(t_start), 0));
		Vector3 center = Vector3();

		vts.push_back(center);
		vts.push_back(mouth);
	}

	// Stack rings.
	for (int i = 1; i <= 3; i++) {
		for (int sgn = -1; sgn <= 1; sgn += 2) {
			real_t y = (real_t)sgn * sphere_r * ((real_t)i / (real_t)4.0);
			if (y >= y_cap - CMP_EPSILON) {
				continue;
			}
			real_t ring_r2 = sphere_r * sphere_r - y * y;
			if (ring_r2 <= (real_t)0.0) {
				continue;
			}
			real_t ring_r = Math::sqrt(ring_r2);

			for (int j = 0; j < N; j++) {
				real_t a0 = (real_t)j * DP;
				real_t a1 = (real_t)((j + 1) % N) * DP;
				Vector3 p0 = Vector3(ring_r * Math::cos(a0), y, ring_r * Math::sin(a0));
				Vector3 p1 = Vector3(ring_r * Math::cos(a1), y, ring_r * Math::sin(a1));

				vts.push_back(p0);
				vts.push_back(p1);
			}
		}
	}

	for (int64_t i = 0; i < vts.size(); i++) {
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(vts[i]));
	}
}
#endif // TOOLS_ENABLED

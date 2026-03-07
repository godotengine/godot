/**************************************************************************/
/*  tapered_capsule_shape_3d.cpp                                          */
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

#include "tapered_capsule_shape_3d.h"

#include "core/object/class_db.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/tapered_capsule_mesh.h"

Vector<Vector3> TaperedCapsuleShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;

	const int points_per_circle = 60;
	const int circle_step = 360 / points_per_circle;
	const int num_connecting_lines = 6;
	points.reserve(points_per_circle * (2 * 3 + num_connecting_lines));

	// Top circle
	const real_t angle = TaperedCapsuleMesh::get_tangent_angle(radius_top, radius_bottom, mid_height); //0 on horizontal, pi/2 on top, -pi/2 on bottom
	const real_t rtop = get_radius_top();
	const real_t r1 = rtop * cos(angle);
	const real_t h = get_mid_height();
	const Vector3 top_center(0, h / 2 + sin(angle) * rtop, 0);

	// Bottom circle
	const real_t rbottom = get_radius_bottom();
	const real_t r2 = rbottom * cos(angle);
	const Vector3 bottom_center(0, -h / 2 + sin(angle) * rbottom, 0);
	Vector<Pair<real_t, Vector3>> circles = { { r1, top_center }, { r2, bottom_center } };

	if (Math::abs(angle) > Math::PI / 12) { //15deg
		//circle on the biggest end of the tapered cylinder
		const bool circle_at_top = rtop > rbottom;
		const Vector3 center(0, circle_at_top ? h / 2 : -h / 2, 0);
		const real_t radius = circle_at_top ? rtop : rbottom;
		circles.push_back({ radius, center });
	}
	for (const auto k : circles) {
		const Vector3 &center = k.second;
		const real_t &radius = k.first;
		if (radius <= CMP_EPSILON) {
			continue;
		}

		for (int i = 0; i <= 360; i += circle_step) {
			real_t ra = Math::deg_to_rad((real_t)i);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;

			Vector3 newpoint = Vector3(a.x, 0, a.y) + center;
			points.push_back(newpoint);
			switch (i) {
				case 360:
				case 0:
					break;

				default:
					points.push_back(newpoint);
					break;
			}
		}
	}

	const int points_per_connecting_line = points_per_circle / 2;
	// Connecting lines
	for (size_t i = 0; i < num_connecting_lines; i++) {
		//points.push_back(points[0 + points_per_circle * 2 * i / num_connecting_lines]);
		//points.push_back(points[points_per_circle * 2 * (num_connecting_lines + i) / num_connecting_lines]);
		real_t ra = Math::deg_to_rad((real_t)360 * i / num_connecting_lines);
		Point2 a = Vector2(Math::cos(ra), Math::sin(ra));
		int top_transition = Math::round(points_per_connecting_line * (Math::PI / 2 - angle) / Math::PI);
		int start = top_transition == 0;
		int end = points_per_connecting_line + (top_transition != points_per_connecting_line);
		for (int j = start; j <= end; j++) {
			bool is_top = j <= top_transition;
			real_t phi = (j - !is_top) * Math::PI / points_per_connecting_line; //0 on top, pi on bottom
			if (j == top_transition || j == top_transition + 1) {
				phi = Math::PI / 2 - angle;
			}
			real_t radius = is_top ? rtop : rbottom;
			real_t rxz = sin(phi) * radius;
			real_t y = (is_top ? h / 2 : -h / 2) + cos(phi) * radius;
			Vector3 newpoint(a.x * rxz, y, a.y * rxz);
			points.push_back(newpoint);
			if (unlikely(j == start || j == end)) {
			} else {
				points.push_back(newpoint);
			}
		}
	}

	return points;
}

Ref<ArrayMesh> TaperedCapsuleShape3D::get_debug_arraymesh_faces(const Color &p_modulate) const {
	Array capsule_array;
	capsule_array.resize(RSE::ARRAY_MAX);
	TaperedCapsuleMesh::create_mesh_array(capsule_array, radius_top, radius_bottom, mid_height, 32, 8);

	Vector<Color> colors;
	const PackedVector3Array &verts = capsule_array[RSE::ARRAY_VERTEX];
	const int32_t verts_size = verts.size();
	for (int i = 0; i < verts_size; i++) {
		colors.append(p_modulate);
	}

	Ref<ArrayMesh> capsule_mesh = memnew(ArrayMesh);
	capsule_array[RSE::ARRAY_COLOR] = colors;
	capsule_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, capsule_array);
	return capsule_mesh;
}

real_t TaperedCapsuleShape3D::get_enclosing_radius() const {
	return MAX(radius_top, radius_bottom) + mid_height * 0.5f;
}

void TaperedCapsuleShape3D::_update_shape() {
	Dictionary d;
	if (PhysicsServer3D::get_singleton()->shape_get_type(get_shape()) == PhysicsServer3D::SHAPE_TAPERED_CAPSULE) {
		d["radius_top"] = radius_top;
		d["radius_bottom"] = radius_bottom;
		d["height"] = mid_height;
	} else {
		d["radius"] = (radius_top + radius_bottom) / 2.0;
		d["height"] = get_height();
	}
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void TaperedCapsuleShape3D::set_radius_top(real_t p_radius_top) {
	ERR_FAIL_COND_MSG(p_radius_top < 0.0f, "TaperedCapsuleShape3D radius_top cannot be negative.");
	radius_top = p_radius_top;
	_update_shape();
	emit_changed();
}

real_t TaperedCapsuleShape3D::get_radius_top() const {
	return radius_top;
}

void TaperedCapsuleShape3D::set_radius_bottom(real_t p_radius_bottom) {
	ERR_FAIL_COND_MSG(p_radius_bottom < 0.0f, "TaperedCapsuleShape3D radius_bottom cannot be negative.");
	radius_bottom = p_radius_bottom;
	_update_shape();
	emit_changed();
}

real_t TaperedCapsuleShape3D::get_radius_bottom() const {
	return radius_bottom;
}

void TaperedCapsuleShape3D::set_mid_height(real_t p_mid_height) {
	ERR_FAIL_COND_MSG(p_mid_height <= 0.0f, "TaperedCapsuleShape3D mid_height must be positive.");
	mid_height = p_mid_height;
	_update_shape();
	emit_changed();
}

real_t TaperedCapsuleShape3D::get_mid_height() const {
	return mid_height;
}

void TaperedCapsuleShape3D::set_height(real_t p_height) {
	real_t new_mid_height = p_height - radius_top - radius_bottom;
	if (new_mid_height <= 0) {
		new_mid_height = 0.001f; // Minimum
	}
	set_mid_height(new_mid_height);
}

real_t TaperedCapsuleShape3D::get_height() const {
	return mid_height + radius_top + radius_bottom;
}

void TaperedCapsuleShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius_top", "radius_top"), &TaperedCapsuleShape3D::set_radius_top);
	ClassDB::bind_method(D_METHOD("get_radius_top"), &TaperedCapsuleShape3D::get_radius_top);
	ClassDB::bind_method(D_METHOD("set_radius_bottom", "radius_bottom"), &TaperedCapsuleShape3D::set_radius_bottom);
	ClassDB::bind_method(D_METHOD("get_radius_bottom"), &TaperedCapsuleShape3D::get_radius_bottom);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &TaperedCapsuleShape3D::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &TaperedCapsuleShape3D::get_mid_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &TaperedCapsuleShape3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &TaperedCapsuleShape3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_top", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius_top", "get_radius_top");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_bottom", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius_bottom", "get_radius_bottom");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_mid_height", "get_mid_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m", PROPERTY_USAGE_NONE), "set_height", "get_height");

	ADD_LINKED_PROPERTY("radius_top", "height");
	ADD_LINKED_PROPERTY("radius_bottom", "height");
	ADD_LINKED_PROPERTY("mid_height", "height");
	ADD_LINKED_PROPERTY("height", "radius_top");
	ADD_LINKED_PROPERTY("height", "radius_bottom");
	ADD_LINKED_PROPERTY("height", "mid_height");
}

TaperedCapsuleShape3D::TaperedCapsuleShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_TAPERED_CAPSULE)) {
	_update_shape();
}

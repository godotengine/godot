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

#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/tapered_capsule_mesh.h" // Include the header for TaperedCapsuleMesh
#include "servers/physics_3d/physics_server_3d.h"

Vector<Vector3> TaperedCapsuleShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;

	// Top hemisphere
	real_t r1 = get_radius_top();
	real_t h = get_mid_height();
	Vector3 top_center = Vector3(0, h * 0.5f, 0);
	for (int i = 0; i < 360; i++) {
		real_t ra = Math::deg_to_rad((real_t)i);
		real_t rb = Math::deg_to_rad((real_t)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r1;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r1;

		points.push_back(Vector3(a.x, 0, a.y) + top_center);
		points.push_back(Vector3(b.x, 0, b.y) + top_center);

		if (i % 90 == 0) {
			points.push_back(Vector3(0, a.x, a.y) + top_center);
			points.push_back(Vector3(0, b.x, b.y) + top_center);
			points.push_back(Vector3(a.y, a.x, 0) + top_center);
			points.push_back(Vector3(b.y, b.x, 0) + top_center);
		}
	}

	// Bottom hemisphere
	real_t r2 = get_radius_bottom();
	Vector3 bottom_center = Vector3(0, -h * 0.5f, 0);
	for (int i = 0; i < 360; i++) {
		real_t ra = Math::deg_to_rad((real_t)i);
		real_t rb = Math::deg_to_rad((real_t)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r2;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r2;

		points.push_back(Vector3(a.x, 0, a.y) + bottom_center);
		points.push_back(Vector3(b.x, 0, b.y) + bottom_center);

		if (i % 90 == 0) {
			points.push_back(Vector3(0, a.x, a.y) + bottom_center);
			points.push_back(Vector3(0, b.x, b.y) + bottom_center);
			points.push_back(Vector3(a.y, a.x, 0) + bottom_center);
			points.push_back(Vector3(b.y, b.x, 0) + bottom_center);
		}
	}

	// Connecting lines (cylinder part)
	points.push_back(Vector3(r1, h * 0.5f, 0));
	points.push_back(Vector3(r2, -h * 0.5f, 0));
	points.push_back(Vector3(-r1, h * 0.5f, 0));
	points.push_back(Vector3(-r2, -h * 0.5f, 0));
	points.push_back(Vector3(0, h * 0.5f, r1));
	points.push_back(Vector3(0, -h * 0.5f, r2));
	points.push_back(Vector3(0, h * 0.5f, -r1));
	points.push_back(Vector3(0, -h * 0.5f, -r2));

	return points;
}

Ref<ArrayMesh> TaperedCapsuleShape3D::get_debug_arraymesh_faces(const Color &p_modulate) const {
	Array capsule_array;
	capsule_array.resize(RS::ARRAY_MAX);
	TaperedCapsuleMesh::create_mesh_array(capsule_array, radius_top, radius_bottom, mid_height, 32, 8);

	Vector<Color> colors;
	const PackedVector3Array &verts = capsule_array[RS::ARRAY_VERTEX];
	const int32_t verts_size = verts.size();
	for (int i = 0; i < verts_size; i++) {
		colors.append(p_modulate);
	}

	Ref<ArrayMesh> capsule_mesh = memnew(ArrayMesh);
	capsule_array[RS::ARRAY_COLOR] = colors;
	capsule_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, capsule_array);
	return capsule_mesh;
}

real_t TaperedCapsuleShape3D::get_enclosing_radius() const {
	return MAX(radius_top, radius_bottom) + mid_height * 0.5f;
}

void TaperedCapsuleShape3D::_update_shape() {
	Dictionary d;
	d["radius"] = (radius_top + radius_bottom) / 2.0;
	d["height"] = mid_height;
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

Variant TaperedCapsuleShape3D::get_data() const {
	Vector<real_t> data;
	data.resize(3);
	data.write[0] = radius_top;
	data.write[1] = radius_bottom;
	data.write[2] = mid_height;
	return data;
}

void TaperedCapsuleShape3D::set_data(const Variant &p_data) {
	if (p_data.get_type() == Variant::ARRAY) {
		// Backward compatibility with dictionary
		const Dictionary data = p_data;
		const Variant maybe_radius_top = data.get("radius_top", Variant());
		const Variant maybe_radius_bottom = data.get("radius_bottom", Variant());
		const Variant maybe_height = data.get("height", Variant());
		if (maybe_radius_top.get_type() == Variant::FLOAT && maybe_radius_bottom.get_type() == Variant::FLOAT && maybe_height.get_type() == Variant::FLOAT) {
			set_radius_top(maybe_radius_top);
			set_radius_bottom(maybe_radius_bottom);
			set_height(maybe_height);
			return;
		}
	}

	ERR_FAIL_COND(p_data.get_type() != Variant::PACKED_FLOAT64_ARRAY);

	const Vector<real_t> data = p_data;
	ERR_FAIL_COND(data.size() != 3);

	set_radius_top(data[0]);
	set_radius_bottom(data[1]);
	set_mid_height(data[2]);
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
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_height", "get_height");

	ADD_LINKED_PROPERTY("radius_top", "height");
	ADD_LINKED_PROPERTY("radius_bottom", "height");
	ADD_LINKED_PROPERTY("mid_height", "height");
	ADD_LINKED_PROPERTY("height", "radius_top");
	ADD_LINKED_PROPERTY("height", "radius_bottom");
	ADD_LINKED_PROPERTY("height", "mid_height");
}

TaperedCapsuleShape3D::TaperedCapsuleShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_CAPSULE)) {
	_update_shape();
}

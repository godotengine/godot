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
#include "servers/physics_server_3d.h"

Vector<Vector3> TaperedCapsuleShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;

	// Top hemisphere
	float r1 = get_radius_top();
	float h = get_height();
	Vector3 top_center = Vector3(0, h * 0.5f, 0);
	for (int i = 0; i < 360; i++) {
		float ra = Math::deg_to_rad((float)i);
		float rb = Math::deg_to_rad((float)i + 1);
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
	float r2 = get_radius_bottom();
	Vector3 bottom_center = Vector3(0, -h * 0.5f, 0);
	for (int i = 0; i < 360; i++) {
		float ra = Math::deg_to_rad((float)i);
		float rb = Math::deg_to_rad((float)i + 1);
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
	TaperedCapsuleMesh::create_mesh_array(capsule_array, radius_top, radius_bottom, height, 32, 8);

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
	return MAX(radius_top, radius_bottom) + height * 0.5f;
}

void TaperedCapsuleShape3D::_update_shape() {
	Dictionary d;
	d["radius_top"] = radius_top;
	d["radius_bottom"] = radius_bottom;
	d["height"] = height;
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void TaperedCapsuleShape3D::set_radius_top(float p_radius_top) {
	ERR_FAIL_COND_MSG(p_radius_top < 0.0f, "TaperedCapsuleShape3D radius_top cannot be negative.");
	radius_top = p_radius_top;
	_update_shape();
	emit_changed();
}

float TaperedCapsuleShape3D::get_radius_top() const {
	return radius_top;
}

void TaperedCapsuleShape3D::set_radius_bottom(float p_radius_bottom) {
	ERR_FAIL_COND_MSG(p_radius_bottom < 0.0f, "TaperedCapsuleShape3D radius_bottom cannot be negative.");
	radius_bottom = p_radius_bottom;
	_update_shape();
	emit_changed();
}

float TaperedCapsuleShape3D::get_radius_bottom() const {
	return radius_bottom;
}

void TaperedCapsuleShape3D::set_height(float p_height) {
	ERR_FAIL_COND_MSG(p_height <= 0.0f, "TaperedCapsuleShape3D height must be positive.");
	height = p_height;
	_update_shape();
	emit_changed();
}

float TaperedCapsuleShape3D::get_height() const {
	return height;
}

void TaperedCapsuleShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius_top", "radius_top"), &TaperedCapsuleShape3D::set_radius_top);
	ClassDB::bind_method(D_METHOD("get_radius_top"), &TaperedCapsuleShape3D::get_radius_top);
	ClassDB::bind_method(D_METHOD("set_radius_bottom", "radius_bottom"), &TaperedCapsuleShape3D::set_radius_bottom);
	ClassDB::bind_method(D_METHOD("get_radius_bottom"), &TaperedCapsuleShape3D::get_radius_bottom);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &TaperedCapsuleShape3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &TaperedCapsuleShape3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_top", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius_top", "get_radius_top");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_bottom", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius_bottom", "get_radius_bottom");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_height", "get_height");
}

TaperedCapsuleShape3D::TaperedCapsuleShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_CAPSULE)) {
	_update_shape();
}

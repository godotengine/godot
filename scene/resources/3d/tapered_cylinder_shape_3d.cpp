/**************************************************************************/
/*  tapered_cylinder_shape_3d.cpp                                         */
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

#include "tapered_cylinder_shape_3d.h"

#include "core/object/class_db.h"
#include "scene/resources/3d/primitive_meshes.h"

Vector<Vector3> TaperedCylinderShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;
	const int points_per_circle = 60;
	const int circle_step = 360 / points_per_circle;
	const int num_connecting_lines = 6;
	// Top and bottom circles
	real_t h = get_height();
	for (size_t k = 0; k < 2; k++) {
		bool is_top = !k;
		real_t radius = is_top ? get_top_radius() : get_bottom_radius();
		real_t y = is_top ? h / 2 : -h / 2;
		for (int i = 0; i <= 360; i += circle_step) {
			real_t ra = Math::deg_to_rad((real_t)i);
			Point2 a = Vector2(Math::cos(ra), Math::sin(ra)) * radius;

			Vector3 newpoint(a.x, y, a.y);
			points.push_back(newpoint);
			switch (i) {
				case 0:
				case 360:
					break;

				default:
					points.push_back(newpoint);
					break;
			}
		}
	}

	// Connecting lines from top to bottom
	for (size_t i = 0; i < num_connecting_lines; i++) {
		points.push_back(points[i * points_per_circle * 2 / num_connecting_lines]);
		points.push_back(points[(i + num_connecting_lines) * points_per_circle * 2 / num_connecting_lines]);
	}

	return points;
}

Ref<ArrayMesh> TaperedCylinderShape3D::get_debug_arraymesh_faces(const Color &p_modulate) const {
	Array cylinder_array;
	cylinder_array.resize(RSE::ARRAY_MAX);
	CylinderMesh::create_mesh_array(cylinder_array, top_radius, bottom_radius, height, 32);

	Vector<Color> colors;
	const PackedVector3Array &verts = cylinder_array[RSE::ARRAY_VERTEX];
	const int32_t verts_size = verts.size();
	for (int i = 0; i < verts_size; i++) {
		colors.append(p_modulate);
	}

	Ref<ArrayMesh> cylinder_mesh = memnew(ArrayMesh);
	cylinder_array[RSE::ARRAY_COLOR] = colors;
	cylinder_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, cylinder_array);
	return cylinder_mesh;
}

real_t TaperedCylinderShape3D::get_enclosing_radius() const {
	return MAX(top_radius, bottom_radius) + height * 0.5f;
}

void TaperedCylinderShape3D::_update_shape() {
	Dictionary d;
	static PhysicsServer3D::ShapeType created_shape = PhysicsServer3D::get_singleton()->shape_get_type(get_shape());
	static bool can_make_tapered_shapes = created_shape == PhysicsServer3D::SHAPE_TAPERED_CYLINDER; // Jolt can do that
	static bool can_make_basic_shape = created_shape == PhysicsServer3D::SHAPE_CYLINDER; // Dummy can't do that, but GodotPhysics3D can
	if (can_make_tapered_shapes) {
		d["radius_top"] = top_radius;
		d["radius_bottom"] = bottom_radius;
	} else if (can_make_basic_shape) {
		d["radius"] = (top_radius + bottom_radius) / 2.0;
		WARN_PRINT_ONCE("Tapered cylinder shapes aren't supported by the current physics engine (" + PhysicsServer3DManager::get_singleton()->get_default_server_name() + "). They will behave as regular cylinders instead.");
	}
	d["height"] = height;
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void TaperedCylinderShape3D::set_top_radius(real_t p_top_radius) {
	ERR_FAIL_COND_MSG(p_top_radius < 0, "TaperedCylinderShape3D top_radius cannot be negative.");
	top_radius = p_top_radius < 0 ? 0 : p_top_radius;
	_update_shape();
	emit_changed();
}

real_t TaperedCylinderShape3D::get_top_radius() const {
	return top_radius;
}

void TaperedCylinderShape3D::set_bottom_radius(real_t p_bottom_radius) {
	ERR_FAIL_COND_MSG(p_bottom_radius < 0, "TaperedCylinderShape3D bottom_radius cannot be negative.");
	bottom_radius = p_bottom_radius < 0 ? 0 : p_bottom_radius;
	_update_shape();
	emit_changed();
}

real_t TaperedCylinderShape3D::get_bottom_radius() const {
	return bottom_radius;
}

void TaperedCylinderShape3D::set_height(real_t p_height) {
	ERR_FAIL_COND_MSG(p_height <= 0.0f, "TaperedCylinderShape3D height must be positive.");
	height = p_height < CMP_EPSILON ? CMP_EPSILON : p_height;
	_update_shape();
	emit_changed();
}

real_t TaperedCylinderShape3D::get_height() const {
	return height;
}

void TaperedCylinderShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_top_radius", "top_radius"), &TaperedCylinderShape3D::set_top_radius);
	ClassDB::bind_method(D_METHOD("get_top_radius"), &TaperedCylinderShape3D::get_top_radius);
	ClassDB::bind_method(D_METHOD("set_bottom_radius", "bottom_radius"), &TaperedCylinderShape3D::set_bottom_radius);
	ClassDB::bind_method(D_METHOD("get_bottom_radius"), &TaperedCylinderShape3D::get_bottom_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &TaperedCylinderShape3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &TaperedCylinderShape3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_radius", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater,suffix:m"), "set_top_radius", "get_top_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bottom_radius", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater,suffix:m"), "set_bottom_radius", "get_bottom_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater,suffix:m"), "set_height", "get_height");
}

TaperedCylinderShape3D::TaperedCylinderShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_TAPERED_CYLINDER)) {
	_update_shape();
}

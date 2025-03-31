/**************************************************************************/
/*  world_boundary_shape_2d.cpp                                           */
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

#include "world_boundary_shape_2d.h"

#include "core/math/geometry_2d.h"
#include "servers/physics_server_2d.h"
#include "servers/rendering_server.h"

bool WorldBoundaryShape2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	Vector2 point = distance * normal;
	Vector2 l[2][2] = { { point - normal.orthogonal() * 100, point + normal.orthogonal() * 100 }, { point, point + normal * 30 } };

	for (int i = 0; i < 2; i++) {
		Vector2 closest = Geometry2D::get_closest_point_to_segment(p_point, l[i]);
		if (p_point.distance_to(closest) < p_tolerance) {
			return true;
		}
	}

	return false;
}

void WorldBoundaryShape2D::_update_shape() {
	Array arr = { normal, distance };
	PhysicsServer2D::get_singleton()->shape_set_data(get_rid(), arr);
	emit_changed();
}

void WorldBoundaryShape2D::set_normal(const Vector2 &p_normal) {
	// Can be non-unit but prevent zero.
	ERR_FAIL_COND(p_normal.is_zero_approx());
	if (normal == p_normal) {
		return;
	}
	normal = p_normal;
	_update_shape();
}

void WorldBoundaryShape2D::set_distance(real_t p_distance) {
	if (distance == p_distance) {
		return;
	}
	distance = p_distance;
	_update_shape();
}

Vector2 WorldBoundaryShape2D::get_normal() const {
	return normal;
}

real_t WorldBoundaryShape2D::get_distance() const {
	return distance;
}

void WorldBoundaryShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	Vector2 point = distance * normal;
	real_t line_width = 3.0;

	// Draw collision shape line.
	PackedVector2Array line_points = {
		point - normal.orthogonal() * 100,
		point - normal.orthogonal() * 60,
		point + normal.orthogonal() * 60,
		point + normal.orthogonal() * 100
	};

	Color transparent_color = Color(p_color, 0);
	PackedColorArray line_colors = {
		transparent_color,
		p_color,
		p_color,
		transparent_color
	};

	RS::get_singleton()->canvas_item_add_polyline(p_to_rid, line_points, line_colors, line_width);

	// Draw arrow.
	Color arrow_color = p_color.inverted();

	Transform2D xf;
	xf.rotate(normal.angle());

	Vector<Vector2> arrow_points = {
		xf.xform(Vector2(distance + line_width / 2, -2.5)),
		xf.xform(Vector2(distance + 20, -2.5)),
		xf.xform(Vector2(distance + 20, -10)),
		xf.xform(Vector2(distance + 40, 0)),
		xf.xform(Vector2(distance + 20, 10)),
		xf.xform(Vector2(distance + 20, 2.5)),
		xf.xform(Vector2(distance + line_width / 2, 2.5)),
	};

	RS::get_singleton()->canvas_item_add_polyline(p_to_rid, arrow_points, { arrow_color }, line_width / 2);
}

Rect2 WorldBoundaryShape2D::get_rect() const {
	Vector2 point = distance * normal;

	Vector2 l1[2] = { point - normal.orthogonal() * 100, point + normal.orthogonal() * 100 };
	Vector2 l2[2] = { point, point + normal * 30 };
	Rect2 rect;
	rect.position = l1[0];
	rect.expand_to(l1[1]);
	rect.expand_to(l2[0]);
	rect.expand_to(l2[1]);
	return rect;
}

real_t WorldBoundaryShape2D::get_enclosing_radius() const {
	return distance;
}

void WorldBoundaryShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_normal", "normal"), &WorldBoundaryShape2D::set_normal);
	ClassDB::bind_method(D_METHOD("get_normal"), &WorldBoundaryShape2D::get_normal);

	ClassDB::bind_method(D_METHOD("set_distance", "distance"), &WorldBoundaryShape2D::set_distance);
	ClassDB::bind_method(D_METHOD("get_distance"), &WorldBoundaryShape2D::get_distance);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "normal"), "set_normal", "get_normal");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance", PROPERTY_HINT_RANGE, "-1024,1024,0.01,or_greater,or_less,suffix:px"), "set_distance", "get_distance");
}

WorldBoundaryShape2D::WorldBoundaryShape2D() :
		Shape2D(PhysicsServer2D::get_singleton()->world_boundary_shape_create()) {
	_update_shape();
}

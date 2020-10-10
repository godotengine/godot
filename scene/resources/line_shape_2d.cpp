/*************************************************************************/
/*  line_shape_2d.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "line_shape_2d.h"

#include "core/math/geometry_2d.h"
#include "servers/physics_server_2d.h"
#include "servers/rendering_server.h"

bool LineShape2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	Vector2 point = get_distance() * get_normal();
	Vector2 l[2][2] = { { point - get_normal().tangent() * 100, point + get_normal().tangent() * 100 }, { point, point + get_normal() * 30 } };

	for (int i = 0; i < 2; i++) {
		Vector2 closest = Geometry2D::get_closest_point_to_segment(p_point, l[i]);
		if (p_point.distance_to(closest) < p_tolerance) {
			return true;
		}
	}

	return false;
}

void LineShape2D::_update_shape() {
	Array arr;
	arr.push_back(normal);
	arr.push_back(distance);
	PhysicsServer2D::get_singleton()->shape_set_data(get_rid(), arr);
	emit_changed();
}

void LineShape2D::set_normal(const Vector2 &p_normal) {
	normal = p_normal;
	_update_shape();
}

void LineShape2D::set_distance(real_t p_distance) {
	distance = p_distance;
	_update_shape();
}

Vector2 LineShape2D::get_normal() const {
	return normal;
}

real_t LineShape2D::get_distance() const {
	return distance;
}

void LineShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	Vector2 point = get_distance() * get_normal();

	Vector2 l1[2] = { point - get_normal().tangent() * 100, point + get_normal().tangent() * 100 };
	RS::get_singleton()->canvas_item_add_line(p_to_rid, l1[0], l1[1], p_color, 3);
	Vector2 l2[2] = { point, point + get_normal() * 30 };
	RS::get_singleton()->canvas_item_add_line(p_to_rid, l2[0], l2[1], p_color, 3);
}

Rect2 LineShape2D::get_rect() const {
	Vector2 point = get_distance() * get_normal();

	Vector2 l1[2] = { point - get_normal().tangent() * 100, point + get_normal().tangent() * 100 };
	Vector2 l2[2] = { point, point + get_normal() * 30 };
	Rect2 rect;
	rect.position = l1[0];
	rect.expand_to(l1[1]);
	rect.expand_to(l2[0]);
	rect.expand_to(l2[1]);
	return rect;
}

real_t LineShape2D::get_enclosing_radius() const {
	return distance;
}

void LineShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_normal", "normal"), &LineShape2D::set_normal);
	ClassDB::bind_method(D_METHOD("get_normal"), &LineShape2D::get_normal);

	ClassDB::bind_method(D_METHOD("set_distance", "distance"), &LineShape2D::set_distance);
	ClassDB::bind_method(D_METHOD("get_distance"), &LineShape2D::get_distance);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "normal"), "set_normal", "get_normal");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance"), "set_distance", "get_distance");
}

LineShape2D::LineShape2D() :
		Shape2D(PhysicsServer2D::get_singleton()->line_shape_create()) {
	normal = Vector2(0, 1);
	distance = 0;
	_update_shape();
}

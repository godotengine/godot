/**************************************************************************/
/*  capsule_shape_2d.cpp                                                  */
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

#include "capsule_shape_2d.h"

#include "servers/physics_2d_server.h"
#include "servers/visual_server.h"

Vector<Vector2> CapsuleShape2D::_get_points() const {
	Vector<Vector2> points;
	for (int i = 0; i < 24; i++) {
		Vector2 ofs = Vector2(0, (i > 6 && i <= 18) ? -get_height() * 0.5 : get_height() * 0.5);

		points.push_back(Vector2(Math::sin(i * Math_PI * 2 / 24.0), Math::cos(i * Math_PI * 2 / 24.0)) * get_radius() + ofs);
		if (i == 6 || i == 18) {
			points.push_back(Vector2(Math::sin(i * Math_PI * 2 / 24.0), Math::cos(i * Math_PI * 2 / 24.0)) * get_radius() - ofs);
		}
	}

	return points;
}

bool CapsuleShape2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	return Geometry::is_point_in_polygon(p_point, _get_points());
}

void CapsuleShape2D::_update_shape() {
	Physics2DServer::get_singleton()->shape_set_data(get_rid(), Vector2(radius, height));
	emit_changed();
}

void CapsuleShape2D::set_radius(real_t p_radius) {
	radius = p_radius;
	_update_shape();
}

real_t CapsuleShape2D::get_radius() const {
	return radius;
}

void CapsuleShape2D::set_height(real_t p_height) {
	height = p_height;
	if (height < 0) {
		height = 0;
	}

	_update_shape();
}

real_t CapsuleShape2D::get_height() const {
	return height;
}

void CapsuleShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	Vector<Vector2> points = _get_points();
	Vector<Color> col;
	col.push_back(p_color);
	VisualServer::get_singleton()->canvas_item_add_polygon(p_to_rid, points, col);
	if (is_collision_outline_enabled()) {
		VisualServer::get_singleton()->canvas_item_add_polyline(p_to_rid, points, col, 1.0, true);
		// Draw the last segment as it's not drawn by `canvas_item_add_polyline()`.
		VisualServer::get_singleton()->canvas_item_add_line(p_to_rid, points[points.size() - 1], points[0], p_color, 1.0, true);
	}
}

Rect2 CapsuleShape2D::get_rect() const {
	Vector2 he = Point2(get_radius(), get_radius() + get_height() * 0.5);
	Rect2 rect;
	rect.position = -he;
	rect.size = he * 2.0;
	return rect;
}

real_t CapsuleShape2D::get_enclosing_radius() const {
	return radius + height * 0.5;
}

void CapsuleShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CapsuleShape2D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CapsuleShape2D::get_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CapsuleShape2D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CapsuleShape2D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater"), "set_height", "get_height");
}

CapsuleShape2D::CapsuleShape2D() :
		Shape2D(Physics2DServer::get_singleton()->capsule_shape_create()) {
	radius = 10;
	height = 20;
	_update_shape();
}

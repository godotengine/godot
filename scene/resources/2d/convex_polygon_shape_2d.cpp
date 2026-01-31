/**************************************************************************/
/*  convex_polygon_shape_2d.cpp                                           */
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

#include "convex_polygon_shape_2d.h"

#include "core/math/geometry_2d.h"
#include "servers/physics_2d/physics_server_2d.h"
#include "servers/rendering/rendering_server.h"

bool ConvexPolygonShape2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	return Geometry2D::is_point_in_polygon(p_point, points);
}

#ifdef DEBUG_ENABLED
// Check if point p3 is to the left of [p1, p2] segment or on it.
bool left_test(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3) {
	Vector2 p12 = p2 - p1;
	Vector2 p13 = p3 - p1;
	// If the value of the cross product is positive or zero; p3 is to the left or on the segment, respectively.
	return p12.cross(p13) >= 0;
}

bool is_convex(const Vector<Vector2> &p_points) {
	// Pre-condition: Polygon is in counter-clockwise order.
	int polygon_size = p_points.size();
	for (int i = 0; i < polygon_size && polygon_size > 3; i++) {
		int j = (i + 1) % polygon_size;
		int k = (j + 1) % polygon_size;
		// If any consecutive three points fail left-test, then there is a concavity.
		if (!left_test(p_points[i], p_points[j], p_points[k])) {
			return false;
		}
	}

	return true;
}
#endif

void ConvexPolygonShape2D::_update_shape() {
	Vector<Vector2> final_points = points;
	if (Geometry2D::is_polygon_clockwise(final_points)) { //needs to be counter clockwise
		final_points.reverse();
	}
#ifdef DEBUG_ENABLED
	if (!is_convex(final_points)) {
		WARN_PRINT("Concave polygon is assigned to ConvexPolygonShape2D.");
	}
#endif
	PhysicsServer2D::get_singleton()->shape_set_data(get_rid(), final_points);
	emit_changed();
}

void ConvexPolygonShape2D::set_point_cloud(const Vector<Vector2> &p_points) {
	Vector<Point2> hull = Geometry2D::convex_hull(p_points);
	ERR_FAIL_COND(hull.size() < 3);
	set_points(hull);
}

void ConvexPolygonShape2D::set_points(const Vector<Vector2> &p_points) {
	points = p_points;

	_update_shape();
}

Vector<Vector2> ConvexPolygonShape2D::get_points() const {
	return points;
}

bool ConvexPolygonShape2D::contains_point(const Vector2 &p_point) const {
	return false; //Requires rewriting base class to include normals of points
}

void ConvexPolygonShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_point_cloud", "point_cloud"), &ConvexPolygonShape2D::set_point_cloud);
	ClassDB::bind_method(D_METHOD("set_points", "points"), &ConvexPolygonShape2D::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &ConvexPolygonShape2D::get_points);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "points"), "set_points", "get_points");
}

void ConvexPolygonShape2D::draw(const RID &p_to_rid, const Color &p_color) {
	if (points.size() < 3) {
		return;
	}

	Vector<Color> col = { p_color };
	RenderingServer::get_singleton()->canvas_item_add_polygon(p_to_rid, points, col);

	if (is_collision_outline_enabled()) {
		col = { Color(p_color, 1.0) };
		RenderingServer::get_singleton()->canvas_item_add_polyline(p_to_rid, points, col);
		// Draw the last segment.
		RenderingServer::get_singleton()->canvas_item_add_line(p_to_rid, points[points.size() - 1], points[0], Color(p_color, 1.0));
	}
}

Rect2 ConvexPolygonShape2D::get_rect() const {
	Rect2 rect;
	for (int i = 0; i < points.size(); i++) {
		if (i == 0) {
			rect.position = points[i];
		} else {
			rect.expand_to(points[i]);
		}
	}

	return rect;
}

real_t ConvexPolygonShape2D::get_enclosing_radius() const {
	real_t r = 0.0;
	for (int i(0); i < get_points().size(); i++) {
		r = MAX(get_points()[i].length_squared(), r);
	}
	return Math::sqrt(r);
}

ConvexPolygonShape2D::ConvexPolygonShape2D() :
		Shape2D(PhysicsServer2D::get_singleton()->convex_polygon_shape_create()) {
}

/*************************************************************************/
/*  convex_polygon_shape_2d.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "convex_polygon_shape_2d.h"

#include "geometry.h"
#include "servers/physics_2d_server.h"
#include "servers/visual_server.h"

void ConvexPolygonShape2D::_update_shape() {

	Physics2DServer::get_singleton()->shape_set_data(get_rid(), points);
	emit_changed();
}

void ConvexPolygonShape2D::set_point_cloud(const Vector<Vector2> &p_points) {

	Vector<Point2> hull = Geometry::convex_hull_2d(p_points);
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

void ConvexPolygonShape2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_point_cloud", "point_cloud"), &ConvexPolygonShape2D::set_point_cloud);
	ClassDB::bind_method(D_METHOD("set_points", "points"), &ConvexPolygonShape2D::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &ConvexPolygonShape2D::get_points);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "points"), "set_points", "get_points");
}

void ConvexPolygonShape2D::draw(const RID &p_to_rid, const Color &p_color) {

	Vector<Color> col;
	col.push_back(p_color);
	VisualServer::get_singleton()->canvas_item_add_polygon(p_to_rid, points, col);
}

Rect2 ConvexPolygonShape2D::get_rect() const {

	Rect2 rect;
	for (int i = 0; i < points.size(); i++) {
		if (i == 0)
			rect.pos = points[i];
		else
			rect.expand_to(points[i]);
	}

	return rect;
}

ConvexPolygonShape2D::ConvexPolygonShape2D()
	: Shape2D(Physics2DServer::get_singleton()->shape_create(Physics2DServer::SHAPE_CONVEX_POLYGON)) {

	int pcount = 3;
	for (int i = 0; i < pcount; i++)
		points.push_back(Vector2(Math::sin(i * Math_PI * 2 / pcount), -Math::cos(i * Math_PI * 2 / pcount)) * 10);

	_update_shape();
}

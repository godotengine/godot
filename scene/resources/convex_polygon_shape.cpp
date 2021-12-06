/*************************************************************************/
/*  convex_polygon_shape.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "convex_polygon_shape.h"
#include "core/math/convex_hull.h"
#include "servers/physics_server.h"

Vector<Vector3> ConvexPolygonShape::get_debug_mesh_lines() {
	PoolVector<Vector3> points = get_points();

	if (points.size() > 3) {
		Vector<Vector3> varr = Variant(points);
		Geometry::MeshData md;
		Error err = ConvexHullComputer::convex_hull(varr, md);
		if (err == OK) {
			Vector<Vector3> lines;
			lines.resize(md.edges.size() * 2);
			for (int i = 0; i < md.edges.size(); i++) {
				lines.write[i * 2 + 0] = md.vertices[md.edges[i].a];
				lines.write[i * 2 + 1] = md.vertices[md.edges[i].b];
			}
			return lines;
		}
	}

	return Vector<Vector3>();
}

void ConvexPolygonShape::_update_shape() {
	PhysicsServer::get_singleton()->shape_set_data(get_shape(), points);
	Shape::_update_shape();
}

void ConvexPolygonShape::set_points(const PoolVector<Vector3> &p_points) {
	points = p_points;
	_update_shape();
	notify_change_to_owners();
}

PoolVector<Vector3> ConvexPolygonShape::get_points() const {
	return points;
}

void ConvexPolygonShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_points", "points"), &ConvexPolygonShape::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &ConvexPolygonShape::get_points);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "points"), "set_points", "get_points");
}

ConvexPolygonShape::ConvexPolygonShape() :
		Shape(RID_PRIME(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_CONVEX_POLYGON))) {
}

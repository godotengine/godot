/*************************************************************************/
/*  concave_polygon_shape.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "concave_polygon_shape.h"

#include "servers/physics_server.h"

Vector<Vector3> ConcavePolygonShape::get_debug_mesh_lines() {
	Set<DrawEdge> edges;

	PoolVector<Vector3> data = get_faces();
	int datalen = data.size();
	ERR_FAIL_COND_V((datalen % 3) != 0, Vector<Vector3>());

	PoolVector<Vector3>::Read r = data.read();

	for (int i = 0; i < datalen; i += 3) {
		for (int j = 0; j < 3; j++) {
			DrawEdge de(r[i + j], r[i + ((j + 1) % 3)]);
			edges.insert(de);
		}
	}

	Vector<Vector3> points;
	points.resize(edges.size() * 2);
	int idx = 0;
	for (Set<DrawEdge>::Element *E = edges.front(); E; E = E->next()) {
		points.write[idx + 0] = E->get().a;
		points.write[idx + 1] = E->get().b;
		idx += 2;
	}

	return points;
}

real_t ConcavePolygonShape::get_enclosing_radius() const {
	PoolVector<Vector3> data = get_faces();
	PoolVector<Vector3>::Read read = data.read();
	real_t r = 0;
	for (int i(0); i < data.size(); i++) {
		r = MAX(read[i].length_squared(), r);
	}
	return Math::sqrt(r);
}

void ConcavePolygonShape::_update_shape() {
	Shape::_update_shape();
}

void ConcavePolygonShape::set_faces(const PoolVector<Vector3> &p_faces) {
	PhysicsServer::get_singleton()->shape_set_data(get_shape(), p_faces);
	_update_shape();
	notify_change_to_owners();
}

PoolVector<Vector3> ConcavePolygonShape::get_faces() const {
	return PhysicsServer::get_singleton()->shape_get_data(get_shape());
}

void ConcavePolygonShape::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_faces", "faces"), &ConcavePolygonShape::set_faces);
	ClassDB::bind_method(D_METHOD("get_faces"), &ConcavePolygonShape::get_faces);
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR3_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_faces", "get_faces");
}

ConcavePolygonShape::ConcavePolygonShape() :
		Shape(RID_PRIME(PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_CONCAVE_POLYGON))) {
	//set_planes(Vector3(1,1,1));
}

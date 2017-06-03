/*************************************************************************/
/*  quad.cpp                                                             */
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
#include "quad.h"
#include "servers/visual_server.h"

void Quad::_update() {

	if (!is_inside_tree())
		return;

	Vector3 normal;
	normal[axis] = 1.0;

	const int axis_order_1[3] = { 1, 2, 0 };
	const int axis_order_2[3] = { 2, 0, 1 };
	const int a1 = axis_order_1[axis];
	const int a2 = axis_order_2[axis];

	PoolVector<Vector3> points;
	points.resize(4);
	PoolVector<Vector3>::Write pointsw = points.write();

	Vector2 s2 = size * 0.5;
	Vector2 o = offset;
	if (!centered)
		o += s2;

	pointsw[0][a1] = -s2.x + offset.x;
	pointsw[0][a2] = s2.y + offset.y;

	pointsw[1][a1] = s2.x + offset.x;
	pointsw[1][a2] = s2.y + offset.y;

	pointsw[2][a1] = s2.x + offset.x;
	pointsw[2][a2] = -s2.y + offset.y;

	pointsw[3][a1] = -s2.x + offset.x;
	pointsw[3][a2] = -s2.y + offset.y;

	aabb = Rect3(pointsw[0], Vector3());
	for (int i = 1; i < 4; i++)
		aabb.expand_to(pointsw[i]);

	pointsw = PoolVector<Vector3>::Write();

	PoolVector<Vector3> normals;
	normals.resize(4);
	PoolVector<Vector3>::Write normalsw = normals.write();

	for (int i = 0; i < 4; i++)
		normalsw[i] = normal;

	normalsw = PoolVector<Vector3>::Write();

	PoolVector<Vector2> uvs;
	uvs.resize(4);
	PoolVector<Vector2>::Write uvsw = uvs.write();

	uvsw[0] = Vector2(0, 0);
	uvsw[1] = Vector2(1, 0);
	uvsw[2] = Vector2(1, 1);
	uvsw[3] = Vector2(0, 1);

	uvsw = PoolVector<Vector2>::Write();

	PoolVector<int> indices;
	indices.resize(6);

	PoolVector<int>::Write indicesw = indices.write();
	indicesw[0] = 0;
	indicesw[1] = 1;
	indicesw[2] = 2;
	indicesw[3] = 2;
	indicesw[4] = 3;
	indicesw[5] = 0;

	indicesw = PoolVector<int>::Write();

	Array arr;
	arr.resize(VS::ARRAY_MAX);
	arr[VS::ARRAY_VERTEX] = points;
	arr[VS::ARRAY_NORMAL] = normals;
	arr[VS::ARRAY_TEX_UV] = uvs;
	arr[VS::ARRAY_INDEX] = indices;

	if (configured) {
		VS::get_singleton()->mesh_remove_surface(mesh, 0);
	} else {
		configured = true;
	}
	VS::get_singleton()->mesh_add_surface_from_arrays(mesh, VS::PRIMITIVE_TRIANGLES, arr);

	pending_update = false;
}

void Quad::set_axis(Vector3::Axis p_axis) {

	axis = p_axis;
	_update();
}

Vector3::Axis Quad::get_axis() const {

	return axis;
}

void Quad::set_size(const Vector2 &p_size) {

	size = p_size;
	_update();
}
Vector2 Quad::get_size() const {

	return size;
}

void Quad::set_offset(const Vector2 &p_offset) {

	offset = p_offset;
	_update();
}
Vector2 Quad::get_offset() const {

	return offset;
}

void Quad::set_centered(bool p_enabled) {

	centered = p_enabled;
	_update();
}
bool Quad::is_centered() const {

	return centered;
}

void Quad::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (pending_update)
				_update();

		} break;
		case NOTIFICATION_EXIT_TREE: {

			pending_update = true;

		} break;
	}
}

PoolVector<Face3> Quad::get_faces(uint32_t p_usage_flags) const {

	return PoolVector<Face3>();
}

Rect3 Quad::get_aabb() const {

	return aabb;
}

void Quad::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_axis", "axis"), &Quad::set_axis);
	ClassDB::bind_method(D_METHOD("get_axis"), &Quad::get_axis);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &Quad::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &Quad::get_size);

	ClassDB::bind_method(D_METHOD("set_centered", "centered"), &Quad::set_centered);
	ClassDB::bind_method(D_METHOD("is_centered"), &Quad::is_centered);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Quad::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Quad::get_offset);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis", PROPERTY_HINT_ENUM, "X,Y,Z"), "set_axis", "get_axis");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "centered"), "set_centered", "is_centered");
}

Quad::Quad() {

	pending_update = true;
	centered = true;
	//offset=0;
	size = Vector2(1, 1);
	axis = Vector3::AXIS_Z;
	mesh = VisualServer::get_singleton()->mesh_create();
	set_base(mesh);
	configured = false;
}

Quad::~Quad() {
	VisualServer::get_singleton()->free(mesh);
}

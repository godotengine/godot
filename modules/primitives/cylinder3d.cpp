/*************************************************************************/
/*  cylinder3d.cpp                                                       */
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
#include "cylinder3d.h"
#include "servers/visual_server.h"

void Cylinder3D::_update() {
	int   i, j, prevrow, thisrow, point;
	float x, y, z, u, v, radius;

	if (!is_inside_tree()) {
		pending_update = true; // try again once we enter our tree...
		return;
	}

	radius = bottom_radius > top_radius ? bottom_radius : top_radius;

	aabb = AABB(Vector3(-radius, height * -0.5, -radius), Vector3(2.0 * radius, height, 2.0 * radius));

	DVector<Vector3> points;
	DVector<Vector3> normals;
	DVector<float> tangents;
	DVector<Vector2> uvs;
	DVector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x); \
	tangents.push_back(m_y); \
	tangents.push_back(m_z); \
	tangents.push_back(m_d);

	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		v = j;
		v /= (rings + 1);

		radius = top_radius + ((bottom_radius - top_radius) * v);

		y = height * v;
		y = (height * 0.5) - y;

		for (i = 0; i <= segments; i++) {
			u = i;
			u /= segments;

			x = sin(u * (Math_PI * 2.0));
			z = cos(u * (Math_PI * 2.0));

			Vector3 p = Vector3(x * radius, y, z * radius);
			points.push_back(p);
			normals.push_back(p.normalized());
			ADD_TANGENT(-z, 0.0, x, -1.0)
			uvs.push_back(Vector2(u, v * 0.5));
			point++;

			if (i>0 && j>0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			};
		};

		prevrow = thisrow;
		thisrow = point;
	};

	// add top
	if (top_radius > 0.0) {
		y = height * 0.5;

		thisrow = point;
		points.push_back(Vector3(0.0, y, 0.0));
		normals.push_back(Vector3(0.0, 1.0, 0.0));
		ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
		uvs.push_back(Vector2(0.25, 0.75));
		point++;

		for (i = 0; i <= segments; i++) {
			float r = i;
			r /= segments;

			x = sin(r * (Math_PI * 2.0));
			z = cos(r * (Math_PI * 2.0));

			u = ((x + 1.0) * 0.25);
			v = 0.5 + ((z + 1.0) * 0.25);

			Vector3 p = Vector3(x * top_radius, y, z * top_radius);
			points.push_back(p);
			normals.push_back(Vector3(0.0, 1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
			uvs.push_back(Vector2(u, v));
			point++;

			if (i > 0) {
				indices.push_back(thisrow);
				indices.push_back(point - 1);
				indices.push_back(point - 2);
			};
		};
	};

	// add bottom
	if (bottom_radius > 0.0) {
		y = height * -0.5;

		thisrow = point;
		points.push_back(Vector3(0.0, y, 0.0));
		normals.push_back(Vector3(0.0, -1.0, 0.0));
		ADD_TANGENT(-1.0, 0.0, 0.0, -1.0)
		uvs.push_back(Vector2(0.75, 0.75));
		point++;

		for (i = 0; i <= segments; i++) {
			float r = i;
			r /= segments;

			x = sin(r * (Math_PI * 2.0));
			z = cos(r * (Math_PI * 2.0));

			u = 0.5 + ((x + 1.0) * 0.25);
			v = 1.0 - ((z + 1.0) * 0.25);

			Vector3 p = Vector3(x * bottom_radius, y, z * bottom_radius);
			points.push_back(p);
			normals.push_back(Vector3(0.0, -1.0, 0.0));
			ADD_TANGENT(-1.0, 0.0, 0.0, -1.0)
			uvs.push_back(Vector2(u, v));
			point++;

			if (i > 0) {
				indices.push_back(thisrow);
				indices.push_back(point - 2);
				indices.push_back(point - 1);
			};
		};
	};

	Array arr;
	arr.resize(VS::ARRAY_MAX);
	arr[VS::ARRAY_VERTEX] = points;
	arr[VS::ARRAY_NORMAL] = normals;
	arr[VS::ARRAY_TANGENT] = tangents;
	arr[VS::ARRAY_TEX_UV] = uvs;
	arr[VS::ARRAY_INDEX] = indices;

	if (configured) {
		VS::get_singleton()->mesh_remove_surface(mesh, 0);
	} else {
		configured = true;
	}
	VS::get_singleton()->mesh_add_surface(mesh, VS::PRIMITIVE_TRIANGLES, arr);

	pending_update = false;
}

void Cylinder3D::_notification(int p_what) {

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

void Cylinder3D::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("set_top_radius", "radius"), &Cylinder3D::set_top_radius);
	ObjectTypeDB::bind_method(_MD("get_top_radius"), &Cylinder3D::get_top_radius);
	ObjectTypeDB::bind_method(_MD("set_bottom_radius", "radius"), &Cylinder3D::set_bottom_radius);
	ObjectTypeDB::bind_method(_MD("get_bottom_radius"), &Cylinder3D::get_bottom_radius);
	ObjectTypeDB::bind_method(_MD("set_height", "height"), &Cylinder3D::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"), &Cylinder3D::get_height);

	ObjectTypeDB::bind_method(_MD("set_segments", "segments"), &Cylinder3D::set_segments);
	ObjectTypeDB::bind_method(_MD("get_segments"), &Cylinder3D::get_segments);
	ObjectTypeDB::bind_method(_MD("set_rings", "rings"), &Cylinder3D::set_rings);
	ObjectTypeDB::bind_method(_MD("get_rings"), &Cylinder3D::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "top_radius"), _SCS("set_top_radius"), _SCS("get_top_radius"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bottom_radius"), _SCS("set_bottom_radius"), _SCS("get_bottom_radius"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height"), _SCS("set_height"), _SCS("get_height"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "segments"), _SCS("set_segments"), _SCS("get_segments"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings"), _SCS("set_rings"), _SCS("get_rings"));
}

void Cylinder3D::set_top_radius(const float p_radius) {
	top_radius = p_radius;
	_update();
}

float Cylinder3D::get_top_radius() const {
	return top_radius;
}

void Cylinder3D::set_bottom_radius(const float p_radius) {
	bottom_radius = p_radius;
	_update();
}

float Cylinder3D::get_bottom_radius() const {
	return bottom_radius;
}

void Cylinder3D::set_height(const float p_height) {
	height = p_height;
	_update();
}

float Cylinder3D::get_height() const {
	return height;
}

void Cylinder3D::set_segments(const int p_segments) {
	segments = p_segments > 4 ? p_segments : 4;
	_update();
}

int Cylinder3D::get_segments() const {
	return segments;
}

void Cylinder3D::set_rings(const int p_rings) {
	rings = p_rings > 0 ? p_rings : 0;
	_update();
}

int Cylinder3D::get_rings() const {
	return rings;
}

AABB Cylinder3D::get_aabb() const {

	return aabb;
}

DVector<Face3> Cylinder3D::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
}

Cylinder3D::Cylinder3D() {
	// defaults
	top_radius = 0.5;
	bottom_radius = 0.5;
	height = 1.0;
	segments = 16;
	rings = 8;

	// empty mesh until we update
	pending_update = true;
	mesh = VisualServer::get_singleton()->mesh_create();
	set_base(mesh);
	configured = false;
}

Cylinder3D::~Cylinder3D() {
	VisualServer::get_singleton()->free(mesh);
}

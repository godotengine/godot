/*************************************************************************/
/*  sphere3d.cpp                                                         */
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
#include "sphere3d.h"
#include "servers/visual_server.h"

void Sphere3D::_update() {
	int   i, j, prevrow, thisrow, point;
	float x, y, z;

	if (!is_inside_tree()) {
		pending_update = true; // try again once we enter our tree...
		return;
	}

	aabb = AABB(Vector3(-radius, height * (is_hemisphere ? 0.0 : -0.5), -radius), Vector3(2.0 * radius, height, 2.0 * radius));

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
		float v = j;
		float w;

		v /= (rings + 1);
		w = sin(Math_PI * v);
		y =  height * (is_hemisphere ? 1.0 : 0.5) * cos(Math_PI * v);

		for (i = 0; i <= segments; i++) {
			float u = i;
			u /= segments;

			x = sin(u * (Math_PI * 2.0));
			z = cos(u * (Math_PI * 2.0));

			if (is_hemisphere && y < 0.0) {
				points.push_back(Vector3(x * radius * w, 0.0, z * radius * w));
				normals.push_back(Vector3(0.0, -1.0, 0.0));
			} else {
				Vector3 p = Vector3(x * radius * w, y, z * radius * w);
				points.push_back(p);
				normals.push_back(p.normalized());
			};
			ADD_TANGENT(-z, 0.0, x, -1.0)
			uvs.push_back(Vector2(u, v));
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

void Sphere3D::_notification(int p_what) {

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

void Sphere3D::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("set_radius", "radius"), &Sphere3D::set_radius);
	ObjectTypeDB::bind_method(_MD("get_radius"), &Sphere3D::get_radius);
	ObjectTypeDB::bind_method(_MD("set_height", "height"), &Sphere3D::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"), &Sphere3D::get_height);

	ObjectTypeDB::bind_method(_MD("set_segments", "segments"), &Sphere3D::set_segments);
	ObjectTypeDB::bind_method(_MD("get_segments"), &Sphere3D::get_segments);
	ObjectTypeDB::bind_method(_MD("set_rings", "rings"), &Sphere3D::set_rings);
	ObjectTypeDB::bind_method(_MD("get_rings"), &Sphere3D::get_rings);

	ObjectTypeDB::bind_method(_MD("set_is_hemisphere", "is_hemisphere"), &Sphere3D::set_is_hemisphere);
	ObjectTypeDB::bind_method(_MD("get_is_hemisphere"), &Sphere3D::get_is_hemisphere);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius"), _SCS("set_radius"), _SCS("get_radius"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height"), _SCS("set_height"), _SCS("get_height"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "segments"), _SCS("set_segments"), _SCS("get_segments"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings"), _SCS("set_rings"), _SCS("get_rings"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_hemisphere"), _SCS("set_is_hemisphere"), _SCS("get_is_hemisphere"));
}

void Sphere3D::set_radius(const float p_radius) {
	radius = p_radius;
	_update();
}

float Sphere3D::get_radius() const {
	return radius;
}

void Sphere3D::set_height(const float p_height) {
	height = p_height;
	_update();
}

float Sphere3D::get_height() const {
	return height;
}

void Sphere3D::set_segments(const int p_segments) {
	segments = p_segments > 4 ? p_segments : 4;
	_update();
}

int Sphere3D::get_segments() const {
	return segments;
}

void Sphere3D::set_rings(const int p_rings) {
	rings = p_rings > 1 ? p_rings : 1;
	_update();
}

int Sphere3D::get_rings() const {
	return rings;
}

void Sphere3D::set_is_hemisphere(const bool p_is_hemisphere) {
	is_hemisphere = p_is_hemisphere;
	_update();
}

bool Sphere3D::get_is_hemisphere() const {
	return is_hemisphere;
}

AABB Sphere3D::get_aabb() const {

	return aabb;
}

DVector<Face3> Sphere3D::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
}

Sphere3D::Sphere3D() {
	// defaults
	radius = 0.5;
	height = 1.0;
	segments = 16;
	rings = 8;
	is_hemisphere = false;

	// empty mesh until we update
	pending_update = true;
	mesh = VisualServer::get_singleton()->mesh_create();
	set_base(mesh);
	configured = false;
}

Sphere3D::~Sphere3D() {
	VisualServer::get_singleton()->free(mesh);
}

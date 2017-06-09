/*************************************************************************/
/*  cube3d.cpp                                                           */
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
#include "prism3d.h"
#include "servers/visual_server.h"

void Prism3D::_update() {
	int   i, j, prevrow, thisrow, point;
	float x, y, z;
	float onethird = 1.0 / 3.0;
	float twothirds = 2.0 / 3.0;

	if (!is_inside_tree()) {
		pending_update = true; // try again once we enter our tree...
		return;
	}

	Vector3 start_pos = size * -0.5;
	aabb = AABB(start_pos, size);

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

	/* front + back */
	y = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= segments_h; j++) {
		float scale = (y - start_pos.y) / size.y;
		float scaled_size_x = size.x * scale;
		float start_x = start_pos.x;
		float offset_front = 0.0;
		float offset_back = 0.0;

		if (mode == PRISM_LEFT) {
			offset_back = (1.0 - scale) * onethird;
		} else if (mode == PRISM_CENTER) {
			start_x += (1.0 - scale) * size.x * 0.5;
			offset_front += (1.0 - scale) * onethird * 0.5;
			offset_back = (1.0 - scale) * onethird * 0.5;
		} else if (mode == PRISM_RIGHT) {
			start_x += (1.0 - scale) * size.x;
			offset_front += (1.0 - scale) * onethird;
		}

		x = 0.0;
		for (i = 0; i <= segments_w; i++) {
			float u = i;
			float v = j;
			u /= (3.0 * segments_w);
			v /= (2.0 * segments_h);

			u *= scale;

			/* front */
			points.push_back(Vector3(start_x + x, -y, -start_pos.z));  // double negative on the Z!
			normals.push_back(Vector3(0.0, 0.0, 1.0));
			ADD_TANGENT(-1.0, 0.0, 0.0, -1.0);
			uvs.push_back(Vector2(offset_front + u, v));
			point++;

			/* back */
			points.push_back(Vector3(start_x + scaled_size_x - x, -y, start_pos.z));
			normals.push_back(Vector3(0.0, 0.0, -1.0));
			ADD_TANGENT(1.0, 0.0, 0.0, -1.0);
			uvs.push_back(Vector2(twothirds + offset_back + u, v));
			point++;

			if (i>0 && j == 1) {
				int i2 = i * 2;

				/* front */
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				/* back */
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			} else if (i>0 && j>0) {
				int i2 = i * 2;

				/* front */
				indices.push_back(prevrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				/* back */
				indices.push_back(prevrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			};

			x += scale * size.x / segments_w;
		};

		y += size.y / segments_h;
		prevrow = thisrow;
		thisrow = point;
	};

	/* left + right */
	Vector3 normal_left, normal_right;

	if (mode == PRISM_LEFT) {
		normal_left = Vector3(-1.0, 0.0, 0.0);
		normal_right = Vector3(size.y, size.x, 0.0);		
	} else if (mode == PRISM_CENTER) {
		normal_left = Vector3(-size.y, size.x * 0.5, 0.0);
		normal_right = Vector3(size.y, size.x * 0.5, 0.0);		
	} else if (mode == PRISM_RIGHT) {
		normal_left = Vector3(-size.y, size.x, 0.0);
		normal_right = Vector3(1.0, 0.0, 0.0);
	};

	normal_left.normalize();
	normal_right.normalize();

	y = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= segments_h; j++) {
		float left, right;
		float scale = (y - start_pos.y) / size.y;

		if (mode == PRISM_LEFT) {
			left = start_pos.x;
		} else if (mode == PRISM_CENTER) {
			left = start_pos.x + (size.x * (1.0 - scale) * 0.5);
		} else if (mode == PRISM_RIGHT) {
			left = start_pos.x + (size.x * (1.0 - scale));
		};
		right = left + (size.x * scale);

		z = start_pos.z ;
		for (i = 0; i <= segments_d; i++) {
			float u = i;
			float v = j;
			u /= (3.0 * segments_d);
			v /= (2.0 * segments_h);

			/* right */
			points.push_back(Vector3(right, -y, -z));
			normals.push_back(normal_right);
			ADD_TANGENT(0.0, 0.0, 1.0, -1.0);
			uvs.push_back(Vector2(onethird + u, v));
			point++;

			/* left */
			points.push_back(Vector3(left, -y, z));
			normals.push_back(normal_left);
			ADD_TANGENT(0.0, 0.0, -1.0, -1.0);
			uvs.push_back(Vector2(u, 0.5 + v));
			point++;

			if (i>0 && j>0) {
				int i2 = i * 2;

				/* right */
				indices.push_back(prevrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				/* left */
				indices.push_back(prevrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			};

			z += size.z / segments_d;
		};

		y += size.y / segments_h;
		prevrow = thisrow;
		thisrow = point;
	};

	/* bottom */
	z = start_pos.z;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= segments_d; j++) {
		x = start_pos.x;
		for (i = 0; i <= segments_w; i++) {
			float u = i;
			float v = j;
			u /= (3.0 * segments_w);
			v /= (2.0 * segments_d);

			/* bottom */
			points.push_back(Vector3(x, start_pos.y, -z));
			normals.push_back(Vector3(0.0, -1.0, 0.0));
			ADD_TANGENT(-1.0, 0.0, 0.0, -1.0);
			uvs.push_back(Vector2(twothirds + u, 0.5 + v));
			point++;

			if (i>0 && j>0) {
				/* bottom */
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			};

			x += size.x / segments_w;
		};

		z += size.z / segments_d;
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

void Prism3D::_notification(int p_what) {

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

void Prism3D::_bind_methods() {
	BIND_CONSTANT(PRISM_LEFT);
	BIND_CONSTANT(PRISM_CENTER);
	BIND_CONSTANT(PRISM_RIGHT);

	ObjectTypeDB::bind_method(_MD("set_mode", "mode"), &Prism3D::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"), &Prism3D::get_mode);

	ObjectTypeDB::bind_method(_MD("set_size", "size"), &Prism3D::set_size);
	ObjectTypeDB::bind_method(_MD("get_size"), &Prism3D::get_size);

	ObjectTypeDB::bind_method(_MD("set_segments_width", "segments"), &Prism3D::set_segments_width);
	ObjectTypeDB::bind_method(_MD("get_segments_width"), &Prism3D::get_segments_width);
	ObjectTypeDB::bind_method(_MD("set_segments_height", "segments"), &Prism3D::set_segments_height);
	ObjectTypeDB::bind_method(_MD("get_segments_height"), &Prism3D::get_segments_height);
	ObjectTypeDB::bind_method(_MD("set_segments_depth", "segments"), &Prism3D::set_segments_depth);
	ObjectTypeDB::bind_method(_MD("get_segments_depth"), &Prism3D::get_segments_depth);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "left,center,right"), _SCS("set_mode"), _SCS("get_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), _SCS("set_size"), _SCS("get_size"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "segments_width"), _SCS("set_segments_width"), _SCS("get_segments_width"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "segments_height"), _SCS("set_segments_height"), _SCS("get_segments_height"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "segments_depth"), _SCS("set_segments_depth"), _SCS("get_segments_depth"));
}

void Prism3D::set_mode(const Prism3D::Mode p_mode) {
	mode = p_mode;
	_update();
}

Prism3D::Mode Prism3D::get_mode() {
	return mode;
}

void Prism3D::set_size(const Vector3 &p_size) {
	size = p_size;
	_update();
}

Vector3 Prism3D::get_size() const {
	return size;
}

void Prism3D::set_segments_width(const int p_segments) {
	segments_w = p_segments > 1 ? p_segments : 1;
	_update();
}

int Prism3D::get_segments_width() const {
	return segments_w;
}

void Prism3D::set_segments_height(const int p_segments) {
	segments_h = p_segments > 1 ? p_segments : 1;
	_update();
}

int Prism3D::get_segments_height() const {
	return segments_h;
}

void Prism3D::set_segments_depth(const int p_segments) {
	segments_d = p_segments > 1 ? p_segments : 1;
	_update();
}

int Prism3D::get_segments_depth() const {
	return segments_d;
}

AABB Prism3D::get_aabb() const {

	return aabb;
}

DVector<Face3> Prism3D::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();
}

Prism3D::Prism3D() {
	// defaults
	mode = PRISM_LEFT;
	size = Vector3(1.0, 1.0, 1.0);
	segments_w = 1;
	segments_h = 1;
	segments_d = 1;

	// empty mesh until we update
	pending_update = true;
	mesh = VisualServer::get_singleton()->mesh_create();
	set_base(mesh);
	configured = false;
}

Prism3D::~Prism3D() {
	VisualServer::get_singleton()->free(mesh);
}

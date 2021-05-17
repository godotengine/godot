/*************************************************************************/
/*  primitive_meshes.cpp                                                 */
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

#include "primitive_meshes.h"
#include "servers/rendering_server.h"

/**
  PrimitiveMesh
*/
void PrimitiveMesh::_update() const {
	Array arr;
	arr.resize(RS::ARRAY_MAX);
	_create_mesh_array(arr);

	Vector<Vector3> points = arr[RS::ARRAY_VERTEX];

	aabb = AABB();

	int pc = points.size();
	ERR_FAIL_COND(pc == 0);
	{
		const Vector3 *r = points.ptr();
		for (int i = 0; i < pc; i++) {
			if (i == 0) {
				aabb.position = r[i];
			} else {
				aabb.expand_to(r[i]);
			}
		}
	}

	Vector<int> indices = arr[RS::ARRAY_INDEX];

	if (flip_faces) {
		Vector<Vector3> normals = arr[RS::ARRAY_NORMAL];

		if (normals.size() && indices.size()) {
			{
				int nc = normals.size();
				Vector3 *w = normals.ptrw();
				for (int i = 0; i < nc; i++) {
					w[i] = -w[i];
				}
			}

			{
				int ic = indices.size();
				int *w = indices.ptrw();
				for (int i = 0; i < ic; i += 3) {
					SWAP(w[i + 0], w[i + 1]);
				}
			}
			arr[RS::ARRAY_NORMAL] = normals;
			arr[RS::ARRAY_INDEX] = indices;
		}
	}

	array_len = pc;
	index_array_len = indices.size();
	// in with the new
	RenderingServer::get_singleton()->mesh_clear(mesh);
	RenderingServer::get_singleton()->mesh_add_surface_from_arrays(mesh, (RenderingServer::PrimitiveType)primitive_type, arr);
	RenderingServer::get_singleton()->mesh_surface_set_material(mesh, 0, material.is_null() ? RID() : material->get_rid());

	pending_request = false;

	clear_cache();

	const_cast<PrimitiveMesh *>(this)->emit_changed();
}

void PrimitiveMesh::_request_update() {
	if (pending_request) {
		return;
	}
	_update();
}

int PrimitiveMesh::get_surface_count() const {
	if (pending_request) {
		_update();
	}
	return 1;
}

int PrimitiveMesh::surface_get_array_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, -1);
	if (pending_request) {
		_update();
	}

	return array_len;
}

int PrimitiveMesh::surface_get_array_index_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, -1);
	if (pending_request) {
		_update();
	}

	return index_array_len;
}

Array PrimitiveMesh::surface_get_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, 1, Array());
	if (pending_request) {
		_update();
	}

	return RenderingServer::get_singleton()->mesh_surface_get_arrays(mesh, 0);
}

Dictionary PrimitiveMesh::surface_get_lods(int p_surface) const {
	return Dictionary(); //not really supported
}

Array PrimitiveMesh::surface_get_blend_shape_arrays(int p_surface) const {
	return Array(); //not really supported
}

uint32_t PrimitiveMesh::surface_get_format(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, 0);

	return RS::ARRAY_FORMAT_VERTEX | RS::ARRAY_FORMAT_NORMAL | RS::ARRAY_FORMAT_TANGENT | RS::ARRAY_FORMAT_TEX_UV | RS::ARRAY_FORMAT_INDEX;
}

Mesh::PrimitiveType PrimitiveMesh::surface_get_primitive_type(int p_idx) const {
	return primitive_type;
}

void PrimitiveMesh::surface_set_material(int p_idx, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX(p_idx, 1);

	set_material(p_material);
}

Ref<Material> PrimitiveMesh::surface_get_material(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, nullptr);

	return material;
}

int PrimitiveMesh::get_blend_shape_count() const {
	return 0;
}

StringName PrimitiveMesh::get_blend_shape_name(int p_index) const {
	return StringName();
}

void PrimitiveMesh::set_blend_shape_name(int p_index, const StringName &p_name) {
}

AABB PrimitiveMesh::get_aabb() const {
	if (pending_request) {
		_update();
	}

	return aabb;
}

RID PrimitiveMesh::get_rid() const {
	if (pending_request) {
		_update();
	}
	return mesh;
}

void PrimitiveMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update"), &PrimitiveMesh::_update);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &PrimitiveMesh::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &PrimitiveMesh::get_material);

	ClassDB::bind_method(D_METHOD("get_mesh_arrays"), &PrimitiveMesh::get_mesh_arrays);

	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &PrimitiveMesh::set_custom_aabb);
	ClassDB::bind_method(D_METHOD("get_custom_aabb"), &PrimitiveMesh::get_custom_aabb);

	ClassDB::bind_method(D_METHOD("set_flip_faces", "flip_faces"), &PrimitiveMesh::set_flip_faces);
	ClassDB::bind_method(D_METHOD("get_flip_faces"), &PrimitiveMesh::get_flip_faces);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,StandardMaterial3D"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "custom_aabb", PROPERTY_HINT_NONE, ""), "set_custom_aabb", "get_custom_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_faces"), "set_flip_faces", "get_flip_faces");
}

void PrimitiveMesh::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (!pending_request) {
		// just apply it, else it'll happen when _update is called.
		RenderingServer::get_singleton()->mesh_surface_set_material(mesh, 0, material.is_null() ? RID() : material->get_rid());
		notify_property_list_changed();
		emit_changed();
	};
}

Ref<Material> PrimitiveMesh::get_material() const {
	return material;
}

Array PrimitiveMesh::get_mesh_arrays() const {
	return surface_get_arrays(0);
}

void PrimitiveMesh::set_custom_aabb(const AABB &p_custom) {
	custom_aabb = p_custom;
	RS::get_singleton()->mesh_set_custom_aabb(mesh, custom_aabb);
	emit_changed();
}

AABB PrimitiveMesh::get_custom_aabb() const {
	return custom_aabb;
}

void PrimitiveMesh::set_flip_faces(bool p_enable) {
	flip_faces = p_enable;
	_request_update();
}

bool PrimitiveMesh::get_flip_faces() const {
	return flip_faces;
}

PrimitiveMesh::PrimitiveMesh() {
	mesh = RenderingServer::get_singleton()->mesh_create();
}

PrimitiveMesh::~PrimitiveMesh() {
	RenderingServer::get_singleton()->free(mesh);
}

/**
	CapsuleMesh
*/

void CapsuleMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, y, z, u, v, w;
	float onethird = 1.0 / 3.0;
	float twothirds = 2.0 / 3.0;

	// note, this has been aligned with our collision shape but I've left the descriptions as top/middle/bottom

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	/* top hemisphere */
	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		v = j;

		v /= (rings + 1);
		w = sin(0.5 * Math_PI * v);
		y = radius * cos(0.5 * Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = -sin(u * Math_TAU);
			z = cos(u * Math_TAU);

			Vector3 p = Vector3(x * radius * w, y, -z * radius * w);
			points.push_back(p + Vector3(0.0, 0.5 * mid_height, 0.0));
			normals.push_back(p.normalized());
			ADD_TANGENT(z, 0.0, x, 1.0)
			uvs.push_back(Vector2(u, v * onethird));
			point++;

			if (i > 0 && j > 0) {
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

	/* cylinder */
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		v = j;
		v /= (rings + 1);

		y = mid_height * v;
		y = (mid_height * 0.5) - y;

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = -sin(u * Math_TAU);
			z = cos(u * Math_TAU);

			Vector3 p = Vector3(x * radius, y, -z * radius);
			points.push_back(p);
			normals.push_back(Vector3(x, 0.0, -z));
			ADD_TANGENT(z, 0.0, x, 1.0)
			uvs.push_back(Vector2(u, onethird + (v * onethird)));
			point++;

			if (i > 0 && j > 0) {
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

	/* bottom hemisphere */
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		v = j;

		v /= (rings + 1);
		v += 1.0;
		w = sin(0.5 * Math_PI * v);
		y = radius * cos(0.5 * Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			float u2 = i;
			u2 /= radial_segments;

			x = -sin(u2 * Math_TAU);
			z = cos(u2 * Math_TAU);

			Vector3 p = Vector3(x * radius * w, y, -z * radius * w);
			points.push_back(p + Vector3(0.0, -0.5 * mid_height, 0.0));
			normals.push_back(p.normalized());
			ADD_TANGENT(z, 0.0, x, 1.0)
			uvs.push_back(Vector2(u2, twothirds + ((v - 1.0) * onethird)));
			point++;

			if (i > 0 && j > 0) {
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

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void CapsuleMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CapsuleMesh::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CapsuleMesh::get_radius);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &CapsuleMesh::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &CapsuleMesh::get_mid_height);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "segments"), &CapsuleMesh::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &CapsuleMesh::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &CapsuleMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &CapsuleMesh::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_mid_height", "get_mid_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_rings", "get_rings");
}

void CapsuleMesh::set_radius(const float p_radius) {
	radius = p_radius;
	_request_update();
}

float CapsuleMesh::get_radius() const {
	return radius;
}

void CapsuleMesh::set_mid_height(const float p_mid_height) {
	mid_height = p_mid_height;
	_request_update();
}

float CapsuleMesh::get_mid_height() const {
	return mid_height;
}

void CapsuleMesh::set_radial_segments(const int p_segments) {
	radial_segments = p_segments > 4 ? p_segments : 4;
	_request_update();
}

int CapsuleMesh::get_radial_segments() const {
	return radial_segments;
}

void CapsuleMesh::set_rings(const int p_rings) {
	rings = p_rings > 1 ? p_rings : 1;
	_request_update();
}

int CapsuleMesh::get_rings() const {
	return rings;
}

CapsuleMesh::CapsuleMesh() {}

/**
  BoxMesh
*/

void BoxMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, y, z;
	float onethird = 1.0 / 3.0;
	float twothirds = 2.0 / 3.0;

	Vector3 start_pos = size * -0.5;

	// set our bounding box

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	// front + back
	y = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= subdivide_h + 1; j++) {
		x = start_pos.x;
		for (i = 0; i <= subdivide_w + 1; i++) {
			float u = i;
			float v = j;
			u /= (3.0 * (subdivide_w + 1.0));
			v /= (2.0 * (subdivide_h + 1.0));

			// front
			points.push_back(Vector3(x, -y, -start_pos.z)); // double negative on the Z!
			normals.push_back(Vector3(0.0, 0.0, 1.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(u, v));
			point++;

			// back
			points.push_back(Vector3(-x, -y, start_pos.z));
			normals.push_back(Vector3(0.0, 0.0, -1.0));
			ADD_TANGENT(-1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(twothirds + u, v));
			point++;

			if (i > 0 && j > 0) {
				int i2 = i * 2;

				// front
				indices.push_back(prevrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				// back
				indices.push_back(prevrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			};

			x += size.x / (subdivide_w + 1.0);
		};

		y += size.y / (subdivide_h + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	// left + right
	y = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (subdivide_h + 1); j++) {
		z = start_pos.z;
		for (i = 0; i <= (subdivide_d + 1); i++) {
			float u = i;
			float v = j;
			u /= (3.0 * (subdivide_d + 1.0));
			v /= (2.0 * (subdivide_h + 1.0));

			// right
			points.push_back(Vector3(-start_pos.x, -y, -z));
			normals.push_back(Vector3(1.0, 0.0, 0.0));
			ADD_TANGENT(0.0, 0.0, -1.0, 1.0);
			uvs.push_back(Vector2(onethird + u, v));
			point++;

			// left
			points.push_back(Vector3(start_pos.x, -y, z));
			normals.push_back(Vector3(-1.0, 0.0, 0.0));
			ADD_TANGENT(0.0, 0.0, 1.0, 1.0);
			uvs.push_back(Vector2(u, 0.5 + v));
			point++;

			if (i > 0 && j > 0) {
				int i2 = i * 2;

				// right
				indices.push_back(prevrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				// left
				indices.push_back(prevrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			};

			z += size.z / (subdivide_d + 1.0);
		};

		y += size.y / (subdivide_h + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	// top + bottom
	z = start_pos.z;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (subdivide_d + 1); j++) {
		x = start_pos.x;
		for (i = 0; i <= (subdivide_w + 1); i++) {
			float u = i;
			float v = j;
			u /= (3.0 * (subdivide_w + 1.0));
			v /= (2.0 * (subdivide_d + 1.0));

			// top
			points.push_back(Vector3(-x, -start_pos.y, -z));
			normals.push_back(Vector3(0.0, 1.0, 0.0));
			ADD_TANGENT(-1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(onethird + u, 0.5 + v));
			point++;

			// bottom
			points.push_back(Vector3(x, start_pos.y, -z));
			normals.push_back(Vector3(0.0, -1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(twothirds + u, 0.5 + v));
			point++;

			if (i > 0 && j > 0) {
				int i2 = i * 2;

				// top
				indices.push_back(prevrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2 - 2);
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				// bottom
				indices.push_back(prevrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			};

			x += size.x / (subdivide_w + 1.0);
		};

		z += size.z / (subdivide_d + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void BoxMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &BoxMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &BoxMesh::get_size);

	ClassDB::bind_method(D_METHOD("set_subdivide_width", "subdivide"), &BoxMesh::set_subdivide_width);
	ClassDB::bind_method(D_METHOD("get_subdivide_width"), &BoxMesh::get_subdivide_width);
	ClassDB::bind_method(D_METHOD("set_subdivide_height", "divisions"), &BoxMesh::set_subdivide_height);
	ClassDB::bind_method(D_METHOD("get_subdivide_height"), &BoxMesh::get_subdivide_height);
	ClassDB::bind_method(D_METHOD("set_subdivide_depth", "divisions"), &BoxMesh::set_subdivide_depth);
	ClassDB::bind_method(D_METHOD("get_subdivide_depth"), &BoxMesh::get_subdivide_depth);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_width", "get_subdivide_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_height", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_height", "get_subdivide_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_depth", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_depth", "get_subdivide_depth");
}

void BoxMesh::set_size(const Vector3 &p_size) {
	size = p_size;
	_request_update();
}

Vector3 BoxMesh::get_size() const {
	return size;
}

void BoxMesh::set_subdivide_width(const int p_divisions) {
	subdivide_w = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int BoxMesh::get_subdivide_width() const {
	return subdivide_w;
}

void BoxMesh::set_subdivide_height(const int p_divisions) {
	subdivide_h = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int BoxMesh::get_subdivide_height() const {
	return subdivide_h;
}

void BoxMesh::set_subdivide_depth(const int p_divisions) {
	subdivide_d = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int BoxMesh::get_subdivide_depth() const {
	return subdivide_d;
}

BoxMesh::BoxMesh() {}

/**
  CylinderMesh
*/

void CylinderMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, y, z, u, v, radius;

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		v = j;
		v /= (rings + 1);

		radius = top_radius + ((bottom_radius - top_radius) * v);

		y = height * v;
		y = (height * 0.5) - y;

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = sin(u * Math_TAU);
			z = cos(u * Math_TAU);

			Vector3 p = Vector3(x * radius, y, z * radius);
			points.push_back(p);
			normals.push_back(Vector3(x, 0.0, z));
			ADD_TANGENT(z, 0.0, -x, 1.0)
			uvs.push_back(Vector2(u, v * 0.5));
			point++;

			if (i > 0 && j > 0) {
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

		for (i = 0; i <= radial_segments; i++) {
			float r = i;
			r /= radial_segments;

			x = sin(r * Math_TAU);
			z = cos(r * Math_TAU);

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
		ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
		uvs.push_back(Vector2(0.75, 0.75));
		point++;

		for (i = 0; i <= radial_segments; i++) {
			float r = i;
			r /= radial_segments;

			x = sin(r * Math_TAU);
			z = cos(r * Math_TAU);

			u = 0.5 + ((x + 1.0) * 0.25);
			v = 1.0 - ((z + 1.0) * 0.25);

			Vector3 p = Vector3(x * bottom_radius, y, z * bottom_radius);
			points.push_back(p);
			normals.push_back(Vector3(0.0, -1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
			uvs.push_back(Vector2(u, v));
			point++;

			if (i > 0) {
				indices.push_back(thisrow);
				indices.push_back(point - 2);
				indices.push_back(point - 1);
			};
		};
	};

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void CylinderMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_top_radius", "radius"), &CylinderMesh::set_top_radius);
	ClassDB::bind_method(D_METHOD("get_top_radius"), &CylinderMesh::get_top_radius);
	ClassDB::bind_method(D_METHOD("set_bottom_radius", "radius"), &CylinderMesh::set_bottom_radius);
	ClassDB::bind_method(D_METHOD("get_bottom_radius"), &CylinderMesh::get_bottom_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &CylinderMesh::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CylinderMesh::get_height);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "segments"), &CylinderMesh::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &CylinderMesh::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &CylinderMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &CylinderMesh::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_top_radius", "get_top_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bottom_radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_bottom_radius", "get_bottom_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_rings", "get_rings");
}

void CylinderMesh::set_top_radius(const float p_radius) {
	top_radius = p_radius;
	_request_update();
}

float CylinderMesh::get_top_radius() const {
	return top_radius;
}

void CylinderMesh::set_bottom_radius(const float p_radius) {
	bottom_radius = p_radius;
	_request_update();
}

float CylinderMesh::get_bottom_radius() const {
	return bottom_radius;
}

void CylinderMesh::set_height(const float p_height) {
	height = p_height;
	_request_update();
}

float CylinderMesh::get_height() const {
	return height;
}

void CylinderMesh::set_radial_segments(const int p_segments) {
	radial_segments = p_segments > 4 ? p_segments : 4;
	_request_update();
}

int CylinderMesh::get_radial_segments() const {
	return radial_segments;
}

void CylinderMesh::set_rings(const int p_rings) {
	rings = p_rings > 0 ? p_rings : 0;
	_request_update();
}

int CylinderMesh::get_rings() const {
	return rings;
}

CylinderMesh::CylinderMesh() {}

/**
  PlaneMesh
*/

void PlaneMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, z;

	Size2 start_pos = size * -0.5;

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	/* top + bottom */
	z = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (subdivide_d + 1); j++) {
		x = start_pos.x;
		for (i = 0; i <= (subdivide_w + 1); i++) {
			float u = i;
			float v = j;
			u /= (subdivide_w + 1.0);
			v /= (subdivide_d + 1.0);

			points.push_back(Vector3(-x, 0.0, -z));
			normals.push_back(Vector3(0.0, 1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(1.0 - u, 1.0 - v)); /* 1.0 - uv to match orientation with Quad */
			point++;

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			};

			x += size.x / (subdivide_w + 1.0);
		};

		z += size.y / (subdivide_d + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void PlaneMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &PlaneMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &PlaneMesh::get_size);

	ClassDB::bind_method(D_METHOD("set_subdivide_width", "subdivide"), &PlaneMesh::set_subdivide_width);
	ClassDB::bind_method(D_METHOD("get_subdivide_width"), &PlaneMesh::get_subdivide_width);
	ClassDB::bind_method(D_METHOD("set_subdivide_depth", "subdivide"), &PlaneMesh::set_subdivide_depth);
	ClassDB::bind_method(D_METHOD("get_subdivide_depth"), &PlaneMesh::get_subdivide_depth);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_width", "get_subdivide_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_depth", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_depth", "get_subdivide_depth");
}

void PlaneMesh::set_size(const Size2 &p_size) {
	size = p_size;
	_request_update();
}

Size2 PlaneMesh::get_size() const {
	return size;
}

void PlaneMesh::set_subdivide_width(const int p_divisions) {
	subdivide_w = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int PlaneMesh::get_subdivide_width() const {
	return subdivide_w;
}

void PlaneMesh::set_subdivide_depth(const int p_divisions) {
	subdivide_d = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int PlaneMesh::get_subdivide_depth() const {
	return subdivide_d;
}

PlaneMesh::PlaneMesh() {}

/**
  PrismMesh
*/

void PrismMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, y, z;
	float onethird = 1.0 / 3.0;
	float twothirds = 2.0 / 3.0;

	Vector3 start_pos = size * -0.5;

	// set our bounding box

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	/* front + back */
	y = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (subdivide_h + 1); j++) {
		float scale = (y - start_pos.y) / size.y;
		float scaled_size_x = size.x * scale;
		float start_x = start_pos.x + (1.0 - scale) * size.x * left_to_right;
		float offset_front = (1.0 - scale) * onethird * left_to_right;
		float offset_back = (1.0 - scale) * onethird * (1.0 - left_to_right);

		x = 0.0;
		for (i = 0; i <= (subdivide_w + 1); i++) {
			float u = i;
			float v = j;
			u /= (3.0 * (subdivide_w + 1.0));
			v /= (2.0 * (subdivide_h + 1.0));

			u *= scale;

			/* front */
			points.push_back(Vector3(start_x + x, -y, -start_pos.z)); // double negative on the Z!
			normals.push_back(Vector3(0.0, 0.0, 1.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(offset_front + u, v));
			point++;

			/* back */
			points.push_back(Vector3(start_x + scaled_size_x - x, -y, start_pos.z));
			normals.push_back(Vector3(0.0, 0.0, -1.0));
			ADD_TANGENT(-1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(twothirds + offset_back + u, v));
			point++;

			if (i > 0 && j == 1) {
				int i2 = i * 2;

				/* front */
				indices.push_back(prevrow + i2);
				indices.push_back(thisrow + i2);
				indices.push_back(thisrow + i2 - 2);

				/* back */
				indices.push_back(prevrow + i2 + 1);
				indices.push_back(thisrow + i2 + 1);
				indices.push_back(thisrow + i2 - 1);
			} else if (i > 0 && j > 0) {
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

			x += scale * size.x / (subdivide_w + 1.0);
		};

		y += size.y / (subdivide_h + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	/* left + right */
	Vector3 normal_left, normal_right;

	normal_left = Vector3(-size.y, size.x * left_to_right, 0.0);
	normal_right = Vector3(size.y, size.x * (1.0 - left_to_right), 0.0);
	normal_left.normalize();
	normal_right.normalize();

	y = start_pos.y;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (subdivide_h + 1); j++) {
		float left, right;
		float scale = (y - start_pos.y) / size.y;

		left = start_pos.x + (size.x * (1.0 - scale) * left_to_right);
		right = left + (size.x * scale);

		z = start_pos.z;
		for (i = 0; i <= (subdivide_d + 1); i++) {
			float u = i;
			float v = j;
			u /= (3.0 * (subdivide_d + 1.0));
			v /= (2.0 * (subdivide_h + 1.0));

			/* right */
			points.push_back(Vector3(right, -y, -z));
			normals.push_back(normal_right);
			ADD_TANGENT(0.0, 0.0, -1.0, 1.0);
			uvs.push_back(Vector2(onethird + u, v));
			point++;

			/* left */
			points.push_back(Vector3(left, -y, z));
			normals.push_back(normal_left);
			ADD_TANGENT(0.0, 0.0, 1.0, 1.0);
			uvs.push_back(Vector2(u, 0.5 + v));
			point++;

			if (i > 0 && j > 0) {
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

			z += size.z / (subdivide_d + 1.0);
		};

		y += size.y / (subdivide_h + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	/* bottom */
	z = start_pos.z;
	thisrow = point;
	prevrow = 0;
	for (j = 0; j <= (subdivide_d + 1); j++) {
		x = start_pos.x;
		for (i = 0; i <= (subdivide_w + 1); i++) {
			float u = i;
			float v = j;
			u /= (3.0 * (subdivide_w + 1.0));
			v /= (2.0 * (subdivide_d + 1.0));

			/* bottom */
			points.push_back(Vector3(x, start_pos.y, -z));
			normals.push_back(Vector3(0.0, -1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0);
			uvs.push_back(Vector2(twothirds + u, 0.5 + v));
			point++;

			if (i > 0 && j > 0) {
				/* bottom */
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			};

			x += size.x / (subdivide_w + 1.0);
		};

		z += size.z / (subdivide_d + 1.0);
		prevrow = thisrow;
		thisrow = point;
	};

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void PrismMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_left_to_right", "left_to_right"), &PrismMesh::set_left_to_right);
	ClassDB::bind_method(D_METHOD("get_left_to_right"), &PrismMesh::get_left_to_right);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &PrismMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &PrismMesh::get_size);

	ClassDB::bind_method(D_METHOD("set_subdivide_width", "segments"), &PrismMesh::set_subdivide_width);
	ClassDB::bind_method(D_METHOD("get_subdivide_width"), &PrismMesh::get_subdivide_width);
	ClassDB::bind_method(D_METHOD("set_subdivide_height", "segments"), &PrismMesh::set_subdivide_height);
	ClassDB::bind_method(D_METHOD("get_subdivide_height"), &PrismMesh::get_subdivide_height);
	ClassDB::bind_method(D_METHOD("set_subdivide_depth", "segments"), &PrismMesh::set_subdivide_depth);
	ClassDB::bind_method(D_METHOD("get_subdivide_depth"), &PrismMesh::get_subdivide_depth);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "left_to_right", PROPERTY_HINT_RANGE, "-2.0,2.0,0.1"), "set_left_to_right", "get_left_to_right");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_width", "get_subdivide_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_height", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_height", "get_subdivide_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_depth", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_depth", "get_subdivide_depth");
}

void PrismMesh::set_left_to_right(const float p_left_to_right) {
	left_to_right = p_left_to_right;
	_request_update();
}

float PrismMesh::get_left_to_right() const {
	return left_to_right;
}

void PrismMesh::set_size(const Vector3 &p_size) {
	size = p_size;
	_request_update();
}

Vector3 PrismMesh::get_size() const {
	return size;
}

void PrismMesh::set_subdivide_width(const int p_divisions) {
	subdivide_w = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int PrismMesh::get_subdivide_width() const {
	return subdivide_w;
}

void PrismMesh::set_subdivide_height(const int p_divisions) {
	subdivide_h = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int PrismMesh::get_subdivide_height() const {
	return subdivide_h;
}

void PrismMesh::set_subdivide_depth(const int p_divisions) {
	subdivide_d = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int PrismMesh::get_subdivide_depth() const {
	return subdivide_d;
}

PrismMesh::PrismMesh() {}

/**
  QuadMesh
*/

void QuadMesh::_create_mesh_array(Array &p_arr) const {
	Vector<Vector3> faces;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;

	faces.resize(6);
	normals.resize(6);
	tangents.resize(6 * 4);
	uvs.resize(6);

	Vector2 _size = Vector2(size.x / 2.0f, size.y / 2.0f);

	Vector3 quad_faces[4] = {
		Vector3(-_size.x, -_size.y, 0),
		Vector3(-_size.x, _size.y, 0),
		Vector3(_size.x, _size.y, 0),
		Vector3(_size.x, -_size.y, 0),
	};

	static const int indices[6] = {
		0, 1, 2,
		0, 2, 3
	};

	for (int i = 0; i < 6; i++) {
		int j = indices[i];
		faces.set(i, quad_faces[j]);
		normals.set(i, Vector3(0, 0, 1));
		tangents.set(i * 4 + 0, 1.0);
		tangents.set(i * 4 + 1, 0.0);
		tangents.set(i * 4 + 2, 0.0);
		tangents.set(i * 4 + 3, 1.0);

		static const Vector2 quad_uv[4] = {
			Vector2(0, 1),
			Vector2(0, 0),
			Vector2(1, 0),
			Vector2(1, 1),
		};

		uvs.set(i, quad_uv[j]);
	}

	p_arr[RS::ARRAY_VERTEX] = faces;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
}

void QuadMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &QuadMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &QuadMesh::get_size);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
}

QuadMesh::QuadMesh() {
	primitive_type = PRIMITIVE_TRIANGLES;
}

void QuadMesh::set_size(const Size2 &p_size) {
	size = p_size;
	_request_update();
}

Size2 QuadMesh::get_size() const {
	return size;
}

/**
  SphereMesh
*/

void SphereMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, y, z;

	// set our bounding box

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		float v = j;
		float w;

		v /= (rings + 1);
		w = sin(Math_PI * v);
		y = height * (is_hemisphere ? 1.0 : 0.5) * cos(Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			float u = i;
			u /= radial_segments;

			x = sin(u * Math_TAU);
			z = cos(u * Math_TAU);

			if (is_hemisphere && y < 0.0) {
				points.push_back(Vector3(x * radius * w, 0.0, z * radius * w));
				normals.push_back(Vector3(0.0, -1.0, 0.0));
			} else {
				Vector3 p = Vector3(x * radius * w, y, z * radius * w);
				points.push_back(p);
				normals.push_back(p.normalized());
			};
			ADD_TANGENT(z, 0.0, -x, 1.0)
			uvs.push_back(Vector2(u, v));
			point++;

			if (i > 0 && j > 0) {
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

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void SphereMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &SphereMesh::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &SphereMesh::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &SphereMesh::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &SphereMesh::get_height);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "radial_segments"), &SphereMesh::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &SphereMesh::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &SphereMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &SphereMesh::get_rings);

	ClassDB::bind_method(D_METHOD("set_is_hemisphere", "is_hemisphere"), &SphereMesh::set_is_hemisphere);
	ClassDB::bind_method(D_METHOD("get_is_hemisphere"), &SphereMesh::get_is_hemisphere);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_rings", "get_rings");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_hemisphere"), "set_is_hemisphere", "get_is_hemisphere");
}

void SphereMesh::set_radius(const float p_radius) {
	radius = p_radius;
	_request_update();
}

float SphereMesh::get_radius() const {
	return radius;
}

void SphereMesh::set_height(const float p_height) {
	height = p_height;
	_request_update();
}

float SphereMesh::get_height() const {
	return height;
}

void SphereMesh::set_radial_segments(const int p_radial_segments) {
	radial_segments = p_radial_segments > 4 ? p_radial_segments : 4;
	_request_update();
}

int SphereMesh::get_radial_segments() const {
	return radial_segments;
}

void SphereMesh::set_rings(const int p_rings) {
	rings = p_rings > 1 ? p_rings : 1;
	_request_update();
}

int SphereMesh::get_rings() const {
	return rings;
}

void SphereMesh::set_is_hemisphere(const bool p_is_hemisphere) {
	is_hemisphere = p_is_hemisphere;
	_request_update();
}

bool SphereMesh::get_is_hemisphere() const {
	return is_hemisphere;
}

SphereMesh::SphereMesh() {}

/**
  PointMesh
*/

void PointMesh::_create_mesh_array(Array &p_arr) const {
	Vector<Vector3> faces;
	faces.resize(1);
	faces.set(0, Vector3(0.0, 0.0, 0.0));

	p_arr[RS::ARRAY_VERTEX] = faces;
}

PointMesh::PointMesh() {
	primitive_type = PRIMITIVE_POINTS;
}
// TUBE TRAIL

void TubeTrailMesh::set_radius(const float p_radius) {
	radius = p_radius;
	_request_update();
}
float TubeTrailMesh::get_radius() const {
	return radius;
}

void TubeTrailMesh::set_radial_steps(const int p_radial_steps) {
	ERR_FAIL_COND(p_radial_steps < 3 || p_radial_steps > 128);
	radial_steps = p_radial_steps;
	_request_update();
}
int TubeTrailMesh::get_radial_steps() const {
	return radial_steps;
}

void TubeTrailMesh::set_sections(const int p_sections) {
	ERR_FAIL_COND(p_sections < 2 || p_sections > 128);
	sections = p_sections;
	_request_update();
}
int TubeTrailMesh::get_sections() const {
	return sections;
}

void TubeTrailMesh::set_section_length(float p_section_length) {
	section_length = p_section_length;
	_request_update();
}
float TubeTrailMesh::get_section_length() const {
	return section_length;
}

void TubeTrailMesh::set_section_rings(const int p_section_rings) {
	ERR_FAIL_COND(p_section_rings < 1 || p_section_rings > 1024);
	section_rings = p_section_rings;
	_request_update();
}
int TubeTrailMesh::get_section_rings() const {
	return section_rings;
}

void TubeTrailMesh::set_curve(const Ref<Curve> &p_curve) {
	if (curve == p_curve) {
		return;
	}
	if (curve.is_valid()) {
		curve->disconnect("changed", callable_mp(this, &TubeTrailMesh::_curve_changed));
	}
	curve = p_curve;
	if (curve.is_valid()) {
		curve->connect("changed", callable_mp(this, &TubeTrailMesh::_curve_changed));
	}
	_request_update();
}
Ref<Curve> TubeTrailMesh::get_curve() const {
	return curve;
}

void TubeTrailMesh::_curve_changed() {
	_request_update();
}
int TubeTrailMesh::get_builtin_bind_pose_count() const {
	return sections + 1;
}

Transform TubeTrailMesh::get_builtin_bind_pose(int p_index) const {
	float depth = section_length * sections;

	Transform xform;
	xform.origin.y = depth / 2.0 - section_length * float(p_index);
	xform.origin.y = -xform.origin.y; //bind is an inverse transform, so negate y

	return xform;
}

void TubeTrailMesh::_create_mesh_array(Array &p_arr) const {
	PackedVector3Array points;
	PackedVector3Array normals;
	PackedFloat32Array tangents;
	PackedVector2Array uvs;
	PackedInt32Array bone_indices;
	PackedFloat32Array bone_weights;
	PackedInt32Array indices;

	int point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	int thisrow = 0;
	int prevrow = 0;

	int total_rings = section_rings * sections;
	float depth = section_length * sections;

	for (int j = 0; j <= total_rings; j++) {
		float v = j;
		v /= total_rings;

		float y = depth * v;
		y = (depth * 0.5) - y;

		int bone = j / section_rings;
		float blend = 1.0 - float(j % section_rings) / float(section_rings);

		for (int i = 0; i <= radial_steps; i++) {
			float u = i;
			u /= radial_steps;

			float r = radius;
			if (curve.is_valid() && curve->get_point_count() > 0) {
				r *= curve->interpolate_baked(v);
			}
			float x = sin(u * Math_TAU);
			float z = cos(u * Math_TAU);

			Vector3 p = Vector3(x * r, y, z * r);
			points.push_back(p);
			normals.push_back(Vector3(x, 0, z));
			ADD_TANGENT(z, 0.0, -x, 1.0)
			uvs.push_back(Vector2(u, v * 0.5));
			point++;
			{
				bone_indices.push_back(bone);
				bone_indices.push_back(MIN(sections, bone + 1));
				bone_indices.push_back(0);
				bone_indices.push_back(0);

				bone_weights.push_back(blend);
				bone_weights.push_back(1.0 - blend);
				bone_weights.push_back(0);
				bone_weights.push_back(0);
			}

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			}
		}

		prevrow = thisrow;
		thisrow = point;
	}

	// add top
	float scale_pos = 1.0;
	if (curve.is_valid() && curve->get_point_count() > 0) {
		scale_pos = curve->interpolate_baked(0);
	}

	if (scale_pos > CMP_EPSILON) {
		float y = depth * 0.5;

		thisrow = point;
		points.push_back(Vector3(0.0, y, 0));
		normals.push_back(Vector3(0.0, 1.0, 0.0));
		ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
		uvs.push_back(Vector2(0.25, 0.75));
		point++;

		bone_indices.push_back(0);
		bone_indices.push_back(0);
		bone_indices.push_back(0);
		bone_indices.push_back(0);

		bone_weights.push_back(1.0);
		bone_weights.push_back(0);
		bone_weights.push_back(0);
		bone_weights.push_back(0);

		float rm = radius * scale_pos;

		for (int i = 0; i <= radial_steps; i++) {
			float r = i;
			r /= radial_steps;

			float x = sin(r * Math_TAU);
			float z = cos(r * Math_TAU);

			float u = ((x + 1.0) * 0.25);
			float v = 0.5 + ((z + 1.0) * 0.25);

			Vector3 p = Vector3(x * rm, y, z * rm);
			points.push_back(p);
			normals.push_back(Vector3(0.0, 1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
			uvs.push_back(Vector2(u, v));
			point++;

			bone_indices.push_back(0);
			bone_indices.push_back(0);
			bone_indices.push_back(0);
			bone_indices.push_back(0);

			bone_weights.push_back(1.0);
			bone_weights.push_back(0);
			bone_weights.push_back(0);
			bone_weights.push_back(0);

			if (i > 0) {
				indices.push_back(thisrow);
				indices.push_back(point - 1);
				indices.push_back(point - 2);
			};
		};
	};

	float scale_neg = 1.0;
	if (curve.is_valid() && curve->get_point_count() > 0) {
		scale_neg = curve->interpolate_baked(1.0);
	}

	// add bottom
	if (scale_neg > CMP_EPSILON) {
		float y = depth * -0.5;

		thisrow = point;
		points.push_back(Vector3(0.0, y, 0.0));
		normals.push_back(Vector3(0.0, -1.0, 0.0));
		ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
		uvs.push_back(Vector2(0.75, 0.75));
		point++;

		bone_indices.push_back(sections);
		bone_indices.push_back(0);
		bone_indices.push_back(0);
		bone_indices.push_back(0);

		bone_weights.push_back(1.0);
		bone_weights.push_back(0);
		bone_weights.push_back(0);
		bone_weights.push_back(0);

		float rm = radius * scale_neg;

		for (int i = 0; i <= radial_steps; i++) {
			float r = i;
			r /= radial_steps;

			float x = sin(r * Math_TAU);
			float z = cos(r * Math_TAU);

			float u = 0.5 + ((x + 1.0) * 0.25);
			float v = 1.0 - ((z + 1.0) * 0.25);

			Vector3 p = Vector3(x * rm, y, z * rm);
			points.push_back(p);
			normals.push_back(Vector3(0.0, -1.0, 0.0));
			ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
			uvs.push_back(Vector2(u, v));
			point++;

			bone_indices.push_back(sections);
			bone_indices.push_back(0);
			bone_indices.push_back(0);
			bone_indices.push_back(0);

			bone_weights.push_back(1.0);
			bone_weights.push_back(0);
			bone_weights.push_back(0);
			bone_weights.push_back(0);

			if (i > 0) {
				indices.push_back(thisrow);
				indices.push_back(point - 2);
				indices.push_back(point - 1);
			};
		};
	};

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_BONES] = bone_indices;
	p_arr[RS::ARRAY_WEIGHTS] = bone_weights;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void TubeTrailMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &TubeTrailMesh::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &TubeTrailMesh::get_radius);

	ClassDB::bind_method(D_METHOD("set_radial_steps", "radial_steps"), &TubeTrailMesh::set_radial_steps);
	ClassDB::bind_method(D_METHOD("get_radial_steps"), &TubeTrailMesh::get_radial_steps);

	ClassDB::bind_method(D_METHOD("set_sections", "sections"), &TubeTrailMesh::set_sections);
	ClassDB::bind_method(D_METHOD("get_sections"), &TubeTrailMesh::get_sections);

	ClassDB::bind_method(D_METHOD("set_section_length", "section_length"), &TubeTrailMesh::set_section_length);
	ClassDB::bind_method(D_METHOD("get_section_length"), &TubeTrailMesh::get_section_length);

	ClassDB::bind_method(D_METHOD("set_section_rings", "section_rings"), &TubeTrailMesh::set_section_rings);
	ClassDB::bind_method(D_METHOD("get_section_rings"), &TubeTrailMesh::get_section_rings);

	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &TubeTrailMesh::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &TubeTrailMesh::get_curve);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_radius", "get_radius");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_steps", PROPERTY_HINT_RANGE, "3,128,1"), "set_radial_steps", "get_radial_steps");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sections", PROPERTY_HINT_RANGE, "2,128,1"), "set_sections", "get_sections");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "section_length", PROPERTY_HINT_RANGE, "0.001,1024.0,0.001,or_greater"), "set_section_length", "get_section_length");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "section_rings", PROPERTY_HINT_RANGE, "1,128,1"), "set_section_rings", "get_section_rings");

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve", "get_curve");
}

TubeTrailMesh::TubeTrailMesh() {
}

// TUBE TRAIL

void RibbonTrailMesh::set_shape(Shape p_shape) {
	shape = p_shape;
	_request_update();
}
RibbonTrailMesh::Shape RibbonTrailMesh::get_shape() const {
	return shape;
}

void RibbonTrailMesh::set_size(const float p_size) {
	size = p_size;
	_request_update();
}
float RibbonTrailMesh::get_size() const {
	return size;
}

void RibbonTrailMesh::set_sections(const int p_sections) {
	ERR_FAIL_COND(p_sections < 2 || p_sections > 128);
	sections = p_sections;
	_request_update();
}
int RibbonTrailMesh::get_sections() const {
	return sections;
}

void RibbonTrailMesh::set_section_length(float p_section_length) {
	section_length = p_section_length;
	_request_update();
}
float RibbonTrailMesh::get_section_length() const {
	return section_length;
}

void RibbonTrailMesh::set_section_segments(const int p_section_segments) {
	ERR_FAIL_COND(p_section_segments < 1 || p_section_segments > 1024);
	section_segments = p_section_segments;
	_request_update();
}
int RibbonTrailMesh::get_section_segments() const {
	return section_segments;
}

void RibbonTrailMesh::set_curve(const Ref<Curve> &p_curve) {
	if (curve == p_curve) {
		return;
	}
	if (curve.is_valid()) {
		curve->disconnect("changed", callable_mp(this, &RibbonTrailMesh::_curve_changed));
	}
	curve = p_curve;
	if (curve.is_valid()) {
		curve->connect("changed", callable_mp(this, &RibbonTrailMesh::_curve_changed));
	}
	_request_update();
}
Ref<Curve> RibbonTrailMesh::get_curve() const {
	return curve;
}

void RibbonTrailMesh::_curve_changed() {
	_request_update();
}
int RibbonTrailMesh::get_builtin_bind_pose_count() const {
	return sections + 1;
}

Transform RibbonTrailMesh::get_builtin_bind_pose(int p_index) const {
	float depth = section_length * sections;

	Transform xform;
	xform.origin.y = depth / 2.0 - section_length * float(p_index);
	xform.origin.y = -xform.origin.y; //bind is an inverse transform, so negate y

	return xform;
}

void RibbonTrailMesh::_create_mesh_array(Array &p_arr) const {
	PackedVector3Array points;
	PackedVector3Array normals;
	PackedFloat32Array tangents;
	PackedVector2Array uvs;
	PackedInt32Array bone_indices;
	PackedFloat32Array bone_weights;
	PackedInt32Array indices;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	int total_segments = section_segments * sections;
	float depth = section_length * sections;

	for (int j = 0; j <= total_segments; j++) {
		float v = j;
		v /= total_segments;

		float y = depth * v;
		y = (depth * 0.5) - y;

		int bone = j / section_segments;
		float blend = 1.0 - float(j % section_segments) / float(section_segments);

		float s = size;

		if (curve.is_valid() && curve->get_point_count() > 0) {
			s *= curve->interpolate_baked(v);
		}

		points.push_back(Vector3(-s * 0.5, y, 0));
		points.push_back(Vector3(+s * 0.5, y, 0));
		if (shape == SHAPE_CROSS) {
			points.push_back(Vector3(0, y, -s * 0.5));
			points.push_back(Vector3(0, y, +s * 0.5));
		}

		normals.push_back(Vector3(0, 0, 1));
		normals.push_back(Vector3(0, 0, 1));
		if (shape == SHAPE_CROSS) {
			normals.push_back(Vector3(1, 0, 0));
			normals.push_back(Vector3(1, 0, 0));
		}

		uvs.push_back(Vector2(0, v));
		uvs.push_back(Vector2(1, v));
		if (shape == SHAPE_CROSS) {
			uvs.push_back(Vector2(0, v));
			uvs.push_back(Vector2(1, v));
		}

		ADD_TANGENT(0.0, 1.0, 0.0, 1.0)
		ADD_TANGENT(0.0, 1.0, 0.0, 1.0)
		if (shape == SHAPE_CROSS) {
			ADD_TANGENT(0.0, 1.0, 0.0, 1.0)
			ADD_TANGENT(0.0, 1.0, 0.0, 1.0)
		}

		for (int i = 0; i < (shape == SHAPE_CROSS ? 4 : 2); i++) {
			bone_indices.push_back(bone);
			bone_indices.push_back(MIN(sections, bone + 1));
			bone_indices.push_back(0);
			bone_indices.push_back(0);

			bone_weights.push_back(blend);
			bone_weights.push_back(1.0 - blend);
			bone_weights.push_back(0);
			bone_weights.push_back(0);
		}

		if (j > 0) {
			if (shape == SHAPE_CROSS) {
				int base = j * 4 - 4;
				indices.push_back(base + 0);
				indices.push_back(base + 1);
				indices.push_back(base + 4);

				indices.push_back(base + 1);
				indices.push_back(base + 5);
				indices.push_back(base + 4);

				indices.push_back(base + 2);
				indices.push_back(base + 3);
				indices.push_back(base + 6);

				indices.push_back(base + 3);
				indices.push_back(base + 7);
				indices.push_back(base + 6);
			} else {
				int base = j * 2 - 2;
				indices.push_back(base + 0);
				indices.push_back(base + 1);
				indices.push_back(base + 2);

				indices.push_back(base + 1);
				indices.push_back(base + 3);
				indices.push_back(base + 2);
			}
		}
	}

	p_arr[RS::ARRAY_VERTEX] = points;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_BONES] = bone_indices;
	p_arr[RS::ARRAY_WEIGHTS] = bone_weights;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void RibbonTrailMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &RibbonTrailMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &RibbonTrailMesh::get_size);

	ClassDB::bind_method(D_METHOD("set_sections", "sections"), &RibbonTrailMesh::set_sections);
	ClassDB::bind_method(D_METHOD("get_sections"), &RibbonTrailMesh::get_sections);

	ClassDB::bind_method(D_METHOD("set_section_length", "section_length"), &RibbonTrailMesh::set_section_length);
	ClassDB::bind_method(D_METHOD("get_section_length"), &RibbonTrailMesh::get_section_length);

	ClassDB::bind_method(D_METHOD("set_section_segments", "section_segments"), &RibbonTrailMesh::set_section_segments);
	ClassDB::bind_method(D_METHOD("get_section_segments"), &RibbonTrailMesh::get_section_segments);

	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &RibbonTrailMesh::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &RibbonTrailMesh::get_curve);

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &RibbonTrailMesh::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &RibbonTrailMesh::get_shape);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "shape", PROPERTY_HINT_ENUM, "Flat,Cross"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "size", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sections", PROPERTY_HINT_RANGE, "2,128,1"), "set_sections", "get_sections");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "section_length", PROPERTY_HINT_RANGE, "0.001,1024.0,0.001,or_greater"), "set_section_length", "get_section_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "section_segments", PROPERTY_HINT_RANGE, "1,128,1"), "set_section_segments", "get_section_segments");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve", "get_curve");

	BIND_ENUM_CONSTANT(SHAPE_FLAT)
	BIND_ENUM_CONSTANT(SHAPE_CROSS)
}

RibbonTrailMesh::RibbonTrailMesh() {
}

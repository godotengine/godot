/**************************************************************************/
/*  primitive_meshes.cpp                                                  */
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

#include "primitive_meshes.h"

#include "core/core_string_names.h"
#include "core/os/main_loop.h"
#include "scene/resources/theme.h"
#include "servers/visual_server.h"
#include "thirdparty/misc/clipper.hpp"
#include "thirdparty/misc/triangulator.h"

/**
  PrimitiveMesh
*/
void PrimitiveMesh::_update() const {
	Array arr;
	arr.resize(VS::ARRAY_MAX);
	_create_mesh_array(arr);

	PoolVector<Vector3> points = arr[VS::ARRAY_VERTEX];

	aabb = AABB();

	int pc = points.size();
	ERR_FAIL_COND(pc == 0);
	{
		PoolVector<Vector3>::Read r = points.read();
		for (int i = 0; i < pc; i++) {
			if (i == 0) {
				aabb.position = r[i];
			} else {
				aabb.expand_to(r[i]);
			}
		}
	}

	if (flip_faces) {
		PoolVector<Vector3> normals = arr[VS::ARRAY_NORMAL];
		PoolVector<int> indices = arr[VS::ARRAY_INDEX];
		if (normals.size() && indices.size()) {
			{
				int nc = normals.size();
				PoolVector<Vector3>::Write w = normals.write();
				for (int i = 0; i < nc; i++) {
					w[i] = -w[i];
				}
			}

			{
				int ic = indices.size();
				PoolVector<int>::Write w = indices.write();
				for (int i = 0; i < ic; i += 3) {
					SWAP(w[i + 0], w[i + 1]);
				}
			}
			arr[VS::ARRAY_NORMAL] = normals;
			arr[VS::ARRAY_INDEX] = indices;
		}
	}

	// in with the new
	VisualServer::get_singleton()->mesh_clear(mesh);
	VisualServer::get_singleton()->mesh_add_surface_from_arrays(mesh, (VisualServer::PrimitiveType)primitive_type, arr);
	VisualServer::get_singleton()->mesh_surface_set_material(mesh, 0, material.is_null() ? RID() : material->get_rid());

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

	return VisualServer::get_singleton()->mesh_surface_get_array_len(mesh, 0);
}

int PrimitiveMesh::surface_get_array_index_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, -1);
	if (pending_request) {
		_update();
	}

	return VisualServer::get_singleton()->mesh_surface_get_array_index_len(mesh, 0);
}

Array PrimitiveMesh::surface_get_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, 1, Array());
	if (pending_request) {
		_update();
	}

	return VisualServer::get_singleton()->mesh_surface_get_arrays(mesh, 0);
}

Array PrimitiveMesh::surface_get_blend_shape_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, 1, Array());
	if (pending_request) {
		_update();
	}

	return Array();
}

uint32_t PrimitiveMesh::surface_get_format(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, 0);
	if (pending_request) {
		_update();
	}

	return VisualServer::get_singleton()->mesh_surface_get_format(mesh, 0);
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

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SpatialMaterial,ORMSpatialMaterial,ShaderMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "custom_aabb", PROPERTY_HINT_NONE, ""), "set_custom_aabb", "get_custom_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_faces"), "set_flip_faces", "get_flip_faces");
}

void PrimitiveMesh::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (!pending_request) {
		// just apply it, else it'll happen when _update is called.
		VisualServer::get_singleton()->mesh_surface_set_material(mesh, 0, material.is_null() ? RID() : material->get_rid());
		_change_notify();
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
	VS::get_singleton()->mesh_set_custom_aabb(mesh, custom_aabb);
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
	flip_faces = false;
	// defaults
	mesh = RID_PRIME(VisualServer::get_singleton()->mesh_create());

	// assume primitive triangles as the type, correct for all but one and it will change this :)
	primitive_type = Mesh::PRIMITIVE_TRIANGLES;

	// make sure we do an update after we've finished constructing our object
	pending_request = true;
}

PrimitiveMesh::~PrimitiveMesh() {
	VisualServer::get_singleton()->free(mesh);
}

/**
  CapsuleMesh
*/

void CapsuleMesh::_create_mesh_array(Array &p_arr) const {
	create_mesh_array(p_arr, radius, mid_height, radial_segments, rings);
}

void CapsuleMesh::create_mesh_array(Array &p_arr, const float radius, const float mid_height, const int radial_segments, const int rings) {
	int i, j, prevrow, thisrow, point;
	float x, y, z, u, v, w;
	float onethird = 1.0 / 3.0;
	float twothirds = 2.0 / 3.0;

	// note, this has been aligned with our collision shape but I've left the descriptions as top/middle/bottom

	PoolVector<Vector3> points;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;
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
		z = radius * cos(0.5 * Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = sin(u * (Math_PI * 2.0));
			y = -cos(u * (Math_PI * 2.0));

			Vector3 p = Vector3(x * radius * w, y * radius * w, z);
			points.push_back(p + Vector3(0.0, 0.0, 0.5 * mid_height));
			normals.push_back(p.normalized());
			ADD_TANGENT(-y, x, 0.0, 1.0)
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

		z = mid_height * v;
		z = (mid_height * 0.5) - z;

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = sin(u * (Math_PI * 2.0));
			y = -cos(u * (Math_PI * 2.0));

			Vector3 p = Vector3(x * radius, y * radius, z);
			points.push_back(p);
			normals.push_back(Vector3(x, y, 0.0));
			ADD_TANGENT(-y, x, 0.0, 1.0)
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
		z = radius * cos(0.5 * Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			float u2 = i;
			u2 /= radial_segments;

			x = sin(u2 * (Math_PI * 2.0));
			y = -cos(u2 * (Math_PI * 2.0));

			Vector3 p = Vector3(x * radius * w, y * radius * w, z);
			points.push_back(p + Vector3(0.0, 0.0, -0.5 * mid_height));
			normals.push_back(p.normalized());
			ADD_TANGENT(-y, x, 0.0, 1.0)
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

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
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

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mid_height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_mid_height", "get_mid_height");
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

CapsuleMesh::CapsuleMesh() {
	// defaults
	radius = 1.0;
	mid_height = 1.0;
	radial_segments = default_radial_segments;
	rings = default_rings;
}

/**
  CubeMesh
*/

void CubeMesh::_create_mesh_array(Array &p_arr) const {
	create_mesh_array(p_arr, size, subdivide_w, subdivide_h, subdivide_d);
}

void CubeMesh::create_mesh_array(Array &p_arr, const Vector3 size, const int subdivide_w, const int subdivide_h, const int subdivide_d) {
	int i, j, prevrow, thisrow, point;
	float x, y, z;
	float onethird = 1.0 / 3.0;
	float twothirds = 2.0 / 3.0;

	Vector3 start_pos = size * -0.5;

	// set our bounding box

	PoolVector<Vector3> points;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;
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

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
}

void CubeMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &CubeMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &CubeMesh::get_size);

	ClassDB::bind_method(D_METHOD("set_subdivide_width", "subdivide"), &CubeMesh::set_subdivide_width);
	ClassDB::bind_method(D_METHOD("get_subdivide_width"), &CubeMesh::get_subdivide_width);
	ClassDB::bind_method(D_METHOD("set_subdivide_height", "divisions"), &CubeMesh::set_subdivide_height);
	ClassDB::bind_method(D_METHOD("get_subdivide_height"), &CubeMesh::get_subdivide_height);
	ClassDB::bind_method(D_METHOD("set_subdivide_depth", "divisions"), &CubeMesh::set_subdivide_depth);
	ClassDB::bind_method(D_METHOD("get_subdivide_depth"), &CubeMesh::get_subdivide_depth);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_width", "get_subdivide_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_height", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_height", "get_subdivide_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_depth", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_depth", "get_subdivide_depth");
}

void CubeMesh::set_size(const Vector3 &p_size) {
	size = p_size;
	_request_update();
}

Vector3 CubeMesh::get_size() const {
	return size;
}

void CubeMesh::set_subdivide_width(const int p_divisions) {
	subdivide_w = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int CubeMesh::get_subdivide_width() const {
	return subdivide_w;
}

void CubeMesh::set_subdivide_height(const int p_divisions) {
	subdivide_h = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int CubeMesh::get_subdivide_height() const {
	return subdivide_h;
}

void CubeMesh::set_subdivide_depth(const int p_divisions) {
	subdivide_d = p_divisions > 0 ? p_divisions : 0;
	_request_update();
}

int CubeMesh::get_subdivide_depth() const {
	return subdivide_d;
}

CubeMesh::CubeMesh() {
	// defaults
	size = Vector3(2.0, 2.0, 2.0);
	subdivide_w = default_subdivide_w;
	subdivide_h = default_subdivide_h;
	subdivide_d = default_subdivide_d;
}

/**
  CylinderMesh
*/

void CylinderMesh::_create_mesh_array(Array &p_arr) const {
	create_mesh_array(p_arr, top_radius, bottom_radius, height, radial_segments, rings);
}

void CylinderMesh::create_mesh_array(Array &p_arr, float top_radius, float bottom_radius, float height, int radial_segments, int rings) {
	int i, j, prevrow, thisrow, point;
	float x, y, z, u, v, radius;

	PoolVector<Vector3> points;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	thisrow = 0;
	prevrow = 0;
	const real_t side_normal_y = (bottom_radius - top_radius) / height;
	for (j = 0; j <= (rings + 1); j++) {
		v = j;
		v /= (rings + 1);

		radius = top_radius + ((bottom_radius - top_radius) * v);

		y = height * v;
		y = (height * 0.5) - y;

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = sin(u * (Math_PI * 2.0));
			z = cos(u * (Math_PI * 2.0));

			Vector3 p = Vector3(x * radius, y, z * radius);
			points.push_back(p);
			normals.push_back(Vector3(x, side_normal_y, z).normalized());
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
		ADD_TANGENT(1.0, 0.0, 0.0, 1.0)
		uvs.push_back(Vector2(0.75, 0.75));
		point++;

		for (i = 0; i <= radial_segments; i++) {
			float r = i;
			r /= radial_segments;

			x = sin(r * (Math_PI * 2.0));
			z = cos(r * (Math_PI * 2.0));

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

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
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

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "top_radius", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_top_radius", "get_top_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bottom_radius", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_bottom_radius", "get_bottom_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_rings", "get_rings");
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

CylinderMesh::CylinderMesh() {
	// defaults
	top_radius = 1.0;
	bottom_radius = 1.0;
	height = 2.0;
	radial_segments = default_radial_segments;
	rings = default_rings;
}

/**
  PlaneMesh
*/

void PlaneMesh::_create_mesh_array(Array &p_arr) const {
	int i, j, prevrow, thisrow, point;
	float x, z;

	Size2 start_pos = size * -0.5;

	PoolVector<Vector3> points;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;
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

			points.push_back(Vector3(-x, 0.0, -z) + center_offset);
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

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
}

void PlaneMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &PlaneMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &PlaneMesh::get_size);

	ClassDB::bind_method(D_METHOD("set_subdivide_width", "subdivide"), &PlaneMesh::set_subdivide_width);
	ClassDB::bind_method(D_METHOD("get_subdivide_width"), &PlaneMesh::get_subdivide_width);
	ClassDB::bind_method(D_METHOD("set_subdivide_depth", "subdivide"), &PlaneMesh::set_subdivide_depth);
	ClassDB::bind_method(D_METHOD("get_subdivide_depth"), &PlaneMesh::get_subdivide_depth);
	ClassDB::bind_method(D_METHOD("set_center_offset", "offset"), &PlaneMesh::set_center_offset);
	ClassDB::bind_method(D_METHOD("get_center_offset"), &PlaneMesh::get_center_offset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_width", "get_subdivide_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_depth", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_depth", "get_subdivide_depth");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_offset"), "set_center_offset", "get_center_offset");
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

void PlaneMesh::set_center_offset(const Vector3 p_offset) {
	center_offset = p_offset;
	_request_update();
}

Vector3 PlaneMesh::get_center_offset() const {
	return center_offset;
}

PlaneMesh::PlaneMesh() {
	// defaults
	size = Size2(2.0, 2.0);
	subdivide_w = 0;
	subdivide_d = 0;
	center_offset = Vector3(0.0, 0.0, 0.0);
}

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

	PoolVector<Vector3> points;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;
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

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
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

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "left_to_right", PROPERTY_HINT_RANGE, "-2.0,2.0,0.1"), "set_left_to_right", "get_left_to_right");
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

PrismMesh::PrismMesh() {
	// defaults
	left_to_right = 0.5;
	size = Vector3(2.0, 2.0, 2.0);
	subdivide_w = 0;
	subdivide_h = 0;
	subdivide_d = 0;
}

/**
  QuadMesh
*/

void QuadMesh::_create_mesh_array(Array &p_arr) const {
	PoolVector<Vector3> faces;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;

	faces.resize(6);
	normals.resize(6);
	tangents.resize(6 * 4);
	uvs.resize(6);

	Vector2 _size = Vector2(size.x / 2.0f, size.y / 2.0f);

	Vector3 quad_faces[4] = {
		Vector3(-_size.x, -_size.y, 0) + center_offset,
		Vector3(-_size.x, _size.y, 0) + center_offset,
		Vector3(_size.x, _size.y, 0) + center_offset,
		Vector3(_size.x, -_size.y, 0) + center_offset,
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

	p_arr[VS::ARRAY_VERTEX] = faces;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
}

void QuadMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &QuadMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &QuadMesh::get_size);
	ClassDB::bind_method(D_METHOD("set_center_offset", "center_offset"), &QuadMesh::set_center_offset);
	ClassDB::bind_method(D_METHOD("get_center_offset"), &QuadMesh::get_center_offset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_offset"), "set_center_offset", "get_center_offset");
}

QuadMesh::QuadMesh() {
	primitive_type = PRIMITIVE_TRIANGLES;
	size = Size2(1.0, 1.0);
	center_offset = Vector3(0.0, 0.0, 0.0);
}

void QuadMesh::set_size(const Size2 &p_size) {
	size = p_size;
	_request_update();
}

Size2 QuadMesh::get_size() const {
	return size;
}

void QuadMesh::set_center_offset(Vector3 p_center_offset) {
	center_offset = p_center_offset;
	_request_update();
}

Vector3 QuadMesh::get_center_offset() const {
	return center_offset;
}

/**
  SphereMesh
*/

void SphereMesh::_create_mesh_array(Array &p_arr) const {
	create_mesh_array(p_arr, radius, height, radial_segments, rings, is_hemisphere);
}

void SphereMesh::create_mesh_array(Array &p_arr, float radius, float height, int radial_segments, int rings, bool is_hemisphere) {
	int i, j, prevrow, thisrow, point;
	float x, y, z;

	float scale = height * (is_hemisphere ? 1.0 : 0.5);

	// set our bounding box

	PoolVector<Vector3> points;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;
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
		y = scale * cos(Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			float u = i;
			u /= radial_segments;

			x = sin(u * (Math_PI * 2.0));
			z = cos(u * (Math_PI * 2.0));

			if (is_hemisphere && y < 0.0) {
				points.push_back(Vector3(x * radius * w, 0.0, z * radius * w));
				normals.push_back(Vector3(0.0, -1.0, 0.0));
			} else {
				Vector3 p = Vector3(x * radius * w, y, z * radius * w);
				points.push_back(p);
				Vector3 normal = Vector3(x * w * scale, radius * (y / scale), z * w * scale);
				normals.push_back(normal.normalized());
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

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
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

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater"), "set_height", "get_height");
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

SphereMesh::SphereMesh() {
	// defaults
	radius = 1.0;
	height = 2.0;
	radial_segments = default_radial_segments;
	rings = default_rings;
	is_hemisphere = default_is_hemisphere;
}

/**
  TorusMesh
*/

void TorusMesh::_create_mesh_array(Array &p_arr) const {
	// set our bounding box

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int> indices;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	ERR_FAIL_COND_MSG(inner_radius == outer_radius, "Inner radius and outer radius cannot be the same.");

	float min_radius = inner_radius;
	float max_radius = outer_radius;

	if (min_radius > max_radius) {
		SWAP(min_radius, max_radius);
	}

	float radius = (max_radius - min_radius) * 0.5;

	for (int i = 0; i <= rings; i++) {
		int prevrow = (i - 1) * (ring_segments + 1);
		int thisrow = i * (ring_segments + 1);
		float inci = float(i) / rings;
		float angi = inci * Math_TAU;

		Vector2 normali = Vector2(-Math::sin(angi), -Math::cos(angi));

		for (int j = 0; j <= ring_segments; j++) {
			float incj = float(j) / ring_segments;
			float angj = incj * Math_TAU;

			Vector2 normalj = Vector2(-Math::cos(angj), Math::sin(angj));
			Vector2 normalk = normalj * radius + Vector2(min_radius + radius, 0);

			points.push_back(Vector3(normali.x * normalk.x, normalk.y, normali.y * normalk.x));
			normals.push_back(Vector3(normali.x * normalj.x, normalj.y, normali.y * normalj.x));
			ADD_TANGENT(-Math::cos(angi), 0.0, Math::sin(angi), 1.0);
			uvs.push_back(Vector2(inci, incj));

			if (i > 0 && j > 0) {
				indices.push_back(thisrow + j - 1);
				indices.push_back(prevrow + j);
				indices.push_back(prevrow + j - 1);

				indices.push_back(thisrow + j - 1);
				indices.push_back(thisrow + j);
				indices.push_back(prevrow + j);
			}
		}
	}

	p_arr[VS::ARRAY_VERTEX] = points;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
}

void TorusMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "radius"), &TorusMesh::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &TorusMesh::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "radius"), &TorusMesh::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &TorusMesh::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &TorusMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &TorusMesh::get_rings);

	ClassDB::bind_method(D_METHOD("set_ring_segments", "rings"), &TorusMesh::set_ring_segments);
	ClassDB::bind_method(D_METHOD("get_ring_segments"), &TorusMesh::get_ring_segments);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "inner_radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "outer_radius", PROPERTY_HINT_RANGE, "0.001,1000.0,0.001,or_greater"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "3,128,1"), "set_rings", "get_rings");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ring_segments", PROPERTY_HINT_RANGE, "3,64,1"), "set_ring_segments", "get_ring_segments");
}

void TorusMesh::set_inner_radius(const float p_inner_radius) {
	inner_radius = p_inner_radius;
	_request_update();
}

float TorusMesh::get_inner_radius() const {
	return inner_radius;
}

void TorusMesh::set_outer_radius(const float p_outer_radius) {
	outer_radius = p_outer_radius;
	_request_update();
}

float TorusMesh::get_outer_radius() const {
	return outer_radius;
}

void TorusMesh::set_rings(const int p_rings) {
	ERR_FAIL_COND(p_rings < 3);
	rings = p_rings;
	_request_update();
}

int TorusMesh::get_rings() const {
	return rings;
}

void TorusMesh::set_ring_segments(const int p_ring_segments) {
	ERR_FAIL_COND(p_ring_segments < 3);
	ring_segments = p_ring_segments;
	_request_update();
}

int TorusMesh::get_ring_segments() const {
	return ring_segments;
}

TorusMesh::TorusMesh() {}

/**
  PointMesh
*/

void PointMesh::_create_mesh_array(Array &p_arr) const {
	PoolVector<Vector3> faces;
	faces.resize(1);
	faces.set(0, Vector3(0.0, 0.0, 0.0));

	p_arr[VS::ARRAY_VERTEX] = faces;
}

PointMesh::PointMesh() {
	primitive_type = PRIMITIVE_POINTS;
}

/**
  TextMesh
*/

void TextMesh::_generate_glyph_mesh_data(uint32_t p_utf32_char, const Ref<Font> &p_font, CharType p_char, CharType p_next) const {
	if (cache.has(p_utf32_char)) {
		return;
	}

	GlyphMeshData &gl_data = cache[p_utf32_char];

	Dictionary d = p_font->get_char_contours(p_char, p_next);

	PoolVector3Array points = d["points"];
	PoolIntArray contours = d["contours"];
	bool orientation = d["orientation"];

	if (points.size() < 3 || contours.size() < 1) {
		return; // No full contours, only glyph control points (or nothing), ignore.
	}

	// Approximate Bezier curves as polygons.
	// See https://freetype.org/freetype2/docs/glyphs/glyphs-6.html, for more info.
	for (int i = 0; i < contours.size(); i++) {
		int32_t start = (i == 0) ? 0 : (contours[i - 1] + 1);
		int32_t end = contours[i];
		Vector<ContourPoint> polygon;

		for (int32_t j = start; j <= end; j++) {
			if (points[j].z == Font::CONTOUR_CURVE_TAG_ON) {
				// Point on the curve.
				Vector2 p = Vector2(points[j].x, points[j].y) * pixel_size;
				polygon.push_back(ContourPoint(p, true));
			} else if (points[j].z == Font::CONTOUR_CURVE_TAG_OFF_CONIC) {
				// Conic Bezier arc.
				int32_t next = (j == end) ? start : (j + 1);
				int32_t prev = (j == start) ? end : (j - 1);
				Vector2 p0;
				Vector2 p1 = Vector2(points[j].x, points[j].y);
				Vector2 p2;

				// For successive conic OFF points add a virtual ON point in the middle.
				if (points[prev].z == Font::CONTOUR_CURVE_TAG_OFF_CONIC) {
					p0 = (Vector2(points[prev].x, points[prev].y) + Vector2(points[j].x, points[j].y)) / 2.0;
				} else if (points[prev].z == Font::CONTOUR_CURVE_TAG_ON) {
					p0 = Vector2(points[prev].x, points[prev].y);
				} else {
					ERR_FAIL_MSG(vformat("Invalid conic arc point sequence at %d:%d", i, j));
				}
				if (points[next].z == Font::CONTOUR_CURVE_TAG_OFF_CONIC) {
					p2 = (Vector2(points[j].x, points[j].y) + Vector2(points[next].x, points[next].y)) / 2.0;
				} else if (points[next].z == Font::CONTOUR_CURVE_TAG_ON) {
					p2 = Vector2(points[next].x, points[next].y);
				} else {
					ERR_FAIL_MSG(vformat("Invalid conic arc point sequence at %d:%d", i, j));
				}

				real_t step = CLAMP(curve_step / (p0 - p2).length(), 0.01, 0.5);
				real_t t = step;
				while (t < 1.0) {
					real_t omt = (1.0 - t);
					real_t omt2 = omt * omt;
					real_t t2 = t * t;

					Vector2 point = p1 + omt2 * (p0 - p1) + t2 * (p2 - p1);
					Vector2 p = point * pixel_size;
					polygon.push_back(ContourPoint(p, false));
					t += step;
				}
			} else if (points[j].z == Font::CONTOUR_CURVE_TAG_OFF_CUBIC) {
				// Cubic Bezier arc.
				int32_t cur = j;
				int32_t next1 = (j == end) ? start : (j + 1);
				int32_t next2 = (next1 == end) ? start : (next1 + 1);
				int32_t prev = (j == start) ? end : (j - 1);

				// There must be exactly two OFF points and two ON points for each cubic arc.
				if (points[prev].z != Font::CONTOUR_CURVE_TAG_ON) {
					cur = (cur == 0) ? end : cur - 1;
					next1 = (next1 == 0) ? end : next1 - 1;
					next2 = (next2 == 0) ? end : next2 - 1;
					prev = (prev == 0) ? end : prev - 1;
				} else {
					j++;
				}
				ERR_FAIL_COND_MSG(points[prev].z != Font::CONTOUR_CURVE_TAG_ON, vformat("Invalid cubic arc point sequence at %d:%d", i, prev));
				ERR_FAIL_COND_MSG(points[cur].z != Font::CONTOUR_CURVE_TAG_OFF_CUBIC, vformat("Invalid cubic arc point sequence at %d:%d", i, cur));
				ERR_FAIL_COND_MSG(points[next1].z != Font::CONTOUR_CURVE_TAG_OFF_CUBIC, vformat("Invalid cubic arc point sequence at %d:%d", i, next1));
				ERR_FAIL_COND_MSG(points[next2].z != Font::CONTOUR_CURVE_TAG_ON, vformat("Invalid cubic arc point sequence at %d:%d", i, next2));

				Vector2 p0 = Vector2(points[prev].x, points[prev].y);
				Vector2 p1 = Vector2(points[cur].x, points[cur].y);
				Vector2 p2 = Vector2(points[next1].x, points[next1].y);
				Vector2 p3 = Vector2(points[next2].x, points[next2].y);

				real_t step = CLAMP(curve_step / (p0 - p3).length(), 0.01, 0.5);
				real_t t = step;
				while (t < 1.0) {
					real_t omt = (1.0 - t);
					real_t omt2 = omt * omt;
					real_t omt3 = omt2 * omt;
					real_t t2 = t * t;
					real_t t3 = t2 * t;

					Vector2 point = p0 * omt3 + p1 * omt2 * t * 3.0 + p2 * omt * t2 * 3.0 + p3 * t3;
					Vector2 p = point * pixel_size;
					polygon.push_back(ContourPoint(p, false));
					t += step;
				}
			} else {
				ERR_FAIL_MSG(vformat("Unknown point tag at %d:%d", i, j));
			}
		}

		if (polygon.size() < 3) {
			continue; // Skip glyph control points.
		}

		if (!orientation) {
			polygon.invert();
		}

		gl_data.contours.push_back(polygon);
	}

	// Calculate bounds.
	List<TriangulatorPoly> in_poly;
	for (int i = 0; i < gl_data.contours.size(); i++) {
		TriangulatorPoly inp;
		inp.Init(gl_data.contours[i].size());
		real_t length = 0.0;
		for (int j = 0; j < gl_data.contours[i].size(); j++) {
			int next = (j + 1 == gl_data.contours[i].size()) ? 0 : (j + 1);

			gl_data.min_p.x = MIN(gl_data.min_p.x, gl_data.contours[i][j].point.x);
			gl_data.min_p.y = MIN(gl_data.min_p.y, gl_data.contours[i][j].point.y);
			gl_data.max_p.x = MAX(gl_data.max_p.x, gl_data.contours[i][j].point.x);
			gl_data.max_p.y = MAX(gl_data.max_p.y, gl_data.contours[i][j].point.y);
			length += (gl_data.contours[i][next].point - gl_data.contours[i][j].point).length();

			inp.GetPoint(j) = gl_data.contours[i][j].point;
		}
		int poly_orient = inp.GetOrientation();
		if (poly_orient == TRIANGULATOR_CW) {
			inp.SetHole(true);
		}
		in_poly.push_back(inp);
		gl_data.contours_info.push_back(ContourInfo(length, poly_orient == TRIANGULATOR_CCW));
	}

	TriangulatorPartition tpart;

	//Decompose and triangulate.
	List<TriangulatorPoly> out_poly;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) {
		ERR_FAIL_MSG("Convex decomposing failed. Make sure the font doesn't contain self-intersecting lines, as these are not supported in TextMesh.");
	}
	List<TriangulatorPoly> out_tris;
	for (List<TriangulatorPoly>::Element *I = out_poly.front(); I; I = I->next()) {
		if (tpart.Triangulate_OPT(&(I->get()), &out_tris) == 0) {
			ERR_FAIL_MSG("Triangulation failed. Make sure the font doesn't contain self-intersecting lines, as these are not supported in TextMesh.");
		}
	}

	for (List<TriangulatorPoly>::Element *I = out_tris.front(); I; I = I->next()) {
		TriangulatorPoly &tp = I->get();
		ERR_FAIL_COND(tp.GetNumPoints() != 3); // Triangles only.

		for (int i = 0; i < 3; i++) {
			gl_data.triangles.push_back(Vector2(tp.GetPoint(i).x, tp.GetPoint(i).y));
		}
	}
}

void TextMesh::_create_mesh_array(Array &p_arr) const {
	Ref<Font> font = _get_font_or_default();
	ERR_FAIL_COND(font.is_null());

	if (dirty_cache) {
		cache.clear();
		dirty_cache = false;
	}

	String t = (uppercase) ? xl_text.to_upper() : xl_text;

	float line_width = font->get_string_size(t).x * pixel_size;

	Vector2 offset;
	switch (horizontal_alignment) {
		case ALIGN_LEFT:
			offset.x = 0.0;
			break;
		case ALIGN_CENTER: {
			offset.x = -line_width / 2.0;
		} break;
		case ALIGN_RIGHT: {
			offset.x = -line_width;
		} break;
	}

	bool has_depth = !Math::is_zero_approx(depth);

	// Generate glyph data, precalculate size of the arrays and mesh bounds for UV.
	int64_t p_size = 0;
	int64_t i_size = 0;

	Vector2 min_p = Vector2(INFINITY, INFINITY);
	Vector2 max_p = Vector2(-INFINITY, -INFINITY);

	Vector2 offset_pre = offset;
	for (int i = 0; i < t.size(); i++) {
		CharType c = t[i];
		CharType n = t[i + 1];
		uint32_t utf32_char = c;
		if (((c & 0xfffffc00) == 0xd800) && (n & 0xfffffc00) == 0xdc00) { // decode surrogate pair.
			utf32_char = (c << 10UL) + n - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
		}
		if ((c & 0xfffffc00) == 0xdc00) { // skip trail surrogate.
			continue;
		}
		if (utf32_char >= 0x20) {
			_generate_glyph_mesh_data(utf32_char, font, c, n);
			GlyphMeshData &gl_data = cache[utf32_char];

			p_size += gl_data.triangles.size() * ((has_depth) ? 2 : 1);
			i_size += gl_data.triangles.size() * ((has_depth) ? 2 : 1);

			if (has_depth) {
				for (int j = 0; j < gl_data.contours.size(); j++) {
					p_size += gl_data.contours[j].size() * 4;
					i_size += gl_data.contours[j].size() * 6;
				}
			}

			min_p.x = MIN(gl_data.min_p.x + offset_pre.x, min_p.x);
			min_p.y = MIN(gl_data.min_p.y + offset_pre.y, min_p.y);
			max_p.x = MAX(gl_data.max_p.x + offset_pre.x, max_p.x);
			max_p.y = MAX(gl_data.max_p.y + offset_pre.y, max_p.y);
		}

		offset_pre.x += font->get_char_size(c, n).x * pixel_size;
	}

	PoolVector<Vector3> vertices;
	PoolVector<Vector3> normals;
	PoolVector<float> tangents;
	PoolVector<Vector2> uvs;
	PoolVector<int> indices;

	vertices.resize(p_size);
	normals.resize(p_size);
	uvs.resize(p_size);
	tangents.resize(p_size * 4);
	indices.resize(i_size);

	PoolVector<Vector3>::Write vertices_ptr = vertices.write();
	PoolVector<Vector3>::Write normals_ptr = normals.write();
	PoolVector<float>::Write tangents_ptr = tangents.write();
	PoolVector<Vector2>::Write uvs_ptr = uvs.write();
	PoolVector<int>::Write indices_ptr = indices.write();

	// Generate mesh.
	int32_t p_idx = 0;
	int32_t i_idx = 0;

	for (int i = 0; i < t.size(); i++) {
		CharType c = t[i];
		CharType n = t[i + 1];
		uint32_t utf32_char = c;
		if (((c & 0xfffffc00) == 0xd800) && (n & 0xfffffc00) == 0xdc00) { // decode surrogate pair.
			utf32_char = (c << 10UL) + n - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
		}
		if ((c & 0xfffffc00) == 0xdc00) { // skip trail surrogate.
			continue;
		}
		if (utf32_char >= 0x20) {
			_generate_glyph_mesh_data(utf32_char, font, c, n);
			GlyphMeshData &gl_data = cache[utf32_char];

			int64_t ts = gl_data.triangles.size();
			const Vector2 *ts_ptr = gl_data.triangles.ptr();

			for (int k = 0; k < ts; k += 3) {
				// Add front face.
				for (int l = 0; l < 3; l++) {
					Vector3 point = Vector3(ts_ptr[k + l].x + offset.x, -ts_ptr[k + l].y + offset.y, depth / 2.0);
					vertices_ptr[p_idx] = point;
					normals_ptr[p_idx] = Vector3(0.0, 0.0, 1.0);
					if (has_depth) {
						uvs_ptr[p_idx] = Vector2(Math::range_lerp(point.x, min_p.x, max_p.x, real_t(0.0), real_t(1.0)), Math::range_lerp(point.y, -min_p.y, -max_p.y, real_t(0.0), real_t(0.4)));
					} else {
						uvs_ptr[p_idx] = Vector2(Math::range_lerp(point.x, min_p.x, max_p.x, real_t(0.0), real_t(1.0)), Math::range_lerp(point.y, -min_p.y, -max_p.y, real_t(0.0), real_t(1.0)));
					}
					tangents_ptr[p_idx * 4 + 0] = 1.0;
					tangents_ptr[p_idx * 4 + 1] = 0.0;
					tangents_ptr[p_idx * 4 + 2] = 0.0;
					tangents_ptr[p_idx * 4 + 3] = 1.0;
					indices_ptr[i_idx++] = p_idx;
					p_idx++;
				}
				if (has_depth) {
					// Add back face.
					for (int l = 2; l >= 0; l--) {
						Vector3 point = Vector3(ts_ptr[k + l].x + offset.x, -ts_ptr[k + l].y + offset.y, -depth / 2.0);
						vertices_ptr[p_idx] = point;
						normals_ptr[p_idx] = Vector3(0.0, 0.0, -1.0);
						uvs_ptr[p_idx] = Vector2(Math::range_lerp(point.x, min_p.x, max_p.x, real_t(0.0), real_t(1.0)), Math::range_lerp(point.y, -min_p.y, -max_p.y, real_t(0.4), real_t(0.8)));
						tangents_ptr[p_idx * 4 + 0] = -1.0;
						tangents_ptr[p_idx * 4 + 1] = 0.0;
						tangents_ptr[p_idx * 4 + 2] = 0.0;
						tangents_ptr[p_idx * 4 + 3] = 1.0;
						indices_ptr[i_idx++] = p_idx;
						p_idx++;
					}
				}
			}
			// Add sides.
			if (has_depth) {
				for (int k = 0; k < gl_data.contours.size(); k++) {
					int64_t ps = gl_data.contours[k].size();
					const ContourPoint *ps_ptr = gl_data.contours[k].ptr();
					const ContourInfo &ps_info = gl_data.contours_info[k];
					real_t length = 0.0;
					for (int l = 0; l < ps; l++) {
						int prev = (l == 0) ? (ps - 1) : (l - 1);
						int next = (l + 1 == ps) ? 0 : (l + 1);
						Vector2 d1;
						Vector2 d2 = (ps_ptr[next].point - ps_ptr[l].point).normalized();
						if (ps_ptr[l].sharp) {
							d1 = d2;
						} else {
							d1 = (ps_ptr[l].point - ps_ptr[prev].point).normalized();
						}
						real_t seg_len = (ps_ptr[next].point - ps_ptr[l].point).length();

						Vector3 quad_faces[4] = {
							Vector3(ps_ptr[l].point.x + offset.x, -ps_ptr[l].point.y + offset.y, -depth / 2.0),
							Vector3(ps_ptr[next].point.x + offset.x, -ps_ptr[next].point.y + offset.y, -depth / 2.0),
							Vector3(ps_ptr[l].point.x + offset.x, -ps_ptr[l].point.y + offset.y, depth / 2.0),
							Vector3(ps_ptr[next].point.x + offset.x, -ps_ptr[next].point.y + offset.y, depth / 2.0),
						};
						for (int m = 0; m < 4; m++) {
							const Vector2 &d = ((m % 2) == 0) ? d1 : d2;
							real_t u_pos = ((m % 2) == 0) ? length : length + seg_len;
							vertices_ptr[p_idx + m] = quad_faces[m];
							normals_ptr[p_idx + m] = Vector3(d.y, d.x, 0.0);
							if (m < 2) {
								uvs_ptr[p_idx + m] = Vector2(Math::range_lerp(u_pos, 0, ps_info.length, real_t(0.0), real_t(1.0)), (ps_info.ccw) ? 0.8 : 0.9);
							} else {
								uvs_ptr[p_idx + m] = Vector2(Math::range_lerp(u_pos, 0, ps_info.length, real_t(0.0), real_t(1.0)), (ps_info.ccw) ? 0.9 : 1.0);
							}
							tangents_ptr[(p_idx + m) * 4 + 0] = d.x;
							tangents_ptr[(p_idx + m) * 4 + 1] = -d.y;
							tangents_ptr[(p_idx + m) * 4 + 2] = 0.0;
							tangents_ptr[(p_idx + m) * 4 + 3] = 1.0;
						}

						indices_ptr[i_idx++] = p_idx;
						indices_ptr[i_idx++] = p_idx + 1;
						indices_ptr[i_idx++] = p_idx + 2;

						indices_ptr[i_idx++] = p_idx + 1;
						indices_ptr[i_idx++] = p_idx + 3;
						indices_ptr[i_idx++] = p_idx + 2;

						length += seg_len;
						p_idx += 4;
					}
				}
			}
		}
		offset.x += font->get_char_size(c, n).x * pixel_size;
	}

	if (p_size == 0) {
		// If empty, add single triangle to suppress errors.
		vertices.push_back(Vector3());
		normals.push_back(Vector3());
		uvs.push_back(Vector2());
		tangents.push_back(1.0);
		tangents.push_back(0.0);
		tangents.push_back(0.0);
		tangents.push_back(1.0);
		indices.push_back(0);
		indices.push_back(0);
		indices.push_back(0);
	}

	p_arr[VS::ARRAY_VERTEX] = vertices;
	p_arr[VS::ARRAY_NORMAL] = normals;
	p_arr[VS::ARRAY_TANGENT] = tangents;
	p_arr[VS::ARRAY_TEX_UV] = uvs;
	p_arr[VS::ARRAY_INDEX] = indices;
}

void TextMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &TextMesh::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &TextMesh::get_horizontal_alignment);

	ClassDB::bind_method(D_METHOD("set_text", "text"), &TextMesh::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &TextMesh::get_text);

	ClassDB::bind_method(D_METHOD("set_font", "font"), &TextMesh::set_font);
	ClassDB::bind_method(D_METHOD("get_font"), &TextMesh::get_font);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &TextMesh::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &TextMesh::get_depth);

	ClassDB::bind_method(D_METHOD("set_pixel_size", "pixel_size"), &TextMesh::set_pixel_size);
	ClassDB::bind_method(D_METHOD("get_pixel_size"), &TextMesh::get_pixel_size);

	ClassDB::bind_method(D_METHOD("set_curve_step", "curve_step"), &TextMesh::set_curve_step);
	ClassDB::bind_method(D_METHOD("get_curve_step"), &TextMesh::get_curve_step);

	ClassDB::bind_method(D_METHOD("set_uppercase", "enable"), &TextMesh::set_uppercase);
	ClassDB::bind_method(D_METHOD("is_uppercase"), &TextMesh::is_uppercase);

	ClassDB::bind_method(D_METHOD("_font_changed"), &TextMesh::_font_changed);
	ClassDB::bind_method(D_METHOD("_request_update"), &TextMesh::_request_update);

	ADD_GROUP("Text", "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");

	ADD_GROUP("Mesh", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "pixel_size", PROPERTY_HINT_RANGE, "0.0001,128,0.0001"), "set_pixel_size", "get_pixel_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "curve_step", PROPERTY_HINT_RANGE, "0.1,10,0.1"), "set_curve_step", "get_curve_step");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth", PROPERTY_HINT_RANGE, "0.0,100.0,0.001,or_greater"), "set_depth", "get_depth");

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
}

void TextMesh::_notification(int p_what) {
	switch (p_what) {
		case MainLoop::NOTIFICATION_TRANSLATION_CHANGED: {
			String new_text = tr(text);
			if (new_text == xl_text) {
				return; // Nothing new.
			}
			xl_text = new_text;
			_request_update();
		} break;
	}
}

TextMesh::TextMesh() {
	primitive_type = PRIMITIVE_TRIANGLES;
}

TextMesh::~TextMesh() {
}

void TextMesh::set_horizontal_alignment(TextMesh::Align p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 3);
	if (horizontal_alignment != p_alignment) {
		horizontal_alignment = p_alignment;
		_request_update();
	}
}

TextMesh::Align TextMesh::get_horizontal_alignment() const {
	return horizontal_alignment;
}

void TextMesh::set_text(const String &p_string) {
	if (text != p_string) {
		text = p_string;
		xl_text = tr(text);
		_request_update();
	}
}

String TextMesh::get_text() const {
	return text;
}

void TextMesh::_font_changed() {
	dirty_cache = true;
	call_deferred("_request_update");
}

void TextMesh::set_font(const Ref<Font> &p_font) {
	if (font_override != p_font) {
		if (font_override.is_valid()) {
			font_override->disconnect(CoreStringNames::get_singleton()->changed, this, "_font_changed");
		}
		font_override = p_font;
		dirty_cache = true;
		if (font_override.is_valid()) {
			font_override->connect(CoreStringNames::get_singleton()->changed, this, "_font_changed");
		}
		_request_update();
	}
}

Ref<Font> TextMesh::get_font() const {
	return font_override;
}

Ref<Font> TextMesh::_get_font_or_default() const {
	if (font_override.is_valid()) {
		return font_override;
	}

	// Check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		List<StringName> theme_types;
		Theme::get_project_default()->get_type_dependencies(get_class_name(), StringName(), &theme_types);

		for (List<StringName>::Element *E = theme_types.front(); E; E = E->next()) {
			if (Theme::get_project_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", E->get())) {
				return Theme::get_project_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", E->get());
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	{
		List<StringName> theme_types;
		Theme::get_default()->get_type_dependencies(get_class_name(), StringName(), &theme_types);

		for (List<StringName>::Element *E = theme_types.front(); E; E = E->next()) {
			if (Theme::get_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", E->get())) {
				return Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", E->get());
			}
		}
	}

	// If they don't exist, use any type to return the default/empty value.
	return Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", StringName());
}

void TextMesh::set_depth(real_t p_depth) {
	if (depth != p_depth) {
		depth = MAX(p_depth, 0.0);
		_request_update();
	}
}

real_t TextMesh::get_depth() const {
	return depth;
}

void TextMesh::set_pixel_size(real_t p_amount) {
	if (pixel_size != p_amount) {
		pixel_size = CLAMP(p_amount, 0.0001, 128.0);
		dirty_cache = true;
		_request_update();
	}
}

real_t TextMesh::get_pixel_size() const {
	return pixel_size;
}

void TextMesh::set_curve_step(real_t p_step) {
	if (curve_step != p_step) {
		curve_step = CLAMP(p_step, 0.1, 10.0);
		dirty_cache = true;
		_request_update();
	}
}

real_t TextMesh::get_curve_step() const {
	return curve_step;
}

void TextMesh::set_uppercase(bool p_uppercase) {
	if (uppercase != p_uppercase) {
		uppercase = p_uppercase;
		_request_update();
	}
}

bool TextMesh::is_uppercase() const {
	return uppercase;
}

/*************************************************************************/
/*  primitive_meshes.cpp                                                 */
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

#include "primitive_meshes.h"

#include "core/core_string_names.h"
#include "scene/resources/theme.h"
#include "servers/rendering_server.h"
#include "thirdparty/misc/clipper.hpp"
#include "thirdparty/misc/polypartition.h"

/**
  PrimitiveMesh
*/
void PrimitiveMesh::_update() const {
	Array arr;
	if (GDVIRTUAL_CALL(_create_mesh_array, arr)) {
		ERR_FAIL_COND_MSG(arr.size() != RS::ARRAY_MAX, "_create_mesh_array must return an array of Mesh.ARRAY_MAX elements.");
	} else {
		arr.resize(RS::ARRAY_MAX);
		_create_mesh_array(arr);
	}

	Vector<Vector3> points = arr[RS::ARRAY_VERTEX];

	ERR_FAIL_COND_MSG(points.size() == 0, "_create_mesh_array must return at least a vertex array.");

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

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "custom_aabb", PROPERTY_HINT_NONE, "suffix:m"), "set_custom_aabb", "get_custom_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_faces"), "set_flip_faces", "get_flip_faces");

	GDVIRTUAL_BIND(_create_mesh_array);
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
	create_mesh_array(p_arr, radius, height, radial_segments, rings);
}

void CapsuleMesh::create_mesh_array(Array &p_arr, const float radius, const float height, const int radial_segments, const int rings) {
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
			points.push_back(p + Vector3(0.0, 0.5 * height - radius, 0.0));
			normals.push_back(p.normalized());
			ADD_TANGENT(-z, 0.0, -x, 1.0)
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

		y = (height - 2.0 * radius) * v;
		y = (0.5 * height - radius) - y;

		for (i = 0; i <= radial_segments; i++) {
			u = i;
			u /= radial_segments;

			x = -sin(u * Math_TAU);
			z = cos(u * Math_TAU);

			Vector3 p = Vector3(x * radius, y, -z * radius);
			points.push_back(p);
			normals.push_back(Vector3(x, 0.0, -z));
			ADD_TANGENT(-z, 0.0, -x, 1.0)
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
			points.push_back(p + Vector3(0.0, -0.5 * height + radius, 0.0));
			normals.push_back(p.normalized());
			ADD_TANGENT(-z, 0.0, -x, 1.0)
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
	ClassDB::bind_method(D_METHOD("set_height", "height"), &CapsuleMesh::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CapsuleMesh::get_height);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "segments"), &CapsuleMesh::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &CapsuleMesh::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &CapsuleMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &CapsuleMesh::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_rings", "get_rings");
}

void CapsuleMesh::set_radius(const float p_radius) {
	radius = p_radius;
	if (radius > height * 0.5) {
		radius = height * 0.5;
	}
	_request_update();
}

float CapsuleMesh::get_radius() const {
	return radius;
}

void CapsuleMesh::set_height(const float p_height) {
	height = p_height;
	if (radius > height * 0.5) {
		height = radius * 2;
	}
	_request_update();
}

float CapsuleMesh::get_height() const {
	return height;
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
	BoxMesh::create_mesh_array(p_arr, size, subdivide_w, subdivide_h, subdivide_d);
}

void BoxMesh::create_mesh_array(Array &p_arr, Vector3 size, int subdivide_w, int subdivide_h, int subdivide_d) {
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

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
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
	create_mesh_array(p_arr, top_radius, bottom_radius, height, radial_segments, rings, cap_top, cap_bottom);
}

void CylinderMesh::create_mesh_array(Array &p_arr, float top_radius, float bottom_radius, float height, int radial_segments, int rings, bool cap_top, bool cap_bottom) {
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
	if (cap_top && top_radius > 0.0) {
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
	if (cap_bottom && bottom_radius > 0.0) {
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

	ClassDB::bind_method(D_METHOD("set_cap_top", "cap_top"), &CylinderMesh::set_cap_top);
	ClassDB::bind_method(D_METHOD("is_cap_top"), &CylinderMesh::is_cap_top);

	ClassDB::bind_method(D_METHOD("set_cap_bottom", "cap_bottom"), &CylinderMesh::set_cap_bottom);
	ClassDB::bind_method(D_METHOD("is_cap_bottom"), &CylinderMesh::is_cap_bottom);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_radius", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater,suffix:m"), "set_top_radius", "get_top_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bottom_radius", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater,suffix:m"), "set_bottom_radius", "get_bottom_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_rings", "get_rings");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cap_top"), "set_cap_top", "is_cap_top");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cap_bottom"), "set_cap_bottom", "is_cap_bottom");
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

void CylinderMesh::set_cap_top(bool p_cap_top) {
	cap_top = p_cap_top;
	_request_update();
}

bool CylinderMesh::is_cap_top() const {
	return cap_top;
}

void CylinderMesh::set_cap_bottom(bool p_cap_bottom) {
	cap_bottom = p_cap_bottom;
	_request_update();
}

bool CylinderMesh::is_cap_bottom() const {
	return cap_bottom;
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
	ClassDB::bind_method(D_METHOD("set_center_offset", "offset"), &PlaneMesh::set_center_offset);
	ClassDB::bind_method(D_METHOD("get_center_offset"), &PlaneMesh::get_center_offset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_width", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_width", "get_subdivide_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivide_depth", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_subdivide_depth", "get_subdivide_depth");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_offset", PROPERTY_HINT_NONE, "suffix:m"), "set_center_offset", "get_center_offset");
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
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
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

	p_arr[RS::ARRAY_VERTEX] = faces;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
}

void QuadMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &QuadMesh::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &QuadMesh::get_size);
	ClassDB::bind_method(D_METHOD("set_center_offset", "center_offset"), &QuadMesh::set_center_offset);
	ClassDB::bind_method(D_METHOD("get_center_offset"), &QuadMesh::get_center_offset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_offset", PROPERTY_HINT_NONE, "suffix:m"), "set_center_offset", "get_center_offset");
}

uint32_t QuadMesh::surface_get_format(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, 1, 0);

	return RS::ARRAY_FORMAT_VERTEX | RS::ARRAY_FORMAT_NORMAL | RS::ARRAY_FORMAT_TANGENT | RS::ARRAY_FORMAT_TEX_UV;
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
		y = scale * cos(Math_PI * v);

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

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_height", "get_height");
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

Transform3D TubeTrailMesh::get_builtin_bind_pose(int p_index) const {
	float depth = section_length * sections;

	Transform3D xform;
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

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_radius", "get_radius");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_steps", PROPERTY_HINT_RANGE, "3,128,1"), "set_radial_steps", "get_radial_steps");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sections", PROPERTY_HINT_RANGE, "2,128,1"), "set_sections", "get_sections");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "section_length", PROPERTY_HINT_RANGE, "0.001,1024.0,0.001,or_greater,suffix:m"), "set_section_length", "get_section_length");

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

Transform3D RibbonTrailMesh::get_builtin_bind_pose(int p_index) const {
	float depth = section_length * sections;

	Transform3D xform;
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
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "size", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sections", PROPERTY_HINT_RANGE, "2,128,1"), "set_sections", "get_sections");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "section_length", PROPERTY_HINT_RANGE, "0.001,1024.0,0.001,or_greater,suffix:m"), "set_section_length", "get_section_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "section_segments", PROPERTY_HINT_RANGE, "1,128,1"), "set_section_segments", "get_section_segments");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve", "get_curve");

	BIND_ENUM_CONSTANT(SHAPE_FLAT)
	BIND_ENUM_CONSTANT(SHAPE_CROSS)
}

RibbonTrailMesh::RibbonTrailMesh() {
}

/*************************************************************************/
/*  TextMesh                                                             */
/*************************************************************************/

void TextMesh::_generate_glyph_mesh_data(uint32_t p_hash, const Glyph &p_gl) const {
	if (cache.has(p_hash)) {
		return;
	}

	GlyphMeshData &gl_data = cache[p_hash];

	Dictionary d = TS->font_get_glyph_contours(p_gl.font_rid, p_gl.font_size, p_gl.index);
	Vector2 origin = Vector2(p_gl.x_off, p_gl.y_off) * pixel_size;

	PackedVector3Array points = d["points"];
	PackedInt32Array contours = d["contours"];
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
			if (points[j].z == TextServer::CONTOUR_CURVE_TAG_ON) {
				// Point on the curve.
				Vector2 p = Vector2(points[j].x, points[j].y) * pixel_size + origin;
				polygon.push_back(ContourPoint(p, true));
			} else if (points[j].z == TextServer::CONTOUR_CURVE_TAG_OFF_CONIC) {
				// Conic Bezier arc.
				int32_t next = (j == end) ? start : (j + 1);
				int32_t prev = (j == start) ? end : (j - 1);
				Vector2 p0;
				Vector2 p1 = Vector2(points[j].x, points[j].y);
				Vector2 p2;

				// For successive conic OFF points add a virtual ON point in the middle.
				if (points[prev].z == TextServer::CONTOUR_CURVE_TAG_OFF_CONIC) {
					p0 = (Vector2(points[prev].x, points[prev].y) + Vector2(points[j].x, points[j].y)) / 2.0;
				} else if (points[prev].z == TextServer::CONTOUR_CURVE_TAG_ON) {
					p0 = Vector2(points[prev].x, points[prev].y);
				} else {
					ERR_FAIL_MSG(vformat("Invalid conic arc point sequence at %d:%d", i, j));
				}
				if (points[next].z == TextServer::CONTOUR_CURVE_TAG_OFF_CONIC) {
					p2 = (Vector2(points[j].x, points[j].y) + Vector2(points[next].x, points[next].y)) / 2.0;
				} else if (points[next].z == TextServer::CONTOUR_CURVE_TAG_ON) {
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
					Vector2 p = point * pixel_size + origin;
					polygon.push_back(ContourPoint(p, false));
					t += step;
				}
			} else if (points[j].z == TextServer::CONTOUR_CURVE_TAG_OFF_CUBIC) {
				// Cubic Bezier arc.
				int32_t cur = j;
				int32_t next1 = (j == end) ? start : (j + 1);
				int32_t next2 = (next1 == end) ? start : (next1 + 1);
				int32_t prev = (j == start) ? end : (j - 1);

				// There must be exactly two OFF points and two ON points for each cubic arc.
				if (points[prev].z != TextServer::CONTOUR_CURVE_TAG_ON) {
					cur = (cur == 0) ? end : cur - 1;
					next1 = (next1 == 0) ? end : next1 - 1;
					next2 = (next2 == 0) ? end : next2 - 1;
					prev = (prev == 0) ? end : prev - 1;
				} else {
					j++;
				}
				ERR_FAIL_COND_MSG(points[prev].z != TextServer::CONTOUR_CURVE_TAG_ON, vformat("Invalid cubic arc point sequence at %d:%d", i, prev));
				ERR_FAIL_COND_MSG(points[cur].z != TextServer::CONTOUR_CURVE_TAG_OFF_CUBIC, vformat("Invalid cubic arc point sequence at %d:%d", i, cur));
				ERR_FAIL_COND_MSG(points[next1].z != TextServer::CONTOUR_CURVE_TAG_OFF_CUBIC, vformat("Invalid cubic arc point sequence at %d:%d", i, next1));
				ERR_FAIL_COND_MSG(points[next2].z != TextServer::CONTOUR_CURVE_TAG_ON, vformat("Invalid cubic arc point sequence at %d:%d", i, next2));

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
					Vector2 p = point * pixel_size + origin;
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
			polygon.reverse();
		}

		gl_data.contours.push_back(polygon);
	}

	// Calculate bounds.
	List<TPPLPoly> in_poly;
	for (int i = 0; i < gl_data.contours.size(); i++) {
		TPPLPoly inp;
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
		TPPLOrientation poly_orient = inp.GetOrientation();
		if (poly_orient == TPPL_ORIENTATION_CW) {
			inp.SetHole(true);
		}
		in_poly.push_back(inp);
		gl_data.contours_info.push_back(ContourInfo(length, poly_orient == TPPL_ORIENTATION_CCW));
	}

	TPPLPartition tpart;

	//Decompose and triangulate.
	List<TPPLPoly> out_poly;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) {
		ERR_FAIL_MSG("Convex decomposing failed. Make sure the font doesn't contain self-intersecting lines, as these are not supported in TextMesh.");
	}
	List<TPPLPoly> out_tris;
	for (List<TPPLPoly>::Element *I = out_poly.front(); I; I = I->next()) {
		if (tpart.Triangulate_OPT(&(I->get()), &out_tris) == 0) {
			ERR_FAIL_MSG("Triangulation failed. Make sure the font doesn't contain self-intersecting lines, as these are not supported in TextMesh.");
		}
	}

	for (List<TPPLPoly>::Element *I = out_tris.front(); I; I = I->next()) {
		TPPLPoly &tp = I->get();
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

	// Update text buffer.
	if (dirty_text) {
		TS->shaped_text_clear(text_rid);
		TS->shaped_text_set_direction(text_rid, text_direction);

		String text = (uppercase) ? TS->string_to_upper(xl_text, language) : xl_text;
		TS->shaped_text_add_string(text_rid, text, font->get_rids(), font_size, opentype_features, language);

		Array stt;
		if (st_parser == TextServer::STRUCTURED_TEXT_CUSTOM) {
			GDVIRTUAL_CALL(_structured_text_parser, st_args, text, stt);
		} else {
			stt = TS->parse_structured_text(st_parser, st_args, text);
		}
		TS->shaped_text_set_bidi_override(text_rid, stt);

		dirty_text = false;
		dirty_font = false;
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			TS->shaped_text_fit_to_width(text_rid, width, TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA);
		}
	} else if (dirty_font) {
		int spans = TS->shaped_get_span_count(text_rid);
		for (int i = 0; i < spans; i++) {
			TS->shaped_set_span_update_font(text_rid, i, font->get_rids(), font_size, opentype_features);
		}

		dirty_font = false;
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			TS->shaped_text_fit_to_width(text_rid, width, TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA);
		}
	}

	Vector2 offset;
	const Glyph *glyphs = TS->shaped_text_get_glyphs(text_rid);
	int gl_size = TS->shaped_text_get_glyph_count(text_rid);
	float line_width = TS->shaped_text_get_width(text_rid) * pixel_size;

	switch (horizontal_alignment) {
		case HORIZONTAL_ALIGNMENT_LEFT:
			offset.x = 0.0;
			break;
		case HORIZONTAL_ALIGNMENT_FILL:
		case HORIZONTAL_ALIGNMENT_CENTER: {
			offset.x = -line_width / 2.0;
		} break;
		case HORIZONTAL_ALIGNMENT_RIGHT: {
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
	for (int i = 0; i < gl_size; i++) {
		if (glyphs[i].index == 0) {
			offset.x += glyphs[i].advance * pixel_size * glyphs[i].repeat;
			continue;
		}
		if (glyphs[i].font_rid != RID()) {
			uint32_t hash = hash_one_uint64(glyphs[i].font_rid.get_id());
			hash = hash_murmur3_one_32(glyphs[i].index, hash);

			_generate_glyph_mesh_data(hash, glyphs[i]);
			GlyphMeshData &gl_data = cache[hash];

			p_size += glyphs[i].repeat * gl_data.triangles.size() * ((has_depth) ? 2 : 1);
			i_size += glyphs[i].repeat * gl_data.triangles.size() * ((has_depth) ? 2 : 1);

			if (has_depth) {
				for (int j = 0; j < gl_data.contours.size(); j++) {
					p_size += glyphs[i].repeat * gl_data.contours[j].size() * 4;
					i_size += glyphs[i].repeat * gl_data.contours[j].size() * 6;
				}
			}

			for (int j = 0; j < glyphs[i].repeat; j++) {
				min_p.x = MIN(gl_data.min_p.x + offset_pre.x, min_p.x);
				min_p.y = MIN(gl_data.min_p.y + offset_pre.y, min_p.y);
				max_p.x = MAX(gl_data.max_p.x + offset_pre.x, max_p.x);
				max_p.y = MAX(gl_data.max_p.y + offset_pre.y, max_p.y);

				offset_pre.x += glyphs[i].advance * pixel_size;
			}
		} else {
			p_size += glyphs[i].repeat * 4;
			i_size += glyphs[i].repeat * 6;

			offset_pre.x += glyphs[i].advance * pixel_size * glyphs[i].repeat;
		}
	}

	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<float> tangents;
	Vector<Vector2> uvs;
	Vector<int32_t> indices;

	vertices.resize(p_size);
	normals.resize(p_size);
	uvs.resize(p_size);
	tangents.resize(p_size * 4);
	indices.resize(i_size);

	Vector3 *vertices_ptr = vertices.ptrw();
	Vector3 *normals_ptr = normals.ptrw();
	float *tangents_ptr = tangents.ptrw();
	Vector2 *uvs_ptr = uvs.ptrw();
	int32_t *indices_ptr = indices.ptrw();

	// Generate mesh.
	int32_t p_idx = 0;
	int32_t i_idx = 0;
	for (int i = 0; i < gl_size; i++) {
		if (glyphs[i].index == 0) {
			offset.x += glyphs[i].advance * pixel_size * glyphs[i].repeat;
			continue;
		}
		if (glyphs[i].font_rid != RID()) {
			uint32_t hash = hash_one_uint64(glyphs[i].font_rid.get_id());
			hash = hash_murmur3_one_32(glyphs[i].index, hash);

			const GlyphMeshData &gl_data = cache[hash];

			int64_t ts = gl_data.triangles.size();
			const Vector2 *ts_ptr = gl_data.triangles.ptr();

			for (int j = 0; j < glyphs[i].repeat; j++) {
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
				offset.x += glyphs[i].advance * pixel_size;
			}
		} else {
			// Add fallback quad for missing glyphs.
			for (int j = 0; j < glyphs[i].repeat; j++) {
				Size2 sz = TS->get_hex_code_box_size(glyphs[i].font_size, glyphs[i].index) * pixel_size;
				Vector3 quad_faces[4] = {
					Vector3(offset.x, offset.y, 0.0),
					Vector3(offset.x, sz.y + offset.y, 0.0),
					Vector3(sz.x + offset.x, sz.y + offset.y, 0.0),
					Vector3(sz.x + offset.x, offset.y, 0.0),
				};
				for (int k = 0; k < 4; k++) {
					vertices_ptr[p_idx + k] = quad_faces[k];
					normals_ptr[p_idx + k] = Vector3(0.0, 0.0, 1.0);
					if (has_depth) {
						uvs_ptr[p_idx + k] = Vector2(Math::range_lerp(quad_faces[k].x, min_p.x, max_p.x, real_t(0.0), real_t(1.0)), Math::range_lerp(quad_faces[k].y, -min_p.y, -max_p.y, real_t(0.0), real_t(0.4)));
					} else {
						uvs_ptr[p_idx + k] = Vector2(Math::range_lerp(quad_faces[k].x, min_p.x, max_p.x, real_t(0.0), real_t(1.0)), Math::range_lerp(quad_faces[k].y, -min_p.y, -max_p.y, real_t(0.0), real_t(1.0)));
					}
					tangents_ptr[(p_idx + k) * 4 + 0] = 1.0;
					tangents_ptr[(p_idx + k) * 4 + 1] = 0.0;
					tangents_ptr[(p_idx + k) * 4 + 2] = 0.0;
					tangents_ptr[(p_idx + k) * 4 + 3] = 1.0;
				}

				indices_ptr[i_idx++] = p_idx;
				indices_ptr[i_idx++] = p_idx + 1;
				indices_ptr[i_idx++] = p_idx + 2;

				indices_ptr[i_idx++] = p_idx + 0;
				indices_ptr[i_idx++] = p_idx + 2;
				indices_ptr[i_idx++] = p_idx + 3;
				p_idx += 4;

				offset.x += glyphs[i].advance * pixel_size;
			}
		}
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

	p_arr[RS::ARRAY_VERTEX] = vertices;
	p_arr[RS::ARRAY_NORMAL] = normals;
	p_arr[RS::ARRAY_TANGENT] = tangents;
	p_arr[RS::ARRAY_TEX_UV] = uvs;
	p_arr[RS::ARRAY_INDEX] = indices;
}

void TextMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &TextMesh::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &TextMesh::get_horizontal_alignment);

	ClassDB::bind_method(D_METHOD("set_text", "text"), &TextMesh::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &TextMesh::get_text);

	ClassDB::bind_method(D_METHOD("set_font", "font"), &TextMesh::set_font);
	ClassDB::bind_method(D_METHOD("get_font"), &TextMesh::get_font);

	ClassDB::bind_method(D_METHOD("set_font_size", "font_size"), &TextMesh::set_font_size);
	ClassDB::bind_method(D_METHOD("get_font_size"), &TextMesh::get_font_size);

	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &TextMesh::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &TextMesh::get_depth);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &TextMesh::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &TextMesh::get_width);

	ClassDB::bind_method(D_METHOD("set_pixel_size", "pixel_size"), &TextMesh::set_pixel_size);
	ClassDB::bind_method(D_METHOD("get_pixel_size"), &TextMesh::get_pixel_size);

	ClassDB::bind_method(D_METHOD("set_curve_step", "curve_step"), &TextMesh::set_curve_step);
	ClassDB::bind_method(D_METHOD("get_curve_step"), &TextMesh::get_curve_step);

	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &TextMesh::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &TextMesh::get_text_direction);

	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &TextMesh::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &TextMesh::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &TextMesh::clear_opentype_features);

	ClassDB::bind_method(D_METHOD("set_language", "language"), &TextMesh::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &TextMesh::get_language);

	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &TextMesh::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &TextMesh::get_structured_text_bidi_override);

	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &TextMesh::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &TextMesh::get_structured_text_bidi_override_options);

	ClassDB::bind_method(D_METHOD("set_uppercase", "enable"), &TextMesh::set_uppercase);
	ClassDB::bind_method(D_METHOD("is_uppercase"), &TextMesh::is_uppercase);

	ClassDB::bind_method(D_METHOD("_font_changed"), &TextMesh::_font_changed);
	ClassDB::bind_method(D_METHOD("_request_update"), &TextMesh::_request_update);

	ADD_GROUP("Text", "");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "font_size", PROPERTY_HINT_RANGE, "1,127,1,suffix:px"), "set_font_size", "get_font_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");

	ADD_GROUP("Mesh", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pixel_size", PROPERTY_HINT_RANGE, "0.0001,128,0.0001,suffix:m"), "set_pixel_size", "get_pixel_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "curve_step", PROPERTY_HINT_RANGE, "0.1,10,0.1,suffix:px"), "set_curve_step", "get_curve_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth", PROPERTY_HINT_RANGE, "0.0,100.0,0.001,or_greater,suffix:m"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width", PROPERTY_HINT_NONE, "suffix:m"), "set_width", "get_width");

	ADD_GROUP("Locale", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
}

void TextMesh::_notification(int p_what) {
	switch (p_what) {
		case MainLoop::NOTIFICATION_TRANSLATION_CHANGED: {
			String new_text = tr(text);
			if (new_text == xl_text) {
				return; // Nothing new.
			}
			xl_text = new_text;
			dirty_text = true;
			_request_update();
		} break;
	}
}

bool TextMesh::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		int value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				dirty_font = true;
				_request_update();
			}
		} else {
			if (!opentype_features.has(tag) || (int)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				dirty_font = true;
				_request_update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	return false;
}

bool TextMesh::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		if (opentype_features.has(tag)) {
			r_ret = opentype_features[tag];
			return true;
		} else {
			r_ret = -1;
			return true;
		}
	}
	return false;
}

void TextMesh::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::INT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

TextMesh::TextMesh() {
	primitive_type = PRIMITIVE_TRIANGLES;
	text_rid = TS->create_shaped_text();
}

TextMesh::~TextMesh() {
	TS->free_rid(text_rid);
}

void TextMesh::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (horizontal_alignment != p_alignment) {
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL || p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			dirty_text = true;
		}
		horizontal_alignment = p_alignment;
		_request_update();
	}
}

HorizontalAlignment TextMesh::get_horizontal_alignment() const {
	return horizontal_alignment;
}

void TextMesh::set_text(const String &p_string) {
	if (text != p_string) {
		text = p_string;
		xl_text = tr(text);
		dirty_text = true;
		_request_update();
	}
}

String TextMesh::get_text() const {
	return text;
}

void TextMesh::_font_changed() {
	dirty_font = true;
	dirty_cache = true;
	call_deferred(SNAME("_request_update"));
}

void TextMesh::set_font(const Ref<Font> &p_font) {
	if (font_override != p_font) {
		if (font_override.is_valid()) {
			font_override->disconnect(CoreStringNames::get_singleton()->changed, Callable(this, "_font_changed"));
		}
		font_override = p_font;
		dirty_font = true;
		dirty_cache = true;
		if (font_override.is_valid()) {
			font_override->connect(CoreStringNames::get_singleton()->changed, Callable(this, "_font_changed"));
		}
		_request_update();
	}
}

Ref<Font> TextMesh::get_font() const {
	return font_override;
}

Ref<Font> TextMesh::_get_font_or_default() const {
	if (font_override.is_valid() && font_override->get_data_count() > 0) {
		return font_override;
	}

	// Check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		List<StringName> theme_types;
		Theme::get_project_default()->get_type_dependencies(get_class_name(), StringName(), &theme_types);

		for (const StringName &E : theme_types) {
			if (Theme::get_project_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", E)) {
				return Theme::get_project_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", E);
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	{
		List<StringName> theme_types;
		Theme::get_default()->get_type_dependencies(get_class_name(), StringName(), &theme_types);

		for (const StringName &E : theme_types) {
			if (Theme::get_default()->has_theme_item(Theme::DATA_TYPE_FONT, "font", E)) {
				return Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", E);
			}
		}
	}

	// If they don't exist, use any type to return the default/empty value.
	return Theme::get_default()->get_theme_item(Theme::DATA_TYPE_FONT, "font", StringName());
}

void TextMesh::set_font_size(int p_size) {
	if (font_size != p_size) {
		font_size = CLAMP(p_size, 1, 127);
		dirty_font = true;
		dirty_cache = true;
		_request_update();
	}
}

int TextMesh::get_font_size() const {
	return font_size;
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

void TextMesh::set_width(real_t p_width) {
	if (width != p_width) {
		width = p_width;
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			dirty_text = true;
		}
		_request_update();
	}
}

real_t TextMesh::get_width() const {
	return width;
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

void TextMesh::set_text_direction(TextServer::Direction p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		dirty_text = true;
		_request_update();
	}
}

TextServer::Direction TextMesh::get_text_direction() const {
	return text_direction;
}

void TextMesh::clear_opentype_features() {
	opentype_features.clear();
	dirty_font = true;
	_request_update();
}

void TextMesh::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		dirty_font = true;
		_request_update();
	}
}

int TextMesh::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void TextMesh::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		dirty_text = true;
		_request_update();
	}
}

String TextMesh::get_language() const {
	return language;
}

void TextMesh::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		dirty_text = true;
		_request_update();
	}
}

TextServer::StructuredTextParser TextMesh::get_structured_text_bidi_override() const {
	return st_parser;
}

void TextMesh::set_structured_text_bidi_override_options(Array p_args) {
	if (st_args != p_args) {
		st_args = p_args;
		dirty_text = true;
		_request_update();
	}
}

Array TextMesh::get_structured_text_bidi_override_options() const {
	return st_args;
}

void TextMesh::set_uppercase(bool p_uppercase) {
	if (uppercase != p_uppercase) {
		uppercase = p_uppercase;
		dirty_text = true;
		_request_update();
	}
}

bool TextMesh::is_uppercase() const {
	return uppercase;
}

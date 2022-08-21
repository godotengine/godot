/*************************************************************************/
/*  navigation_geometry_parser_3d.cpp                                    */
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

#ifndef _3D_DISABLED

#include "navigation_geometry_parser_3d.h"

void NavigationGeometryParser3D::_add_vertex(const Vector3 &p_vec3) {
	if (total_parsed_vertices == nullptr || total_parsed_indices == nullptr) {
		return;
	}
	total_parsed_vertices->push_back(p_vec3.x);
	total_parsed_vertices->push_back(p_vec3.y);
	total_parsed_vertices->push_back(p_vec3.z);
}

void NavigationGeometryParser3D::_add_mesh(const Ref<Mesh> &p_mesh, const Transform3D &p_xform) {
	if (total_parsed_vertices == nullptr || total_parsed_indices == nullptr) {
		return;
	}
	int current_vertex_count;
	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		current_vertex_count = total_parsed_vertices->size() / 3;

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		int index_count = 0;
		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			index_count = p_mesh->surface_get_array_index_len(i);
		} else {
			index_count = p_mesh->surface_get_array_len(i);
		}

		ERR_CONTINUE((index_count == 0 || (index_count % 3) != 0));

		int face_count = index_count / 3;

		Array a = p_mesh->surface_get_arrays(i);

		Vector<Vector3> mesh_vertices = a[Mesh::ARRAY_VERTEX];
		const Vector3 *vr = mesh_vertices.ptr();

		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			Vector<int> mesh_indices = a[Mesh::ARRAY_INDEX];
			const int *ir = mesh_indices.ptr();

			for (int j = 0; j < mesh_vertices.size(); j++) {
				_add_vertex(p_xform.xform(vr[j]));
			}

			for (int j = 0; j < face_count; j++) {
				// CCW
				total_parsed_indices->push_back(current_vertex_count + (ir[j * 3 + 0]));
				total_parsed_indices->push_back(current_vertex_count + (ir[j * 3 + 2]));
				total_parsed_indices->push_back(current_vertex_count + (ir[j * 3 + 1]));
			}
		} else {
			face_count = mesh_vertices.size() / 3;
			for (int j = 0; j < face_count; j++) {
				_add_vertex(p_xform.xform(vr[j * 3 + 0]));
				_add_vertex(p_xform.xform(vr[j * 3 + 2]));
				_add_vertex(p_xform.xform(vr[j * 3 + 1]));

				total_parsed_indices->push_back(current_vertex_count + (j * 3 + 0));
				total_parsed_indices->push_back(current_vertex_count + (j * 3 + 1));
				total_parsed_indices->push_back(current_vertex_count + (j * 3 + 2));
			}
		}
	}
}

void NavigationGeometryParser3D::_add_mesh_array(const Array &p_array, const Transform3D &p_xform) {
	if (total_parsed_vertices == nullptr || total_parsed_indices == nullptr) {
		return;
	}
	Vector<Vector3> mesh_vertices = p_array[Mesh::ARRAY_VERTEX];
	const Vector3 *vr = mesh_vertices.ptr();

	Vector<int> mesh_indices = p_array[Mesh::ARRAY_INDEX];
	const int *ir = mesh_indices.ptr();

	const int face_count = mesh_indices.size() / 3;
	const int current_vertex_count = total_parsed_vertices->size() / 3;

	for (int j = 0; j < mesh_vertices.size(); j++) {
		_add_vertex(p_xform.xform(vr[j]));
	}

	for (int j = 0; j < face_count; j++) {
		// CCW
		total_parsed_indices->push_back(current_vertex_count + (ir[j * 3 + 0]));
		total_parsed_indices->push_back(current_vertex_count + (ir[j * 3 + 2]));
		total_parsed_indices->push_back(current_vertex_count + (ir[j * 3 + 1]));
	}
}

void NavigationGeometryParser3D::_add_faces(const PackedVector3Array &p_faces, const Transform3D &p_xform) {
	if (total_parsed_vertices == nullptr || total_parsed_indices == nullptr) {
		return;
	}
	int face_count = p_faces.size() / 3;
	int current_vertex_count = total_parsed_vertices->size() / 3;

	for (int j = 0; j < face_count; j++) {
		_add_vertex(p_xform.xform(p_faces[j * 3 + 0]));
		_add_vertex(p_xform.xform(p_faces[j * 3 + 1]));
		_add_vertex(p_xform.xform(p_faces[j * 3 + 2]));

		total_parsed_indices->push_back(current_vertex_count + (j * 3 + 0));
		total_parsed_indices->push_back(current_vertex_count + (j * 3 + 2));
		total_parsed_indices->push_back(current_vertex_count + (j * 3 + 1));
	}
}

bool NavigationGeometryParser3D::parses_node(Node *p_node) {
	bool parses_this_node;
	if (Object::cast_to<Node3D>(p_node) == nullptr) {
		parses_this_node = false;
		return parses_this_node;
	}
	if (GDVIRTUAL_CALL(_parses_node, p_node, parses_this_node)) {
		return parses_this_node;
	}

	return false;
}

void NavigationGeometryParser3D::parse_node_geometry(const Transform3D &p_navmesh_transform, Ref<NavigationMesh> p_navigationmesh, Node *p_node, Vector<float> &p_vertices, Vector<int> &p_indices) {
	// GDScript cannot use large vertices and indices vector array's passed by ref in a performant way without making slow copies
	// to work around this issue we store vector array's internally and only expose add functions to scripting users
	navmesh_transform = p_navmesh_transform;
	total_parsed_vertices = &p_vertices;
	total_parsed_indices = &p_indices;

	parse_geometry(p_node, p_navigationmesh);

	navmesh_transform = Transform3D();
	total_parsed_vertices = nullptr;
	total_parsed_indices = nullptr;
}

void NavigationGeometryParser3D::parse_geometry(Node *p_node, Ref<NavigationMesh> p_navigationmesh) {
	if (GDVIRTUAL_CALL(_parse_geometry, p_node, p_navigationmesh)) {
		return;
	}
}

void NavigationGeometryParser3D::add_mesh(const Ref<Mesh> &p_mesh, const Transform3D &p_xform) {
	_add_mesh(p_mesh, navmesh_transform * p_xform);
}

void NavigationGeometryParser3D::add_mesh_array(const Array &p_array, const Transform3D &p_xform) {
	_add_mesh_array(p_array, navmesh_transform * p_xform);
}

void NavigationGeometryParser3D::add_faces(const PackedVector3Array &p_faces, const Transform3D &p_xform) {
	_add_faces(p_faces, navmesh_transform * p_xform);
}

void NavigationGeometryParser3D::_bind_methods() {
	GDVIRTUAL_BIND(_parses_node, "node");
	GDVIRTUAL_BIND(_parse_geometry, "node", "navigationmesh")

	ClassDB::bind_method(D_METHOD("add_mesh", "mesh", "xform"), &NavigationGeometryParser3D::add_mesh);
	ClassDB::bind_method(D_METHOD("add_mesh_array", "mesh_array", "xform"), &NavigationGeometryParser3D::add_mesh_array);
	ClassDB::bind_method(D_METHOD("add_faces", "faces", "xform"), &NavigationGeometryParser3D::add_faces);
}

#endif // _3D_DISABLED
